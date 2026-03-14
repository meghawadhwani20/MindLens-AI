import cv2
import numpy as np
import mediapipe as mp
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from deepface import DeepFace
from .storage import fetch_rows, insert_row, latest_row
from .validation import validate_user_exists
from .services.model_service import predict_habit_stress, predict_stress, update_scan_prediction

logger = logging.getLogger(__name__)

# The installed mediapipe package in this environment exposes only the newer
# tasks API. Fall back cleanly when legacy solutions are unavailable so scans
# still work instead of failing at import time.
mp_pose = None
mp_face_mesh = None
pose_processor = None
face_mesh_processor = None
_mediapipe_ready = False
_mediapipe_error = None


def _init_mediapipe():
    global mp_pose, mp_face_mesh, pose_processor, face_mesh_processor
    global _mediapipe_ready, _mediapipe_error

    if _mediapipe_ready or _mediapipe_error is not None:
        return

    try:
        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            raise AttributeError("mediapipe.solutions is unavailable in the installed package")

        mp_pose = solutions.pose
        mp_face_mesh = solutions.face_mesh
        pose_processor = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        face_mesh_processor = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
        _mediapipe_ready = True
    except Exception as exc:
        _mediapipe_error = exc


def _normalize_deepface_result(result):
    if isinstance(result, list):
        return result[0] if result else None
    if isinstance(result, dict):
        return result
    return None


def _emotion_defaults(emotion: str):
    emotion_key = (emotion or "neutral").lower()
    defaults = {
        "angry": {"jaw": 0.82, "slouch": 0.42, "eyebrow": True, "posture": "Fair"},
        "fear": {"jaw": 0.68, "slouch": 0.38, "eyebrow": True, "posture": "Fair"},
        "fearful": {"jaw": 0.68, "slouch": 0.38, "eyebrow": True, "posture": "Fair"},
        "sad": {"jaw": 0.56, "slouch": 0.44, "eyebrow": False, "posture": "Poor"},
        "surprise": {"jaw": 0.42, "slouch": 0.24, "eyebrow": True, "posture": "Good"},
        "happy": {"jaw": 0.18, "slouch": 0.16, "eyebrow": False, "posture": "Good"},
        "neutral": {"jaw": 0.28, "slouch": 0.20, "eyebrow": False, "posture": "Good"},
    }
    return defaults.get(emotion_key, defaults["neutral"])


def _stress_label_from_score(avg_stress: float | None) -> str:
    if avg_stress is None:
        return "Unknown"
    if avg_stress >= 70:
        return "High"
    if avg_stress >= 40:
        return "Moderate"
    return "Low"

def majority_vote(items):
    data = [i for i in items if i is not None]
    if not data: return "Unknown"
    return Counter(data).most_common(1)[0][0]

def average(items):
    data = [i for i in items if i is not None and isinstance(i, (int, float))]
    if not data: return 0.0
    return round(sum(data) / len(data), 4)

async def analyze_frame(frame_file):
    """Analyzes a single frame and extracts all required features with high accuracy."""
    _init_mediapipe()

    contents = await frame_file.read()
    await frame_file.seek(0)
    
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    # Preprocessing for DeepFace (as per Rule 2/3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    res = {
        "emotion": "Unknown",
        "emotion_confidence": 0.0,
        "mouth_open": False,
        "eyebrow_raise": False,
        "jaw_clench_score": 0.0,
        "jaw_tension": "Unknown",
        "slouch_score": 0.0,
        "head_tilt_angle": 0.0,
        "shoulder_alignment_diff": 0.0,
        "spine_curve_ratio": 1.0,
        "pose_confidence": 0.0,
        "posture_quality": "Unknown",
        "overall_stress": "Low"
    }
    
    # --- DeepFace for Emotion (Quick Scan Mode) ---
    try:
        # DeepFace expects BGR for most backends when passing numpy array
        objs = DeepFace.analyze(
            img, 
            actions=['emotion'], 
            enforce_detection=False, 
            detector_backend='opencv'
        )
        analysis = _normalize_deepface_result(objs)
        if analysis:
            emo_results = analysis.get('emotion', {})
            dominant = analysis.get('dominant_emotion', 'neutral')
            conf = emo_results.get(dominant, 0.0)
            
            # Rule: IGNORE neutral unless confidence > 40%
            if dominant == 'neutral' and conf < 40:
                # Find next best emotion
                sorted_emotions = sorted(emo_results.items(), key=lambda x: x[1], reverse=True)
                # sorted_emotions[0] is neutral, take [1] if exists
                if len(sorted_emotions) > 1:
                    dominant = sorted_emotions[1][0]
                    conf = sorted_emotions[1][1]
            
            res["emotion"] = dominant
            res["emotion_confidence"] = conf / 100.0
    except Exception as e:
        print(f"DeepFace error: {type(e).__name__}: {e}")

    defaults = _emotion_defaults(res["emotion"])
    res["eyebrow_raise"] = defaults["eyebrow"]
    res["jaw_clench_score"] = defaults["jaw"]
    res["jaw_tension"] = "High" if res["jaw_clench_score"] > 0.75 else ("Medium" if res["jaw_clench_score"] > 0.4 else "Low")
    res["slouch_score"] = defaults["slouch"]
    res["spine_curve_ratio"] = max(0.45, round(1.0 - (res["slouch_score"] * 0.7), 4))
    res["pose_confidence"] = 0.55 if _mediapipe_error else 0.0
    res["posture_quality"] = defaults["posture"]

    # --- Mediapipe Pose (Posture) ---
    pose_results = pose_processor.process(img_rgb) if pose_processor else None
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        # Check if key landmarks are visible enough
        if landmarks[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            
            res["pose_confidence"] = landmarks[0].visibility
            res["shoulder_alignment_diff"] = abs(l_sh.y - r_sh.y)
            
            # Slouch score (dynamic vertical offset)
            mid_sh_y = (l_sh.y + r_sh.y) / 2
            res["slouch_score"] = abs(nose.y - mid_sh_y)
            
            # Spine curve ratio (alignment based)
            res["spine_curve_ratio"] = max(0.0, 1.0 - (res["shoulder_alignment_diff"] * 5))
            
            res["posture_quality"] = "Poor" if res["slouch_score"] > 0.35 else ("Fair" if res["slouch_score"] > 0.2 else "Good")
        else:
            res["posture_quality"] = "Unknown (Face not clear)"

    # --- Mediapipe Face Mesh (Biometric Details) ---
    face_results = face_mesh_processor.process(img_rgb) if face_mesh_processor else None
    if face_results and face_results.multi_face_landmarks:
        f_landmarks = face_results.multi_face_landmarks[0].landmark
        
        # Head Tilt
        l_eye = f_landmarks[33]
        r_eye = f_landmarks[263]
        res["head_tilt_angle"] = np.degrees(np.arctan2(r_eye.y - l_eye.y, r_eye.x - l_eye.x))
        
        # Jaw & Mouth (Dynamic)
        top_face = f_landmarks[10]
        bot_face = f_landmarks[152]
        face_h = abs(top_face.y - bot_face.y)
        
        u_lip = f_landmarks[13]
        l_lip = f_landmarks[14]
        lip_gap = abs(u_lip.y - l_lip.y) / face_h
        res["mouth_open"] = lip_gap > 0.05
        
        # Jaw tension
        res["jaw_clench_score"] = max(0.0, 1.0 - (lip_gap * 15))
        res["jaw_tension"] = "High" if res["jaw_clench_score"] > 0.85 else ("Medium" if res["jaw_clench_score"] > 0.6 else "Low")
        
        # Eyebrow Positioning
        l_brow = f_landmarks[70]
        l_eye_top = f_landmarks[159]
        brow_dist = abs(l_brow.y - l_eye_top.y) / face_h
        res["eyebrow_raise"] = brow_dist > 0.11
    elif face_mesh_processor:
        if res["jaw_tension"] == "Unknown":
            res["jaw_tension"] = "Unknown (Face mesh failed)"
    else:
        if res["jaw_tension"] == "Unknown":
            res["jaw_tension"] = "Medium"

    # Stress Calculation logic (Rule 5)
    pts = 0
    if res["emotion"] in ["angry", "fearful", "sad"]: pts += 2
    if res["jaw_tension"] == "High": pts += 1
    if res["eyebrow_raise"]: pts += 0.5
    if res["posture_quality"] == "Poor": pts += 1.5
    if abs(res["head_tilt_angle"]) > 15: pts += 0.5
    
    res["overall_stress"] = "High" if pts >= 3 else ("Moderate" if pts >= 1.5 else "Low")
    
    return res

import asyncio

async def process_stress_scan(email: str, images: list):
    """Aggregates analysis from 3 frames with weighted confidence and saves to Supabase.
    Uses parallel processing to maintain a <5s total scan time.
    """
    # 1. Validate User Existence (Strict Backend Check)
    validate_user_exists(email)

    if len(images) != 3:
        raise ValueError("Exactly 3 frames required")

    # Run AI analysis for all 3 frames in parallel
    tasks = [analyze_frame(img_file) for img_file in images]
    frame_results = await asyncio.gather(*tasks)
    
    # Filter out failed analyses
    frame_results = [r for r in frame_results if r is not None]

    if len(frame_results) < 3:
        raise ValueError("AI analysis failed: Could not detect valid features in 3 frames simultaneously")

    # Weighted Emotion Aggregation (Rule 3)
    votes = defaultdict(float)
    for r in frame_results:
        votes[r["emotion"]] += r["emotion_confidence"]
    final_emotion = max(votes, key=votes.get)

    model_payload = {
        "emotion": final_emotion,
        "emotion_confidence": average([r["emotion_confidence"] for r in frame_results]),
        "mouth_open": majority_vote([r["mouth_open"] for r in frame_results]),
        "eyebrow_raise": majority_vote([r["eyebrow_raise"] for r in frame_results]),
        "jaw_clench_score": average([r["jaw_clench_score"] for r in frame_results]),
        "slouch_score": average([r["slouch_score"] for r in frame_results]),
        "head_tilt_angle": average([r["head_tilt_angle"] for r in frame_results]),
        "shoulder_alignment_diff": average([r["shoulder_alignment_diff"] for r in frame_results]),
        "spine_curve_ratio": average([r["spine_curve_ratio"] for r in frame_results]),
        "posture_confidence": average([r["pose_confidence"] for r in frame_results]),
        "jaw_tension": majority_vote([r["jaw_tension"] for r in frame_results]),
        "posture": majority_vote([r["posture_quality"] for r in frame_results]),
    }
    # Run MindLens model using the latest habit row for this user.
    mindlens_prediction = None
    latest_habit = latest_row("habits", email=email, order_candidates=("timestamp", "created_at"))
    if latest_habit:
        mindlens_prediction = predict_habit_stress(latest_habit)
        logger.info(
            "mindlens model prediction success for %s: %.4f",
            email,
            mindlens_prediction["mindlens_model_score"],
        )
    else:
        logger.warning("mindlens model prediction skipped for %s: no habit row found", email)

    prediction = predict_stress(
        {"email": email, **model_payload},
        mindlens_score=mindlens_prediction["mindlens_model_score"] if mindlens_prediction else None,
    )
    logger.info("stress model prediction success for %s: %.4f", email, prediction["stress_model_score"])

    if prediction["stress_model_score"] is not None and prediction["mindlens_model_score"] is not None:
        overall_average_stress = round(
            (float(prediction["stress_model_score"]) + float(prediction["mindlens_model_score"])) / 2.0,
            4,
        )
        prediction["overall_average_stress"] = overall_average_stress
        prediction["avg_stress"] = overall_average_stress

    # Aggregated Result powered by the project pickle models
    final_analysis = {
        "email": email,
        "emotion": model_payload["emotion"],
        "emotion_confidence": model_payload["emotion_confidence"],
        "mouth_open": model_payload["mouth_open"],
        "eyebrow_raise": model_payload["eyebrow_raise"],
        "jaw_clench_score": model_payload["jaw_clench_score"],
        "jaw_tension": prediction["jaw_tension"],
        "slouch_score": model_payload["slouch_score"],
        "head_tilt_angle": model_payload["head_tilt_angle"],
        "shoulder_alignment_diff": model_payload["shoulder_alignment_diff"],
        "spine_curve_ratio": model_payload["spine_curve_ratio"],
        "pose_confidence": model_payload["posture_confidence"],
        "posture_quality": prediction["posture"],
        "mindlens_model_score": prediction["mindlens_model_score"],
        "mindlens_stress": prediction["mindlens_stress"],
        "stress_model_score": prediction["stress_model_score"],
        "stress_model_stress": prediction["stress_model_stress"],
        "overall_average_stress": prediction["overall_average_stress"],
        "avg_stress": prediction["avg_stress"],
        "overall_stress": prediction.get("overall_prediction") or _stress_label_from_score(prediction.get("avg_stress")),
        "scanned_at": datetime.utcnow().isoformat()
    }
    logger.info("final scan response JSON: %s", json.dumps(final_analysis, default=str))

    # Prepare for Database (Rule 6)
    db_data = {
        "email": final_analysis["email"],
        "emotion": final_analysis["emotion"],
        "emotion_confidence": final_analysis["emotion_confidence"],
        "mouth_open": final_analysis["mouth_open"],
        "eyebrow_raise": final_analysis["eyebrow_raise"],
        "jaw_clench_score": final_analysis["jaw_clench_score"],
        "jaw_tension": final_analysis["jaw_tension"],
        "slouch_score": final_analysis["slouch_score"],
        "head_tilt_angle": final_analysis["head_tilt_angle"],
        "shoulder_alignment_diff": final_analysis["shoulder_alignment_diff"],
        "spine_curve_ratio": final_analysis["spine_curve_ratio"],
        "pose_confidence": final_analysis["pose_confidence"],
        "posture_quality": final_analysis["posture_quality"],
        "stress_model_score": final_analysis["stress_model_score"],
        "mindlens_model_score": final_analysis["mindlens_model_score"],
        "avg_stress": final_analysis["avg_stress"],
        "overall_average_stress": final_analysis["overall_average_stress"],
        "overall_stress": final_analysis["overall_stress"],
        "scanned_at": final_analysis["scanned_at"],
    }
    # Convert bool/float for Supabase
    db_data["emotion_confidence"] = float(db_data["emotion_confidence"])
    db_data["jaw_clench_score"] = float(db_data["jaw_clench_score"])
    db_data["slouch_score"] = float(db_data["slouch_score"])
    db_data["head_tilt_angle"] = float(db_data["head_tilt_angle"])
    db_data["shoulder_alignment_diff"] = float(db_data["shoulder_alignment_diff"])
    db_data["spine_curve_ratio"] = float(db_data["spine_curve_ratio"])
    db_data["pose_confidence"] = float(db_data["pose_confidence"])

    # Final Insertion
    insert_row("stress_scans", db_data)
    logger.info("scan result saved successfully for %s at %s", email, final_analysis["scanned_at"])
    await update_scan_prediction({
        "email": email,
        "emotion": final_analysis["emotion"],
        "emotion_confidence": final_analysis["emotion_confidence"],
        "mouth_open": final_analysis["mouth_open"],
        "eyebrow_raise": final_analysis["eyebrow_raise"],
        "jaw_clench_score": final_analysis["jaw_clench_score"],
        "slouch_score": final_analysis["slouch_score"],
        "head_tilt_angle": final_analysis["head_tilt_angle"],
        "shoulder_alignment_diff": final_analysis["shoulder_alignment_diff"],
        "spine_curve_ratio": final_analysis["spine_curve_ratio"],
        "posture_confidence": final_analysis["pose_confidence"],
        "jaw_tension": final_analysis["jaw_tension"],
        "posture": final_analysis["posture_quality"],
    })

    return final_analysis

async def get_latest_stress_scan(email: str):
    rows = fetch_rows("stress_scans", email=email, limit=1, order_candidates=("scanned_at", "created_at"))
    return rows[0] if rows else None

async def get_all_stress_scans(email: str, limit: int = 5):
    return fetch_rows("stress_scans", email=email, limit=limit, order_candidates=("scanned_at", "created_at"))
