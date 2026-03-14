import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..storage import fetch_rows, latest_row


logger = logging.getLogger(__name__)

STRESS_FEATURE_KEYS = [
    "emotion_confidence",
    "mouth_open",
    "eyebrow_raise",
    "jaw_clench_score",
    "slouch_score",
    "head_tilt_angle",
    "shoulder_alignment_diff",
    "spine_curve_ratio",
    "posture_confidence",
]

HABIT_FEATURE_KEYS = [
    "age",
    "sleep_hours",
    "work_hours",
    "screen_time",
    "water_intake",
    "exercise",
    "meals_per_day",
    "caffeine_intake",
]

EMOTION_LABELS = {
    "a": "Angry",
    "angry": "Angry",
    "anxious": "Anxious",
    "fear": "Fearful",
    "fearful": "Fearful",
    "h": "Happy",
    "happy": "Happy",
    "n": "Neutral",
    "neutral": "Neutral",
    "sad": "Sad",
    "s": "Sad",
    "stressed": "Stressed",
    "surprise": "Surprised",
    "surprised": "Surprised",
    "t": "Tired",
    "tired": "Tired",
}

POSTURE_WORDS = {
    "upright": "Upright",
    "slouched": "Slouched",
    "tilted": "Tilted",
    "aligned": "Aligned",
    "forward-leaning": "Forward-leaning",
    "slightly hunched": "Slightly hunched",
}

_MODELS: Dict[str, Any] = {}
_MODEL_LOCK = asyncio.Lock()
_SUBSCRIBERS: set[asyncio.Queue] = set()
_LATEST_EVENT: Optional[Dict[str, Any]] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_paths(filename: str) -> Iterable[Path]:
    root = _project_root()
    yield root / filename
    yield root / "models" / filename
    yield root / "Models" / filename


def _resolve_model_path(filename: str) -> Path:
    for path in _candidate_paths(filename):
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in project root, models/, or Models/")


def _set_single_thread_runtime(model: Any) -> None:
    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass

    for estimator in getattr(model, "estimators_", []):
        _set_single_thread_runtime(estimator)

    for _, step_model in getattr(model, "steps", []):
        _set_single_thread_runtime(step_model)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "high"} else 0
    return 1 if float(value) > 0 else 0


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _average(values: list[Optional[float]]) -> Optional[float]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return round(sum(cleaned) / len(cleaned), 4)


def _overall_average_stress(stress_model_score: Optional[float], mindlens_model_score: Optional[float]) -> Optional[float]:
    if stress_model_score is None or mindlens_model_score is None:
        return None
    return round((float(stress_model_score) + float(mindlens_model_score)) / 2.0, 4)


def _normalize_emotion(value: Any) -> str:
    raw = str(value or "Neutral").strip().lower()
    return EMOTION_LABELS.get(raw, raw.title() if raw else "Neutral")


def _sanitize_stress_payload(data: Dict[str, Any]) -> Dict[str, float]:
    sanitized = {}
    for key in STRESS_FEATURE_KEYS:
        raw = data.get(key)
        if key in {"mouth_open", "eyebrow_raise"}:
            sanitized[key] = float(_coerce_bool(raw))
        else:
            sanitized[key] = _coerce_float(raw)
    return sanitized


def _sanitize_habit_payload(data: Dict[str, Any]) -> Dict[str, float]:
    return {
        "age": _coerce_float(data.get("age"), 30.0),
        "sleep_hours": _coerce_float(data.get("sleep_hours"), 7.0),
        "work_hours": _coerce_float(data.get("work_hours"), 8.0),
        "screen_time": _coerce_float(data.get("screen_time"), 5.0),
        "water_intake": _coerce_float(data.get("water_intake"), 2.0),
        "exercise": float(_coerce_bool(data.get("exercise"))),
        "meals_per_day": _coerce_float(data.get("meals_per_day"), 3.0),
        "caffeine_intake": float(_coerce_bool(data.get("caffeine_intake"))),
    }


def _derive_posture_descriptor(features: Dict[str, float]) -> str:
    slouch = features["slouch_score"]
    tilt = abs(features["head_tilt_angle"])
    shoulders = features["shoulder_alignment_diff"]
    spine = features["spine_curve_ratio"]
    confidence = features["posture_confidence"]

    if tilt >= 16:
        return "Tilted"
    if slouch >= 0.38:
        return "Slouched"
    if slouch >= 0.28:
        return "Slightly hunched"
    if slouch >= 0.2:
        return "Forward-leaning"
    if confidence >= 0.45 and spine >= 0.85 and shoulders <= 0.08:
        return "Aligned"
    return "Upright"


def _normalize_posture(value: Any, features: Optional[Dict[str, float]] = None) -> str:
    raw = str(value or "").strip().lower()
    if raw in POSTURE_WORDS:
        return POSTURE_WORDS[raw]
    if raw in {"good", "ok", "okay", "normal"}:
        return "Aligned" if features and features.get("posture_confidence", 0.0) >= 0.45 else "Upright"
    if raw == "fair":
        return "Forward-leaning"
    if raw == "poor":
        return "Slouched"
    if features:
        return _derive_posture_descriptor(features)
    return "Upright"


def _derive_jaw_tension(features: Dict[str, float]) -> str:
    jaw_score = features["jaw_clench_score"]
    if jaw_score >= 0.7:
        return "High"
    if jaw_score >= 0.35:
        return "Medium"
    return "Low"


def _build_stress_model_frame(features: Dict[str, float], data: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "emotion": _normalize_emotion(data.get("emotion")).lower(),
        "emotion_confidence": features["emotion_confidence"],
        "mouth_open": features["mouth_open"],
        "eyebrow_raise": features["eyebrow_raise"],
        "jaw_clench_score": features["jaw_clench_score"],
        "slouch_score": features["slouch_score"],
        "head_tilt_angle": features["head_tilt_angle"],
        "shoulder_alignment_diff": features["shoulder_alignment_diff"],
        "spine_curve_ratio": features["spine_curve_ratio"],
        "pose_confidence": features["posture_confidence"],
        "posture_quality": _normalize_posture(data.get("posture") or data.get("posture_quality"), features).lower(),
        "jaw_tension": str(data.get("jaw_tension") or _derive_jaw_tension(features)).lower(),
        "Unnamed: 14": 0.0,
    }
    return pd.DataFrame([row])


def _build_mindlens_model_frame(habits: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([habits])


def _prediction_to_score(prediction: Any) -> float:
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Model returned no prediction values")

    scaled_values = []
    for value in values:
        if not np.isfinite(value):
            continue
        if -1.0 <= value <= 1.0:
            scaled = max(0.0, min(100.0, value * 100.0 if value >= 0 else ((value + 1.0) / 2.0) * 100.0))
        elif 0.0 <= value <= 10.0:
            scaled = value * 10.0
        else:
            scaled = max(0.0, min(100.0, value))
        scaled_values.append(scaled)

    if not scaled_values:
        raise ValueError("Model returned only non-finite prediction values")

    return round(float(sum(scaled_values) / len(scaled_values)), 4)


def _stress_label(score: Optional[float]) -> str:
    if score is None:
        return "Unknown"
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def _status_from_range(value: float, good_min: float, good_max: float, warn_min: float, warn_max: float) -> str:
    if good_min <= value <= good_max:
        return "Normal"
    if warn_min <= value <= warn_max:
        return "Monitor"
    return "Alert"


def _derive_health_metrics(sensor: Optional[Dict[str, Any]], habit: Optional[Dict[str, Any]], avg_stress: Optional[float]) -> Dict[str, Any]:
    habits = habit or {}
    sensors = sensor or {}

    heart_rate = _coerce_float(sensors.get("heart_rate"), 72.0)
    spo2 = _coerce_float(sensors.get("spo2"), 98.0)
    temperature = _coerce_float(sensors.get("temperature"), 36.9)
    sleep_hours = _coerce_float(habits.get("sleep_hours"), 7.0)
    screen_time = _coerce_float(habits.get("screen_time"), 5.0)
    water_intake = _coerce_float(habits.get("water_intake"), 2.0)
    exercise = _coerce_bool(habits.get("exercise"))
    caffeine = _coerce_bool(habits.get("caffeine_intake"))
    meals = _coerce_float(habits.get("meals_per_day"), 3.0)
    stress = avg_stress or 0.0

    systolic = round(108 + ((heart_rate - 70) * 0.45) + (stress * 0.12))
    diastolic = round(72 + ((heart_rate - 70) * 0.2) + (stress * 0.06))
    cholesterol = round(165 + (screen_time * 3.5) - (exercise * 10) + (caffeine * 8))
    glucose = round(88 + ((7 - min(sleep_hours, 7)) * 4) + (stress * 0.08) + max(0, meals - 3) * 2)
    insulin = round(max(4.5, 6.5 + ((glucose - 90) * 0.05) - (exercise * 0.8)), 1)
    mental_health = round(max(0.0, min(10.0, 9.0 - (stress / 18.0) + ((sleep_hours - 7) * 0.2) + (exercise * 0.4))), 1)

    return {
        "heart_rate": round(heart_rate, 1),
        "spo2": round(spo2, 1),
        "temperature": round(temperature, 1),
        "blood_pressure": f"{systolic}/{diastolic}",
        "bp_badge": _status_from_range(systolic, 90, 120, 121, 135),
        "bp_category": "Optimal" if systolic <= 120 else ("Elevated" if systolic <= 135 else "High"),
        "hr_badge": _status_from_range(heart_rate, 60, 100, 50, 110),
        "hr_category": "Resting" if heart_rate < 85 else "Elevated",
        "cholesterol": cholesterol,
        "chol_badge": "Normal" if cholesterol < 200 else ("Monitor" if cholesterol < 240 else "Alert"),
        "chol_category": "Healthy" if cholesterol < 200 else ("Borderline High" if cholesterol < 240 else "High"),
        "glucose": glucose,
        "glucose_badge": "Normal" if glucose < 100 else ("Monitor" if glucose < 126 else "Alert"),
        "glucose_category": "Fasting" if glucose < 100 else ("Prediabetic" if glucose < 126 else "High"),
        "insulin": insulin,
        "insulin_badge": "Normal" if insulin <= 10 else ("Monitor" if insulin <= 15 else "Alert"),
        "insulin_category": "Healthy" if insulin <= 10 else ("Watch" if insulin <= 15 else "High"),
        "mental_health": mental_health,
        "mental_health_badge": "Good" if mental_health >= 7 else ("Monitor" if mental_health >= 5 else "Low"),
        "mental_health_category": "Balanced" if mental_health >= 7 else ("Needs attention" if mental_health >= 5 else "Strained"),
    }


def _variant(options: list[str], seed: str) -> str:
    return options[sum(ord(ch) for ch in seed) % len(options)]


def _generate_recommendations(
    latest_scan: Optional[Dict[str, Any]],
    latest_habit: Optional[Dict[str, Any]],
    overall_average_stress: Optional[float],
    email: Optional[str],
) -> list[str]:
    seed = f"{email or ''}:{latest_scan.get('timestamp') if latest_scan else ''}:{latest_habit.get('timestamp') if latest_habit else ''}"
    recommendations: list[str] = []
    habits = latest_habit or {}
    posture = str((latest_scan or {}).get("posture") or "")
    emotion = str((latest_scan or {}).get("emotion") or "")
    stress = overall_average_stress or 0.0

    if stress >= 70:
        recommendations.append(
            _variant(
                [
                    "Stress is high. Take two short breaks this hour and slow your breathing for five minutes.",
                    "High stress detected. Step away briefly, relax your shoulders, and reset before the next task.",
                ],
                seed + "stress-high",
            )
        )
    elif stress >= 40:
        recommendations.append(
            _variant(
                [
                    "Stress looks moderate. Add a short walk or stretch break before your next work block.",
                    "Moderate stress detected. Reduce task switching for the next hour and take one intentional pause.",
                ],
                seed + "stress-medium",
            )
        )
    else:
        recommendations.append(
            _variant(
                [
                    "Stress is currently low. Keep the same routine and protect your recovery habits.",
                    "Stress looks steady. Maintain your current pace and keep taking short resets.",
                ],
                seed + "stress-low",
            )
        )

    sleep_hours = _coerce_float(habits.get("sleep_hours"), 0.0)
    if sleep_hours and sleep_hours < 7:
        recommendations.append(
            _variant(
                [
                    "Sleep more tonight and aim for at least 7 to 8 hours of rest.",
                    "Your sleep is below target. Shift bedtime earlier to recover at least one extra hour.",
                ],
                seed + "sleep",
            )
        )

    if not _coerce_bool(habits.get("exercise")):
        recommendations.append(
            _variant(
                [
                    "Exercise at least 40 minutes today to reduce tension and improve recovery.",
                    "Add a 40-minute walk, jog, or workout session to bring stress down.",
                ],
                seed + "exercise",
            )
        )

    water_intake = _coerce_float(habits.get("water_intake"), 0.0)
    if water_intake and water_intake < 2.5:
        recommendations.append(
            _variant(
                [
                    "Drink more water over the next few hours and target at least 2.5 liters today.",
                    "Hydration is low. Keep water nearby and finish another bottle before evening.",
                ],
                seed + "water",
            )
        )

    if _coerce_float(habits.get("screen_time"), 0.0) >= 7:
        recommendations.append(
            _variant(
                [
                    "Reduce screen time this evening and give your eyes a device-free break.",
                    "Screen time is elevated. Break it up with off-screen pauses every 45 to 60 minutes.",
                ],
                seed + "screen",
            )
        )

    if posture in {"Slouched", "Slightly hunched", "Forward-leaning", "Tilted"}:
        recommendations.append(
            _variant(
                [
                    f"Improve posture while sitting. Your latest scan suggests a {posture.lower()} position.",
                    f"Reset your sitting posture. The latest scan shows a {posture.lower()} alignment pattern.",
                ],
                seed + "posture",
            )
        )

    if str((latest_scan or {}).get("jaw_tension") or "").lower() == "high":
        recommendations.append(
            _variant(
                [
                    "Unclench your jaw and relax your face for a minute every time you check your posture.",
                    "Jaw tension is high. Add a brief facial relaxation reset and soften your bite.",
                ],
                seed + "jaw",
            )
        )

    social = str(habits.get("social_interaction") or "").lower()
    if social in {"", "none", "low"} and stress >= 40:
        recommendations.append(
            _variant(
                [
                    "Reach out to someone you trust today. A short conversation can help regulate stress.",
                    "Low social interaction plus higher stress can compound fatigue. Check in with a friend or family member.",
                ],
                seed + "social",
            )
        )

    if _coerce_bool(habits.get("caffeine_intake")) and (stress >= 40 or emotion in {"Anxious", "Fearful", "Stressed"}):
        recommendations.append(
            _variant(
                [
                    "Reduce caffeine later in the day so it does not amplify stress or disrupt sleep.",
                    "Caffeine may be adding to tension. Keep the rest of today low-caffeine if possible.",
                ],
                seed + "caffeine",
            )
        )

    if len(recommendations) < 3:
        recommendations.extend(
            [
                "Take short breaks between tasks instead of pushing through long uninterrupted blocks.",
                "Use a two-minute posture reset before your next work session.",
                "Keep a consistent sleep and hydration routine for the next 24 hours.",
            ]
        )

    deduped: list[str] = []
    for item in recommendations:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _model_is_fitted(model: Any) -> bool:
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False
    except Exception:
        return bool(getattr(model, "feature_names_in_", None) is not None or getattr(model, "estimators_", None))


def _log_fit_status(name: str, model: Any) -> None:
    if _model_is_fitted(model):
        logger.info("Model fit skipped because %s is already fitted.", name)
    else:
        logger.warning("Model %s is not fitted. Historical fit skipped because no safe supervised labels are available.", name)


def _extract_numeric_target(row: Dict[str, Any]) -> Optional[float]:
    for key in ("stress_model_score", "mindlens_model_score", "overall_average_stress", "avg_stress", "stress_score"):
        if row.get(key) is not None:
            return _coerce_float(row.get(key))

    stress_label = str(row.get("overall_stress") or row.get("overall_prediction") or "").strip().lower()
    if stress_label in {"low", "calm"}:
        return 25.0
    if stress_label in {"moderate", "medium"}:
        return 55.0
    if stress_label in {"high", "stressed"}:
        return 85.0
    return None


def _fit_model_from_history(
    model: Any,
    model_name: str,
    table: str,
    feature_builder: Any,
    target_builder: Any,
) -> None:
    if _model_is_fitted(model):
        logger.info("Model fit skipped because %s is already fitted.", model_name)
        return

    if not hasattr(model, "fit"):
        logger.warning("Model fit skipped for %s because fit() is unavailable.", model_name)
        return

    rows = _fetch_rows(table=table, limit=250, order_candidates=("timestamp", "scanned_at", "created_at"))
    if not rows:
        logger.warning("Model fit skipped for %s because no historical rows were found in %s.", model_name, table)
        return

    frames: list[pd.DataFrame] = []
    targets: list[float] = []
    for row in rows:
        target = target_builder(row)
        if target is None:
            continue
        frame = feature_builder(row)
        frames.append(frame)
        targets.append(float(target))

    if not frames or not targets:
        logger.warning("Model fit skipped for %s because no usable supervised targets were available.", model_name)
        return

    try:
        x = pd.concat(frames, ignore_index=True)
        y = np.asarray(targets, dtype=float)
        model.fit(x, y)
        logger.info("Model fitted from historical data: %s (%d records).", model_name, len(targets))
    except Exception as exc:
        logger.warning("Model fit skipped for %s due to training error: %s", model_name, exc)


def _warm_up_models() -> None:
    if not _MODELS:
        return

    try:
        latest_habit = _latest_row("habits", order_candidates=("timestamp", "created_at"))
        if latest_habit:
            predict_habit_stress(latest_habit)
            logger.info("MindLens model warm-up prediction completed.")
        else:
            logger.info("MindLens model warm-up skipped because no habit history was found.")
    except Exception as exc:
        logger.warning("MindLens model warm-up skipped due to error: %s", exc)

    try:
        latest_scan = _latest_row("stress_scans", order_candidates=("scanned_at", "created_at"))
        if latest_scan:
            payload = _scan_payload_from_row(latest_scan)
            latest_habit = _latest_row("habits", email=latest_scan.get("email"), order_candidates=("timestamp", "created_at"))
            mindlens_score = predict_habit_stress(latest_habit)["mindlens_model_score"] if latest_habit else None
            predict_stress(payload, mindlens_score=mindlens_score)
            logger.info("Stress model warm-up prediction completed.")
        else:
            logger.info("Stress model warm-up skipped because no scan history was found.")
    except Exception as exc:
        logger.warning("Stress model warm-up skipped due to error: %s", exc)


def load_models() -> Dict[str, str]:
    if _MODELS:
        return {name: meta["path"] for name, meta in _MODELS.items()}

    mindlens_path = _resolve_model_path("mindlens_model.pkl")
    stress_path = _resolve_model_path("stress_model.pkl")

    mindlens_model = joblib.load(mindlens_path)
    stress_model = joblib.load(stress_path)

    _set_single_thread_runtime(mindlens_model)
    _set_single_thread_runtime(stress_model)
    _log_fit_status("mindlens_model.pkl", mindlens_model)
    _log_fit_status("stress_model.pkl", stress_model)

    _MODELS["mindlens"] = {"model": mindlens_model, "path": str(mindlens_path)}
    _MODELS["stress"] = {"model": stress_model, "path": str(stress_path)}

    _fit_model_from_history(
        model=_MODELS["mindlens"]["model"],
        model_name="mindlens_model.pkl",
        table="habits",
        feature_builder=lambda row: _build_mindlens_model_frame(_sanitize_habit_payload(row)),
        target_builder=_extract_numeric_target,
    )
    _fit_model_from_history(
        model=_MODELS["stress"]["model"],
        model_name="stress_model.pkl",
        table="stress_scans",
        feature_builder=lambda row: _build_stress_model_frame(
            _sanitize_stress_payload(_scan_payload_from_row(row)),
            _scan_payload_from_row(row),
        ),
        target_builder=_extract_numeric_target,
    )
    _warm_up_models()

    logger.info("Models loaded successfully: stress=%s mindlens=%s", stress_path, mindlens_path)
    return {name: meta["path"] for name, meta in _MODELS.items()}


async def ensure_models_loaded() -> Dict[str, str]:
    if _MODELS:
        return {name: meta["path"] for name, meta in _MODELS.items()}
    async with _MODEL_LOCK:
        if _MODELS:
            return {name: meta["path"] for name, meta in _MODELS.items()}
        return load_models()


def predict_habit_stress(data: Dict[str, Any]) -> Dict[str, Any]:
    if not _MODELS:
        load_models()

    habits = _sanitize_habit_payload(data)
    prediction = _MODELS["mindlens"]["model"].predict(_build_mindlens_model_frame(habits))
    score = _prediction_to_score(prediction)
    result = {
        "email": data.get("email"),
        "timestamp": data.get("timestamp") or data.get("created_at") or _timestamp(),
        "mindlens_model_score": score,
        "mindlens_stress": score,
        "habits": habits,
        "social_interaction": data.get("social_interaction"),
    }
    logger.info("MindLens prediction succeeded for %s with score %.2f", data.get("email"), score)
    return result


def predict_stress(data: Dict[str, Any], mindlens_score: Optional[float] = None) -> Dict[str, Any]:
    if not _MODELS:
        load_models()

    features = _sanitize_stress_payload(data)
    stress_frame = _build_stress_model_frame(features, data)
    prediction = _MODELS["stress"]["model"].predict(stress_frame)
    stress_score = _prediction_to_score(prediction)
    overall_average = _overall_average_stress(stress_score, mindlens_score)
    emotion = _normalize_emotion(data.get("emotion"))
    posture = _normalize_posture(data.get("posture") or data.get("posture_quality"), features)
    jaw_tension = str(data.get("jaw_tension") or _derive_jaw_tension(features)).title()

    result = {
        "email": data.get("email"),
        "timestamp": data.get("timestamp") or data.get("scanned_at") or data.get("created_at") or _timestamp(),
        "mindlens_model_score": mindlens_score,
        "mindlens_stress": mindlens_score,
        "stress_model_score": stress_score,
        "stress_model_stress": stress_score,
        "overall_average_stress": overall_average,
        "avg_stress": overall_average,
        "overall_prediction": _stress_label(overall_average),
        "emotion": emotion,
        "emotion_confidence": features["emotion_confidence"],
        "jaw_tension": jaw_tension,
        "posture": posture,
        "features": features,
    }
    logger.info("Stress prediction succeeded for %s with stress score %.2f", data.get("email"), stress_score)
    return result


def _fetch_rows(
    table: str,
    email: Optional[str] = None,
    limit: int = 1,
    order_candidates: tuple[str, ...] = ("created_at",),
) -> list[Dict[str, Any]]:
    rows = fetch_rows(table, email=email, limit=limit, order_candidates=order_candidates)
    if not rows:
        logger.info("No %s rows available for %s.", table, email or "global")
    return rows


def _latest_row(
    table: str,
    email: Optional[str] = None,
    order_candidates: tuple[str, ...] = ("created_at",),
) -> Optional[Dict[str, Any]]:
    return latest_row(table, email=email, order_candidates=order_candidates)


def _normalize_habit_entry(row: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    habits = _sanitize_habit_payload(row)
    score = prediction["mindlens_model_score"] if prediction else None
    return {
        "timestamp": row.get("timestamp") or row.get("created_at"),
        "age": habits["age"],
        "sleep_hours": habits["sleep_hours"],
        "work_hours": habits["work_hours"],
        "screen_time": habits["screen_time"],
        "water_intake": habits["water_intake"],
        "exercise": bool(_coerce_bool(row.get("exercise"))),
        "meals_per_day": habits["meals_per_day"],
        "social_interaction": row.get("social_interaction") or "Unknown",
        "caffeine_intake": bool(_coerce_bool(row.get("caffeine_intake"))),
        "mindlens_model_score": score,
        "mood": _stress_label(score) if score is not None else "Unknown",
    }


def _safe_predict_habit(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    try:
        return predict_habit_stress(row)
    except Exception as exc:
        logger.warning("MindLens prediction skipped for one habit entry due to error: %s", exc)
        return None


def _safe_normalize_scan(row: Optional[Dict[str, Any]], mindlens_score: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    try:
        return _normalize_scan_entry(row, mindlens_score=mindlens_score)
    except Exception as exc:
        logger.warning("Stress prediction skipped for one scan entry due to error: %s", exc)
        try:
            payload = _scan_payload_from_row(row)
            features = _sanitize_stress_payload(payload)
            stress_score = _coerce_float(
                row.get("stress_model_score", row.get("stress_model_stress", row.get("avg_stress"))),
                default=None,
            )
            overall_average = _overall_average_stress(stress_score, mindlens_score)
            if overall_average is None:
                raw_overall = row.get("overall_average_stress", row.get("avg_stress"))
                overall_average = _coerce_float(raw_overall, default=None)
                if overall_average is None:
                    overall_average = stress_score

            return {
                "timestamp": payload.get("timestamp") or _timestamp(),
                "emotion": _normalize_emotion(payload.get("emotion")),
                "emotion_confidence": features["emotion_confidence"],
                "jaw_tension": str(payload.get("jaw_tension") or _derive_jaw_tension(features)).title(),
                "posture": _normalize_posture(payload.get("posture"), features),
                "stress_model_score": stress_score,
                "mindlens_model_score": mindlens_score,
                "overall_average_stress": overall_average,
                "overall_prediction": _stress_label(overall_average),
                "features": features,
            }
        except Exception as fallback_exc:
            logger.warning("Fallback scan normalization also failed: %s", fallback_exc)
            return None


def _scan_payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "email": row.get("email"),
        "timestamp": row.get("scanned_at") or row.get("created_at") or row.get("timestamp"),
        "emotion": row.get("emotion"),
        "emotion_confidence": row.get("emotion_confidence"),
        "mouth_open": row.get("mouth_open"),
        "eyebrow_raise": row.get("eyebrow_raise"),
        "jaw_clench_score": row.get("jaw_clench_score"),
        "slouch_score": row.get("slouch_score"),
        "head_tilt_angle": row.get("head_tilt_angle"),
        "shoulder_alignment_diff": row.get("shoulder_alignment_diff"),
        "spine_curve_ratio": row.get("spine_curve_ratio"),
        "posture_confidence": row.get("posture_confidence", row.get("pose_confidence")),
        "jaw_tension": row.get("jaw_tension"),
        "posture": row.get("posture") or row.get("posture_quality"),
    }


def _normalize_scan_entry(row: Dict[str, Any], mindlens_score: Optional[float] = None) -> Dict[str, Any]:
    payload = _scan_payload_from_row(row)
    saved_stress_score = _coerce_float(
        row.get("stress_model_score", row.get("stress_model_stress", row.get("avg_stress"))),
        default=None,
    )
    saved_mindlens_score = _coerce_float(
        row.get("mindlens_model_score", row.get("mindlens_stress")),
        default=None,
    )
    effective_mindlens_score = saved_mindlens_score if saved_mindlens_score is not None else mindlens_score
    saved_overall_average = _coerce_float(
        row.get("overall_average_stress", row.get("avg_stress")),
        default=None,
    )

    if saved_stress_score is None:
        prediction = predict_stress(payload, mindlens_score=effective_mindlens_score)
        stress_model_score = prediction["stress_model_score"]
        overall_average_stress = prediction["overall_average_stress"]
        emotion = prediction["emotion"]
        emotion_confidence = prediction["emotion_confidence"]
        jaw_tension = prediction["jaw_tension"]
        posture = prediction["posture"]
        features = prediction["features"]
        timestamp = prediction["timestamp"]
    else:
        features = _sanitize_stress_payload(payload)
        stress_model_score = saved_stress_score
        if saved_overall_average is not None:
            overall_average_stress = saved_overall_average
        else:
            overall_average_stress = _overall_average_stress(stress_model_score, effective_mindlens_score)
            if overall_average_stress is None:
                overall_average_stress = stress_model_score
        emotion = _normalize_emotion(payload.get("emotion"))
        emotion_confidence = features["emotion_confidence"]
        jaw_tension = str(payload.get("jaw_tension") or _derive_jaw_tension(features)).title()
        posture = _normalize_posture(payload.get("posture"), features)
        timestamp = payload.get("timestamp") or _timestamp()

    return {
        "timestamp": timestamp,
        "emotion": emotion,
        "emotion_confidence": emotion_confidence,
        "jaw_tension": jaw_tension,
        "posture": posture,
        "stress_model_score": stress_model_score,
        "mindlens_model_score": effective_mindlens_score,
        "overall_average_stress": overall_average_stress,
        "overall_prediction": _stress_label(overall_average_stress),
        "features": features,
    }


def _normalize_sensor_entry(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "heart_rate": _coerce_float(row.get("heart_rate"), 72.0),
        "spo2": _coerce_float(row.get("spo2"), 98.0),
        "temperature": _coerce_float(row.get("temperature"), 36.9),
        "timestamp": row.get("created_at") or row.get("timestamp") or _timestamp(),
    }


async def get_dashboard_summary(email: str, recent_limit: int = 5) -> Dict[str, Any]:
    await ensure_models_loaded()
    recent_limit = max(1, min(recent_limit, 5))

    latest_habit_row = _latest_row("habits", email=email, order_candidates=("timestamp", "created_at"))
    recent_habit_rows = _fetch_rows("habits", email=email, limit=recent_limit, order_candidates=("timestamp", "created_at"))
    latest_scan_row = _latest_row("stress_scans", email=email, order_candidates=("scanned_at", "created_at"))
    recent_scan_rows = _fetch_rows("stress_scans", email=email, limit=recent_limit, order_candidates=("scanned_at", "created_at"))
    latest_sensor = _normalize_sensor_entry(_latest_row("sensor_records", order_candidates=("created_at",)))

    latest_habit_prediction = _safe_predict_habit(latest_habit_row)
    latest_habit = _normalize_habit_entry(latest_habit_row, latest_habit_prediction) if latest_habit_row else None
    latest_scan = _safe_normalize_scan(
        latest_scan_row,
        latest_habit_prediction["mindlens_model_score"] if latest_habit_prediction else None,
    )
    if latest_scan:
        logger.info("dashboard latest scan fetched successfully for %s", email)

    recent_habits: list[Dict[str, Any]] = []
    for row in recent_habit_rows[:recent_limit]:
        pred = _safe_predict_habit(row)
        recent_habits.append(_normalize_habit_entry(row, pred))

    recent_scans: list[Dict[str, Any]] = []
    for row in recent_scan_rows[:recent_limit]:
        normalized = _safe_normalize_scan(row, latest_habit_prediction["mindlens_model_score"] if latest_habit_prediction else None)
        if normalized:
            recent_scans.append(normalized)

    stress_model_score = latest_scan["stress_model_score"] if latest_scan else None
    mindlens_model_score = latest_scan["mindlens_model_score"] if latest_scan else None
    if mindlens_model_score is None and latest_habit_prediction:
        mindlens_model_score = latest_habit_prediction["mindlens_model_score"]
    if mindlens_model_score is None and latest_habit_row:
        raw_habit_score = latest_habit_row.get("mindlens_model_score", latest_habit_row.get("mindlens_stress"))
        if raw_habit_score is not None:
            mindlens_model_score = _coerce_float(raw_habit_score, default=None)
    overall_average_stress = latest_scan["overall_average_stress"] if latest_scan else None
    if overall_average_stress is None:
        overall_average_stress = _overall_average_stress(stress_model_score, mindlens_model_score)
    if overall_average_stress is None:
        overall_average_stress = stress_model_score
    health = _derive_health_metrics(latest_sensor, latest_habit, overall_average_stress)
    recommendations = _generate_recommendations(latest_scan, latest_habit, overall_average_stress, email)

    summary = {
        "timestamp": _timestamp(),
        "email": email,
        "latest_scan": latest_scan,
        "latest_habit_data": latest_habit,
        "recent_stress_scans": recent_scans,
        "recent_habit_entries": recent_habits,
        "sensor": latest_sensor,
        "stress_model_score": stress_model_score,
        "mindlens_model_score": mindlens_model_score,
        "overall_average_stress": overall_average_stress,
        "avg_stress": overall_average_stress,
        "overall_prediction": _stress_label(overall_average_stress),
        "model_predictions": {
            "stress_model_score": stress_model_score,
            "mindlens_model_score": mindlens_model_score,
            "overall_average_stress": overall_average_stress,
            "overall_prediction": _stress_label(overall_average_stress),
        },
        "merged_dashboard_data": {
            "latest_scan": latest_scan,
            "latest_habit_data": latest_habit,
            "model_predictions": {
                "stress_model_score": stress_model_score,
                "mindlens_model_score": mindlens_model_score,
                "overall_average_stress": overall_average_stress,
            },
        },
        "recommendations": recommendations,
        **health,
    }
    logger.info("dashboard response JSON: %s", json.dumps(summary, default=str))
    logger.info("Dashboard data sent successfully for %s", email)
    return summary


def _resolve_dashboard_email(email: Optional[str] = None) -> Optional[str]:
    if email:
        normalized = str(email).strip()
        lowered = normalized.lower()
        if normalized and lowered not in {"undefined", "null", "none"} and "@" in normalized and "." in normalized:
            return normalized

    latest_scan = _latest_row("stress_scans", order_candidates=("scanned_at", "created_at"))
    if latest_scan and latest_scan.get("email"):
        return str(latest_scan.get("email"))

    latest_habit = _latest_row("habits", order_candidates=("timestamp", "created_at"))
    if latest_habit and latest_habit.get("email"):
        return str(latest_habit.get("email"))

    latest_user = _latest_row("users", order_candidates=("created_at",))
    if latest_user and latest_user.get("email"):
        return str(latest_user.get("email"))

    return None


async def get_dashboard_summary_any(email: Optional[str] = None, recent_limit: int = 5) -> Dict[str, Any]:
    resolved_email = _resolve_dashboard_email(email)
    if not resolved_email:
        latest_event_data = (_LATEST_EVENT or {}).get("data") or {}
        event_email = latest_event_data.get("email")
        if event_email:
            resolved_email = str(event_email)
        elif latest_event_data:
            avg_stress = latest_event_data.get("overall_average_stress", latest_event_data.get("avg_stress"))
            return {
                "timestamp": _timestamp(),
                "email": None,
                "latest_scan": latest_event_data,
                "latest_habit_data": None,
                "recent_stress_scans": [latest_event_data],
                "recent_habit_entries": [],
                "sensor": None,
                "stress_model_score": latest_event_data.get("stress_model_score"),
                "mindlens_model_score": latest_event_data.get("mindlens_model_score"),
                "overall_average_stress": avg_stress,
                "avg_stress": avg_stress,
                "overall_prediction": latest_event_data.get("overall_prediction", "Unknown"),
                "model_predictions": {
                    "stress_model_score": latest_event_data.get("stress_model_score"),
                    "mindlens_model_score": latest_event_data.get("mindlens_model_score"),
                    "overall_average_stress": avg_stress,
                    "overall_prediction": latest_event_data.get("overall_prediction", "Unknown"),
                },
                "merged_dashboard_data": {
                    "latest_scan": latest_event_data,
                    "latest_habit_data": None,
                    "model_predictions": {
                        "stress_model_score": latest_event_data.get("stress_model_score"),
                        "mindlens_model_score": latest_event_data.get("mindlens_model_score"),
                        "overall_average_stress": avg_stress,
                    },
                },
                "recommendations": _generate_recommendations(latest_event_data, None, avg_stress, None),
                **_derive_health_metrics(None, None, avg_stress),
            }

    if not resolved_email:
        logger.warning("Dashboard summary requested but no user email could be resolved.")
        return {
            "timestamp": _timestamp(),
            "email": None,
            "latest_scan": None,
            "latest_habit_data": None,
            "recent_stress_scans": [],
            "recent_habit_entries": [],
            "sensor": None,
            "stress_model_score": None,
            "mindlens_model_score": None,
            "overall_average_stress": None,
            "avg_stress": None,
            "overall_prediction": "Unknown",
            "model_predictions": {
                "stress_model_score": None,
                "mindlens_model_score": None,
                "overall_average_stress": None,
                "overall_prediction": "Unknown",
            },
            "merged_dashboard_data": {
                "latest_scan": None,
                "latest_habit_data": None,
                "model_predictions": {
                    "stress_model_score": None,
                    "mindlens_model_score": None,
                    "overall_average_stress": None,
                },
            },
            "recommendations": [
                "Add one habit entry and one stress scan to start live dashboard predictions.",
                "Keep hydration, sleep, and short breaks consistent to improve stress trends.",
                "Complete a fresh scan now to begin real-time monitoring.",
            ],
            **_derive_health_metrics(None, None, None),
        }

    return await get_dashboard_summary(email=resolved_email, recent_limit=recent_limit)


async def _publish_event(event_type: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    global _LATEST_EVENT

    event = {
        "event": event_type,
        "timestamp": _timestamp(),
        "email": (payload or {}).get("email"),
        "data": payload or {},
    }
    _LATEST_EVENT = event

    stale = []
    for queue in list(_SUBSCRIBERS):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            stale.append(queue)
    for queue in stale:
        _SUBSCRIBERS.discard(queue)

    return event


async def update_habit_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    prediction = predict_habit_stress(data)
    return await _publish_event("habit_updated", prediction)


async def update_sensor_state(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_sensor_entry(data)
    return await _publish_event("sensor_updated", normalized)


async def update_scan_prediction(data: Dict[str, Any]) -> Dict[str, Any]:
    mindlens_score = None
    email = data.get("email")
    if email:
        latest_habit_row = _latest_row("habits", email=email, order_candidates=("timestamp", "created_at"))
        if latest_habit_row:
            mindlens_score = predict_habit_stress(latest_habit_row)["mindlens_model_score"]
    prediction = predict_stress(data, mindlens_score=mindlens_score)
    return await _publish_event("scan_updated", prediction)


def get_latest_result() -> Optional[Dict[str, Any]]:
    return _LATEST_EVENT


async def store_latest_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return await _publish_event("stored_result", result)


def register_subscriber() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    _SUBSCRIBERS.add(queue)
    return queue


def unregister_subscriber(queue: asyncio.Queue) -> None:
    _SUBSCRIBERS.discard(queue)


async def stream_events():
    queue = register_subscriber()
    try:
        if _LATEST_EVENT is not None:
            yield f"data: {json.dumps(_LATEST_EVENT)}\n\n"

        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=15)
                yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
    finally:
        unregister_subscriber(queue)
