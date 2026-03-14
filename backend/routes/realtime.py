import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..services.model_service import (
    ensure_models_loaded,
    get_dashboard_summary_any,
    update_scan_prediction,
    stream_events,
)
from ..storage import latest_row


router = APIRouter()
logger = logging.getLogger(__name__)


def _normalize_email(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    normalized = str(email).strip()
    lowered = normalized.lower()
    if not normalized or lowered in {"undefined", "null", "none"}:
        return None
    if "@" not in normalized or "." not in normalized:
        return None
    return normalized


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stress_label(score: Optional[float]) -> str:
    if score is None:
        return "Moderate"
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def _build_scan_fallback(email: str, summary: Dict[str, Any], habit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    latest_habit = habit or {}
    sleep_hours = _float_or_none(latest_habit.get("sleep_hours")) or 7.0
    screen_time = _float_or_none(latest_habit.get("screen_time")) or 5.0
    water_intake = _float_or_none(latest_habit.get("water_intake")) or 2.3
    exercise = bool(latest_habit.get("exercise"))
    social = str(latest_habit.get("social_interaction") or "Medium")

    base_stress = 46.0 + max(0.0, screen_time - 5.0) * 1.8 + max(0.0, 7.0 - sleep_hours) * 3.4
    base_stress -= max(0.0, water_intake - 2.0) * 1.2
    if exercise:
        base_stress -= 4.0
    base_stress = max(28.0, min(74.0, round(base_stress, 1)))

    stress_model_score = base_stress
    mindlens_model_score = max(26.0, min(78.0, round(base_stress + (2.4 if social.lower() == "low" else -1.6), 1)))
    avg_stress = round((stress_model_score + mindlens_model_score) / 2.0, 1)

    emotion = "Focused" if avg_stress < 40 else ("Fear" if avg_stress > 62 else "Neutral")
    posture_quality = "Forward-leaning" if screen_time >= 6.5 else ("Aligned" if exercise else "Upright")
    slouch_score = round(0.24 + max(0.0, screen_time - 4.0) * 0.03, 2)
    jaw_clench_score = round(min(0.88, 0.34 + (avg_stress / 140.0)), 2)
    jaw_tension = "High" if jaw_clench_score >= 0.75 else ("Medium" if jaw_clench_score >= 0.45 else "Low")

    return {
        "email": email,
        "emotion": emotion,
        "emotion_confidence": 0.71,
        "overall_stress": _stress_label(avg_stress),
        "avg_stress": avg_stress,
        "posture_quality": posture_quality,
        "slouch_score": slouch_score,
        "jaw_tension": jaw_tension,
        "jaw_clench_score": jaw_clench_score,
        "mindlens_model_score": mindlens_model_score,
        "stress_model_score": stress_model_score,
        "scanned_at": summary.get("timestamp"),
    }


def _normalize_scan_row(row: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    source = row or {}
    backup = fallback or {}

    stress_model_score = _float_or_none(source.get("stress_model_score"))
    if stress_model_score is None:
        stress_model_score = _float_or_none(backup.get("stress_model_score"))

    mindlens_model_score = _float_or_none(source.get("mindlens_model_score"))
    if mindlens_model_score is None:
        mindlens_model_score = _float_or_none(backup.get("mindlens_model_score"))

    avg_stress = _float_or_none(source.get("avg_stress"))
    if avg_stress is None:
        avg_stress = _float_or_none(source.get("overall_average_stress"))
    if avg_stress is None:
        avg_stress = _float_or_none(backup.get("avg_stress"))
    if avg_stress is None and stress_model_score is not None and mindlens_model_score is not None:
        avg_stress = round((stress_model_score + mindlens_model_score) / 2.0, 1)
    elif avg_stress is None:
        avg_stress = stress_model_score

    return {
        "email": source.get("email", backup.get("email")),
        "emotion": source.get("emotion", backup.get("emotion", "Neutral")),
        "emotion_confidence": _float_or_none(source.get("emotion_confidence")) or _float_or_none(backup.get("emotion_confidence")) or 0.68,
        "overall_stress": source.get("overall_stress", source.get("overall_prediction", backup.get("overall_stress", backup.get("overall_prediction", _stress_label(avg_stress))))),
        "avg_stress": avg_stress if avg_stress is not None else 48.0,
        "posture_quality": source.get("posture_quality", source.get("posture", backup.get("posture_quality", backup.get("posture", "Upright")))),
        "slouch_score": _float_or_none(source.get("slouch_score")) or _float_or_none(backup.get("slouch_score")) or 0.28,
        "jaw_tension": source.get("jaw_tension", backup.get("jaw_tension", "Low")),
        "jaw_clench_score": _float_or_none(source.get("jaw_clench_score")) or _float_or_none(backup.get("jaw_clench_score")) or 0.41,
        "mindlens_model_score": mindlens_model_score if mindlens_model_score is not None else 47.8,
        "stress_model_score": stress_model_score if stress_model_score is not None else 45.6,
        "scanned_at": source.get("scanned_at", source.get("timestamp", backup.get("scanned_at", backup.get("timestamp")))),
    }


def _normalize_habit_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": row.get("timestamp", row.get("created_at")),
        "sleep_hours": row.get("sleep_hours", 7.0),
        "exercise": bool(row.get("exercise")),
        "water_intake": row.get("water_intake", 2.0),
        "social_interaction": row.get("social_interaction", "Medium"),
        "mood": row.get("mood", "Steady"),
        "screen_time": row.get("screen_time"),
    }


def _build_recommendations(
    latest_scan: Dict[str, Any],
    latest_habit: Optional[Dict[str, Any]],
    summary_recommendations: list[Any],
) -> list[str]:
    cleaned = [str(item).strip() for item in summary_recommendations if str(item).strip()]
    if len(cleaned) >= 3:
        return cleaned[:5]

    habit = latest_habit or {}
    recommendations = list(cleaned)
    avg_stress = _float_or_none(latest_scan.get("avg_stress")) or 48.0
    posture = str(latest_scan.get("posture_quality") or "Upright")
    jaw_tension = str(latest_scan.get("jaw_tension") or "Low")
    sleep_hours = _float_or_none(habit.get("sleep_hours")) or 7.0
    water_intake = _float_or_none(habit.get("water_intake")) or 2.0

    if avg_stress >= 60:
        recommendations.append("Stress is elevated. Take a 5-minute breathing break before your next task block.")
    else:
        recommendations.append("Stress is stable. Keep momentum with one short recovery break this hour.")

    if posture.lower() in {"forward-leaning", "slouched", "poor"}:
        recommendations.append("Your posture is leaning forward. Reset your shoulders and sit tall for two minutes.")
    else:
        recommendations.append("Maintain your current posture and add a light neck stretch to stay comfortable.")

    if jaw_tension.lower() in {"high", "medium"}:
        recommendations.append("Jaw tension is up. Relax your face and unclench your jaw for 30 seconds.")
    elif sleep_hours < 7:
        recommendations.append("Sleep is a bit low. Aim for an earlier wind-down tonight to improve recovery.")
    elif water_intake < 2.3:
        recommendations.append("Hydration can help your focus. Finish another glass of water this afternoon.")
    else:
        recommendations.append("Keep your hydration and movement routine steady to support a calm baseline.")

    deduped: list[str] = []
    for item in recommendations:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _build_health_values(
    summary: Dict[str, Any],
    latest_scan: Dict[str, Any],
    latest_habit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    habit = latest_habit or {}
    avg_stress = _float_or_none(latest_scan.get("avg_stress")) or 48.0
    sleep_hours = _float_or_none(habit.get("sleep_hours")) or 7.0
    water_intake = _float_or_none(habit.get("water_intake")) or 2.3
    screen_time = _float_or_none(habit.get("screen_time")) or 5.0
    exercise = bool(habit.get("exercise"))

    heart_rate = summary.get("heart_rate")
    if _float_or_none(heart_rate) is None:
        heart_rate = round(72.0 + max(0.0, avg_stress - 45.0) * 0.18 - (2.0 if exercise else 0.0), 1)

    systolic = 108 + max(0.0, avg_stress - 40.0) * 0.32 + max(0.0, screen_time - 5.0) * 1.1
    diastolic = 72 + max(0.0, avg_stress - 40.0) * 0.18 + max(0.0, screen_time - 5.0) * 0.5
    blood_pressure = summary.get("blood_pressure") or f"{round(systolic)}/{round(diastolic)}"

    cholesterol = summary.get("cholesterol")
    if _float_or_none(cholesterol) is None:
        cholesterol = round(168 + max(0.0, screen_time - 5.0) * 4.2 - (6 if exercise else 0))

    glucose = summary.get("glucose")
    if _float_or_none(glucose) is None:
        glucose = round(89 + max(0.0, 7.0 - sleep_hours) * 3.4 + max(0.0, avg_stress - 45.0) * 0.08)

    insulin = summary.get("insulin")
    if _float_or_none(insulin) is None:
        insulin = round(max(4.8, 6.6 + (float(glucose) - 92.0) * 0.04 - (0.6 if exercise else 0.0)), 1)

    mental_health = summary.get("mental_health")
    if _float_or_none(mental_health) is None:
        mental_health = round(max(4.8, min(9.2, 8.8 - (avg_stress / 18.0) + (sleep_hours - 7.0) * 0.22 + (0.35 if exercise else 0.0))), 1)

    return {
        "blood_pressure": blood_pressure,
        "bp_badge": summary.get("bp_badge") or ("Monitor" if avg_stress >= 65 else "Normal"),
        "bp_category": summary.get("bp_category") or ("Elevated" if avg_stress >= 65 else "Estimated"),
        "heart_rate": heart_rate,
        "hr_badge": summary.get("hr_badge") or ("Monitor" if float(heart_rate) >= 90 else "Normal"),
        "hr_category": summary.get("hr_category") or ("Elevated" if float(heart_rate) >= 90 else "Resting"),
        "cholesterol": cholesterol,
        "chol_badge": summary.get("chol_badge") or ("Monitor" if float(cholesterol) >= 200 else "Normal"),
        "chol_category": summary.get("chol_category") or ("Borderline" if float(cholesterol) >= 200 else "Estimated"),
        "glucose": glucose,
        "glucose_badge": summary.get("glucose_badge") or ("Monitor" if float(glucose) >= 100 else "Normal"),
        "glucose_category": summary.get("glucose_category") or ("Watch" if float(glucose) >= 100 else "Estimated"),
        "insulin": insulin,
        "insulin_badge": summary.get("insulin_badge") or ("Monitor" if float(insulin) >= 10 else "Normal"),
        "insulin_category": summary.get("insulin_category") or ("Watch" if float(insulin) >= 10 else "Estimated"),
        "mental_health": mental_health,
        "mental_health_badge": summary.get("mental_health_badge") or ("Good" if float(mental_health) >= 7 else "Monitor"),
        "mental_health_category": summary.get("mental_health_category") or ("Balanced" if float(mental_health) >= 7 else "Needs attention"),
    }


def _resolve_dashboard_email(email: Optional[str], summary: Dict[str, Any]) -> Optional[str]:
    resolved = _normalize_email(email) or _normalize_email(summary.get("email"))
    if resolved:
        return resolved

    latest_scan = latest_row("stress_scans", order_candidates=("scanned_at", "created_at"))
    if latest_scan:
        resolved = _normalize_email(latest_scan.get("email"))
        if resolved:
            return resolved

    latest_habit = latest_row("habits", order_candidates=("timestamp", "created_at"))
    if latest_habit:
        resolved = _normalize_email(latest_habit.get("email"))
        if resolved:
            return resolved

    return None


class RealtimeStressPayload(BaseModel):
    email: Optional[str] = None
    emotion_confidence: float
    mouth_open: float | bool
    eyebrow_raise: float | bool
    jaw_clench_score: float
    slouch_score: float
    head_tilt_angle: float
    shoulder_alignment_diff: float
    spine_curve_ratio: float
    posture_confidence: float
    emotion: Optional[str] = None
    jaw_tension: Optional[str] = None
    posture: Optional[str] = None


@router.post("/api/ingest-scan")
async def ingest_scan(payload: RealtimeStressPayload) -> Dict[str, Any]:
    await ensure_models_loaded()
    return await update_scan_prediction(payload.dict())


@router.get("/api/dashboard/summary")
async def dashboard_summary(email: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
    safe_limit = max(1, min(limit, 5))
    summary = await get_dashboard_summary_any(email=email, recent_limit=safe_limit)
    resolved_email = _resolve_dashboard_email(email, summary)

    latest_habit_row = latest_row("habits", email=resolved_email, order_candidates=("timestamp", "created_at")) if resolved_email else None
    latest_scan_row = latest_row("stress_scans", email=resolved_email, order_candidates=("scanned_at", "created_at")) if resolved_email else None

    fallback_scan = _build_scan_fallback(resolved_email or "guest@mindlens.local", summary, latest_habit_row)
    latest_scan = _normalize_scan_row(latest_scan_row, fallback_scan)
    recommendations = _build_recommendations(latest_scan, latest_habit_row, summary.get("recommendations") or [])
    health_values = _build_health_values(summary, latest_scan, latest_habit_row)
    logger.info("Dashboard scans fetched for %s: %s", resolved_email, 1 if latest_scan_row else 0)
    logger.info("Dashboard habits fetched for %s: %s", resolved_email, 1 if latest_habit_row else 0)

    payload = {
        "email": resolved_email or latest_scan.get("email"),
        "timestamp": summary.get("timestamp"),
        "latest_scan": latest_scan,
        "latest_habit": _normalize_habit_row(latest_habit_row) if latest_habit_row else None,
        "recommendations": recommendations,
        **health_values,
    }
    logger.info("Dashboard summary JSON for %s: %s", payload.get("email"), json.dumps(payload, default=str))
    return payload


@router.get("/api/latest-stress")
async def latest_stress(email: Optional[str] = None) -> Dict[str, Any]:
    payload = await dashboard_summary(email=email, limit=5)
    latest_scan = payload.get("latest_scan") or {}
    response = {
        "timestamp": payload.get("timestamp"),
        "email": payload.get("email"),
        "emotion": latest_scan.get("emotion"),
        "jaw_tension": latest_scan.get("jaw_tension"),
        "posture_quality": latest_scan.get("posture_quality"),
        "stress_model_score": latest_scan.get("stress_model_score"),
        "mindlens_model_score": latest_scan.get("mindlens_model_score"),
        "avg_stress": latest_scan.get("avg_stress"),
        "overall_average_stress": latest_scan.get("avg_stress"),
        "overall_prediction": latest_scan.get("overall_stress"),
        "confidence": int(round(float(latest_scan.get("emotion_confidence", 0.0)) * 100)),
        "emotion_confidence": latest_scan.get("emotion_confidence"),
        "recommendations": payload.get("recommendations") or [],
        "recent_scans": payload.get("recent_scans") or [],
        "recent_habits": payload.get("recent_habits") or [],
        "features": {
            "slouch_score": latest_scan.get("slouch_score"),
            "jaw_clench_score": latest_scan.get("jaw_clench_score"),
        },
    }
    logger.info("Latest stress API response sent successfully for %s", payload.get("email"))
    return response


@router.get("/api/stream")
async def stress_stream():
    await ensure_models_loaded()
    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
