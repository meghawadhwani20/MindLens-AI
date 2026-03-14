import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from .db import supabase


logger = logging.getLogger(__name__)

_STORE_PATH = Path(__file__).resolve().parent / "data" / "local_store.json"
_TABLES = ("users", "habits", "stress_scans", "sensor_records")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_store() -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STORE_PATH.exists():
        _STORE_PATH.write_text(json.dumps({table: [] for table in _TABLES}, indent=2), encoding="utf-8")


def _load_store() -> Dict[str, list[Dict[str, Any]]]:
    _ensure_store()
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    for table in _TABLES:
        data.setdefault(table, [])
    return data


def _save_store(data: Dict[str, list[Dict[str, Any]]]) -> None:
    _ensure_store()
    _STORE_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    normalized.setdefault("id", str(uuid4()))
    if "created_at" not in normalized and "timestamp" not in normalized and "scanned_at" not in normalized:
        normalized["created_at"] = _utc_now()
    return normalized


def _sort_key(row: Dict[str, Any], order_candidates: Iterable[str]) -> str:
    for key in order_candidates:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def insert_row(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    payload = _normalize_row(row)
    try:
        supabase.table(table).insert(payload).execute()
        return payload
    except Exception as exc:
        logger.warning("Supabase insert failed for %s, using local store: %s", table, exc)
        store = _load_store()
        store.setdefault(table, []).append(payload)
        _save_store(store)
        return payload


def fetch_rows(
    table: str,
    email: Optional[str] = None,
    limit: int = 1,
    order_candidates: tuple[str, ...] = ("created_at",),
) -> list[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for order_column in order_candidates:
        try:
            query = supabase.table(table).select("*")
            if email:
                query = query.eq("email", email)
            response = query.order(order_column, desc=True).limit(limit).execute()
            return response.data or []
        except Exception as exc:
            last_error = exc

    if last_error:
        logger.warning("Supabase fetch failed for %s (%s), using local store.", table, last_error)

    store = _load_store()
    rows = list(store.get(table, []))
    if email:
        rows = [row for row in rows if str(row.get("email") or "").lower() == email.lower()]
    rows.sort(key=lambda row: _sort_key(row, order_candidates), reverse=True)
    return rows[:limit]


def latest_row(
    table: str,
    email: Optional[str] = None,
    order_candidates: tuple[str, ...] = ("created_at",),
) -> Optional[Dict[str, Any]]:
    rows = fetch_rows(table, email=email, limit=1, order_candidates=order_candidates)
    return rows[0] if rows else None


def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    rows = fetch_rows("users", email=email, limit=1, order_candidates=("created_at",))
    return rows[0] if rows else None


def count_rows(table: str, email: Optional[str] = None) -> int:
    try:
        query = supabase.table(table).select("id", count="exact")
        if email:
            query = query.eq("email", email)
        response = query.execute()
        count = getattr(response, "count", None)
        if count is not None:
            return int(count)
    except Exception as exc:
        logger.warning("Supabase count failed for %s, using local store: %s", table, exc)

    store = _load_store()
    rows = list(store.get(table, []))
    if email:
        rows = [row for row in rows if str(row.get("email") or "").lower() == email.lower()]
    return len(rows)


def update_rows(table: str, match: Dict[str, Any], updates: Dict[str, Any]) -> int:
    try:
        query = supabase.table(table).update(updates)
        for key, value in match.items():
            query = query.eq(key, value)
        response = query.execute()
        updated = response.data or []
        if updated:
            return len(updated)
    except Exception as exc:
        logger.warning("Supabase update failed for %s, using local store: %s", table, exc)

    store = _load_store()
    count = 0
    rows = store.get(table, [])
    for row in rows:
        if all(row.get(key) == value for key, value in match.items()):
            row.update(updates)
            count += 1
    if count:
        _save_store(store)
    return count
