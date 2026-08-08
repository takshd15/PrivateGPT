"""Auto-detected signup/registration events from the background email-scan job.

Reads/writes the ``events`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK, jsonb payload). This module never creates or
alters that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_event(user_id: str, type_: str, payload: dict) -> str | None:
    """Insert one event. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        from psycopg.types.json import Json

        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO events (user_id, type, payload) VALUES (%s, %s, %s) RETURNING id;",
                    (user_id, type_, Json(payload)),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_event skipped: {exc}")
        return None


def count_events_since(user_id: str, since_iso: str) -> int:
    """Count events created since a given ISO timestamp. 0 on any failure."""
    if not DATABASE_URL:
        return 0
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM events WHERE user_id = %s AND created_at >= %s;",
                    (user_id, since_iso),
                )
                row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:
        print(f"[memory] count_events_since skipped: {exc}")
        return 0
