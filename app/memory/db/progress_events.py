"""Progress log entries captured via the LOG_PROGRESS voice/typed intent.

Reads/writes the ``progress_events`` table exactly as pre-provisioned in
Supabase (uuid id, uuid user_id/memory_id FKs). This module never creates or
alters that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_progress_event(
    user_id: str,
    entity_name: str,
    event_text: str,
    event_type: str = "update",
    event_date: str | None = None,
    domain: str | None = None,
    memory_id: str | None = None,
    source: str = "voice-confirmed",
) -> str | None:
    """Insert one progress event. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        from datetime import date as date_cls

        _init_all()
        resolved_date = event_date or date_cls.today().isoformat()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO progress_events
                        (user_id, entity_name, domain, event_text, event_type,
                         memory_id, event_date, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, entity_name, domain, event_text, event_type,
                     memory_id, resolved_date, source),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_progress_event skipped: {exc}")
        return None
