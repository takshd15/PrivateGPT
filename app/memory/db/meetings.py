"""Meetings passively synced from Google Calendar (app/main.py's calendar
fetch call sites), plus the once-per-meeting follow-up opt-in/opt-out cadence.

Reads/writes the ``meetings`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK), with ONE deliberate exception: this module adds a
single column, ``followup_status``, via an idempotent ALTER TABLE in
init_schema(). This is the only schema alteration anywhere in this package -
justified because the opt-out-permanence requirement has no other durable,
query-able home, and the pre-provisioned table doesn't have one.

``gcal_event_id`` has no unique constraint in the live schema, so dedup here
is a manual find-then-insert/update, not an ON CONFLICT clause.
"""

from __future__ import annotations
from datetime import date as date_cls

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE meetings ADD COLUMN IF NOT EXISTS followup_status text DEFAULT 'pending';")


def find_meeting_by_gcal_id(user_id: str, gcal_event_id: str) -> dict | None:
    """None if not found or on failure. Never raises."""
    if not DATABASE_URL or not (gcal_event_id or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, notes, location, starts_at, ends_at, followup_status
                    FROM meetings
                    WHERE user_id = %s AND gcal_event_id = %s
                    LIMIT 1;
                    """,
                    (user_id, gcal_event_id),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "title": row[1], "notes": row[2], "location": row[3],
            "starts_at": row[4], "ends_at": row[5], "followup_status": row[6],
        }
    except Exception as exc:
        print(f"[memory] find_meeting_by_gcal_id skipped: {exc}")
        return None


def upsert_meeting(
    user_id: str,
    title: str,
    location: str | None,
    starts_at: str | None,
    ends_at: str | None,
    gcal_event_id: str,
    source: str = "calendar-sync",
) -> str | None:
    """Insert a new meeting, or update title/location/times on an existing
    one matched by gcal_event_id. Never touches notes/followup_status on
    update - those belong to the follow-up flow. Returns the row's uuid (as
    str), or None on failure. Never raises."""
    if not DATABASE_URL or not (gcal_event_id or "").strip():
        return None
    try:
        existing = find_meeting_by_gcal_id(user_id, gcal_event_id)
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                if existing:
                    cur.execute(
                        """
                        UPDATE meetings
                        SET title = %s, location = %s, starts_at = %s, ends_at = %s,
                            updated_at = now()
                        WHERE id = %s
                        RETURNING id;
                        """,
                        (title, location, starts_at, ends_at, existing["id"]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO meetings
                            (user_id, title, location, starts_at, ends_at, gcal_event_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (user_id, title, location, starts_at, ends_at, gcal_event_id),
                    )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] upsert_meeting skipped: {exc}")
        return None


def list_recent(user_id: str, limit: int = 10) -> list[dict]:
    """Return recent meetings, most recently started first. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, notes, location, starts_at, ends_at, followup_status
                    FROM meetings
                    WHERE user_id = %s
                    ORDER BY starts_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "title": r[1], "notes": r[2], "location": r[3],
                "starts_at": r[4], "ends_at": r[5], "followup_status": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_recent skipped: {exc}")
        return []


def list_followup_candidates(user_id: str, days_ago: int = 7, limit: int = 5) -> list[dict]:
    """Meetings that ended >= days_ago days ago, still pending, with no
    linked progress_events/memories entry mentioning them since. [] on any
    failure or if the follow-up feature has nothing to ask about."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT m.id, m.title, m.ends_at
                    FROM meetings m
                    WHERE m.user_id = %s
                      AND m.followup_status = 'pending'
                      AND m.ends_at IS NOT NULL
                      AND m.ends_at <= now() - (%s || ' days')::interval
                      AND NOT EXISTS (
                          SELECT 1 FROM progress_events pe
                          WHERE pe.user_id = m.user_id
                            AND pe.entity_name = m.title
                            AND pe.created_at >= m.ends_at
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM memories mem
                          WHERE mem.user_id = m.user_id
                            AND mem.entity_name = m.title
                            AND mem.created_at >= m.ends_at
                      )
                    ORDER BY m.ends_at ASC
                    LIMIT %s;
                    """,
                    (user_id, days_ago, limit),
                )
                rows = cur.fetchall()
        return [{"id": str(r[0]), "title": r[1], "ends_at": r[2]} for r in rows]
    except Exception as exc:
        print(f"[memory] list_followup_candidates skipped: {exc}")
        return []


def mark_followup_status(meeting_id: str, status: str) -> None:
    """Set followup_status ('declined' | 'done'). Never raises."""
    if not DATABASE_URL or not meeting_id:
        return
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE meetings SET followup_status = %s, updated_at = now() WHERE id = %s;",
                    (status, meeting_id),
                )
            conn.commit()
    except Exception as exc:
        print(f"[memory] mark_followup_status skipped: {exc}")


def append_meeting_notes(meeting_id: str, note_line: str) -> None:
    """Append a dated line to notes rather than overwriting. Never raises."""
    if not DATABASE_URL or not meeting_id:
        return
    try:
        _init_all()
        today = date_cls.today().isoformat()
        stamped = f"[{today}] {note_line}"
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE meetings
                    SET notes = CASE WHEN notes IS NULL OR notes = '' THEN %s
                                     ELSE notes || E'\n' || %s END,
                        updated_at = now()
                    WHERE id = %s;
                    """,
                    (stamped, stamped, meeting_id),
                )
            conn.commit()
    except Exception as exc:
        print(f"[memory] append_meeting_notes skipped: {exc}")
