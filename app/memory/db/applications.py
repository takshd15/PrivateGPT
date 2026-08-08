"""Applications tracked via the background application-tracking email job.

Reads/writes the ``applications`` table exactly as pre-provisioned in
Supabase (uuid id, uuid user_id FK). This module never creates or alters
that table.

Matching is case-insensitive exact match on the (name, program) pair - the
same company can have multiple applications (different roles/programs), and
status stays free text to match the existing varied vocabulary already in
this table (e.g. 'aspirational', 'needs_context') rather than a fixed enum.
"""

from __future__ import annotations
from datetime import date as date_cls

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def find_application(user_id: str, name: str, program: str | None) -> dict | None:
    """Case-insensitive exact match on (name, program). None if not found or on failure."""
    if not DATABASE_URL or not (name or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, program, status, deadline, notes
                    FROM applications
                    WHERE user_id = %s
                      AND lower(trim(name)) = lower(trim(%s))
                      AND lower(trim(coalesce(program, ''))) = lower(trim(coalesce(%s, '')))
                    LIMIT 1;
                    """,
                    (user_id, name, program or ""),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "name": row[1], "program": row[2],
            "status": row[3], "deadline": row[4], "notes": row[5],
        }
    except Exception as exc:
        print(f"[memory] find_application skipped: {exc}")
        return None


def insert_application(
    user_id: str,
    name: str,
    program: str | None = None,
    status: str = "active",
    deadline: str | None = None,
    notes: str = "",
) -> str | None:
    """Insert one application. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO applications (user_id, name, program, status, deadline, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, name, program, status or "active", deadline, notes or None),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_application skipped: {exc}")
        return None


def list_updated_since(user_id: str, since_iso: str, limit: int = 10) -> list[dict]:
    """Applications inserted or status-updated since a given ISO timestamp,
    for the briefing rollup. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT name, program, status
                    FROM applications
                    WHERE user_id = %s AND updated_at >= %s
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (user_id, since_iso, limit),
                )
                rows = cur.fetchall()
        return [{"name": r[0], "program": r[1], "status": r[2]} for r in rows]
    except Exception as exc:
        print(f"[memory] list_updated_since skipped: {exc}")
        return []


def update_application_status(application_id: str, status: str, note_line: str) -> None:
    """Update status and append a dated line to notes rather than overwriting. Never raises."""
    if not DATABASE_URL or not application_id:
        return
    try:
        _init_all()
        today = date_cls.today().isoformat()
        stamped = f"[{today}] {note_line}"
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE applications
                    SET status = %s,
                        notes = CASE WHEN notes IS NULL OR notes = '' THEN %s
                                     ELSE notes || E'\n' || %s END,
                        updated_at = now()
                    WHERE id = %s;
                    """,
                    (status, stamped, stamped, application_id),
                )
            conn.commit()
    except Exception as exc:
        print(f"[memory] update_application_status skipped: {exc}")
