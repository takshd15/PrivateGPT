"""Companies tracked via the background application-tracking email job.

Reads/writes the ``companies`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK). This module never creates or alters that table.

Matching is case-insensitive EXACT name match only (no fuzzy matching) - a
near-miss ("Google" vs "Google LLC") creates a new row rather than risking a
silent merge of two different companies.
"""

from __future__ import annotations
from datetime import date as date_cls

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def find_company_by_name(user_id: str, name: str) -> dict | None:
    """Case-insensitive exact match on name. None if not found or on failure."""
    if not DATABASE_URL or not (name or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, category, website, notes
                    FROM companies
                    WHERE user_id = %s AND lower(trim(name)) = lower(trim(%s))
                    LIMIT 1;
                    """,
                    (user_id, name),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {"id": str(row[0]), "name": row[1], "category": row[2], "website": row[3], "notes": row[4]}
    except Exception as exc:
        print(f"[memory] find_company_by_name skipped: {exc}")
        return None


def insert_company(
    user_id: str, name: str, category: str | None = None, website: str | None = None, notes: str = ""
) -> str | None:
    """Insert one company. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO companies (user_id, name, category, website, notes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, name, category, website, notes or None),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_company skipped: {exc}")
        return None


def append_company_notes(company_id: str, note_line: str) -> None:
    """Append a dated line to notes rather than overwriting. Never raises."""
    if not DATABASE_URL or not company_id:
        return
    try:
        _init_all()
        today = date_cls.today().isoformat()
        stamped = f"[{today}] {note_line}"
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE companies
                    SET notes = CASE WHEN notes IS NULL OR notes = '' THEN %s
                                     ELSE notes || E'\n' || %s END,
                        updated_at = now()
                    WHERE id = %s;
                    """,
                    (stamped, stamped, company_id),
                )
            conn.commit()
    except Exception as exc:
        print(f"[memory] append_company_notes skipped: {exc}")
