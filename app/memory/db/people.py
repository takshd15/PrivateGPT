"""Contacts upserted via the background application-tracking email job.

Reads/writes the ``people`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id/company_id FKs). This module never creates or alters
that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def find_person_by_email(user_id: str, email: str) -> dict | None:
    """Case-insensitive exact match on email. None if not found or on failure."""
    if not DATABASE_URL or not (email or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, email, role, company_id, notes
                    FROM people
                    WHERE user_id = %s AND lower(trim(email)) = lower(trim(%s))
                    LIMIT 1;
                    """,
                    (user_id, email),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "name": row[1], "email": row[2],
            "role": row[3], "company_id": str(row[4]) if row[4] else None, "notes": row[5],
        }
    except Exception as exc:
        print(f"[memory] find_person_by_email skipped: {exc}")
        return None


def find_person_by_name(user_id: str, name: str, limit: int = 5) -> list[dict]:
    """Case-insensitive partial match on name. [] if not found or on failure."""
    if not DATABASE_URL or not (name or "").strip():
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, email, role, company_id, notes
                    FROM people
                    WHERE user_id = %s AND name ILIKE %s
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (user_id, f"%{name.strip()}%", limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "name": r[1], "email": r[2],
                "role": r[3], "company_id": str(r[4]) if r[4] else None, "notes": r[5],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] find_person_by_name skipped: {exc}")
        return []


def upsert_person(
    user_id: str,
    name: str,
    email: str,
    role: str | None = None,
    company_id: str | None = None,
    notes: str = "",
) -> str | None:
    """Insert a new person, or update role/company/notes if the email already
    exists. Returns the row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL or not (email or "").strip():
        return None
    try:
        existing = find_person_by_email(user_id, email)
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                if existing:
                    cur.execute(
                        """
                        UPDATE people
                        SET role = coalesce(%s, role),
                            company_id = coalesce(%s, company_id),
                            notes = CASE WHEN %s = '' THEN notes
                                         WHEN notes IS NULL OR notes = '' THEN %s
                                         ELSE notes || E'\n' || %s END,
                            updated_at = now()
                        WHERE id = %s
                        RETURNING id;
                        """,
                        (role, company_id, notes or "", notes, notes, existing["id"]),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO people (user_id, name, email, role, company_id, notes)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (user_id, name, email, role, company_id, notes or None),
                    )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] upsert_person skipped: {exc}")
        return None
