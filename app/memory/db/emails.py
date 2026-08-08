"""Audit trail of emails scanned by the background email-mining job.

Reads/writes the ``emails`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK). This module never creates or alters that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_email(
    user_id: str,
    gmail_id: str,
    thread_id: str,
    sender: str,
    subject: str,
    snippet: str,
    received_at: str | None,
    is_important: bool = False,
) -> str | None:
    """Insert one scanned-email audit row. Returns the new row's uuid (as
    str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO emails
                        (user_id, gmail_id, thread_id, sender, subject, snippet,
                         received_at, is_important)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gmail_id) DO NOTHING
                    RETURNING id;
                    """,
                    (user_id, gmail_id, thread_id, sender, subject, snippet,
                     received_at, is_important),
                )
                row = cur.fetchone()  # None on conflict (already scanned) - not an error
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_email skipped: {exc}")
        return None
