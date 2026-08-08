"""Opportunities found via the FIND_OPPORTUNITIES voice intent (app/tools/websearch.py).

Reads/writes the ``opportunities`` table exactly as pre-provisioned in
Supabase (uuid id, uuid user_id FK, jsonb raw). This module never creates or
alters that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_opportunity(
    user_id: str,
    name: str,
    type_: str,
    url: str,
    deadline: str | None,
    fit_score: float,
    reason: str,
    next_action: str,
    source: str,
    raw: dict,
) -> str | None:
    """Insert one opportunity (status always 'found'). Returns the new row's
    uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        from psycopg.types.json import Json

        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO opportunities
                        (user_id, name, type, url, deadline, fit_score, status,
                         reason, next_action, source, raw)
                    VALUES (%s, %s, %s, %s, %s, %s, 'found', %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, name, type_ or None, url or None, deadline,
                     fit_score, reason or None, next_action or None, source, Json(raw)),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_opportunity skipped: {exc}")
        return None


def list_recent_opportunities(user_id: str, limit: int = 10) -> list[dict]:
    """Return recent opportunities, most recent first. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, type, url, deadline, fit_score, status, created_at
                    FROM opportunities
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "name": r[1], "type": r[2], "url": r[3],
                "deadline": r[4], "fit_score": r[5], "status": r[6], "created_at": r[7],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_recent_opportunities skipped: {exc}")
        return []
