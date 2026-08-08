"""Tasks captured via the REMEMBER voice/typed intent (app/brain/structuring.py).

Reads/writes the ``tasks`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id/project_id FKs). This module never creates or alters
that table - only app/memory/db/interactions.py owns a CREATE TABLE.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_task(
    user_id: str,
    title: str,
    details: str = "",
    due_date: str | None = None,
    priority: int = 0,
    entity_name: str | None = None,
    domain: str | None = None,
    project_id: str | None = None,
) -> str | None:
    """Insert one task. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks
                        (user_id, project_id, title, details, due_date, priority,
                         entity_name, domain)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, project_id, title, details or None, due_date, priority,
                     entity_name, domain),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_task skipped: {exc}")
        return None


def list_open_tasks(user_id: str, limit: int = 20) -> list[dict]:
    """Return open tasks, most recent first. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, details, due_date, priority, entity_name, domain
                    FROM tasks
                    WHERE user_id = %s AND status = 'open'
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "title": r[1], "details": r[2], "due_date": r[3],
                "priority": r[4], "entity_name": r[5], "domain": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_open_tasks skipped: {exc}")
        return []
