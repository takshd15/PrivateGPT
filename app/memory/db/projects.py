"""Projects captured via the NEW_PROJECT voice/typed intent (app/brain/structuring.py).

Reads/writes the ``projects`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK). This module never creates or alters that table.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def insert_project(
    user_id: str,
    name: str,
    description: str = "",
    status: str = "active",
) -> str | None:
    """Insert one project. Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO projects (user_id, name, description, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, name, description or None, status),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_project skipped: {exc}")
        return None


def list_projects(user_id: str, limit: int = 50) -> list[dict]:
    """Return projects, most recent first. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, description, status
                    FROM projects
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {"id": str(r[0]), "name": r[1], "description": r[2], "status": r[3]}
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_projects skipped: {exc}")
        return []
