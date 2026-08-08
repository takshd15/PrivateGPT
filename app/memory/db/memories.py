"""Memories: read for opportunity-search context, written by passive memory
auto-update (app/brain/structuring.py's detect_memory_and_goal) and the
meeting follow-up flow (both via main.py).

Reads/writes the ``memories`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK). This module never creates or alters that table.
``embedding`` is intentionally left null on inserts here - these are short
structured facts/notes, not conversation transcripts (unlike interactions.py,
which embeds for RAG similarity search); nothing currently queries memories
by vector similarity.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all


def list_recent_memories(user_id: str, limit: int = 10) -> list[dict]:
    """Return the most important/recent active memories. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, kind, content, domain, entity_name, importance, created_at
                    FROM memories
                    WHERE user_id = %s AND status = 'active'
                    ORDER BY importance DESC, created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "kind": r[1], "content": r[2], "domain": r[3],
                "entity_name": r[4], "importance": r[5], "created_at": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_recent_memories skipped: {exc}")
        return []


def insert_memory(
    user_id: str,
    content: str,
    kind: str = "fact",
    domain: str | None = None,
    entity_name: str | None = None,
    importance: float = 0.5,
    source: str = "auto-convo",
) -> str | None:
    """Insert one memory. ``kind`` matches the live data's actual vocabulary
    (fact/task/progress/decision), not the column's unused 'profile' default.
    Returns the new row's uuid (as str), or None on failure. Never raises."""
    if not DATABASE_URL or not (content or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories
                        (user_id, kind, content, domain, entity_name, importance, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (user_id, kind, content, domain, entity_name, importance, source),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_memory skipped: {exc}")
        return None
