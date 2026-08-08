"""Goals: read for opportunity-search context, written by passive goal
detection (app/brain/structuring.py's detect_memory_and_goal, via main.py).

Reads/writes the ``goals`` table exactly as pre-provisioned in Supabase
(uuid id, uuid user_id FK). This module never creates or alters that table.

Duplicate suppression uses difflib.SequenceMatcher (same technique/threshold
style as app/tools/dedupe.py's calendar-event title matching) - not another
LLM call, per the spec.
"""

from __future__ import annotations
from difflib import SequenceMatcher

from app.config import DATABASE_URL

from . import _connect, init_schema as _init_all

SIMILARITY_THRESHOLD = 0.6


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def list_goals(user_id: str, limit: int = 10) -> list[dict]:
    """Return active goals ordered by priority, then recency. [] on any failure."""
    if not DATABASE_URL:
        return []
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, description, priority, status, created_at
                    FROM goals
                    WHERE user_id = %s AND status = 'active'
                    ORDER BY priority DESC, created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "title": r[1], "description": r[2],
                "priority": r[3], "status": r[4], "created_at": r[5],
            }
            for r in rows
        ]
    except Exception as exc:
        print(f"[memory] list_goals skipped: {exc}")
        return []


def find_similar_goal(user_id: str, title: str, threshold: float = SIMILARITY_THRESHOLD) -> dict | None:
    """Return the closest existing goal if its title similarity is >=
    threshold, else None. Compares against all goals (not just active), so a
    completed/dropped goal still suppresses a re-detected duplicate. [] on
    any failure treated as "no match" (never blocks an insert on a DB hiccup
    - a rare duplicate is far better than silently losing a real goal)."""
    if not DATABASE_URL or not (title or "").strip():
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, description, priority, status FROM goals WHERE user_id = %s;",
                    (user_id,),
                )
                rows = cur.fetchall()
        best = None
        best_score = 0.0
        for r in rows:
            score = _title_similarity(title, r[1])
            if score >= threshold and score > best_score:
                best_score = score
                best = {"id": str(r[0]), "title": r[1], "description": r[2], "priority": r[3], "status": r[4]}
        return best
    except Exception as exc:
        print(f"[memory] find_similar_goal skipped: {exc}")
        return None


def insert_goal(
    user_id: str, title: str, description: str = "", priority: int = 0, source: str = "auto-goal"
) -> str | None:
    """Insert one goal (status='active'). Returns the new row's uuid (as
    str), or None on failure. Never raises. ``source`` isn't a real column on
    goals (no such column exists) - callers wanting an audit trail should
    fold it into ``description`` if needed; kept as a parameter for call-site
    consistency with every other insert_* function in this package."""
    if not DATABASE_URL:
        return None
    try:
        _init_all()
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO goals (user_id, title, description, status, priority)
                    VALUES (%s, %s, %s, 'active', %s)
                    RETURNING id;
                    """,
                    (user_id, title, description or None, priority),
                )
                row = cur.fetchone()
            conn.commit()
        return str(row[0]) if row else None
    except Exception as exc:
        print(f"[memory] insert_goal skipped: {exc}")
        return None
