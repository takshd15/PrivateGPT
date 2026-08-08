"""RAG memory of past interactions: log every exchange, retrieve similar ones.

Before answering QUESTION/CONVERSATION intents, the most similar past
interactions are retrieved and given to the model as context, so Jarvix
"remembers" things across sessions instead of starting cold each time.
"""

from __future__ import annotations

from app.config import DATABASE_URL

from . import _connect


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id bigserial PRIMARY KEY,
                ts timestamptz NOT NULL DEFAULT now(),
                transcript text NOT NULL,
                intent text,
                response text,
                embedding vector(1536)
            );
            """
        )


def log_interaction(transcript: str, intent: str, response: str) -> None:
    """Embed and store one interaction. Never raises."""
    if not DATABASE_URL or not (transcript or "").strip():
        return
    try:
        from pgvector import Vector

        from app.brain.llm_client import embed
        from . import init_schema as _init_all

        _init_all()
        vector = Vector(embed(transcript))
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO interactions (transcript, intent, response, embedding) "
                    "VALUES (%s, %s, %s, %s);",
                    (transcript, intent, response, vector),
                )
            conn.commit()
    except Exception as exc:
        print(f"[memory] log_interaction skipped: {exc}")


def retrieve_similar(query: str, k: int = 3) -> list[dict]:
    """Return up to ``k`` most-similar past interactions.

    Returns [] on any failure (no DB configured, embedding failed, DB down).
    """
    if not DATABASE_URL or not (query or "").strip():
        return []
    try:
        from pgvector import Vector

        from app.brain.llm_client import embed
        from . import init_schema as _init_all

        _init_all()
        vector = Vector(embed(query))
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT transcript, response, ts FROM interactions "
                    "ORDER BY embedding <=> %s LIMIT %s;",
                    (vector, k),
                )
                rows = cur.fetchall()
        return [{"transcript": r[0], "response": r[1], "ts": r[2]} for r in rows]
    except Exception as exc:
        print(f"[memory] retrieve_similar skipped: {exc}")
        return []
