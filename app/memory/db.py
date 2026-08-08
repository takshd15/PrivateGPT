"""Postgres (Supabase) RAG memory of past interactions.

Every voice/typed interaction is logged with an embedding of its transcript.
Before answering QUESTION/CONVERSATION intents, the most similar past
interactions are retrieved and given to the model as context, so Jarvix
"remembers" things across sessions instead of starting cold each time.

Every function here degrades silently on failure (missing DATABASE_URL, DB
down, embedding call failed) - memory is a nice-to-have, never a reason to
break the voice loop.
"""

from __future__ import annotations

from app.config import DATABASE_URL

_SCHEMA_READY = False


def _raw_connect():
    import psycopg

    return psycopg.connect(DATABASE_URL, connect_timeout=5, sslmode="require")


def _connect():
    """Connection with the pgvector type registered. Schema must exist already."""
    from pgvector.psycopg import register_vector

    conn = _raw_connect()
    register_vector(conn)
    return conn


def init_schema() -> None:
    """Create the pgvector extension and interactions table if missing. Idempotent."""
    global _SCHEMA_READY
    if _SCHEMA_READY or not DATABASE_URL:
        return
    try:
        with _raw_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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
            conn.commit()
        _SCHEMA_READY = True
    except Exception as exc:
        print(f"[memory] schema init skipped: {exc}")


def log_interaction(transcript: str, intent: str, response: str) -> None:
    """Embed and store one interaction. Never raises."""
    if not DATABASE_URL or not (transcript or "").strip():
        return
    try:
        from pgvector import Vector

        from app.brain.llm_client import embed

        init_schema()
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
    """Return up to ``k`` most-similar past interactions, newest ties broken last.

    Returns [] on any failure (no DB configured, embedding failed, DB down).
    """
    if not DATABASE_URL or not (query or "").strip():
        return []
    try:
        from pgvector import Vector

        from app.brain.llm_client import embed

        init_schema()
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
