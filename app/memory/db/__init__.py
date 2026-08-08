"""Postgres (Supabase) persistence for Jarvix: RAG memory + structured capture.

The Supabase project already has the full application schema provisioned
(tasks, projects, progress_events, opportunities, events, companies,
applications, emails, meetings, people, memories, goals, users, ...) with real
uuid primary keys and foreign keys to ``users``. Jarvix only creates the one
table it genuinely owns and that doesn't pre-exist: ``interactions`` (RAG
conversation memory), plus the ``vector`` extension both it and ``memories``
depend on. Every other submodule here (tasks, projects, progress_events, ...)
reads/writes the existing tables as-is and must never attempt to create or
alter them.

Every write/read function in every submodule degrades silently on failure
(missing DATABASE_URL, DB down, embedding call failed) - persistence here is
additive, never a reason to break the voice loop.
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
    """Create the vector extension + interactions table if missing. Idempotent, never raises.

    Only touches tables Jarvix itself owns. tasks/projects/progress_events/etc.
    are pre-provisioned in Supabase and are never created or altered here, with
    one deliberate exception: meetings.followup_status, added by
    meetings.py's own migration step below - the opt-out-permanence feature
    has no other durable, query-able home, and this is the only ALTER TABLE
    anywhere in this package.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY or not DATABASE_URL:
        return
    from . import interactions, meetings

    try:
        with _raw_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            interactions.init_schema(conn)
            meetings.init_schema(conn)
            conn.commit()
        _SCHEMA_READY = True
    except Exception as exc:
        print(f"[memory] schema init skipped: {exc}")


def log_interaction(transcript: str, intent: str, response: str) -> None:
    from . import interactions

    interactions.log_interaction(transcript, intent, response)


def retrieve_similar(query: str, k: int = 3) -> list[dict]:
    from . import interactions

    return interactions.retrieve_similar(query, k=k)
