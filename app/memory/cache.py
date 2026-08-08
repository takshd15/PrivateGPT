"""Local SQLite cache for email extraction results.

Re-scanning the inbox shouldn't re-run the model on emails we've already seen.
We key cached extraction by Gmail message id + a hash of the email content, so a
cache hit only happens when the same message is unchanged.

The DB stores email snippets/extractions (private data) and is gitignored.
"""
import hashlib
import json
import sqlite3
import time

from app.config import ROOT

DB_PATH = ROOT / "jarvix_cache.sqlite"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_cache (
            message_id    TEXT PRIMARY KEY,
            subject       TEXT,
            date          TEXT,
            snippet_hash  TEXT,
            extracted_json TEXT,
            extracted_at  REAL
        )
        """
    )
    return conn


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def get_cached(message_id: str, snippet_hash: str) -> list[dict] | None:
    """Return cached candidate dicts for an unchanged message, else None."""
    if not message_id:
        return None
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT extracted_json FROM email_cache WHERE message_id=? AND snippet_hash=?",
            (message_id, snippet_hash),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        data = json.loads(row[0])
        return data if isinstance(data, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


def set_cached(
    message_id: str,
    subject: str,
    date: str,
    snippet_hash: str,
    candidates: list[dict],
) -> None:
    if not message_id:
        return
    conn = _conn()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO email_cache
                (message_id, subject, date, snippet_hash, extracted_json, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, date, snippet_hash, json.dumps(candidates), time.time()),
        )
        conn.commit()
    finally:
        conn.close()
