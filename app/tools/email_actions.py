"""Email drafting and sending for Jarvix v2.

Drafting (Phase 2.9): resolve the recipient from a small contacts allowlist,
then send the recipient name and instruction to OpenAI to write the subject
and body. Sending (Phase 2.10) uses the Gmail API and is only ever called
after an explicit confirmation in the caller - this module never auto-sends.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.utils import parseaddr
from pathlib import Path

from app.brain.llm_client import ask_llm
from app.tools.google_auth import gmail_service

CONTACTS_FILE = Path(__file__).resolve().parents[1] / "memory" / "contacts.json"


@dataclass
class EmailDraft:
    to_name: str
    to_email: str
    subject: str
    body: str


# --------------------------------------------------------------------------- #
# Contacts
# --------------------------------------------------------------------------- #
def load_contacts() -> dict[str, str]:
    if not CONTACTS_FILE.exists():
        return {}
    return json.loads(CONTACTS_FILE.read_text(encoding="utf-8"))


def resolve_recipient(name: str) -> str | None:
    """Look up an email by contact name (case-insensitive). None if unknown."""
    if not name:
        return None
    value = name.strip()
    if _valid_email(value):
        return value
    return load_contacts().get(value.lower())


def _valid_email(value: str) -> bool:
    parsed = parseaddr(value.strip())[1]
    local, separator, domain = parsed.rpartition("@")
    return bool(separator and local and "." in domain and " " not in parsed)


def save_contact(name: str, email: str) -> None:
    if not _valid_email(email):
        raise ValueError("That doesn't look like a valid email address.")
    contacts = load_contacts()
    contacts[name.strip().lower()] = email.strip()
    CONTACTS_FILE.write_text(json.dumps(contacts, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Drafting
# --------------------------------------------------------------------------- #
def compose(to_name: str, instruction: str) -> tuple[str, str]:
    """Ask the model to write a short email. Returns (subject, body)."""
    system_prompt = (
        "You write short, polite, professional emails. "
        "Return ONLY JSON with keys 'subject' and 'body'. "
        "Keep the body to 2-4 sentences. Sign off as the sender's name is unknown, "
        "so end with 'Best regards,' on its own line and nothing after it. "
        "Do not invent facts beyond the instruction."
    )
    user_prompt = f"Recipient first name: {to_name}\nInstruction: {instruction}"

    raw = ask_llm(system_prompt, user_prompt, json_mode=True, num_predict=300)
    try:
        data = json.loads(raw)
        subject = (data.get("subject") or "").strip() or "(no subject)"
        body = (data.get("body") or "").strip()
    except (json.JSONDecodeError, AttributeError):
        # Fall back to a plain draft if the model didn't return clean JSON.
        subject = "(no subject)"
        body = raw.strip()
    return subject, body


def build_draft(to_name: str, to_email: str, instruction: str) -> EmailDraft:
    subject, body = compose(to_name, instruction)
    return EmailDraft(to_name=to_name, to_email=to_email, subject=subject, body=body)


# --------------------------------------------------------------------------- #
# Sending (Phase 2.10) - caller must confirm first
# --------------------------------------------------------------------------- #
def send_email(draft: EmailDraft) -> dict:
    """Send the draft via Gmail. Raises on failure; returns the API response."""
    message = MIMEText(draft.body)
    message["to"] = draft.to_email
    message["subject"] = draft.subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

    service = gmail_service()
    return (
        service.users()
        .messages()
        .send(userId="me", body={"raw": raw})
        .execute()
    )
