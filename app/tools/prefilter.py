"""Cheap, deterministic prefilter to avoid sending junk emails to the model.

An email is worth extracting only if it looks like it carries a real commitment
(event/deadline keyword, or an explicit date/time). Obvious promo/receipt mail is
dropped unless it also has a strong commitment keyword.
"""
import re

EVENT_KEYWORDS = {
    "deadline", "due", "interview", "meeting", "appointment", "booking",
    "reservation", "confirmed", "calendar", "event", "hackathon", "submission",
    "application", "accepted", "rejected", "schedule", "invite", "webinar",
    "flight", "ticket", "exam", "payment due", "reminder",
}

JUNK_KEYWORDS = {
    "promo", "sale", "discount", "% off", "offer", "unsubscribe", "newsletter",
    "receipt", "delivery", "order confirmation",
}

# Strong signals that override a junk match (a real interview can arrive in a
# templated "no-reply" email that also looks promotional).
STRONG_KEYWORDS = {
    "interview", "hackathon", "exam", "deadline", "submission", "appointment",
}

_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}"
    r"|\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}\s*(am|pm)?|\d{1,2}\s*(am|pm))\b", re.IGNORECASE)


def _text(email: dict) -> str:
    body = (email.get("body") or "")[:600]
    return " ".join(
        [email.get("subject", ""), email.get("snippet", ""), body]
    ).lower()


def is_worthy(email: dict) -> bool:
    text = _text(email)
    strong = any(k in text for k in STRONG_KEYWORDS)
    if any(j in text for j in JUNK_KEYWORDS) and not strong:
        return False
    has_keyword = any(k in text for k in EVENT_KEYWORDS)
    has_datetime = bool(_DATE_RE.search(text) or _TIME_RE.search(text))
    return has_keyword or has_datetime


def prefilter(emails: list[dict]) -> list[dict]:
    return [e for e in emails if is_worthy(e)]
