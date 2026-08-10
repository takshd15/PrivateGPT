"""High-impact action gate for the browser agent loop.

Separate from app/safety/permissions.py (which gates a small fixed set of
existing non-browser actions like send_email) - this module classifies
actions an open-ended browser agent proposes, which are inherently more
varied (any button on any website), so it works off both an explicit
tool-level flag the LLM sets and a keyword backstop over the element the
agent is about to interact with. Only used to decide whether to PAUSE for
confirmation; it never itself executes or blocks anything.
"""

from __future__ import annotations

import re

from app.config import JARVIX_BROWSER_BLOCKED_DOMAINS

# Verbs/phrases on the button/link/element label itself that mean "this click
# has a real-world consequence." Checked against the target element's label
# as a backstop even when the LLM didn't flag its own action as high-impact -
# a model can be wrong about how consequential a click is, especially on an
# unfamiliar site.
_HIGH_IMPACT_LABEL_PATTERNS = [
    r"\bsend\b",
    r"\bsubmit\b",
    r"\bpay\b",
    r"\bpurchase\b",
    r"\bbuy\s*(now)?\b",
    r"\bcheckout\b",
    r"\bplace\s+order\b",
    r"\bconfirm\s+(order|purchase|booking|payment)\b",
    r"\bbook\s*(now|flight|hotel)?\b",
    r"\breserve\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\btrash\b",
    r"\bdiscard\b",
    r"\bpublish\b",
    r"\bpost\b",
    r"\btweet\b",
    r"\bshare\b",
    r"\bchange\s+password\b",
    r"\breset\s+password\b",
    r"\bupdate\s+password\b",
    r"\bchange\s+email\b",
    r"\bdeactivate\b",
    r"\bdelete\s+account\b",
    r"\baccept\s+(terms|agreement|contract)\b",
    r"\bagree\b",
    r"\bi\s+agree\b",
    r"\btransfer\b",
    r"\bwithdraw\b",
    r"\bdonate\b",
    r"\bsubscribe\b",
    r"\bcancel\s+subscription\b",
    r"\bunsubscribe\b",
]
_HIGH_IMPACT_RE = re.compile("|".join(_HIGH_IMPACT_LABEL_PATTERNS), re.I)

# Signals a page needs a human (CAPTCHA/2FA/etc) - Jarvix must pause and hand
# off rather than attempt to solve or click through these.
_AUTH_CHALLENGE_PATTERNS = [
    r"\bcaptcha\b",
    r"\bi'?m not a robot\b",
    r"\bverify (you'?re|it'?s) (a human|you)\b",
    r"\btwo-factor\b",
    r"\b2fa\b",
    r"\bone-time (code|password)\b",
    r"\bone time (code|password)\b",
    r"\bverification code\b",
    r"\bauthenticator\b",
    r"\bsecurity key\b",
    r"\bconfirm your identity\b",
    r"\bbiometric\b",
    r"\bre-?enter your password\b",
    r"\bunusual (sign-?in|login) activity\b",
]
_AUTH_CHALLENGE_RE = re.compile("|".join(_AUTH_CHALLENGE_PATTERNS), re.I)


def label_is_high_impact(label: str) -> bool:
    return bool(_HIGH_IMPACT_RE.search(label or ""))


def page_has_auth_challenge(page_text: str, page_title: str = "") -> bool:
    combined = f"{page_title} {page_text}"
    return bool(_AUTH_CHALLENGE_RE.search(combined))


def describe_high_impact_action(action: str, target_label: str, extra: dict | None = None) -> str:
    """Short human-readable description of a paused action, for the spoken
    confirmation prompt and jarvix.log."""
    extra = extra or {}
    details = ", ".join(f"{k}={v}" for k, v in extra.items() if v)
    base = f"{action} on '{target_label}'" if target_label else action
    return f"{base} ({details})" if details else base


def blocked_domain_for(url: str) -> str | None:
    """The blocklist entry a URL matches, or None if it's allowed. Substring
    match on the whole URL (lowercased) so an entry like 'chase.com' catches
    'www.chase.com', 'secure.chase.com/login', etc. Enforced in real-Chrome
    mode by app/browser/actions.goto and the agent loop - a hard wall the LLM
    cannot argue its way past, protecting the accounts most costly to have an
    agent poking at (banking, brokerages, password managers)."""
    lowered = (url or "").lower()
    for entry in JARVIX_BROWSER_BLOCKED_DOMAINS:
        if entry and entry in lowered:
            return entry
    return None


def is_blocked_url(url: str) -> bool:
    return blocked_domain_for(url) is not None
