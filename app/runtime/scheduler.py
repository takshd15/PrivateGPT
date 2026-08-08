"""Background autopilot: one daemon thread running scheduled jobs alongside
the wake listener - not a separate process the user has to launch.

Each job wraps its body in a broad try/except (same "never crash the caller"
philosophy as app/memory/db/*) and logs to jarvix.log via app/runtime/log.py.
A failing job just waits for its next interval rather than retry-storming.
Every write from a job is tagged source='autopilot:<job-name>' so it's
distinguishable from voice-confirmed writes in the same tables.
"""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import (
    AUTOPILOT_APPLICATIONS,
    AUTOPILOT_EMAIL_SCAN_MINUTES,
    AUTOPILOT_ENABLED,
    AUTOPILOT_EVENTS,
    ROOT,
)
from app.runtime.log import log

STATE_FILE = ROOT / "app" / "memory" / "autopilot_state.json"

_NOREPLY_PATTERN = re.compile(r"no[-._]?reply|do[-._]?not[-._]?reply", re.I)


def _write_state(job_name: str, **counts) -> None:
    try:
        state = {}
        if STATE_FILE.exists():
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state[job_name] = {
            "last_run": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **counts,
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"[autopilot] state write failed: {exc}")


def read_state() -> dict:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _is_noreply(email_address: str) -> bool:
    return bool(_NOREPLY_PATTERN.search(email_address or ""))


def _parse_email_date(raw: str | None):
    """Gmail's Date header is RFC 2822 ("Sat, 08 Aug 2026 07:23:57 +0000
    (UTC)"), which timestamptz columns don't accept as a bare string via
    psycopg. Parse it into a real datetime; None if unparseable."""
    if not raw:
        return None
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def run_email_scan(user_id: str) -> dict:
    """Fetch recent email once, run the combined signup+application extraction
    per new/changed email, write to emails/events/companies/applications/people.

    Returns a dict of counts for autopilot-status. Never raises.
    """
    from app.memory import cache
    from app.memory.db import applications as db_applications
    from app.memory.db import companies as db_companies
    from app.memory.db import emails as db_emails
    from app.memory.db import events as db_events
    from app.memory.db import people as db_people
    from app.tools.gmail import get_recent_emails
    from app.brain.structuring import structure_email_signals

    counts = {
        "emails_scanned": 0, "emails_new": 0, "events_added": 0,
        "companies_added": 0, "applications_added": 0, "applications_updated": 0,
        "people_upserted": 0, "errors": 0,
    }
    try:
        emails = get_recent_emails(limit=10, days=7)
        counts["emails_scanned"] = len(emails)

        for email in emails:
            message_id = email.get("id", "")
            chash = cache.content_hash((email.get("snippet") or "") + (email.get("body") or ""))
            if cache.get_cached(message_id, chash) is not None:
                continue  # already processed, unchanged
            counts["emails_new"] += 1

            try:
                signals = structure_email_signals(email)
            except Exception as exc:
                log(f"[autopilot:email-scan] extraction failed for {message_id}: {exc}")
                counts["errors"] += 1
                cache.set_cached(message_id, email.get("subject", ""), email.get("date", ""), chash, [])
                continue

            db_emails.insert_email(
                user_id,
                gmail_id=message_id,
                thread_id=email.get("thread_id", ""),
                sender=email.get("from", ""),
                subject=email.get("subject", ""),
                snippet=email.get("snippet", ""),
                received_at=_parse_email_date(email.get("date")),
                is_important=bool(signals.signup or signals.application),
            )

            if AUTOPILOT_EVENTS and signals.signup:
                from app.tools import live_info

                s = signals.signup
                event_date = None
                if s.date_phrase:
                    resolved = live_info.resolve_date_phrase(s.date_phrase)
                    event_date = resolved.isoformat() if resolved else s.date_phrase
                event_id = db_events.insert_event(
                    user_id,
                    "signup",
                    {
                        "name": s.name, "event_type": s.event_type, "date": event_date,
                        "location": s.location, "confirmation_details": s.confirmation_details,
                        "source_email_id": message_id, "source": "autopilot:event-scan",
                    },
                )
                if event_id:
                    counts["events_added"] += 1

            if AUTOPILOT_APPLICATIONS and signals.application:
                a = signals.application
                if a.company_name.strip():
                    company = db_companies.find_company_by_name(user_id, a.company_name)
                    company_id = company["id"] if company else None
                    if not company:
                        company_id = db_companies.insert_company(user_id, a.company_name)
                        if company_id:
                            counts["companies_added"] += 1
                    else:
                        db_companies.append_company_notes(
                            company_id, f"Email: {email.get('subject', '')}"
                        )

                    existing_app = db_applications.find_application(user_id, a.company_name, a.program)
                    note_line = f"Email: {email.get('subject', '')} - status: {a.status}"
                    if existing_app:
                        db_applications.update_application_status(existing_app["id"], a.status, note_line)
                        counts["applications_updated"] += 1
                    else:
                        new_app_id = db_applications.insert_application(
                            user_id, a.company_name, program=a.program, status=a.status, notes=note_line
                        )
                        if new_app_id:
                            counts["applications_added"] += 1

                    if a.contact_name and a.contact_email and not _is_noreply(a.contact_email):
                        person_id = db_people.upsert_person(
                            user_id, a.contact_name, a.contact_email, company_id=company_id
                        )
                        if person_id:
                            counts["people_upserted"] += 1

            cache.set_cached(message_id, email.get("subject", ""), email.get("date", ""), chash, [])

        log(f"[autopilot:email-scan] {counts}")
    except Exception as exc:
        log(f"[autopilot:email-scan] job failed: {exc}")
        counts["errors"] += 1

    _write_state("email-scan", **counts)
    return counts


class _Job:
    def __init__(self, name: str, interval_seconds: float, fn) -> None:
        self.name = name
        self.interval_seconds = interval_seconds
        self.fn = fn
        self.next_due = time.monotonic()  # due immediately on first check

    def run_if_due(self, user_id: str) -> None:
        if time.monotonic() < self.next_due:
            return
        # Always advance next_due before running, so a failing job can't retry-storm.
        self.next_due = time.monotonic() + self.interval_seconds
        try:
            self.fn(user_id)
        except Exception as exc:
            log(f"[autopilot] job {self.name} raised: {exc}")


class BackgroundScheduler:
    """One daemon thread, checked every ~60s, running any due jobs."""

    def __init__(self) -> None:
        self._jobs: list[_Job] = [
            _Job("email-scan", AUTOPILOT_EMAIL_SCAN_MINUTES * 60, run_email_scan),
        ]
        self._thread: threading.Thread | None = None

    def _loop(self, user_id: str) -> None:
        while True:
            for job in self._jobs:
                job.run_if_due(user_id)
            time.sleep(60)

    def start(self, user_id: str) -> None:
        if not AUTOPILOT_ENABLED or not user_id:
            log("[autopilot] scheduler not started (disabled or no USER_ID)")
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, args=(user_id,), name="jarvix-autopilot", daemon=True
        )
        self._thread.start()
        log("[autopilot] scheduler started")

    def run_job_now(self, job_name: str, user_id: str) -> dict:
        """Manual trigger (autopilot-run CLI command) - runs a job's function
        directly, bypassing the interval check."""
        for job in self._jobs:
            if job.name == job_name:
                return job.fn(user_id)
        raise ValueError(f"Unknown job '{job_name}'. Known jobs: {[j.name for j in self._jobs]}")


_scheduler = BackgroundScheduler()


def start(user_id: str) -> None:
    _scheduler.start(user_id)


def run_job_now(job_name: str, user_id: str) -> dict:
    return _scheduler.run_job_now(job_name, user_id)
