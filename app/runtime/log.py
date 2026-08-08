"""Shared minimal file logger for background/headless code paths.

Both the login entry point (startup.py, no console) and the background
scheduler (scheduler.py, runs inside `wake`) need somewhere to record what
happened without a visible terminal. Appends timestamped lines to jarvix.log
next to the project. Never raises - a logging failure must never take down
whatever it was logging for.
"""

from __future__ import annotations

import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "jarvix.log"


def log(msg: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    except Exception:
        pass
