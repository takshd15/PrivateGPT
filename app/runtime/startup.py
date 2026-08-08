"""Login entry point for Jarvix: starts the wake loop in the background.

Launched on login by ``start_jarvix.vbs`` (hidden, no console). It runs in
``wakeword`` mode ("hey jarvis") - the ``enter`` mode needs a terminal, which a
hidden process doesn't have. Errors are written to ``jarvix.log`` next to the
project so a silent background crash is still debuggable.
"""

import datetime
import sys
import traceback
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "jarvix.log"


def _log(msg: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    except Exception:
        pass


def main() -> None:
    _log("startup: launching wake loop (wakeword mode)")
    try:
        from app.main import app
        from app.memory import db

        db.init_schema()
        # Force wakeword: a hidden background process has no stdin for Enter.
        sys.argv = ["jarvix", "wake", "--mode", "wakeword"]
        app()
    except SystemExit:
        pass
    except Exception:
        _log("CRASH:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
