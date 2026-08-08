"""Login entry point for Jarvix: starts the wake loop in the background.

Launched on login by ``start_jarvix.vbs`` (hidden, no console). It runs in
``wakeword`` mode ("hey jarvis") - the ``enter`` mode needs a terminal, which a
hidden process doesn't have. Errors are written to ``jarvix.log`` next to the
project so a silent background crash is still debuggable.
"""

import sys
import traceback

from app.runtime.log import log as _log


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
