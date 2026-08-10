"""Owns the single Playwright browser context Jarvix drives.

A dedicated, persistent Chrome profile (JARVIX_CHROME_PROFILE_DIR) keeps
Jarvix logged into whatever the user signs into through it, separate from
their normal Chrome profile. The context is started lazily on first use and
kept alive across voice turns/tabs so a login survives between commands -
only `shutdown()` (process exit) tears it down.

Playwright's sync API is bound to whichever OS thread started it - EVERY
call on that context/page/locator must happen on that same thread, or it
raises. app/main.py's CLI/wake-loop paths are naturally single-threaded so
this was never an issue there, but app/server.py's /api/command handler runs
each request's app/brain/orchestrator.py loop on a fresh worker thread
(see app/server.py's `worker()`), so a naive singleton would start Chrome on
request #1's thread and then fail on request #2's different thread. Fixed by
running Playwright itself on ONE dedicated background thread
(`_BrowserThread`) owned by this module, and marshalling every call onto it
via `run()` - callers (app/browser/actions.py, state.py, tools.py) don't
need to know this happens; they just call Page/BrowserContext methods as
normal Python objects from whatever thread they're on.
"""

from __future__ import annotations

import json
import queue
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, TypeVar

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from app.config import (
    JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS,
    JARVIX_BROWSER_CDP_URL,
    JARVIX_BROWSER_CHANNEL,
    JARVIX_BROWSER_HEADLESS,
    JARVIX_BROWSER_MODE,
    JARVIX_CHROME_PROFILE_DIR,
)
from app.runtime.log import log as _file_log

T = TypeVar("T")


class BrowserUnavailable(Exception):
    """Raised when the browser context can't be started (Playwright/Chrome not installed)."""


def _log(msg: str) -> None:
    _file_log(f"[JARVIS] {msg}")


def _probe_cdp_available(cdp_url: str, timeout: float) -> bool:
    """Cheap, fast reachability check for the classic Chrome DevTools
    Protocol HTTP endpoint (/json/version) - the one Playwright's
    connect_over_cdp needs. Used by 'auto' mode to decide, on every fresh
    browser start, whether the user's real Chrome is actually attachable
    right now, without paying the cost of a full connect_over_cdp() attempt
    (which is slower to fail and would otherwise delay every task back to
    the dedicated profile whenever the user hasn't launched Chrome with
    --remote-debugging-port). Never raises - any failure just means "not
    available", which is exactly what a probe should report."""
    url = cdp_url.rstrip("/") + "/json/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            # A real CDP endpoint's /json/version always has this key - a
            # 200 from some unrelated service on that port shouldn't count.
            return "webSocketDebuggerUrl" in payload or "Browser" in payload
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return False


class _BrowserThreadHung(Exception):
    """Raised when a Playwright call didn't return within the hard timeout -
    the underlying connection is presumed dead (e.g. the Chrome process
    crashed or its pipe broke mid-call), not just slow. Distinct from a
    normal Playwright timeout (which raises promptly and cleanly): this is
    for the case a call never returns AT ALL, which would otherwise wedge
    the single dedicated browser thread forever and silently fail every
    future browser task with no error shown to the user (see jarvix.log
    2026-08-10: a mid-task 'Connection closed while reading from the driver'
    was handled, but the NEXT action after it hung with zero log output)."""


# Hard ceiling on any single Playwright call. Generous relative to
# BROWSER_ACTION_TIMEOUT_SECONDS/BROWSER_NAV_TIMEOUT_SECONDS (which bound
# well-behaved Playwright waits) - this is a last-resort backstop for calls
# that don't respect their own timeout because the connection itself died.
_HARD_CALL_TIMEOUT_SECONDS = 45.0


class _BrowserThread:
    """A single, long-lived worker thread that owns Playwright end to end.
    Every Playwright call from anywhere in app/browser/* is marshalled onto
    this thread via `run()`, so it never matters which thread the caller
    (a CLI command, a wake-loop turn, or one of app/server.py's per-request
    worker threads) happens to be running on.

    If a call doesn't return within _HARD_CALL_TIMEOUT_SECONDS, `run()`
    raises _BrowserThreadHung to its caller AND retires this worker thread,
    replacing it with a fresh one - the old thread may still be stuck
    running the hung call forever (Python has no way to forcibly cancel a
    running thread), so it's abandoned as a daemon rather than reused, which
    would otherwise silently wedge every future browser task behind it with
    no error at all.
    """

    def __init__(self) -> None:
        self._jobs: "queue.Queue[tuple[Callable, tuple, dict, queue.Queue]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="jarvix-browser", daemon=True)
        self._thread.start()
        # Set by BrowserManager after both singletons exist - lets a hang
        # invalidate the manager's cached Playwright/context/page references
        # too, not just this thread. Those objects belong to the abandoned
        # thread's dead connection; a fresh thread can't use them, so
        # _start_impl must be told to relaunch from scratch rather than
        # trying (and failing forever) to reuse them.
        self.on_hang: Callable[[], None] | None = None

    def _run(self) -> None:
        while True:
            fn, args, kwargs, result_q = self._jobs.get()
            try:
                result_q.put((True, fn(*args, **kwargs)))
            except BaseException as exc:  # noqa: BLE001 - must relay every failure to the caller, never swallow
                result_q.put((False, exc))

    def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if threading.current_thread() is self._thread:
            # Already on the browser thread (e.g. a nested call from within
            # another run()) - call directly instead of deadlocking on our
            # own queue.
            return fn(*args, **kwargs)
        result_q: "queue.Queue[tuple[bool, object]]" = queue.Queue()
        self._jobs.put((fn, args, kwargs, result_q))
        try:
            ok, value = result_q.get(timeout=_HARD_CALL_TIMEOUT_SECONDS)
        except queue.Empty:
            _log(
                f"Browser call hung for over {_HARD_CALL_TIMEOUT_SECONDS:.0f}s "
                "(connection likely dead) - abandoning it and starting a fresh browser thread"
            )
            self._replace_thread()
            if self.on_hang is not None:
                try:
                    self.on_hang()
                except Exception:
                    pass
            raise _BrowserThreadHung(
                f"A browser action didn't respond within {_HARD_CALL_TIMEOUT_SECONDS:.0f}s - "
                "the connection to Chrome was likely lost."
            )
        if not ok:
            raise value
        return value

    def _replace_thread(self) -> None:
        """Abandon the (possibly still-stuck) worker thread and start a
        fresh one with an empty job queue, so the NEXT browser call gets a
        clean thread instead of queuing up behind a call that may never
        return. The old thread is a daemon, so it won't block process exit
        even if it never finishes."""
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="jarvix-browser", daemon=True)
        self._thread.start()


_browser_thread = _BrowserThread()


class BrowserManager:
    """Lazily-started, singleton-per-process Playwright Chrome context.

    Every public method (including `run`) marshals its actual Playwright
    work onto the single `_browser_thread` via `_browser_thread.run(...)` -
    callers can invoke these from any thread (a CLI command, the wake loop,
    or one of app/server.py's per-request worker threads) and it's always
    safe, because the Playwright objects themselves are only ever touched
    from the one thread that created them.
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._active_page: Optional[Page] = None
        # True when attached to the user's own running Chrome over CDP rather
        # than owning a launched instance - governs teardown (disconnect, do
        # NOT close their browser) and is surfaced so the agent loop can apply
        # the stricter real-Chrome safety gates.
        self._attached_to_real_chrome: bool = False
        self._browser = None  # the connect_over_cdp Browser handle, CDP mode only
        # Flipped by the context's own "close" event - fires whether the
        # window was closed by the user, crashed, or (real-Chrome mode) the
        # whole browser process quit. _start_impl checks this and relaunches
        # instead of handing back a dead reference that fails every call
        # forever after (see the "Target page, context or browser has been
        # closed" bug this fixes).
        self._context_closed: bool = False
        # True only when _context_closed was set by a HUNG call (vs. a
        # normal page/context "close" event) - see _mark_dead. Distinguishes
        # the two in _start_impl because a hang means self._playwright lives
        # on an abandoned, possibly-still-stuck thread and must not be
        # touched from the new thread during teardown (see _teardown_partial).
        self._dead_from_hang: bool = False
        _browser_thread.on_hang = self._mark_dead

    def _mark_dead(self) -> None:
        """Called from _BrowserThread when a call hung and the worker thread
        was abandoned - the cached Playwright/context/page here belong to
        that now-abandoned thread's connection and are unusable from the
        fresh replacement thread, so mark them dead the same way an
        externally-closed browser is: _start_impl sees this and relaunches
        from scratch on the next call instead of reusing broken references."""
        self._context_closed = True
        self._dead_from_hang = True

    @property
    def is_running(self) -> bool:
        return self._context is not None

    @property
    def is_real_chrome(self) -> bool:
        return self._attached_to_real_chrome

    def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run any Playwright-touching callable on the browser thread. Use
        this from app/browser/actions.py and state.py for anything beyond
        the plain page/context accessors already provided below (e.g.
        page.evaluate, locator.click, page.screenshot)."""
        return _browser_thread.run(fn, *args, **kwargs)

    def start(self) -> BrowserContext:
        """Start (or return the already-running) persistent Chrome context."""
        return _browser_thread.run(self._start_impl)

    def _start_impl(self) -> BrowserContext:
        if self._context is not None and not self._context_closed:
            return self._context

        if self._context is not None and self._context_closed:
            # The window was closed/crashed since last time - the cached
            # context/playwright handles are dead and every call on them
            # would keep raising "Target page, context or browser has been
            # closed" forever. Drop them and relaunch fresh instead.
            if self._dead_from_hang:
                _log("Previous browser connection hung and was abandoned - relaunching")
            else:
                _log("Browser was closed since last use - relaunching")
            self._teardown_partial(skip_stop=self._dead_from_hang)

        if JARVIX_BROWSER_MODE == "dedicated":
            return self._start_dedicated_impl()

        if JARVIX_BROWSER_MODE == "real":
            # Explicit opt-in to ALWAYS use real Chrome - fail loudly (not a
            # silent fallback) so setup problems are obvious, not masked.
            return self._start_real_chrome_impl()

        # "auto" (the default): a fast, cheap reachability probe decides.
        # Nothing is assumed - if the user hasn't launched Chrome with
        # --remote-debugging-port, this silently and correctly lands on the
        # dedicated profile, same as always worked before real-Chrome mode
        # existed. If they have, Jarvix gets their actual logged-in session
        # with no flag to remember to flip.
        if _probe_cdp_available(JARVIX_BROWSER_CDP_URL, JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS):
            try:
                return self._start_real_chrome_impl()
            except BrowserUnavailable as exc:
                _log(f"Real Chrome was reachable but attach failed, falling back to dedicated profile: {exc}")
        else:
            _log(f"No real Chrome detected at {JARVIX_BROWSER_CDP_URL} - using the dedicated profile")
        return self._start_dedicated_impl()

    def _on_context_closed(self) -> None:
        self._context_closed = True

    def _start_dedicated_impl(self) -> BrowserContext:
        profile_dir = Path(JARVIX_CHROME_PROFILE_DIR)
        profile_dir.mkdir(parents=True, exist_ok=True)

        _log(f"Starting Chrome (dedicated profile: {profile_dir})")
        try:
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                channel=JARVIX_BROWSER_CHANNEL or None,
                headless=JARVIX_BROWSER_HEADLESS,
                viewport={"width": 1366, "height": 900},
                args=["--start-maximized"] if not JARVIX_BROWSER_HEADLESS else [],
            )
        except Exception as exc:
            self._teardown_partial()
            raise BrowserUnavailable(
                f"Couldn't start Chrome for browser control: {exc}. "
                "Run 'python -m playwright install chrome' and try again."
            ) from exc

        self._context_closed = False
        self._context.on("page", self._on_new_page)
        self._context.on("close", self._on_context_closed)
        if self._context.pages:
            self._active_page = self._context.pages[0]
        else:
            self._active_page = self._context.new_page()
        return self._context

    def _start_real_chrome_impl(self) -> BrowserContext:
        """Attach to the user's ALREADY-RUNNING Chrome over CDP. We adopt its
        existing context/tabs rather than creating our own, and mark
        _attached_to_real_chrome so teardown only disconnects (never closes
        the user's browser) and the agent loop applies the stricter gates."""
        _log(f"Attaching to your real Chrome over CDP ({JARVIX_BROWSER_CDP_URL})")
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.connect_over_cdp(JARVIX_BROWSER_CDP_URL)
        except Exception as exc:
            self._teardown_partial()
            raise BrowserUnavailable(
                f"Couldn't attach to your real Chrome at {JARVIX_BROWSER_CDP_URL}: {exc}. "
                "Note: the in-browser 'Allow remote debugging for this browser instance' "
                "toggle (the DevTools MCP one) does NOT expose the endpoint this needs. "
                "Fully quit Chrome, then relaunch it from a terminal with:  "
                'chrome.exe --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\\Google\\Chrome\\User Data"  '
                "and try again."
            ) from exc

        contexts = self._browser.contexts
        if not contexts:
            self._teardown_partial()
            raise BrowserUnavailable("Attached to Chrome but it has no open window/context to drive.")
        self._context = contexts[0]
        self._attached_to_real_chrome = True
        self._context_closed = False

        self._context.on("page", self._on_new_page)
        self._context.on("close", self._on_context_closed)
        open_pages = [p for p in self._context.pages if not p.is_closed()]
        self._active_page = open_pages[0] if open_pages else self._context.new_page()
        return self._context

    def _on_new_page(self, page: Page) -> None:
        # Popups/new tabs opened by page JS (e.g. an OAuth window, a link with
        # target=_blank) become the active page automatically so the agent
        # loop's next read_page() sees them instead of a stale background tab.
        # Fired by Playwright ON the browser thread already - no marshalling needed.
        self._active_page = page

    def _teardown_partial(self, skip_stop: bool = False) -> None:
        """skip_stop=True after a hung-call recovery: self._playwright was
        created on the OLD, now-abandoned browser thread (see _mark_dead) -
        calling .stop() on it from the NEW thread would violate Playwright's
        single-thread-per-instance requirement and could itself hang or raise
        unpredictably. The old thread's process/pipes are simply leaked as a
        daemon thread; not calling .stop() just means we don't wait around
        for a connection that's already known to be dead."""
        if self._playwright is not None and not skip_stop:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._playwright = None
        self._context = None
        self._active_page = None
        self._browser = None
        self._attached_to_real_chrome = False
        self._context_closed = False
        self._dead_from_hang = False

    def ensure_started(self) -> BrowserContext:
        try:
            return self.start()
        except BrowserUnavailable:
            raise
        except Exception as exc:
            raise BrowserUnavailable(str(exc)) from exc

    @property
    def page(self) -> Page:
        """The current active tab. Starts the browser if it isn't running yet."""
        return _browser_thread.run(self._page_impl)

    def _page_impl(self) -> Page:
        self._start_impl()
        if self._active_page is None or self._active_page.is_closed():
            open_pages = [p for p in (self._context.pages if self._context else []) if not p.is_closed()]
            self._active_page = open_pages[0] if open_pages else self._context.new_page()
        return self._active_page

    def set_active_page(self, page: Page) -> None:
        self._active_page = page

    def pages(self) -> list[Page]:
        return _browser_thread.run(self._pages_impl)

    def _pages_impl(self) -> list[Page]:
        self._start_impl()
        return [p for p in self._context.pages if not p.is_closed()]

    def new_tab(self, url: str | None = None) -> Page:
        return _browser_thread.run(self._new_tab_impl, url)

    def _new_tab_impl(self, url: str | None) -> Page:
        self._start_impl()
        page = self._context.new_page()
        self._active_page = page
        if url:
            page.goto(url)
        return page

    def close_tab(self, page: Page) -> None:
        _browser_thread.run(self._close_tab_impl, page)

    def _close_tab_impl(self, page: Page) -> None:
        if self._attached_to_real_chrome:
            # Refuse to close tabs in the user's own Chrome - they may be the
            # user's own work, not something Jarvix opened. Just switch away.
            raise RuntimeError(
                "I won't close tabs in your real Chrome - close it yourself if you want it gone."
            )
        if page.is_closed():
            return
        was_active = page is self._active_page
        page.close()
        if was_active:
            remaining = [p for p in self._context.pages if not p.is_closed()]
            self._active_page = remaining[-1] if remaining else self._context.new_page()

    def shutdown(self) -> None:
        _browser_thread.run(self._shutdown_impl)

    def _shutdown_impl(self) -> None:
        # In real-Chrome mode we only DISCONNECT - closing the context/browser
        # here would close the user's own Chrome window and everything they
        # had open. Only a Jarvix-launched dedicated context is ours to close.
        if self._attached_to_real_chrome:
            _log("Detaching from your real Chrome (leaving it open)")
        else:
            _log("Closing Chrome")
            if self._context is not None:
                try:
                    self._context.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._context = None
        self._active_page = None
        self._playwright = None
        self._browser = None
        self._attached_to_real_chrome = False
        self._context_closed = False


# Module-level singleton - one browser context per Jarvix process, matching
# the persistent-login goal (a fresh context per call would lose cookies).
_manager = BrowserManager()


def get_manager() -> BrowserManager:
    return _manager
