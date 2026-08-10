"""Primitive browser actions - thin wrappers over a Playwright Page.

Every function here does ONE physical thing (navigate, click, type, ...) and
returns a short human-readable result string, catching Playwright's own
exceptions and turning them into an ActionError with an actionable message
instead of a raw stack trace. app/browser/tools.py's agent loop is the only
caller with LLM involvement - these functions are dumb executors, no
decision-making.

Every function's actual Playwright work is wrapped in an inner `_do()` and
run via `get_manager().run(_do)` - Playwright's sync API is bound to one OS
thread (see app/browser/manager.py's `_BrowserThread`), and these functions
can be called from any thread (CLI, wake loop, or one of app/server.py's
per-request worker threads), so the call has to be marshalled rather than
touching Page/Locator objects directly on the caller's own thread.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PWTimeoutError

from app.browser.manager import _BrowserThreadHung, get_manager
from app.browser.safety import blocked_domain_for
from app.browser.state import find_element
from app.config import BROWSER_ACTION_TIMEOUT_SECONDS, BROWSER_NAV_TIMEOUT_SECONDS


class ActionError(Exception):
    """A browser action failed in a way the agent loop should see and react to
    (retry, re-read the page, or give up) rather than crash."""


def _run(do):
    """Shared call site for every action's get_manager().run(_do) - converts
    a hung/dead browser connection (app.browser.manager._BrowserThreadHung)
    into a normal ActionError so it's handled the exact same way as any other
    action failure by app/browser/tools.py's agent loop, instead of an
    unhandled exception silently killing the whole task with no log line
    (the actual live bug this fixes - see jarvix.log 2026-08-10, a hung
    second goto() produced zero output and the task just stopped)."""
    try:
        return get_manager().run(do)
    except _BrowserThreadHung as exc:
        raise ActionError(str(exc)) from exc


def _element_timeout_ms() -> float:
    return BROWSER_ACTION_TIMEOUT_SECONDS * 1000


def _locate(page: Page, element_id: int):
    """Must be called on the browser thread - only safe from inside a
    manager.run() callback, never directly."""
    locator = find_element(page, element_id)
    if locator is None:
        raise ActionError(
            f"Element [{element_id}] no longer exists on the page - it may have changed. "
            "Read the page again to get current element IDs."
        )
    return locator


def goto(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    # In real-Chrome mode, a hard wall the LLM can't navigate past. Enforced
    # here (not just in the prompt) so it holds even if the model ignores it.
    if get_manager().is_real_chrome:
        blocked = blocked_domain_for(url)
        if blocked:
            raise ActionError(
                f"Blocked: '{url}' matches the protected-site list ('{blocked}'). "
                "I won't open banking, brokerage, or password-manager sites in your real Chrome. "
                "Do that yourself, or adjust JARVIX_BROWSER_BLOCKED_DOMAINS if it's a false match."
            )

    def _do() -> str:
        page = get_manager().page
        try:
            page.goto(url, timeout=BROWSER_NAV_TIMEOUT_SECONDS * 1000, wait_until="domcontentloaded")
        except PWTimeoutError as exc:
            raise ActionError(f"Navigation to {url} timed out.") from exc
        except Exception as exc:
            raise ActionError(f"Couldn't open {url}: {exc}") from exc
        return f"Opened {page.url}"

    return _run(_do)


def go_back() -> str:
    def _do() -> str:
        page = get_manager().page
        try:
            page.go_back(timeout=BROWSER_NAV_TIMEOUT_SECONDS * 1000, wait_until="domcontentloaded")
        except Exception as exc:
            raise ActionError(f"Couldn't go back: {exc}") from exc
        return f"Went back to {page.url}"

    return _run(_do)


def go_forward() -> str:
    def _do() -> str:
        page = get_manager().page
        try:
            page.go_forward(timeout=BROWSER_NAV_TIMEOUT_SECONDS * 1000, wait_until="domcontentloaded")
        except Exception as exc:
            raise ActionError(f"Couldn't go forward: {exc}") from exc
        return f"Went forward to {page.url}"

    return _run(_do)


def reload() -> str:
    def _do() -> str:
        page = get_manager().page
        try:
            page.reload(timeout=BROWSER_NAV_TIMEOUT_SECONDS * 1000, wait_until="domcontentloaded")
        except Exception as exc:
            raise ActionError(f"Couldn't reload the page: {exc}") from exc
        return "Reloaded the page."

    return _run(_do)


def click(element_id: int) -> str:
    def _do() -> str:
        page = get_manager().page
        locator = _locate(page, element_id)
        try:
            locator.click(timeout=_element_timeout_ms())
        except PWTimeoutError as exc:
            raise ActionError(f"Element [{element_id}] wasn't clickable in time - it may be hidden or covered.") from exc
        except Exception as exc:
            raise ActionError(f"Couldn't click element [{element_id}]: {exc}") from exc
        return f"Clicked [{element_id}]"

    return _run(_do)


def type_text(element_id: int, text: str, submit: bool = False) -> str:
    def _do() -> str:
        page = get_manager().page
        locator = _locate(page, element_id)
        try:
            locator.click(timeout=_element_timeout_ms())
            locator.fill("", timeout=_element_timeout_ms())
            locator.type(text, timeout=_element_timeout_ms())
            if submit:
                locator.press("Enter", timeout=_element_timeout_ms())
        except PWTimeoutError as exc:
            raise ActionError(f"Couldn't type into element [{element_id}] in time.") from exc
        except Exception as exc:
            raise ActionError(f"Couldn't type into element [{element_id}]: {exc}") from exc
        return f"Typed into [{element_id}]" + (" and submitted" if submit else "")

    return _run(_do)


def select_option(element_id: int, value: str) -> str:
    def _do() -> str:
        page = get_manager().page
        locator = _locate(page, element_id)
        try:
            locator.select_option(label=value, timeout=_element_timeout_ms())
        except Exception:
            try:
                locator.select_option(value=value, timeout=_element_timeout_ms())
            except Exception as exc:
                raise ActionError(f"Couldn't select '{value}' in [{element_id}]: {exc}") from exc
        return f"Selected '{value}' in [{element_id}]"

    return _run(_do)


def press_key(key: str) -> str:
    def _do() -> str:
        page = get_manager().page
        try:
            page.keyboard.press(key)
        except Exception as exc:
            raise ActionError(f"Couldn't press key '{key}': {exc}") from exc
        return f"Pressed {key}"

    return _run(_do)


def scroll(direction: str = "down", amount_px: int = 800) -> str:
    def _do() -> str:
        page = get_manager().page
        delta = amount_px if direction == "down" else -amount_px
        try:
            page.mouse.wheel(0, delta)
        except Exception as exc:
            raise ActionError(f"Couldn't scroll: {exc}") from exc
        return f"Scrolled {direction}"

    return _run(_do)


def wait_for_load(seconds: float = 3.0) -> str:
    def _do() -> str:
        page = get_manager().page
        try:
            page.wait_for_load_state("networkidle", timeout=seconds * 1000)
        except PWTimeoutError:
            pass  # not every page reaches network-idle (polling widgets, etc.) - not fatal
        except Exception as exc:
            raise ActionError(f"Error waiting for the page to load: {exc}") from exc
        return "Waited for the page to settle."

    return _run(_do)


def screenshot(label: str = "screenshot") -> Path:
    from datetime import datetime

    out_dir = Path(__file__).resolve().parents[2] / "browser_screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    path = out_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{safe_label}.png"

    def _do() -> Path:
        page = get_manager().page
        try:
            page.screenshot(path=str(path))
        except Exception as exc:
            raise ActionError(f"Couldn't take a screenshot: {exc}") from exc
        return path

    return _run(_do)


def upload_file(element_id: int, file_path: str) -> str:
    resolved = Path(file_path).expanduser()
    if not resolved.exists():
        raise ActionError(f"File not found: {resolved}")

    def _do() -> str:
        page = get_manager().page
        locator = _locate(page, element_id)
        try:
            locator.set_input_files(str(resolved), timeout=_element_timeout_ms())
        except Exception as exc:
            raise ActionError(f"Couldn't upload {resolved.name} to [{element_id}]: {exc}") from exc
        return f"Uploaded {resolved.name} to [{element_id}]"

    return _run(_do)


def download_via_click(element_id: int, save_as: str | None = None) -> str:
    downloads_dir = Path(__file__).resolve().parents[2] / "browser_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    def _do() -> str:
        page = get_manager().page
        locator = _locate(page, element_id)
        try:
            with page.expect_download(timeout=BROWSER_NAV_TIMEOUT_SECONDS * 1000) as dl_info:
                locator.click(timeout=_element_timeout_ms())
            download = dl_info.value
            dest = downloads_dir / (save_as or download.suggested_filename)
            download.save_as(str(dest))
        except PWTimeoutError as exc:
            raise ActionError(f"Clicking [{element_id}] didn't start a download in time.") from exc
        except Exception as exc:
            raise ActionError(f"Download failed: {exc}") from exc
        return f"Downloaded to {dest}"

    return _run(_do)


def list_tabs() -> list[dict]:
    def _do() -> list[dict]:
        manager = get_manager()
        pages = manager.pages()
        active = manager.page
        return [
            {"index": i, "title": p.title(), "url": p.url, "active": p is active}
            for i, p in enumerate(pages)
        ]

    return _run(_do)


def switch_tab(index: int) -> str:
    def _do() -> str:
        manager = get_manager()
        pages = manager.pages()
        if not (0 <= index < len(pages)):
            raise ActionError(f"No tab at index {index}. There are {len(pages)} open tab(s).")
        manager.set_active_page(pages[index])
        return f"Switched to tab {index}: {pages[index].title()}"

    return _run(_do)


def new_tab(url: str | None = None) -> str:
    def _do() -> str:
        page = get_manager().new_tab(url)
        return f"Opened new tab: {page.url}"

    return _run(_do)


def close_tab(index: int | None = None) -> str:
    def _do() -> str:
        manager = get_manager()
        pages = manager.pages()
        if not pages:
            raise ActionError("No tabs are open.")
        if index is not None and not (0 <= index < len(pages)):
            raise ActionError(f"No tab at index {index}. There are {len(pages)} open tab(s).")
        target = pages[index] if index is not None else manager.page
        manager.close_tab(target)
        return "Closed the tab."

    return _run(_do)
