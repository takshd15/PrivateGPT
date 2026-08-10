"""Turns a live Playwright Page into the compact, structured snapshot the
browser agent loop (app/browser/tools.py) feeds the LLM, instead of raw HTML.

Interactive elements get small stable-for-this-snapshot integer IDs
(data-jarvix-id) so the LLM can reference "[3]" in a click/type call and the
executor (actions.py) can look the element back up reliably, even though
Playwright has no built-in numeric handle.
"""

from __future__ import annotations

from dataclasses import dataclass

from playwright.sync_api import Page

_MAX_TEXT_CHARS = 3000
_MAX_ELEMENTS = 60

# Elements a user could plausibly act on. Kept intentionally small/generic -
# this is a heuristic for "what's worth numbering," not a full accessibility tree.
_INTERACTIVE_SELECTOR = ",".join(
    [
        "a[href]",
        "button",
        "input:not([type=hidden])",
        "select",
        "textarea",
        "[role=button]",
        "[role=link]",
        "[role=checkbox]",
        "[role=menuitem]",
        "[role=tab]",
        "[contenteditable=true]",
        "summary",
    ]
)

_MARK_ELEMENTS_JS = """
(selector) => {
    const els = Array.from(document.querySelectorAll(selector));
    const results = [];
    let id = 0;
    for (const el of els) {
        const rect = el.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0 &&
            getComputedStyle(el).visibility !== 'hidden' &&
            getComputedStyle(el).display !== 'none';
        if (!visible) continue;
        const label = (
            el.getAttribute('aria-label') ||
            el.getAttribute('placeholder') ||
            el.getAttribute('title') ||
            el.innerText ||
            el.value ||
            el.getAttribute('alt') ||
            ''
        ).trim().replace(/\\s+/g, ' ').slice(0, 80);
        if (!label && el.tagName !== 'INPUT' && el.tagName !== 'SELECT' && el.tagName !== 'TEXTAREA') continue;
        el.setAttribute('data-jarvix-id', String(id));
        const tag = el.tagName.toLowerCase();
        let kind = tag;
        if (tag === 'input') kind = 'input[' + (el.getAttribute('type') || 'text') + ']';
        if (tag === 'a') kind = 'link';
        if (el.getAttribute('role')) kind = el.getAttribute('role');
        results.push({id, kind, label, disabled: !!el.disabled});
        id += 1;
        if (id >= %d) break;
    }
    return results;
}
""" % _MAX_ELEMENTS


@dataclass
class ElementRef:
    id: int
    kind: str
    label: str
    disabled: bool


@dataclass
class PageSnapshot:
    title: str
    url: str
    elements: list[ElementRef]
    text: str
    truncated: bool

    def as_prompt_text(self) -> str:
        lines = [f"Page title: {self.title}", f"Current URL: {self.url}", "", "Interactive elements:"]
        if self.elements:
            for el in self.elements:
                flag = " (disabled)" if el.disabled else ""
                lines.append(f"[{el.id}] {el.kind}: {el.label}{flag}")
        else:
            lines.append("(none found)")
        lines.append("")
        lines.append("Text content:")
        lines.append(self.text if self.text else "(empty)")
        if self.truncated:
            lines.append("...(page text truncated)")
        return "\n".join(lines)


def snapshot(page: Page, max_text_chars: int = _MAX_TEXT_CHARS) -> PageSnapshot:
    """Read the current page into a structured, LLM-friendly snapshot.

    Clears any stale data-jarvix-id marks from a previous snapshot first, so
    IDs always refer to what the page shows RIGHT NOW - stale IDs from a page
    that has since navigated/re-rendered are the single biggest cause of
    "click element 4" silently hitting the wrong thing.
    """
    try:
        page.evaluate(
            "() => document.querySelectorAll('[data-jarvix-id]').forEach(e => e.removeAttribute('data-jarvix-id'))"
        )
    except Exception:
        pass

    raw_elements = page.evaluate(_MARK_ELEMENTS_JS, _INTERACTIVE_SELECTOR)
    elements = [
        ElementRef(id=e["id"], kind=e["kind"], label=e["label"] or "(unlabeled)", disabled=e["disabled"])
        for e in raw_elements
    ]

    try:
        body_text = page.evaluate("() => document.body ? document.body.innerText : ''") or ""
    except Exception:
        body_text = ""
    body_text = " ".join(body_text.split())
    truncated = len(body_text) > max_text_chars
    if truncated:
        body_text = body_text[:max_text_chars]

    return PageSnapshot(
        title=page.title() or "(untitled)",
        url=page.url,
        elements=elements,
        text=body_text,
        truncated=truncated,
    )


def find_element(page: Page, element_id: int):
    """Locator for a data-jarvix-id set by the most recent snapshot(). None if
    the page has since navigated/re-rendered and the ID no longer exists."""
    locator = page.locator(f"[data-jarvix-id='{element_id}']")
    if locator.count() == 0:
        return None
    return locator.first
