"""Resolves a spoken/typed short site name ("github", "my bank", "youtube")
to the URL the user actually visits most, by reading their REAL Chrome
browsing history - separate from JARVIX_CHROME_PROFILE_DIR (Jarvix's own,
initially-empty browser profile), which this module never touches.

Read-only and copy-first: Chrome holds its History sqlite file open while
running, so it's always copied to a temp file before querying (sqlite3 can't
reliably open a file another process has locked for writing on Windows) -
this module never opens or writes to the real file in place.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.config import CHROME_REAL_PROFILE_DIR, CHROME_REAL_PROFILES

# Chrome's last_visit_time is microseconds since 1601-01-01 (WebKit/Windows
# FILETIME epoch), not Unix time - this is the offset to convert one to the
# other.
_WEBKIT_EPOCH_OFFSET_SECONDS = 11644473600


@dataclass
class SiteMatch:
    url: str
    title: str
    visit_count: int
    last_visit_unix: float
    profile: str

    @property
    def domain(self) -> str:
        m = re.search(r"://(?:www\.)?([^/]+)", self.url)
        return m.group(1) if m else self.url


def _profile_dirs() -> list[Path]:
    base = Path(CHROME_REAL_PROFILE_DIR)
    return [base / name for name in CHROME_REAL_PROFILES if (base / name / "History").exists()]


def _copy_history_file(history_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="jarvix_hist_")) / "History"
    shutil.copy2(history_path, tmp)
    return tmp


def _query_profile(history_path: Path, profile_name: str) -> list[SiteMatch]:
    tmp_copy = _copy_history_file(history_path)
    try:
        con = sqlite3.connect(f"file:{tmp_copy}?mode=ro", uri=True)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT url, title, visit_count, last_visit_time FROM urls "
                "WHERE hidden = 0 AND visit_count > 0"
            )
            rows = cur.fetchall()
        finally:
            con.close()
    finally:
        shutil.rmtree(tmp_copy.parent, ignore_errors=True)

    results = []
    for url, title, visit_count, last_visit_time in rows:
        last_visit_unix = (last_visit_time / 1_000_000) - _WEBKIT_EPOCH_OFFSET_SECONDS if last_visit_time else 0.0
        results.append(SiteMatch(url=url or "", title=title or "", visit_count=visit_count, last_visit_unix=last_visit_unix, profile=profile_name))
    return results


def _load_all_history() -> list[SiteMatch]:
    """Every profile's history, merged. Never raises - a missing/unreadable
    Chrome install just means no history to match against, not a crash."""
    all_matches: list[SiteMatch] = []
    for profile_dir in _profile_dirs():
        try:
            all_matches.extend(_query_profile(profile_dir / "History", profile_dir.name))
        except Exception:
            continue
    return all_matches


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _name_matches(query_norm: str, match: SiteMatch) -> bool:
    domain_norm = _normalize(match.domain)
    title_norm = _normalize(match.title)
    return query_norm in domain_norm or query_norm in title_norm


def find_site(name: str, top_n: int = 5) -> list[SiteMatch]:
    """Rank the user's history for a short spoken name, best match first.

    A match is ranked by (matches the domain rather than just the title,
    total visit count, most recent visit) - domain matches beat title-only
    matches so "open github" prefers github.com over some page that merely
    mentions GitHub in its title. Multiple URLs on the same domain are
    collapsed into the single most-visited one (e.g. gmail.com vs
    mail.google.com vs mail.google.com/mail/u/0/#inbox all count as "gmail").
    """
    query_norm = _normalize(name)
    if not query_norm:
        return []

    candidates = [m for m in _load_all_history() if _name_matches(query_norm, m)]

    best_by_domain: dict[str, SiteMatch] = {}
    for m in candidates:
        domain_norm = _normalize(m.domain)
        existing = best_by_domain.get(domain_norm)
        if existing is None or m.visit_count > existing.visit_count:
            best_by_domain[domain_norm] = m

    ranked = sorted(
        best_by_domain.values(),
        key=lambda m: (_normalize(m.domain) == query_norm or query_norm in _normalize(m.domain), m.visit_count, m.last_visit_unix),
        reverse=True,
    )
    return ranked[:top_n]


def best_site_url(name: str) -> SiteMatch | None:
    matches = find_site(name, top_n=1)
    return matches[0] if matches else None


def canonical_url(match: SiteMatch) -> str:
    """The domain root (https://domain/) rather than whatever deep/logged-in
    path happened to have the most visits (e.g. a specific inbox URL) - a
    fresh Jarvix session isn't logged in yet, so opening the root and letting
    the site redirect (to its own login page if needed) is more reliable
    than replaying a stale deep link."""
    return f"https://{match.domain}/"
