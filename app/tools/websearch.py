"""Web search via Exa (app/tools/websearch.py -> intent FIND_OPPORTUNITIES).

Exa is used (not Tavily/WEB_SEARCH_API_KEY) because the opportunities table
already has 21 existing rows all sourced from Exa - a consistent provenance
matters more than which search API's docs read better.
"""

from __future__ import annotations

import requests

from app.config import EXA_API_KEY

_SEARCH_URL = "https://api.exa.ai/search"
_TIMEOUT = 10


class WebSearchError(Exception):
    """Raised when a search can't be run. Caller shows the message."""


def search(query: str, num_results: int = 8) -> list[dict]:
    """Run a web search. Each item: {title, url, snippet}."""
    if not EXA_API_KEY:
        raise WebSearchError("Opportunity search isn't configured yet - EXA_API_KEY is missing from .env.")

    response = requests.post(
        _SEARCH_URL,
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        json={
            "query": query,
            "numResults": num_results,
            "contents": {"text": {"maxCharacters": 500}},
        },
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": (item.get("text") or "")[:500],
            }
        )
    return results
