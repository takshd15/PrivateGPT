"""News headlines via NewsAPI.org (app/tools/news.py -> intent NEWS)."""

from __future__ import annotations

import requests

from app.config import NEWS_API_KEY

_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
_TIMEOUT = 6


class NewsError(Exception):
    """Raised when headlines can't be fetched. Caller shows the message."""


def get_top_headlines(country: str = "us", limit: int = 5) -> list[dict]:
    """Fetch top headlines. Each item: {title, source, description, url}."""
    if not NEWS_API_KEY:
        raise NewsError("News isn't configured yet - NEWS_API_KEY is missing from .env.")

    response = requests.get(
        _TOP_HEADLINES_URL,
        params={"country": country, "pageSize": limit, "apiKey": NEWS_API_KEY},
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "ok":
        raise NewsError(data.get("message") or "News service returned an error.")

    articles = []
    for item in data.get("articles", [])[:limit]:
        articles.append(
            {
                "title": item.get("title") or "",
                "source": (item.get("source") or {}).get("name") or "",
                "description": item.get("description") or "",
                "url": item.get("url") or "",
            }
        )
    return articles
