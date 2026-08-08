"""Deterministic time, date, and weather helpers for spoken answers."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from app.config import DEFAULT_WEATHER_LOCATION, TIMEZONE, WEATHER_TIMEOUT_SECONDS


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    95: "a thunderstorm",
    96: "a thunderstorm with hail",
    99: "a thunderstorm with hail",
}


def local_now() -> datetime:
    try:
        return datetime.now(ZoneInfo(TIMEZONE))
    except ZoneInfoNotFoundError:
        return datetime.now().astimezone()


def spoken_time() -> str:
    now = local_now()
    value = now.strftime("%I:%M %p").lstrip("0")
    return f"It's {value}."


def resolve_date_phrase(phrase: str, today: date | None = None) -> date | None:
    """Resolve the small, predictable date vocabulary accepted by the router."""
    base = today or local_now().date()
    text = " ".join((phrase or "").lower().split())
    if text == "today":
        return base
    if text == "tomorrow":
        return base + timedelta(days=1)
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass

    is_next = text.startswith("next ")
    weekday = text.removeprefix("next ")
    if weekday not in _WEEKDAYS:
        return None
    days = (_WEEKDAYS[weekday] - base.weekday()) % 7
    if days == 0 or is_next:
        days += 7
    return base + timedelta(days=days)


_TIME_PATTERN = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b", re.I
)


def parse_spoken_time(phrase: str) -> str | None:
    """Best-effort spoken time -> "HH:MM" (24h). None if unparseable.

    Handles "3pm", "3:30 pm", "15:00", "noon", "midnight". Ambiguous bare
    hours (e.g. "3") are assumed PM for anything 1-7 and AM otherwise, since
    that matches how people usually say event times without a period.
    """
    text = " ".join((phrase or "").lower().split())
    if not text:
        return None
    if text in {"noon", "12pm", "12 pm"}:
        return "12:00"
    if text in {"midnight", "12am", "12 am"}:
        return "00:00"

    match = _TIME_PATTERN.search(text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    period = (match.group(3) or "").replace(".", "")
    if hour > 23 or minute > 59:
        return None

    if period == "pm" and hour != 12:
        hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    elif not period and 1 <= hour <= 7:
        hour += 12  # bare small hours default to afternoon/evening

    if hour > 23:
        return None
    return f"{hour:02d}:{minute:02d}"


def weather(location: str | None) -> str:
    requested = (location or DEFAULT_WEATHER_LOCATION).strip()
    if not requested:
        return "Which city should I check?"

    try:
        geocode = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": requested, "count": 1, "language": "en", "format": "json"},
            timeout=WEATHER_TIMEOUT_SECONDS,
        )
        geocode.raise_for_status()
        matches = geocode.json().get("results") or []
        if not matches:
            return f"I couldn't find a place called {requested}. Which city did you mean?"

        place = matches[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code",
                "timezone": "auto",
            },
            timeout=WEATHER_TIMEOUT_SECONDS,
        )
        forecast.raise_for_status()
        current = forecast.json()["current"]
    except (requests.RequestException, AttributeError, KeyError, TypeError, ValueError):
        return "I couldn't reach the weather service right now."

    try:
        name = place.get("name") or requested
        region = place.get("admin1") or place.get("country") or ""
        label = f"{name}, {region}" if region and region.lower() != name.lower() else name
        temp = round(float(current["temperature_2m"]))
        feels = round(float(current["apparent_temperature"]))
        condition = _WEATHER_CODES.get(int(current.get("weather_code", -1)), "mixed conditions")
    except (AttributeError, KeyError, TypeError, ValueError):
        return "The weather service returned an incomplete result."
    return f"In {label}, it's {temp} degrees Celsius and {condition}. It feels like {feels}."
