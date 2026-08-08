"""Music control for Jarvix v2.

Three layers:

1. Media keys (play/pause, next, previous) via the OS virtual-key codes.
   These are global media keys, so they reach Spotify even when it is not the
   focused window. They control whatever media app is playing.

2. Spotify Web API playback using the local OAuth token in secrets/.

3. ``play(query)`` falls back to the installed desktop app (via its
   ``spotify:`` URI scheme) when the Web API cannot find an active Spotify
   device, and only opens the open.spotify.com web player if the desktop app
   isn't installed.
"""

from __future__ import annotations

import ctypes
import json
import os
import secrets
import time
import webbrowser
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlencode, urlparse

import requests

from app.config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_REDIRECT_URI,
    SPOTIFY_TOKEN_FILE,
)
from app.tools.desktop import resolve_app

# Windows virtual-key codes for global media controls.
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

KEYEVENTF_KEYUP = 0x0002
SPOTIFY_SCOPES = [
    "user-read-playback-state",
    "user-modify-playback-state",
]
SPOTIFY_API = "https://api.spotify.com/v1"
SPOTIFY_ACCOUNTS = "https://accounts.spotify.com"
SPOTIFY_TIMEOUT = 4


def _tap_key(vk: int) -> None:
    """Press and release a virtual key."""
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]  # Windows-only
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def _spotify_headers() -> dict[str, str]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError("Spotify API credentials are not configured in .env.")
    raw = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode("utf-8")
    return {
        "Authorization": "Basic " + b64encode(raw).decode("ascii"),
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _load_token() -> dict:
    if not SPOTIFY_TOKEN_FILE.exists():
        return {}
    try:
        return json.loads(SPOTIFY_TOKEN_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_token(token: dict) -> None:
    SPOTIFY_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPOTIFY_TOKEN_FILE.write_text(json.dumps(token, indent=2), encoding="utf-8")


def _with_expiry(token: dict) -> dict:
    token["expires_at"] = time.time() + int(token.get("expires_in", 3600))
    return token


def _refresh_access_token(token: dict) -> dict:
    refresh_token = token.get("refresh_token")
    if not refresh_token:
        raise ValueError("Spotify refresh token is missing.")

    response = requests.post(
        f"{SPOTIFY_ACCOUNTS}/api/token",
        headers=_spotify_headers(),
        data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        timeout=SPOTIFY_TIMEOUT,
    )
    response.raise_for_status()
    fresh = _with_expiry(response.json())
    if "refresh_token" not in fresh:
        fresh["refresh_token"] = refresh_token
    _save_token(fresh)
    return fresh


def _wait_for_spotify_code(state: str) -> str:
    parsed = urlparse(SPOTIFY_REDIRECT_URI)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8888
    callback_path = parsed.path or "/callback"
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            query = parse_qs(urlparse(self.path).query)
            if urlparse(self.path).path != callback_path:
                self.send_response(404)
                self.end_headers()
                return
            if query.get("state", [""])[0] != state:
                result["error"] = "Spotify authorization state mismatch."
            elif "error" in query:
                result["error"] = query["error"][0]
            else:
                result["code"] = query.get("code", [""])[0]

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Spotify authorized.</h1>"
                b"<p>You can close this tab and return to Jarvix.</p></body></html>"
            )

        def log_message(self, format: str, *args) -> None:
            return

    with HTTPServer((host, port), CallbackHandler) as server:
        server.timeout = 180
        while "code" not in result and "error" not in result:
            server.handle_request()

    if result.get("error"):
        raise ValueError(result["error"])
    return result["code"]


def _authorize_spotify() -> dict:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError("Spotify API credentials are not configured in .env.")

    state = secrets.token_urlsafe(16)
    url = f"{SPOTIFY_ACCOUNTS}/authorize?" + urlencode(
        {
            "response_type": "code",
            "client_id": SPOTIFY_CLIENT_ID,
            "scope": " ".join(SPOTIFY_SCOPES),
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "state": state,
        }
    )
    print("Please authorize Spotify playback in your browser.")
    webbrowser.open(url)
    code = _wait_for_spotify_code(state)

    response = requests.post(
        f"{SPOTIFY_ACCOUNTS}/api/token",
        headers=_spotify_headers(),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
        },
        timeout=SPOTIFY_TIMEOUT,
    )
    response.raise_for_status()
    token = _with_expiry(response.json())
    _save_token(token)
    return token


def get_spotify_access_token(allow_authorize: bool = True) -> str:
    token = _load_token()
    if token.get("access_token") and token.get("expires_at", 0) > time.time() + 60:
        return token["access_token"]
    if token.get("refresh_token"):
        try:
            return _refresh_access_token(token)["access_token"]
        except requests.HTTPError:
            pass
    if not allow_authorize:
        raise ValueError("Spotify is not authorized.")
    return _authorize_spotify()["access_token"]


def _spotify_request(
    method: str,
    path: str,
    request_timeout: float = SPOTIFY_TIMEOUT,
    allow_authorize: bool = True,
    **kwargs,
) -> requests.Response:
    access_token = get_spotify_access_token(allow_authorize=allow_authorize)
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {access_token}"
    response = requests.request(
        method,
        f"{SPOTIFY_API}{path}",
        headers=headers,
        timeout=request_timeout,
        **kwargs,
    )
    if response.status_code == 401:
        token = _refresh_access_token(_load_token())
        headers["Authorization"] = f"Bearer {token['access_token']}"
        response = requests.request(
            method,
            f"{SPOTIFY_API}{path}",
            headers=headers,
            timeout=request_timeout,
            **kwargs,
        )
    return response


def _spotify_track_uri(value: str) -> str | None:
    value = value.strip()
    if value.startswith("spotify:track:"):
        return value
    parsed = urlparse(value)
    if parsed.netloc.endswith("spotify.com") and parsed.path.startswith("/track/"):
        track_id = parsed.path.split("/")[2]
        return f"spotify:track:{track_id}" if track_id else None
    return None


def _search_best(query: str) -> dict | None:
    """Search Spotify for tracks and artists and return the best match.

    Returns a dict with ``display`` (the real name to speak back) and a play
    target: ``uri`` for a single track, or ``context_uri`` for an artist whose
    top tracks should play. An artist is preferred only when its name matches
    the query, so "play Travis Scott" plays the artist while "play Sicko Mode"
    plays the track.
    """
    response = _spotify_request(
        "GET",
        "/search",
        params={"q": query, "type": "track,artist", "limit": 5},
    )
    response.raise_for_status()
    data = response.json()
    tracks = data.get("tracks", {}).get("items", [])
    artists = data.get("artists", {}).get("items", [])

    wanted = query.strip().lower()
    for artist in artists:
        if (artist.get("name") or "").strip().lower() == wanted:
            return {"display": artist["name"], "uri": None, "context_uri": artist["uri"]}

    if tracks:
        top = tracks[0]
        artist_name = (top.get("artists") or [{}])[0].get("name", "")
        display = f"{top['name']} by {artist_name}" if artist_name else top["name"]
        return {"display": display, "uri": top["uri"], "context_uri": None}

    if artists:
        top = artists[0]
        return {"display": top["name"], "uri": None, "context_uri": top["uri"]}

    return None


def _match_play_body(match: dict) -> dict:
    return {"uris": [match["uri"]]} if match.get("uri") else {"context_uri": match["context_uri"]}


def _active_device_id() -> str | None:
    response = _spotify_request("GET", "/me/player/devices")
    response.raise_for_status()
    devices = response.json().get("devices", [])
    for device in devices:
        if device.get("is_active") and not device.get("is_restricted"):
            return device.get("id")
    for device in devices:
        if not device.get("is_restricted"):
            return device.get("id")
    return None


def _play_body_with_api(body: dict) -> bool:
    open_spotify()
    time.sleep(1.5)
    device_id = _active_device_id()
    params = {"device_id": device_id} if device_id else None
    response = _spotify_request(
        "PUT",
        "/me/player/play",
        params=params,
        json=body,
    )
    if response.status_code in (200, 202, 204):
        return True
    if response.status_code in (404, 429):
        return False
    response.raise_for_status()
    return True


def _play_uri_with_api(uri: str) -> bool:
    return _play_body_with_api({"uris": [uri]})


def _web_player_url(spotify_uri: str) -> str:
    """Convert a spotify:track:ID URI to its open.spotify.com web-player URL."""
    kind_and_id = spotify_uri.removeprefix("spotify:")
    return f"https://open.spotify.com/{kind_and_id.replace(':', '/')}"


def _play_uri_fallback(uri: str) -> None:
    """Open a spotify: URI in the desktop app if installed, else the web player."""
    if _spotify_app_path() is not None:
        os.startfile(uri)  # type: ignore[attr-defined]  # Windows-only; spotify: protocol handler
    else:
        webbrowser.open(_web_player_url(uri))
    time.sleep(1.5)
    _tap_key(VK_MEDIA_PLAY_PAUSE)


def is_playing() -> bool | None:
    """Return Spotify playback state when the API can tell us."""
    try:
        response = _spotify_request(
            "GET",
            "/me/player",
            request_timeout=1.0,
            allow_authorize=False,
        )
        if response.status_code == 204:
            return False
        response.raise_for_status()
        return bool(response.json().get("is_playing"))
    except Exception:
        return None


def pause_if_playing() -> bool:
    """Pause Spotify only when it is definitely playing."""
    if is_playing() is not True:
        return False
    try:
        response = _spotify_request(
            "PUT",
            "/me/player/pause",
            request_timeout=1.0,
            allow_authorize=False,
        )
        return response.status_code in (200, 202, 204)
    except Exception:
        return False


def resume_playback() -> bool:
    """Resume Spotify when Jarvix temporarily paused it to listen."""
    try:
        response = _spotify_request(
            "PUT",
            "/me/player/play",
            request_timeout=1.0,
            allow_authorize=False,
        )
        return response.status_code in (200, 202, 204)
    except Exception:
        return False


def play_pause() -> str:
    _tap_key(VK_MEDIA_PLAY_PAUSE)
    return "Toggled play/pause"


def _spotify_app_path():
    """Resolve the installed Spotify.exe, or None if it isn't installed."""
    return resolve_app("spotify")


def open_spotify() -> str:
    app_path = _spotify_app_path()
    if app_path is not None:
        os.startfile(str(app_path))  # type: ignore[attr-defined]  # Windows-only
        return "Opening Spotify"
    webbrowser.open("https://open.spotify.com")
    return "Opening Spotify"


def next_track() -> str:
    _tap_key(VK_MEDIA_NEXT_TRACK)
    return "Skipped to next track"


def previous_track() -> str:
    _tap_key(VK_MEDIA_PREV_TRACK)
    return "Went to previous track"


def volume_up(steps: int = 2) -> str:
    for _ in range(max(1, steps)):
        _tap_key(VK_VOLUME_UP)
    return "Turned the volume up"


def volume_down(steps: int = 2) -> str:
    for _ in range(max(1, steps)):
        _tap_key(VK_VOLUME_DOWN)
    return "Turned the volume down"


def mute() -> str:
    _tap_key(VK_VOLUME_MUTE)
    return "Toggled mute"


def play(query: str) -> str:
    """Play a Spotify track URL/URI or search for ``query`` and start playback.

    Prefers the Spotify Web API; falls back to the desktop URI + media key.
    """
    query = query.strip()
    if not query:
        open_spotify()
        _tap_key(VK_MEDIA_PLAY_PAUSE)
        return "Opening Spotify"

    uri = _spotify_track_uri(query)
    if uri:
        try:
            if _play_uri_with_api(uri):
                return "Playing the Spotify track"
        except Exception as exc:
            print(f"[spotify api unavailable, falling back] {exc}")
        _play_uri_fallback(uri)
        return "Opening the Spotify track"

    try:
        match = _search_best(query)
        if match and _play_body_with_api(_match_play_body(match)):
            # Speak the name Spotify actually matched, not the raw mishear.
            return f"Playing {match['display']} on Spotify"
    except Exception as exc:
        print(f"[spotify api unavailable, falling back] {exc}")

    if _spotify_app_path() is not None:
        # spotify:search: opens more reliably than a raw search URI on some
        # installs, but the interactive search UI still needs the browser as
        # a reliable fallback when the desktop app can't be pointed at the
        # result directly - open the desktop app first so it's frontmost.
        os.startfile(str(_spotify_app_path()))  # type: ignore[attr-defined]  # Windows-only
    webbrowser.open(f"https://open.spotify.com/search/{quote(query)}")
    # Give the app a moment to focus the search before nudging playback.
    time.sleep(1.5)
    _tap_key(VK_MEDIA_PLAY_PAUSE)
    return f"Searching Spotify for {query}"
