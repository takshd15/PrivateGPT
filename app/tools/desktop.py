"""Safe desktop control for Jarvix v2.

Only opens apps and folders that are explicitly listed in
``app/memory/aliases.json``. There is intentionally NO path to run an
arbitrary shell command or open an arbitrary path the user names, so the
model can never turn the laptop into spicy toast.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
from pathlib import Path

ALIASES_FILE = Path(__file__).resolve().parents[1] / "memory" / "aliases.json"
_DETACHED_FLAGS = 0
if os.name == "nt":
    _DETACHED_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP


class DesktopError(Exception):
    """Raised for unknown aliases or missing targets. Caller shows the message."""


def _load_aliases() -> dict:
    if not ALIASES_FILE.exists():
        raise DesktopError(f"Alias file not found: {ALIASES_FILE}")
    with open(ALIASES_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize(name: str) -> str:
    return name.strip().lower()


def list_folders() -> list[str]:
    return sorted(_load_aliases().get("folders", {}))


def list_apps() -> list[str]:
    return sorted(_load_aliases().get("apps", {}))


def open_folder(alias: str) -> str:
    """Open a folder from the allowlist in Explorer. Returns a message."""
    folders = _load_aliases().get("folders", {})
    key = _normalize(alias)
    if key not in folders:
        known = ", ".join(sorted(folders)) or "(none configured)"
        raise DesktopError(f"Unknown folder '{alias}'. Known folders: {known}")

    path = Path(folders[key])
    if not path.is_dir():
        raise DesktopError(f"Folder for '{alias}' does not exist: {path}")

    os.startfile(str(path))  # type: ignore[attr-defined]  # Windows-only
    return f"Opening folder {path}"


def _resolve_app(value: str) -> Path | None:
    """Resolve an allowlisted app value to a real executable, or None."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    if value.lower() in {"code", "vscode"}:
        for path in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "Code.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft VS Code" / "Code.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft VS Code" / "Code.exe",
        ):
            if path.exists():
                return path

    if value.lower() in {"claude", "claude.exe"}:
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Claude" / "Claude.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "AnthropicClaude" / "Claude.exe",
        ]
        install_root = Path(os.environ.get("LOCALAPPDATA", "")) / "AnthropicClaude"
        if install_root.is_dir():
            candidates.extend(sorted(install_root.glob("app-*/Claude.exe"), reverse=True))
        for path in candidates:
            if path.exists():
                return path

    if value.lower() in {"spotify", "spotify.exe"}:
        for path in (
            Path(os.environ.get("APPDATA", "")) / "Spotify" / "Spotify.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps" / "Spotify.exe",
        ):
            if path.exists():
                return path

    found = shutil.which(value)
    return Path(found) if found else None


def _resolve_vscode_cli() -> Path | str | None:
    for name in ("code.cmd", "code"):
        found = shutil.which(name)
        if found:
            return found

    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code" / "bin"
    for name in ("code.cmd", "code"):
        path = local / name
        if path.exists():
            return path
    return None


def resolve_app(value: str) -> Path | None:
    """Public lookup for an app executable by raw value (not alias). None if not found."""
    return _resolve_app(value)


def open_app(alias: str) -> str:
    """Launch an app from the allowlist. Returns a message."""
    apps = _load_aliases().get("apps", {})
    key = _normalize(alias)
    if key not in apps:
        known = ", ".join(sorted(apps)) or "(none configured)"
        raise DesktopError(f"Unknown app '{alias}'. Known apps: {known}")

    resolved = _resolve_app(apps[key])
    if resolved is None:
        raise DesktopError(
            f"App '{alias}' is configured as '{apps[key]}' but it was not found. "
            f"Is it installed and on PATH?"
        )

    os.startfile(str(resolved))  # type: ignore[attr-defined]  # Windows-only
    return f"Opening {alias}"


def open_workspace(app_alias: str, folder_alias: str) -> str:
    """Open an allowlisted folder directly in an allowlisted editor app."""
    apps = _load_aliases().get("apps", {})
    folders = _load_aliases().get("folders", {})
    app_key = _normalize(app_alias)
    folder_key = _normalize(folder_alias)

    if app_key not in apps:
        known = ", ".join(sorted(apps)) or "(none configured)"
        raise DesktopError(f"Unknown app '{app_alias}'. Known apps: {known}")
    if folder_key not in folders:
        known = ", ".join(sorted(folders)) or "(none configured)"
        raise DesktopError(f"Unknown folder '{folder_alias}'. Known folders: {known}")

    app_path = _resolve_app(apps[app_key])
    if app_path is None:
        raise DesktopError(
            f"App '{app_alias}' is configured as '{apps[app_key]}' but it was not found. "
            f"Is it installed and on PATH?"
        )

    folder_path = Path(folders[folder_key])
    if not folder_path.is_dir():
        raise DesktopError(f"Folder for '{folder_alias}' does not exist: {folder_path}")

    if app_key in {"vscode", "code"}:
        code_cli = _resolve_vscode_cli()
        if code_cli is not None:
            command = [str(code_cli), "--new-window", str(folder_path)]
            if os.name == "nt" and str(code_cli).lower().endswith((".cmd", ".bat")):
                command = ["cmd.exe", "/d", "/c", "start", "", *command]
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_DETACHED_FLAGS,
            )
            return f"Opening {folder_path} in {app_alias}"

    args = [str(app_path), str(folder_path)]

    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_DETACHED_FLAGS,
    )
    return f"Opening {folder_path} in {app_alias}"
