from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Brain: OpenAI's chat completions API.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# News - app/tools/news.py (NewsAPI.org top-headlines).
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Opportunity search - app/tools/websearch.py (Exa), intent FIND_OPPORTUNITIES.
EXA_API_KEY = os.getenv("EXA_API_KEY", "")

# Postgres (Supabase) - RAG memory + structured capture, app/memory/db/.
DATABASE_URL = os.getenv("DATABASE_URL", "")

# The Supabase users.id (uuid) that owns every row Jarvix writes to
# tasks/projects/progress_events/etc. Single-user today; kept explicit so
# the schema (which already has real per-user foreign keys) stays correct.
USER_ID = os.getenv("USER_ID", "")

# Background autopilot (app/runtime/scheduler.py). AUTOPILOT_ENABLED gates the
# whole scheduler; the per-job flags only gate whether that job's DB write
# happens - email extraction still runs either way since it's one combined
# LLM call per email.
def _truthy(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


AUTOPILOT_ENABLED = _truthy("AUTOPILOT_ENABLED", "true")
AUTOPILOT_EVENTS = _truthy("AUTOPILOT_EVENTS", "true")
AUTOPILOT_APPLICATIONS = _truthy("AUTOPILOT_APPLICATIONS", "true")
AUTOPILOT_EMAIL_SCAN_MINUTES = int(os.getenv("AUTOPILOT_EMAIL_SCAN_MINUTES", "25"))

GOOGLE_CREDENTIALS_FILE = ROOT / os.getenv(
    "GOOGLE_CREDENTIALS_FILE",
    "secrets/google_credentials.json",
)

GOOGLE_TOKEN_FILE = ROOT / os.getenv(
    "GOOGLE_TOKEN_FILE",
    "secrets/google_token.json",
)

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI",
    "http://127.0.0.1:8888/callback",
)
SPOTIFY_TOKEN_FILE = ROOT / os.getenv(
    "SPOTIFY_TOKEN_FILE",
    "secrets/spotify_token.json",
)

TIMEZONE = os.getenv("TIMEZONE", "Europe/Amsterdam")
DEFAULT_WEATHER_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "")
WEATHER_TIMEOUT_SECONDS = float(os.getenv("WEATHER_TIMEOUT_SECONDS", "5"))

USER_DISPLAY_NAME = os.getenv("USER_DISPLAY_NAME", "Mr Taksh")
JARVIX_GREETING = os.getenv("JARVIX_GREETING", "Welcome back")

# When true, app/brain/orchestrator.py writes concise per-request tool-call
# decisions to jarvix.log via app/runtime/log.py (never raw prompts/chain of
# thought - see orchestrator.py's _debug_log).
JARVIX_DEBUG_ORCHESTRATOR = _truthy("JARVIX_DEBUG_ORCHESTRATOR", "false")

# Text-to-speech: OpenAI's TTS API is primary; local Windows SAPI is a soft
# fallback so Jarvix keeps talking through a network/quota/account hiccup.
# Voice hints (SAPI only) are matched against installed voice names/ids.
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "onyx")
OPENAI_TTS_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TTS_TIMEOUT_SECONDS", "15"))
# OpenAI's speech "speed" param, 0.25-4.0 (1.0 = normal pace).
OPENAI_TTS_SPEED = float(os.getenv("OPENAI_TTS_SPEED", "1.3"))
TTS_RATE = int(os.getenv("TTS_RATE", "165"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))
TTS_VOICE_HINTS = [
    hint.strip().lower()
    for hint in os.getenv("TTS_VOICE_HINTS", "george,david,male,english,british,uk").split(",")
    if hint.strip()
]

# Speech-to-text: AssemblyAI Sync API for voice commands (transcribe() below),
# faster-whisper for the always-on wake-word listener (transcribe_wake_word()).
# "tiny.en" is fastest for faster-whisper; "base.en" more accurate.
STT_MODEL = os.getenv("STT_MODEL", "base.en")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

# Max seconds to record one voice command (recorder also stops early on silence).
VOICE_RECORD_SECONDS = int(os.getenv("VOICE_RECORD_SECONDS", "5"))
MIC_SILENCE_THRESHOLD = float(os.getenv("MIC_SILENCE_THRESHOLD", "0.0015"))
WAKE_DEBUG = os.getenv("WAKE_DEBUG", "false").lower() in ("1", "true", "yes", "on")

# How Jarvix is woken:
#   "wakeword" = say "hey jarvis" (default, most reliable)
#   "clap"     = double-clap detector
#   "enter"    = press Enter (foreground terminal only)
WAKE_MODE = os.getenv("WAKE_MODE", "wakeword")

# The spoken wake word (matched loosely against the STT transcript).
WAKE_WORD = os.getenv("WAKE_WORD", "jarvis")

# Clap loudness floor (0..1 peak amplitude) for clap mode. Adaptive above this.
CLAP_THRESHOLD = float(os.getenv("CLAP_THRESHOLD", "0.06"))

# Welcome routine: only SAFE actions auto-run on wake. Set any to "" to disable.
AUTO_OPEN_APP_ON_WAKE = os.getenv("AUTO_OPEN_APP_ON_WAKE", "cursor")
AUTO_OPEN_FOLDER_ON_WAKE = os.getenv("AUTO_OPEN_FOLDER_ON_WAKE", "jarvix")
AUTO_START_MUSIC_ON_WAKE = os.getenv("AUTO_START_MUSIC_ON_WAKE", "true").lower() in (
    "1", "true", "yes", "on",
)
AUTO_MUSIC_QUERY_ON_WAKE = os.getenv(
    "AUTO_MUSIC_QUERY_ON_WAKE",
    "Should I Stay or Should I Go",
)
AUTO_MUSIC_URI_ON_WAKE = os.getenv("AUTO_MUSIC_URI_ON_WAKE", "")

# Browser control (app/browser/*). Dedicated, persistent Chrome/Chromium
# profile so Jarvix can stay logged into sites without touching the user's
# normal browser profile. Headless is off by default per the feature spec -
# the user wants to see what Jarvix is doing.
JARVIX_CHROME_PROFILE_DIR = os.getenv("JARVIX_CHROME_PROFILE_DIR", "") or str(ROOT / "chrome-profile")
JARVIX_BROWSER_HEADLESS = _truthy("JARVIX_BROWSER_HEADLESS", "false")
JARVIX_BROWSER_CHANNEL = os.getenv("JARVIX_BROWSER_CHANNEL", "chrome")
BROWSER_ACTION_TIMEOUT_SECONDS = float(os.getenv("BROWSER_ACTION_TIMEOUT_SECONDS", "15"))
BROWSER_NAV_TIMEOUT_SECONDS = float(os.getenv("BROWSER_NAV_TIMEOUT_SECONDS", "30"))
MAX_BROWSER_AGENT_STEPS = int(os.getenv("MAX_BROWSER_AGENT_STEPS", "20"))

# --- Real-Chrome mode ("auto" by default) -----------------------------------
# Which browser Jarvix drives:
#   "dedicated" - ALWAYS use the isolated JARVIX_CHROME_PROFILE_DIR. Never
#                 touches the user's real Chrome. Safest, but starts logged
#                 out of everything until the user signs in there once.
#   "real"      - ALWAYS attach to the user's real running Chrome over CDP
#                 (see JARVIX_BROWSER_CDP_URL). Errors out if that endpoint
#                 isn't reachable - use this to catch setup problems loudly.
#   "auto"      - try a real-Chrome CDP attach first (a fast, cheap probe -
#                 see app/browser/manager.py:_probe_cdp_available); if it's
#                 not reachable, silently fall back to the dedicated profile.
#                 DEFAULT. No flag-flipping needed: if the user has launched
#                 Chrome with --remote-debugging-port, Jarvix uses it and
#                 gets their existing logins; if not, it just works anyway
#                 on the isolated profile like before.
# Attaching to the real browser trades away the profile-isolation safety
# boundary - Jarvix can then see and act through EVERY account it's signed
# into - so real-Chrome access (in "real" OR "auto" mode, whenever it's
# actually used) is guarded three ways:
#   1. It's never silently assumed - "auto" falls back safely, "real" fails
#      loudly, and only an actual live CDP endpoint ever engages it.
#   2. Every task pauses for an explicit "use my real Chrome for this?" gate
#      before the agent loop touches the browser (see app/browser/tools.py).
#   3. A code-enforced domain blocklist (below) that the agent physically
#      cannot navigate to or act on, independent of the LLM's judgement.
JARVIX_BROWSER_MODE = os.getenv("JARVIX_BROWSER_MODE", "").strip().lower()
if not JARVIX_BROWSER_MODE:
    # Back-compat: the old on/off flag from before JARVIX_BROWSER_MODE existed.
    JARVIX_BROWSER_MODE = "real" if _truthy("JARVIX_BROWSER_USE_REAL_CHROME", "false") else "auto"
if JARVIX_BROWSER_MODE not in ("dedicated", "real", "auto"):
    JARVIX_BROWSER_MODE = "auto"
# Kept for any external code still reading the old flag name.
JARVIX_BROWSER_USE_REAL_CHROME = JARVIX_BROWSER_MODE == "real"
# CDP endpoint of the user's running Chrome (the "Server running at" address on
# chrome://inspect#devices -> Remote debugging). host:port form.
JARVIX_BROWSER_CDP_URL = os.getenv("JARVIX_BROWSER_CDP_URL", "http://127.0.0.1:9222")
# How long "auto" mode's cheap reachability probe waits before giving up and
# falling back to the dedicated profile. Kept short - this runs on every
# fresh browser start, so a slow/wrong URL shouldn't stall every task.
JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS = float(os.getenv("JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS", "1.5"))
# Domains Jarvix is HARD-BLOCKED from in real-Chrome mode (substring match on
# the hostname). Enforced in app/browser/safety.py + actions.goto, not merely
# requested of the model. Defaults cover banking, brokerages, and password
# managers - the accounts most costly to have an agent poking at. Editable via
# JARVIX_BROWSER_BLOCKED_DOMAINS (comma-separated, REPLACES the defaults) or
# JARVIX_BROWSER_EXTRA_BLOCKED_DOMAINS (comma-separated, ADDED to them).
_DEFAULT_BLOCKED_DOMAINS = [
    "bank", "chase.com", "wellsfargo.com", "bankofamerica.com", "citi.com",
    "capitalone.com", "hsbc", "barclays", "paypal.com", "venmo.com", "wise.com",
    "revolut.com", "coinbase.com", "binance.com", "kraken.com", "robinhood.com",
    "fidelity.com", "schwab.com", "vanguard.com", "e-trade.com", "etrade.com",
    "1password.com", "lastpass.com", "bitwarden.com", "dashlane.com",
    "keepersecurity.com", "accounts.google.com/signin/v2/challenge",
    "myaccount.google.com/security", "icicibank", "hdfcbank", "sbi.co", "axisbank",
    "paytm", "phonepe",
]
_blocked_override = [d.strip().lower() for d in os.getenv("JARVIX_BROWSER_BLOCKED_DOMAINS", "").split(",") if d.strip()]
_blocked_extra = [d.strip().lower() for d in os.getenv("JARVIX_BROWSER_EXTRA_BLOCKED_DOMAINS", "").split(",") if d.strip()]
JARVIX_BROWSER_BLOCKED_DOMAINS = (_blocked_override or _DEFAULT_BLOCKED_DOMAINS) + _blocked_extra

# "Open <name>" site resolution (app/browser/history.py) - ranks a spoken
# short name like "github" against the user's REAL Chrome browsing history
# (visit count + recency), so "open github" opens the account/URL they
# actually use instead of a generic search. Read-only: the reader always
# copies Chrome's locked History sqlite file first, never opens it in place,
# and Jarvix's own browser profile (JARVIX_CHROME_PROFILE_DIR above) is never
# touched by this - only used to read where to navigate it to.
# Comma-separated list of "User Data" profile folder names to search (Chrome
# usually has "Default" plus "Profile 1", "Profile 2", ... for extra people).
CHROME_REAL_PROFILE_DIR = os.getenv("CHROME_REAL_PROFILE_DIR", "") or str(
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
)
CHROME_REAL_PROFILES = [p.strip() for p in os.getenv("CHROME_REAL_PROFILES", "Default,Profile 1,Profile 2").split(",") if p.strip()]

# Applicant profile (app/browser/tools.py) - filled into job/internship
# application forms and used to draft outreach messages. Never guessed or
# invented; a field left blank here means the browser agent must ask you
# instead of making one up.
APPLICANT_NAME = os.getenv("APPLICANT_NAME", "") or USER_DISPLAY_NAME
APPLICANT_EMAIL = os.getenv("APPLICANT_EMAIL", "")
APPLICANT_PHONE = os.getenv("APPLICANT_PHONE", "")
APPLICANT_RESUME_PATH = os.getenv("APPLICANT_RESUME_PATH", "")
APPLICANT_LINKEDIN_URL = os.getenv("APPLICANT_LINKEDIN_URL", "")
APPLICANT_PORTFOLIO_URL = os.getenv("APPLICANT_PORTFOLIO_URL", "")
