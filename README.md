# Jarvix 🧠🎙️ — a JARVIS-style voice assistant that actually runs your laptop

Say **"Hey Jarvis"** and it answers back — in a JARVIS voice, JARVIS wit — then
briefs your calendar, reads your email, manages your projects, and drives a
**real Chrome browser** to get things done on any website: apply to
internships, check LinkedIn, find flights, log into services — like a person
would, not an API integration. It remembers everything you tell it across
sessions, asks before anything risky, and it's fully open source.

```
"Hey Jarvis, find opportunities for me and open the best one in my browser."
   → searches, ranks by fit, opens the real page, reads it back to you.

"Apply to this internship — use my resume."
   → fills the form itself, drafts the pitch, pauses right before Submit.

"Save this as a new project: an AI poker trainer that reads betting odds."
   → remembered. Ask about it next week — it still knows.
```

**One weekend build. Local wake-word detection, an LLM tool-calling brain,
persistent memory, and an agentic browser with its own safety rails —
all wired into one voice loop.**

---

## Highlights

- 🌐 **Agentic browser control** — a real, visible Chrome window that navigates,
  reads, clicks, types, and fills out forms on any site. It resolves "open
  &lt;name&gt;" against your actual browsing history, applies to jobs with
  your saved resume, and **always pauses for your go-ahead** before anything
  irreversible (send, buy, submit, delete, publish, change a password).
- 🧠 **Real memory, not a context window** — every conversation, project, goal,
  and task lives in Postgres/pgvector. Tell it about a project once; ask
  about it next week and it still knows.
- 🗣️ **JARVIS voice and personality** — OpenAI TTS tuned to a calm, dry-witted
  British-butler cadence, driven by a local always-on wake-word listener
  (`faster-whisper`) so ambient room audio never leaves your machine.
- 🛠️ **A real tool-calling brain**, not a chatbot with plugins — an LLM
  orchestrator chains calendar, email, news, memory, and browser tools in one
  request, deciding what it needs and when, with a hard confirmation gate in
  front of every mutating action.
- 🔒 **Safety by construction** — an allowlist for openable apps/folders, a
  code-enforced blocklist for banking/password-manager sites in browser mode,
  and every risky action requires an explicit spoken "yes."

Full capability + confirmation-policy table → [What it does in detail](#what-it-does-in-detail).

---

## The brain: how a spoken command becomes an action

`app/brain/intent_router.py` turns raw text into one of ~20 named intents
(`open_app`, `add_event`, `read_emails`, `news`, `weather`, `question`, ...),
in two stages:

1. **Deterministic rules first** (`_parse_rules`). A sequence of keyword/regex
   checks — e.g. "weather"/"forecast" → `weather`, "add"+"event" (but not
   "email") → `add_event`, "news"/"headlines" → `news`. Rules run first
   because they're instant, free, and predictable for the commands people say
   most often. Order matters: more specific rules (email drafting, add-event,
   news) are checked before broader catch-alls (the generic "today" schedule
   rule) so a specific request never gets swallowed by a vaguer one later in
   the list.
2. **LLM fallback** (`_parse_with_llm`), only when no rule matches. OpenAI is
   given the full list of intents with descriptions and an explicit decision
   procedure — identify the request, check whether exactly one intent is the
   right tool for it, and if none genuinely fits, return `question` so the
   assistant answers directly from its own knowledge instead of forcing a
   wrong tool. This keeps the model from mis-routing things it wasn't built to
   handle.

Once an intent is produced, `app/main.py:run_intent` dispatches it: safe,
instant intents (open app/folder, all music actions) run directly inside the
router; everything else (calendar, email, news, questions) is dispatched to a
handler function in `main.py`. Two safety gates sit in front of anything
irreversible:

- `app/safety/permissions.py` marks `send_email` and `create_calendar_event`
  (among others) as actions that need confirmation.
- The **voice path** confirms by ear — Jarvix speaks the details back and
  requires a spoken "yes" (handled by `app/brain/dialogue.py`'s multi-turn
  state machine) before writing anything.
- The **`scan-mail` CLI/voice command** additionally requires a typed
  `all`/`none`/`1,3`-style approval in the terminal before any of the
  proposed events are created — this path never auto-writes on a spoken yes
  alone.

`app/brain/llm_client.py` is the only place that talks to OpenAI's Chat
Completions and Embeddings APIs (plain `requests` calls, no SDK). Every LLM
call in the app — intent fallback classification, calendar-candidate
extraction from email, email/news summarization, draft-email composition,
general question answering, and memory embeddings — goes through this one
module.

---

## Browser control: how Jarvix operates a real website

`app/browser/` gives the orchestrator one more tool, `browser_task`, for
anything that means actually *using* a website rather than a canned API call
(Gmail/Calendar/Spotify already have direct tools and stay on those — browser
control is for everything else: GitHub, university portals, flight search,
shopping, arbitrary web apps).

```
Jarvix (voice/text) → app/brain/orchestrator.py (outer tool loop)
                           ↓ selects browser_task(instruction)
                     app/browser/tools.py (inner browser agent loop)
                           ↓ read_page / click / type / goto / ...
                     app/browser/actions.py → Playwright → dedicated Chrome profile
```

**Two loops, not one.** The outer orchestrator (the same one that answers
"what's the weather" or drafts a calendar event) picks `browser_task` like any
other tool, handing it one plain-language instruction. From there, a *second*,
independent agent loop (`app/browser/tools.py:run_browser_task`) takes over:
it reads the page (`app/browser/state.py` turns the DOM into a short numbered
list of interactive elements + visible text, not raw HTML), decides the next
click/type/navigate, executes it, and reads again — repeating until the task
is done, fails, or needs you. This mirrors the outer orchestrator's own
iterative tool-calling design (`app/brain/orchestrator.py`), just over a
browser-specific action set instead of the data-lookup tool registry.

**Pause-and-resume confirmation.** Before any step with a real-world
consequence (send, submit, buy, delete, publish, agree to terms, change a
password/security setting, ...), the loop stops **before** touching the page —
flagged either because the model marked the action `high_impact: true`, or
because `app/browser/safety.py`'s keyword backstop recognizes the target
label ("Send", "Delete Account", "Place Order", ...) regardless of what the
model thought. The browser tab is left open exactly as it was; Jarvix asks
you aloud ("I'm about to click 'Send', sir. Shall I proceed?") through the
same `VoiceDialogue` pending-confirmation mechanism as every other
confirmation, and on "yes" resumes the **same** loop to actually perform that
one action and continue — it never restarts the task or re-fills the form.

**CAPTCHA/2FA/security checks:** the loop never attempts to solve these. If
`read_page` detects challenge language on the page, or the model calls
`needs_human`, Jarvix stops and tells you what's blocking it so you can handle
it by hand, then ask Jarvix to continue.

**Recovery:** a failed click/type (stale element, timeout, page changed) is
reported back to the model as a tool error, not raised — it reads the page
again and tries a sensible alternative before giving up (`task_failed`).

**"Open &lt;name&gt;" by your own browsing habits.** Saying "open github" or
"open my bank" doesn't need an exact URL — the `resolve_site_from_history`
tool (`app/browser/history.py`) reads your **real** Chrome profile's history
(copied read-only each time, since Chrome keeps the file locked while
running — never opened in place) and ranks matches by visit count/recency, so
it opens the actual site/account you use, not a generic guess. This is
completely separate from `JARVIX_CHROME_PROFILE_DIR` (Jarvix's own,
initially-empty browsing profile) — history is only ever *read* from your
normal Chrome, never written to.

**Job/internship applications, applying to part-time work, and messaging
leads.** The browser agent can fill out an entire application — name, email,
resume upload, a drafted cover letter/outreach message — using your saved
`APPLICANT_*` profile (see [Setup](#setup)) via the `get_applicant_profile`
tool, never inventing a name, email, or work history it wasn't given. It
**always** pauses for confirmation on the final Submit/Apply/Send click, no
matter how much of the form it filled in on its own. If a form asks for
something with no saved answer (e.g. "why do you want to work here" or a
portfolio URL you haven't set), it calls `ask_user` to ask you directly
instead of making one up — answered the same way as a yes/no confirmation,
just with an open-ended reply.

**Thread-safety note (only matters if you're reading the code):**
Playwright's sync API is bound to whichever OS thread starts it, but
`app/server.py`'s `/api/command` runs each request on a fresh worker thread.
`app/browser/manager.py` handles this with one dedicated background thread
that owns Chrome end-to-end; every Playwright call from `actions.py`/
`state.py`/`tools.py` is marshalled onto it via `BrowserManager.run(...)`, so
callers never need to think about which thread they're on.

### Real-Chrome mode — driving your actual browser

`JARVIX_BROWSER_MODE` controls which browser Jarvix drives, and defaults to
`auto`: on every fresh browser start, it runs a fast, cheap reachability check
against your real, already-running Chrome; if that succeeds, it attaches to
it over the DevTools Protocol (CDP) and gets your existing logins; if not, it
silently falls back to the isolated profile, exactly like Jarvix always
worked. No flag to remember to flip either way — it adapts to whatever's
actually running.

| Mode | Behavior |
|---|---|
| `auto` (default) | Try real Chrome first, fall back to the isolated profile if it's not reachable |
| `dedicated` | Always use the isolated profile — never touches your real Chrome |
| `real` | Always require the real Chrome connection — fails loudly (not a silent fallback) if it isn't reachable, useful for confirming your CDP setup actually works |

Driving your real Chrome removes the profile-isolation boundary — Jarvix can
then see and act through *every* account that browser is signed into — so
whenever it's actually used (`real` mode, or `auto` mode when reachable) it's
guarded three ways:

1. **Never silently assumed.** `auto` only engages it when a live CDP endpoint
   is actually found; otherwise nothing about your setup changes.
2. **Per-task approval gate.** Before the agent touches your live browser for
   a task, Jarvix stops and asks *"this will use your real Chrome, with
   everything you're signed into — shall I go ahead and &lt;task&gt;?"* Only a
   clean "yes" proceeds; anything ambiguous declines. Runs *before* any
   navigation, on top of the existing per-action high-impact gate.
3. **Code-enforced blocklist.** A protected-site list (banking, brokerages,
   password managers by default; editable via `JARVIX_BROWSER_BLOCKED_DOMAINS`
   / `JARVIX_BROWSER_EXTRA_BLOCKED_DOMAINS`) that Jarvix physically cannot
   navigate to or act on — enforced in `app/browser/safety.py` +
   `actions.goto` + the agent loop, not merely requested of the model. It also
   refuses to close your existing tabs, and detaching never closes your Chrome.

**Enabling real Chrome:** Chrome must be relaunched with the classic debug
flag — the in-browser *"Allow remote debugging for this browser instance"*
toggle (the Chrome DevTools MCP feature) is a **different, newer protocol**
that Playwright cannot speak; it will not work here no matter how it's
enabled. Fully quit Chrome, then relaunch it from a terminal with:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="$env:LOCALAPPDATA\Google\Chrome\User Data"
```

and point `JARVIX_BROWSER_CDP_URL` at it if you used a different port (default
`http://127.0.0.1:9222`). This closes all existing Chrome windows first — save
your work before running it.

> **A real limit, not a bug:** Google (and some other sites) actively detect
> and block sign-in attempts from automation-controlled browser sessions —
> "This browser or app may not be secure" — *even when attached to your real
> Chrome over CDP*, since the CDP attachment itself is part of what's
> detected. Jarvix will not attempt to disguise or evade this; it's a
> deliberate security control on Google's end. The reliable way to get
> Jarvix authenticated anywhere (including with real-Chrome mode on) is to
> **sign in yourself, once, in the actual browser window** — after that the
> session persists and future tasks reuse it without hitting sign-in again.

---

## Memory: how Jarvix remembers things

There are two separate memory systems doing different jobs:

**1. RAG conversation memory (Postgres + pgvector, `app/memory/db.py`)**
Every command you give Jarvix — voice or typed — is logged after it's
answered: the transcript, which intent it routed to, the response, and an
OpenAI embedding (`text-embedding-3-small`) of the transcript, stored in a
Supabase Postgres `interactions` table. Before answering a `question` or
`conversation` intent, Jarvix embeds your new request and pulls the most
similar past interactions (`embedding <=> query` cosine search) into the
system prompt as short context, so it can recall things you told it in
earlier sessions instead of starting cold every time. Logging happens on a
background thread so a slow database or embedding call never delays the
spoken reply, and every function in `db.py` degrades silently — a missing
`DATABASE_URL`, an unreachable database, or a failed embedding call simply
means "no memory this time," never a crash or a stalled response.

**2. Email-extraction cache (SQLite, `app/memory/cache.py`)**
A much narrower, local cache that has nothing to do with conversation memory:
it keys on Gmail message ID + a content hash, so re-scanning your inbox for
calendar-worthy events doesn't re-run the (paid) OpenAI extraction call on a
message it has already processed and that hasn't changed.

Two flat files round out the "memory" folder and are pure configuration, not
learned/logged data: `app/memory/aliases.json` (the allowlist of openable
apps/folders) and `app/memory/contacts.json` (name → email address, gitignored,
built from `contacts.example.json`).

---

## What it does in detail

| Tool | What it does | Confirmation needed? |
|---|---|---|
| Calendar read | Speaks today's/tomorrow's/a specific day's Google Calendar events | No |
| **Add calendar event** | Multi-turn voice dialogue (title → date → time) that creates a real Google Calendar event | Yes — spoken "yes" after Jarvix reads back the details |
| Scan mail for events | Reads recent Gmail, has OpenAI propose calendar-worthy items (meetings, deadlines, bookings...), shows them, and only adds the ones you approve | Yes — typed approval in the terminal |
| Read/summarize email | Reads recent Gmail and gives a one-sentence, OpenAI-generated summary of each message's actual point | No (read-only) |
| News | Fetches top headlines (NewsAPI) and gives a one-sentence OpenAI summary of each | No (read-only) |
| Draft email | Has OpenAI write a short email from your instruction, reads it aloud | No (never sends) |
| Send email | Same as draft, then actually sends via Gmail | Yes — spoken "should I send this?" |
| Open app / folder | Launches an app or opens a folder, from an **allowlist only** (`app/memory/aliases.json`) — no arbitrary paths or shell commands | No |
| Music control | Play/pause/next/prev/volume/mute (OS media keys) and play a specific song/artist (Spotify) | No |
| Weather | Current conditions for a spoken or default city (Open-Meteo, no key needed) | No |
| Time | Current local time | No |
| General questions / chat | Answered directly by OpenAI, with relevant past conversations pulled in as context | No |
| **Browser control** | Drives a real Chrome browser to complete open-ended tasks on any website — navigate, read, click, type, fill forms, upload/download, compare options, apply to jobs | Reads/navigation: no. Any high-impact step (send, buy, delete, publish, change a password/security setting, agree to terms, ...): yes — spoken confirmation, mid-task |

Multi-turn follow-ups fill in missing details when you leave something out —
e.g. asking to email someone without saying what to say, or asking for the
weather without naming a city.

> **Reality check:** Jarvix can't run while the laptop is off or asleep. It
> runs once you're logged in and the background process has been started (see
> [Run on startup](#run-on-startup)). It needs microphone permission.

---

## Requirements

- Windows 10/11, a working microphone and speakers
- Python 3.12 + the bundled virtual env (`.venv`)
- Required API keys: OpenAI (brain, TTS, embeddings), AssemblyAI (voice-command transcription)
- A Google Cloud OAuth client (Desktop app) with Gmail + Calendar scopes
- Optional: Spotify Web API credentials (search/play-by-name — media-key
  control works without them), NewsAPI key (news headlines), a Supabase/Postgres
  database with the `pgvector` extension (conversation memory)
- Optional (browser control): Google Chrome installed (Playwright drives your
  real Chrome via `channel="chrome"`, not a bundled Chromium)

---

## Setup

```powershell
cd path\to\jarvix
python -m venv .venv             # if .venv doesn't exist yet
.\.venv\Scripts\activate
pip install -r requirements.txt

# Config
copy .env.example .env           # then edit values
```

At minimum, set `OPENAI_API_KEY` and `ASSEMBLYAI_API_KEY` in `.env`. Every
other tool (Spotify, news, memory) degrades gracefully — Jarvix runs and
speaks without them, it just can't do that one thing until configured.

**Google credentials:** put your OAuth client file at
`secrets/google_credentials.json`. The token is created on first auth and
saved to `secrets/google_token.json`. Both are gitignored.

First run authorizes Google in your browser:

```powershell
python -m app.main reauth
```

Make sure your Google Cloud OAuth consent screen includes all four scopes:

```
gmail.readonly   gmail.send   calendar.readonly   calendar.events
```

**Conversation memory (optional):** set `DATABASE_URL` to a Postgres
connection string (a free Supabase project works) with the `pgvector`
extension available. Jarvix creates the extension and table itself on first
use (`app/memory/db.py:init_schema`) — no manual migration needed. Leave it
blank to run without memory.

**Browser control (optional):** Playwright's Python package is already in
`requirements.txt`; it still needs Chrome registered with it once:

```powershell
python -m playwright install chrome
```

`JARVIX_CHROME_PROFILE_DIR` (in `.env`) controls where Jarvix's own,
dedicated Chrome profile lives — defaults to `jarvix\chrome-profile` if left
blank. This is a brand-new, empty profile; it is **not** your normal Chrome
profile and starts out logged into nothing.

**Logging into websites the first time:**

```powershell
python -m app.main route "open gmail"
```

The Chrome window that opens (`JARVIX_BROWSER_HEADLESS=false` by default, so
you can watch it) is Jarvix's dedicated profile. Log into Gmail, GitHub, your
university portal, etc. by hand in that window, same as any browser — the
session/cookies are saved to `JARVIX_CHROME_PROFILE_DIR` and reused
automatically on every future run, so you only do this once per site. Jarvix
never sees or stores your password — it only ever reads the resulting page
after you've signed in.

**"Open &lt;name&gt;" by browsing history (optional):** works out of the box
against the default Chrome install location. If Chrome is installed
somewhere nonstandard, set `CHROME_REAL_PROFILE_DIR`; `CHROME_REAL_PROFILES`
(default `Default,Profile 1,Profile 2`) controls which of Chrome's own
profile folders get searched — add more names if you use additional Chrome
profiles/people.

**Applicant profile (optional, for job/internship applications):** set
`APPLICANT_NAME`, `APPLICANT_EMAIL`, `APPLICANT_PHONE`,
`APPLICANT_RESUME_PATH` (a local file path), `APPLICANT_LINKEDIN_URL`, and
`APPLICANT_PORTFOLIO_URL` in `.env`. Any left blank simply means Jarvix will
ask you for that detail instead of filling it in — it's never invented.

---

## How to run

```powershell
cd path\to\jarvix
.\.venv\Scripts\activate

python -m app.main wake          # full experience: wake word -> welcome -> commands
```

### Command reference

| Command | What it does |
|---|---|
| `wake` | First wake trigger runs the welcome routine, then trigger again to talk |
| `welcome` | Run the welcome routine once (brief + open app/folder + music) |
| `brief` | Print **and speak** the daily calendar briefing |
| `today` | Spoken schedule summary for today |
| `today-fast` | Instant deterministic plan (calendar + deadlines + top emails, no LLM) |
| `listen` | Record one spoken command and print the transcription |
| `route "<text>"` | Parse a text command, show the matched intent, and execute it (test routing without voice) |
| `open-app <alias>` | Open an allowlisted app (e.g. `vscode`, `chrome`, `spotify`) |
| `open-folder <alias>` | Open an allowlisted folder (e.g. `jarvix`, `downloads`) |
| `music <action>` | `playpause` \| `next` \| `prev` \| `volume-up` \| `volume-down` \| `mute` |
| `play "<song>"` | Search Spotify for a song/artist and start playback |
| `scan-mail` | Read Gmail → propose calendar events via OpenAI → **typed confirmation** → write |
| `scan-mail-deep` | Same as `scan-mail`, but skips the keyword prefilter (model sees every fetched email) |
| `draft-email <to> "<msg>"` | Draft an email (does **not** send) |
| `send-email <to> "<msg>"` | Draft, read aloud, **confirm**, then send |
| `full-run` | `scan-mail` followed by `today` |
| `reauth` | Refresh Google OAuth (needed after scope changes) |
| `spotify-auth` | Authorize the Spotify Web API and save a refresh token |
| `say "<text>"` | Speak text aloud (TTS test) |
| `test-brain` | One-shot OpenAI call to confirm the brain is reachable |
| `clap-calibrate` | Live mic monitor for tuning `CLAP_THRESHOLD` |
| `mic-debug` | Record a few seconds, print mic levels, and transcribe |

Voice examples: *"what's my calendar looking like today"*, *"add an event to
my calendar"*, *"what's the news today"*, *"summarize my last 3 emails"*,
*"weather in Enschede"*, *"play Travis Scott"*, *"send an email to Tisha
saying I'll be late"*. Jarvix asks a short follow-up when a command like that
is missing a detail it needs.

Browser examples: *"open Gmail and find my latest email from the University
of Twente"*, *"check my GitHub notifications"*, *"find the cheapest flight
from Amsterdam to Mumbai next month"* (Jarvix compares options and reports
back — it will not book anything without asking first), *"go to my
university's portal and check if grades are posted"*, *"open github"* /
*"open my bank"* (opens the actual site you visit most under that name, per
your real Chrome history), *"apply to the software intern role at this
company"* / *"message this recruiter about the opening"* (fills the form with
your saved applicant profile, drafts the message, and always asks before the
final Submit/Send).

---

## Web frontend (optional)

`app/server.py` is a local FastAPI bridge that exposes the exact same
`VoiceDialogue`/orchestrator stack the `route`/`wake` commands drive, over
HTTP + Server-Sent Events instead of a terminal or microphone —
`web/jarvix-ui.html` is a single-file browser UI built against it (mic input,
TTS playback, live activity log, and a side panel that renders real
tool/browser-agent events as they stream in). Single-user, no auth, CORS wide
open — a localhost dev tool, not a deployed service, matching the rest of
this codebase's single-tenant design.

```powershell
cd path\to\jarvix
.\.venv\Scripts\activate
.venv\Scripts\python.exe -m uvicorn app.server:app --reload --port 8000
```

Then open `web/jarvix-ui.html` directly in a browser (or serve it from any
static file server). Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Connection check for the frontend |
| `POST /api/command` | `{session_id, text}` → SSE stream of live orchestrator/browser-agent progress, ending with `{"type": "final", "text": ...}` |
| `POST /api/transcribe` | Browser mic recording (multipart) → `{text}` via AssemblyAI |
| `POST /api/speak` | `{text}` → WAV audio bytes via OpenAI TTS |
| `GET /api/briefing` | The same daily briefing text `brief`/`wake`'s welcome routine speaks |

A confirmation (calendar write, email send, or a browser agent pausing before
a high-impact click) works exactly like any other reply — the frontend just
sends your next spoken/typed "yes"/"no" as a normal `/api/command` call; there's
no separate confirm/deny button, `VoiceDialogue`'s existing pending-state
machine handles it the same way it does for the terminal/voice paths.

---

## The wake / welcome flow

```
wake word ("hey jarvis") or double clap or Enter, per WAKE_MODE
  → speak greeting + today's calendar
  → open AUTO_OPEN_APP_ON_WAKE        (e.g. vscode)
  → open AUTO_OPEN_FOLDER_ON_WAKE     (e.g. jarvix, in Explorer)
  → start AUTO_MUSIC_QUERY_ON_WAKE / AUTO_MUSIC_URI_ON_WAKE on Spotify,
    or just launch Spotify if neither is set
  → listen for the next command (wake again to talk)
```

Spotify is launched via the installed desktop app when it's found on the
machine (checked in `app/tools/desktop.py`), falling back to the
open.spotify.com web player only if the desktop app isn't installed.

---

## Configuration (`.env`)

See `.env.example` for the full, commented list with setup links for every
key. The short version:

| Area | Keys |
|---|---|
| Brain / TTS / embeddings | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| News | `NEWS_API_KEY` |
| Conversation memory | `DATABASE_URL` (Postgres/Supabase, needs `pgvector`) |
| Google | `GOOGLE_CREDENTIALS_FILE`, `GOOGLE_TOKEN_FILE` |
| Spotify | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`, `SPOTIFY_TOKEN_FILE` |
| Weather/time | `TIMEZONE`, `DEFAULT_WEATHER_LOCATION`, `WEATHER_TIMEOUT_SECONDS` |
| Persona | `USER_DISPLAY_NAME`, `JARVIX_GREETING` |
| Speech-to-text | `STT_MODEL`, `STT_LANGUAGE`, `ASSEMBLYAI_API_KEY`, `VOICE_RECORD_SECONDS`, `MIC_SILENCE_THRESHOLD`, `WAKE_DEBUG` |
| Wake trigger | `WAKE_MODE`, `WAKE_WORD`, `CLAP_THRESHOLD` |
| Text-to-speech | `OPENAI_TTS_MODEL`, `OPENAI_TTS_VOICE`, `OPENAI_TTS_TIMEOUT_SECONDS`, `TTS_RATE`, `TTS_VOLUME`, `TTS_VOICE_HINTS` |
| Wake routine | `AUTO_OPEN_APP_ON_WAKE`, `AUTO_OPEN_FOLDER_ON_WAKE`, `AUTO_START_MUSIC_ON_WAKE`, `AUTO_MUSIC_QUERY_ON_WAKE`, `AUTO_MUSIC_URI_ON_WAKE` |
| Browser control | `JARVIX_CHROME_PROFILE_DIR`, `JARVIX_BROWSER_HEADLESS`, `JARVIX_BROWSER_CHANNEL`, `BROWSER_ACTION_TIMEOUT_SECONDS`, `BROWSER_NAV_TIMEOUT_SECONDS`, `MAX_BROWSER_AGENT_STEPS` |
| "Open &lt;name&gt;" history matching | `CHROME_REAL_PROFILE_DIR`, `CHROME_REAL_PROFILES` |
| Applicant profile (job applications) | `APPLICANT_NAME`, `APPLICANT_EMAIL`, `APPLICANT_PHONE`, `APPLICANT_RESUME_PATH`, `APPLICANT_LINKEDIN_URL`, `APPLICANT_PORTFOLIO_URL` |
| Real-Chrome mode (default `auto`) | `JARVIX_BROWSER_MODE`, `JARVIX_BROWSER_CDP_URL`, `JARVIX_BROWSER_CDP_PROBE_TIMEOUT_SECONDS`, `JARVIX_BROWSER_BLOCKED_DOMAINS`, `JARVIX_BROWSER_EXTRA_BLOCKED_DOMAINS` |

`WEB_SEARCH_API_KEY` is declared in `.env.example` but not wired into any
tool yet — reserved for a future web-search intent.

### Aliases & contacts (user-editable)

- **Apps/folders:** `app/memory/aliases.json` — add your own. Only listed
  aliases can be opened; there is intentionally no path to run an arbitrary
  shell command or open an arbitrary path the model names.
- **Contacts:** `app/memory/contacts.json` (gitignored — holds real emails).
  Copy `contacts.example.json` to start. Unknown recipients prompt you to type
  the address, which is then saved for next time.

---

## Safety model

| Action | Policy |
|---|---|
| Read calendar/Gmail/news, speak, answer questions | ✅ auto |
| Open allowlisted apps/folders | ✅ auto |
| Music play/pause/next/prev/volume | ✅ auto |
| Add a calendar event you describe | ⚠️ requires a spoken "yes" after Jarvix reads back the details |
| Scan mail → add calendar events | ⚠️ requires **typed** approval in the terminal (`all`/`none`/`1,3`) |
| Send email | ⚠️ requires a spoken "should I send this?" confirmation |
| Browser: navigate, search, read, click, scroll, fill in forms | ✅ auto (no confirmation for browsing/research itself) |
| Browser: send/submit/buy/book/delete/publish/agree/change password or security settings | ⚠️ pauses mid-task, requires a spoken "yes" before that one step runs |
| Browser: CAPTCHA, 2FA, password re-entry, biometric checks | ⛔ never attempted — Jarvix stops and asks you to handle it |
| Arbitrary terminal commands | ⛔ blocked — not implemented |
| Delete/move files, delete/archive email, delete a calendar event | ⛔ not implemented |

Jarvix says "sent" only **after** the Gmail API confirms success, and "added"
only after the Calendar API confirms the event was created.

---

## Voice tuning

- **Wake-name variations:** common mishears such as Jarvix, Jervis, Jorvix,
  Garvis, and Zarvis are accepted near the start of the utterance. Broad
  fragments such as "jar" and "travis" are intentionally rejected so music and
  ordinary speech don't wake the assistant.
- **Noisy ignored transcripts:** keep `WAKE_DEBUG=false` (the default).
  Rejected speech is discarded silently and never routed as a command.
- **Muffled wake speech:** `base.en` is the recommended `STT_MODEL`. The
  wake-word pass uses a wider search, Jarvis-oriented hot words, quiet-audio
  normalization, and `MIC_SILENCE_THRESHOLD` (default `0.0015`).
- **Clap too sensitive / missed:** run `clap-calibrate` to see live peak
  values, then adjust `CLAP_THRESHOLD` in `.env` accordingly.
- **Mishears speech:** `STT_MODEL=base.en` (slower, more accurate) downloads
  automatically on first use.
- **No microphone / no audio device:** `listen`/`wake` fail with a clear
  message instead of crashing. TTS falls back to local Windows SAPI, then to
  printing, if the OpenAI TTS call fails.

---

## Run on startup

`start_jarvix.bat` activates the venv and runs `python -m app.runtime.startup`,
which force-starts `wake --mode wakeword` (a hidden background process has no
stdin for `enter` mode). To launch it on login:

1. Press `Win + R`, type `shell:startup`, press Enter.
2. Put a shortcut to `start_jarvix.bat` (or the `.vbs` launcher, which hides
   the console window) in that folder.

On login Jarvix starts and waits silently for the wake trigger — it doesn't
greet until then, so it never spams you. Startup errors are written to
`jarvix.log` next to the project since a hidden process has no visible console.

To disable: delete the shortcut from the Startup folder.

---

## Troubleshooting

- **`reauth` / 403 on send:** add `gmail.send` (and the other 3 scopes) in
  Google Cloud, then run `python -m app.main reauth`.
- **News says it isn't configured / returns an auth error:** `NEWS_API_KEY` is
  missing or invalid — get a free key at newsapi.org and put it in `.env`.
- **No memory / "what did I just ask" doesn't work:** `DATABASE_URL` is blank,
  the database is unreachable, or `pgvector` isn't enabled. Memory failures
  are silent by design (Jarvix still answers, just without recall) — check the
  console for `[memory] ... skipped: ...` lines to see the actual cause.
- **LLM slow / errors:** check `OPENAI_API_KEY` is set and has quota; use
  `today-fast` for an instant, LLM-free plan.
- **No voice / TTS silent:** check `OPENAI_API_KEY`; on failure it falls back
  to local Windows SAPI automatically, so total silence usually means both
  failed (e.g. no audio output device).
- **`open-app <alias>` fails:** that alias isn't installed where
  `app/tools/desktop.py` looks for it — add the real `.exe` path under `apps`
  in `aliases.json`.
- **Browser control fails to start / "Couldn't start Chrome":** run
  `python -m playwright install chrome`. Make sure Google Chrome is actually
  installed (Playwright drives your real Chrome, not a bundled browser).
- **Browser task keeps saying it needs a human:** that's by design for
  CAPTCHA/2FA/password-reentry pages — log in manually in the Jarvix Chrome
  window once (see [Setup](#setup)), then ask Jarvix to continue.
- **Browser session isn't staying logged in:** confirm
  `JARVIX_CHROME_PROFILE_DIR` points at the same folder every run (leave it
  blank for the default `jarvix\chrome-profile`) and that nothing is deleting
  that folder between runs.

---

## Project layout

```
app/
  main.py            CLI commands (Typer) + intent dispatch + memory logging
  server.py          FastAPI/SSE bridge for web/jarvix-ui.html (optional)
  config.py          env-driven settings (loads .env)
  models.py          EventCandidate schema (Pydantic)
  brain/
    intent_router.py   rule-based parser + OpenAI JSON-classifier fallback
    dialogue.py         multi-turn state machine (email, weather, add-event follow-ups)
    llm_client.py        OpenAI chat completions + embeddings (the only OpenAI caller)
    voice_assistant.py   answers QUESTION/CONVERSATION, pulls in RAG memory context
    command_parser.py    thin stable wrapper around intent_router.parse
  voice/
    tts.py, stt.py, recorder.py, wakeword.py, clap_detector.py, wake_loop.py
  tools/
    calendar.py, gmail.py, google_auth.py     Google Calendar + Gmail
    music.py                                   Spotify (desktop app + Web API) + media keys
    desktop.py                                  allowlisted app/folder launcher
    email_actions.py                            draft/send email, contacts
    extractor.py, dedupe.py, prefilter.py       scan-mail pipeline (extract, dedupe, prefilter)
    news.py                                     NewsAPI top headlines
    summarize.py                                shared one-line LLM summarizer (email + news)
    live_info.py                                weather, time, date parsing (no LLM)
  safety/
    permissions.py     confirmation gates for dangerous actions
  browser/
    manager.py           persistent Chrome profile lifecycle (Playwright), one dedicated
                          background thread owns Chrome; every call is marshalled onto it
    state.py              DOM -> structured page snapshot (numbered elements + text)
    actions.py             primitive browser actions (goto, click, type, scroll, ...)
    safety.py               high-impact action / CAPTCHA-2FA keyword classifiers
    history.py                "open <name>" resolution against your REAL Chrome history
    tools.py                 inner browser agent loop + its own tool schema
  memory/
    db.py               Postgres/pgvector RAG conversation memory
    cache.py             SQLite cache for email->calendar-candidate extraction
    aliases.json, contacts.json, preferences.json
  runtime/
    startup.py          login entry point (forces wakeword mode)
web/
  jarvix-ui.html       single-file browser frontend for app/server.py
tests/
  test_voice_regressions.py   intent routing, dialogue, wake-word regression tests
  test_orchestrator.py        LLM-first orchestrator + confirmation-flow wiring
  test_browser.py             browser agent loop, history matcher, applicant profile,
                               ask_user flow, safety classifier, confirm/resume wiring
```
