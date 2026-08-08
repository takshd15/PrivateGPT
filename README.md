# Jarvix — Local Voice Laptop Operator

Jarvix is a **voice-first** assistant for your Windows laptop. Say the wake
word ("hey jarvis", or double-clap / press Enter depending on `WAKE_MODE`) and
it greets you, speaks a calendar briefing, opens your editor and project
folder, starts music, and takes spoken commands from there — routing each one
to a specific tool. It also remembers past conversations (Postgres/pgvector
RAG memory) and asks for explicit confirmation before any calendar write or
email send.

The wake-word listener runs **locally** (`faster-whisper`, always-on, no cloud
calls for ambient room noise). Everything after a confirmed wake is
cloud-backed: voice commands are transcribed by AssemblyAI, the brain and
text-to-speech run on OpenAI's API, plus Google (Gmail/Calendar), Spotify, and
NewsAPI for the tools that need them.

> **Reality check:** Jarvix can't run while the laptop is off or asleep. It
> runs once you're logged in and the background process has been started (see
> [Run on startup](#run-on-startup)). It needs microphone permission.

---

## What it can do

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

Multi-turn follow-ups fill in missing details when you leave something out —
e.g. asking to email someone without saying what to say, or asking for the
weather without naming a city.

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

## Requirements

- Windows 10/11, a working microphone and speakers
- Python 3.12 + the bundled virtual env (`.venv`)
- Required API keys: OpenAI (brain, TTS, embeddings), AssemblyAI (voice-command transcription)
- A Google Cloud OAuth client (Desktop app) with Gmail + Calendar scopes
- Optional: Spotify Web API credentials (search/play-by-name — media-key
  control works without them), NewsAPI key (news headlines), a Supabase/Postgres
  database with the `pgvector` extension (conversation memory)

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

---

## Project layout

```
app/
  main.py            CLI commands (Typer) + intent dispatch + memory logging
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
  memory/
    db.py               Postgres/pgvector RAG conversation memory
    cache.py             SQLite cache for email->calendar-candidate extraction
    aliases.json, contacts.json, preferences.json
  runtime/
    startup.py          login entry point (forces wakeword mode)
tests/
  test_voice_regressions.py   intent routing, dialogue, wake-word regression tests
```
