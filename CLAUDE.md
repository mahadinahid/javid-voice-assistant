# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules that override your instincts

- **The voice is locked.** `TTS_ENGINE="edge"`, `EDGE_VOICE="en-US-AndrewNeural"`, `LOCAL_FALLBACK=False` is the voice the user approved as "perfect". Never change the voice, engine or fallback behaviour — not even to fix latency or flakiness — unless the user explicitly tells you to. (Details under *Speech* below.)
- **The app runs on real data**: `tasks.txt`, `memory.json`, `chat_log.jsonl`, a connected Google Calendar, and the user's API keys in `.env`. See *Testing safely*.
- **Quota is scarce.** Groq's free tier caps each model at 200k tokens/day (a turn is ~2–3 requests of ~2.3k tokens); Gemini's free tier is 20 requests/day. Don't loop LLM tests.

## Running and checking

```bash
source .venv/bin/activate
python server.py                 # web UI at http://localhost:8000 — the main way the app is used
python javid.py                  # original terminal push-to-talk loop (Enter to record/stop)
lsof -ti :8000 | xargs kill      # stop a server left running in the background
PYTHONUNBUFFERED=1 python server.py   # when debugging: [tool]/[retry]/[llm]/[tts]/[hands-free] lines are plain prints
```

There is no linter or build step. `index.html` is one hand-written file (inline CSS + JS, no framework). Syntax-check its script with:

```bash
python3 -c "s=open('index.html').read(); open('/tmp/ui.js','w').write(s[s.index('<script>')+8:s.index('</script>')])" && node --check /tmp/ui.js
```

**`python regression.py` is the behaviour check** — 20 prompts through the real LLM and tool loop, against scratch task/memory/history files and a fake in-memory calendar (your tasks, memory and calendar are never touched; token usage is still logged to `llm_usage.jsonl` because the tokens are really spent). Run it before/after prompt, tool or agent-loop changes.

```bash
python regression.py -k "delete needs"                      # one case (substring of its name)
python regression.py -v                                     # full run, show every reply
python regression.py -n 3                                   # repeat each case (model is non-deterministic)
python regression.py --models gemini:gemini-3-flash-preview # that chain only, no Groq fallback masking failures
```

A full run is ~40–50 LLM requests (~100k tokens). Exit code 0 = all passed, 1 = a check failed, 3 = inconclusive (429/network — never reported as a pass or fail). Per-minute 429s are waited out; a longer wait stops the suite.

Setup needs `GROQ_API_KEY` in `.env` (also used for speech recognition). `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` are optional. `README.md` describes only the original CLI version (10 tools, macOS `say`); the web UI, calendar, hands-free and model chain are documented here.

## Architecture

**`javid.py` is the shared core** for the CLI and the web server: tool implementations, `TOOLS` schemas / `TOOL_MAP` dispatch, conversation state (module-level `messages`), STT (`transcribe`), TTS (`synthesize` for web, `speak` via macOS `say` for CLI), the LLM provider chain, and the agent loop. `server.py` (FastAPI) and `index.html` wrap it; `calendar_tools.py` adds the Google Calendar tools.

### Agent loop — `think_stream(user_text)`
A generator yielding `("tool", name)` and `("delta", text)`; `think()` joins the deltas (CLI). Behaviours that aren't obvious:
- Up to `MAX_TOOL_ROUNDS` tool rounds per turn (`add_task` is followed by `list_tasks`).
- Each round's text is **buffered** and emitted only if that round made no tool calls — text alongside a tool call is the model guessing before it saw results.
- One retry each for an empty reply, for an `APIError` (malformed tool-call JSON), and for a **garbled** reply (`_is_garbled`: mostly dots/ellipses or one repeated word — seen from gpt-oss-120b); a second garbled reply becomes a short apology. Garbled assistant messages are also skipped by `_restore_context()`.
- `finally` appends the reply to `messages` and `chat_log.jsonl`, so aborted streams still record what was said.
- The **system prompt is rebuilt every turn** (`build_system_prompt()` overwrites `messages[0]`): current time, a 14-day weekday table, and `memory.json` facts — so sidebar edits to memory apply immediately.

### Adding a tool
Implement it, add a schema to `TOOLS`, add a lambda to `TOOL_MAP`, mention it in the system prompt. Calendar tools live in `calendar_tools.py`, which exports its own `TOOLS`/`TOOL_MAP` that `javid.py` splices in; they're wrapped in `_guard`, which turns auth/API errors into sentences the model can speak.

### Calendar edit tools (`move_event`, `delete_event`)
Events are found by title words in Python (`_find_events`/`_pick`), never by model-supplied IDs; zero or several matches are reported back, not guessed. The model often fills optional `day`/`time` with an invented "today", so a hint that matches nothing falls back to a title-only search. `delete_event` is two-step **and enforced in code**: the first call only describes the event; deletion is accepted only if `confirm=true` arrives on the *next* user message (`calendar_tools._turn`, bumped by `think_stream` via `new_turn()`). Don't weaken this to a prompt instruction — the model ignores those.

### LLM provider chain (`javid.py`)
- **Order:** `current_chain()` = the user's saved order (`llm_chain.json`, gitignored) else `default_chain()` = `[LLM_MODEL, *LLM_FALLBACKS]`. Entries are bare Groq ids, or `gemini:<model>` / `deepseek:<model>` (OpenAI-compatible endpoints via the `openai` SDK). A provider without its key in `.env` is skipped.
- **Routing:** `_open_stream` → `_create` → `_provider_client`, chosen by id prefix. A 429, 5xx/network error, 404, 401/403 or DeepSeek's 402 ("insufficient balance") puts that model on a cooldown (`_cooldown_for`; `_retry_after` parses `retry-after`, Groq's "try again in 7m11s" and Google's "retry in 6h21m37s") and the next model answers. A 400 is *not* "unavailable" — it propagates to the malformed-tool-call retry.
- **Two SDKs:** the `API_ERRORS`/`RATE_LIMITED`/`UNAVAILABLE`… tuples cover both `groq` and `openai` exception classes. Use them, not the bare class names.
- **Per-model request options** (`_model_kwargs`): only gpt-oss gets `reasoning_effort="low"` (Qwen given `"none"` skipped the tool call and claimed it was done); Gemini gets `reasoning_effort="low"`. `extra_content` on Gemini tool calls must be echoed back and is stripped for other providers (`_messages_for`); Gemini chunks may lack `index`/`id` (handled in `_stream_completion`/`think_stream`).
- **Provider limits measured on this key (2026-10-13):** Gemini `gemini-3.7-flash` (the configured `GEMINI_MODEL`, highest that answered) = 20 requests/day then "retry in 6h"; `gemini-3-flash-preview` = 5 requests/min, ~2 s; `gemini-3.8-flash` 503s; 3.1-pro/4-argon not free; `gemini-2.5-*` 404 (retired). Google retires IDs quickly — check `models.list()` before trusting a name; `llama-3.3-70b-versatile` isn't on the Groq key either. Gemini/DeepSeek are fake-tested plus a partial live run; Gemini is a thin safety net, not a replacement.
- **Models panel & key management:** `llm_status()` (served at `GET /llm`) merges the chain, cooldowns, `llm_usage.jsonl` (Groq's token counts per request, counted only since logging began), per-minute/per-day headroom from `x-ratelimit-*` headers, and the daily cap — which Groq only reveals inside a 429 message (parsed into `llm_quota.json`; `KNOWN_TPD_LIMITS` seeds it). A model under `LLM_MIN_REQUEST_TOKENS` remaining shows as `low` even though a tiny "Test all now" ping (`POST /llm/ping`) succeeds — a successful ping is not capacity. The dialog can reorder/add/remove models (`PUT /llm/chain`) and paste keys (`PUT/DELETE /llm/key/{provider}` → `set_provider_key`: verified with a token-free `models.list()`, written by `_write_env` which preserves other lines; the key is never returned, only a 4-char hint; `.env` is always rewritten mode 0600, and provider error text passes through `_redact` before it is stored, logged or shown, since a provider may echo part of a key). Every mutating `/llm/*` route goes through `_local_page_only` (Host must be localhost:8000 and Origin must match). The Groq key is read-only there.

### `server.py` routes
- `/type` and `/talk` stream **NDJSON events** — `transcript`, `tool`, `delta`, `error`, `ignored`, `done` (with fresh `tasks`/`memory`) — consumed by `streamReply()` in `index.html`.
- `/talk` takes `mode`: `push` (always answer), `wake` (hands-free; only if the transcript starts with the wake word, via `strip_wake_word`), `followup` (hands-free window right after a reply; no wake word needed).
- `/speak` returns neural TTS audio for one sentence; 503 on failure. Sidebar REST routes (`/tasks`, `/memory`, `/calendar`, `/history`, `/reset`) edit data directly without the LLM. Startup threads warm TTS and ping every model.

### Frontend (`index.html`)
- **Speech:** replies are split into sentences as they stream (`feedSpeech` — boundary is `.!?` + whitespace, so "27.3" doesn't split); each sentence is fetched from `/speak` immediately and played in order (`speechQueue`/`playNext`); `speechGen` invalidates stale callbacks after Stop.
- **Hands-free:** an AudioWorklet taps the mic; `onFrame` runs an energy VAD (adaptive noise floor, pre-roll, thresholds in `VAD`) and encodes 16 kHz WAV. Listening pauses while `javidIsBusy()` so Javid never hears himself.
- **Models dialog:** the cards re-render every second (countdowns) but are skipped when unchanged; the key/add controls re-render only when their data changes and never while an input is focused — keep it that way or typed keys get wiped.

### Persistent data (all gitignored, repo root)
`tasks.txt` (one per line; `✓ ` prefix = done; `[YYYY-MM-DD HH:MM] ` timestamp prefix), `memory.json` (`{"facts": [...]}`), `chat_log.jsonl` (`user`/`assistant`/`divider`; on startup `_restore_context()` reloads the last `CONTEXT_TURNS` messages after the latest divider), `credentials.json` + `token.json` (Google OAuth, Desktop client, scope `calendar.events`; kept mode 0600 by `_save`/`_lock_down`), `llm_usage.jsonl`, `llm_quota.json`, `llm_chain.json`, `.env`.

## Conventions driven by model quirks (gpt-oss-20b on Groq)

- **It miscounts and gets weekdays wrong.** Do counting and date math in Python and put the answer in the tool result: `list_tasks` leads with "N pending, M done"; calendar tools accept the user's words (`"friday"`, `"3pm"`) and resolve them in `_parse_day`/`_parse_time`; `list_holidays` computes long breaks itself.
- **It guesses when data is thin** (`web_search` returns ~150-char snippets). Holidays come from Google's public holiday calendars (`en.<code>#holiday@group.v.calendar.google.com`), not web search. Home country is Bangladesh (`HOME_COUNTRY`, Friday–Saturday weekend `HOME_WEEKEND`).
- Replies are spoken: the prompt demands 1–3 short sentences, no markdown or lists.

## Speech (voice is locked — see top)

- `TTS_ENGINE`: `"edge"` (default; free Microsoft neural voice `EDGE_VOICE`), `"groq"` (Orpheus; needs terms accepted in the Groq console), `"local"` (macOS `say`, `LOCAL_VOICE`, instant).
- edge-tts is an unofficial endpoint and erratic (normally ~1 s, but 3–5 s stalls and `NoAudioReceived` streaks; roughly half of calls failed during one test). `synthesize` abandons a try after `EDGE_ATTEMPT_SEC` and retries up to `EDGE_ATTEMPTS` times. The user prefers the natural voice over speed, so `LOCAL_FALLBACK` (speak with macOS `say`, plus a cooldown breaker) is **off**. Parallel/hedged edge-tts requests made it slower — don't add them back.
- The page drops to the browser voice after 3 consecutive `/speak` failures (`ttsFails` in `index.html`) — this is why the voice can suddenly change mid-session. Changing that is a voice-behaviour change: ask first.
- Short phrases like the "Yes?" ack are cached and pre-warmed at startup.
- **Wake word:** variants and Whisper noise phrases are `_WAKE`/`_NOISE`; `_WAKE` is deliberately phonetic (`_NAME`) because without a hint Whisper spells the name many ways. Hands-free clips are transcribed with `bias_name=False`: the `prompt="Javid"` hint makes Whisper answer silence/noise with "Javid", which the matcher accepts (spurious "Yes?"). Keep the hint for push-to-talk only.

## Testing safely

When exercising the agent from your own scripts (not `regression.py`, which does this already): point `javid.HISTORY_FILE`, `TASKS_FILE`, `MEMORY_FILE` at temp files, never call `add_event` against the real calendar — replace `calendar_tools._service` with a fake — and for anything touching `.env` set `javid.ENV_FILE` to a temp path and use fake keys. Restarting the server reloads recent chat history into context, so a bad past answer can resurface; `POST /reset` (the UI's "New chat") clears it. After editing `javid.py`/`server.py`, restart the server — it doesn't hot-reload.
