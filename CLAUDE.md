# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
source .venv/bin/activate
python server.py        # web UI at http://localhost:8000 (main way the app is used)
python javid.py         # original terminal push-to-talk loop (Enter to record/stop)
lsof -ti :8000 | xargs kill   # stop a server left running in the background
```

Run the server with `PYTHONUNBUFFERED=1` when debugging: `[tool]`, `[retry]`, `[tts]` and `[hands-free]` lines are plain `print`s and won't reach a redirected log otherwise.

There is no linter or build step. `python regression.py` is the behaviour check: 20 prompts through the real LLM and tool loop against scratch task/memory files and a fake calendar (`-n 3` repeats, `-k name` filters, `-v` shows replies). It spends ~100k Groq tokens per run — half the free tier's 200k/day for this model — so run it before/after prompt or tool changes, not in a loop; it exits 3 ("inconclusive") on 429s instead of reporting false failures. `index.html` is a single hand-written file (inline CSS + JS, no framework) served by FastAPI. Syntax-check its script with:

```bash
python3 -c "s=open('index.html').read(); open('/tmp/ui.js','w').write(s[s.index('<script>')+8:s.index('</script>')])" && node --check /tmp/ui.js
```

Requires `GROQ_API_KEY` in `.env`. The README describes only the original CLI version; the web UI, calendar and hands-free features are documented here.

## Architecture

**`javid.py` is the shared core** for both the CLI and the web server: tool implementations, the `TOOLS` schemas / `TOOL_MAP` dispatch table, the conversation state (module-level `messages` list), STT (`transcribe`), TTS (`synthesize` for web, `speak` via macOS `say` for CLI), and the agent loop.

**Agent loop — `think_stream(user_text)`** is a generator yielding `("tool", name)` and `("delta", text)` events. `think()` is just a wrapper that joins the deltas (used by the CLI). Key behaviors:
- Up to `MAX_TOOL_ROUNDS` tool rounds per turn (needed: e.g. `add_task` is followed by `list_tasks`).
- Each round's text is **buffered** and only emitted if that round made no tool calls — text produced alongside a tool call is the model guessing before seeing results and is discarded.
- One retry each for an empty response and for Groq `APIError` (malformed tool-call JSON).
- `finally` appends the reply to `messages` and `chat_log.jsonl`, so aborted streams still record what was said.

**System prompt is rebuilt every turn** (`build_system_prompt()` overwrites `messages[0]`): it injects the current time, a 14-day weekday table, and `memory.json` facts. Sidebar edits to memory therefore apply immediately.

**Adding a tool:** implement it, add a schema to `TOOLS`, add a lambda to `TOOL_MAP`, and mention it in the system prompt. Calendar tools live in `calendar_tools.py`, which exports its own `TOOLS` / `TOOL_MAP` that `javid.py` splices in (`*calendar_tools.TOOLS`, `**calendar_tools.TOOL_MAP`). Calendar functions are wrapped in `_guard`, which turns auth/API errors into sentences the model can speak.

**Calendar edit tools (`move_event`, `delete_event`)** find events by title words in Python (`_find_events` / `_pick`), never by model-supplied IDs; zero or several matches are reported back, not guessed. The model often fills optional `day`/`time` with an invented "today", so a hint that matches nothing falls back to a title-only search. `delete_event` is deliberately two-step and enforced in code: the first call only describes the event, and deletion is accepted only if `confirm=true` arrives on the *next* user message (`calendar_tools._turn`, bumped by `think_stream` via `new_turn()`). Don't weaken this to a prompt instruction — the model ignores those.

**Model fallback chain** (`LLM_MODEL` then `LLM_FALLBACKS` in `javid.py`, via `_open_stream`): a 429 (per-model free-tier quota), 5xx or 404 puts that model on a cooldown (from `retry-after` / "try again in 7m11s") and the next model answers; cooled-down models are skipped, not retried every request. Only gpt-oss models get `reasoning_effort` — Qwen given `"none"` skipped the tool call and claimed the task was added. Model IDs must exist on the key (`client.models.list()`; `llama-3.3-70b-versatile` does not). `llm_usage` counts requests per model.

**`server.py`** (FastAPI) wraps the core:
- `/type` and `/talk` stream **NDJSON events** — `transcript`, `tool`, `delta`, `error`, `ignored`, `done` (with fresh `tasks`/`memory`) — consumed by `streamReply()` in `index.html`.
- `/talk` takes a `mode` form field: `push` (always answer), `wake` (hands-free; answer only if the transcript starts with the wake word, via `strip_wake_word`), `followup` (hands-free window right after a reply; no wake word needed).
- `/speak` returns neural TTS audio for one sentence; a 503 makes the browser fall back to `speechSynthesis` for the rest of the session.
- Sidebar REST routes (`/tasks`, `/memory`, `/calendar`, `/history`, `/reset`) edit data directly without the LLM.

**Frontend (`index.html`)** pieces that span multiple functions:
- Speech: replies are split into sentences as they stream (`feedSpeech` — boundary is `.!?` followed by whitespace, so "27.3" doesn't split), each sentence is fetched from `/speak` immediately and played in order (`speechQueue`/`playNext`). `speechGen` invalidates stale callbacks after Stop.
- Hands-free: an AudioWorklet taps the mic; `onFrame` runs an energy VAD (adaptive noise floor, pre-roll, thresholds in `VAD`) and encodes 16 kHz WAV. Listening pauses while `javidIsBusy()` so Javid never hears himself.

**Persistent data** (all gitignored, in the repo root): `tasks.txt` (one task per line, `✓ ` prefix = done, `[YYYY-MM-DD HH:MM] ` timestamp prefix), `memory.json` (`{"facts": [...]}`), `chat_log.jsonl` (`user`/`assistant`/`divider` entries; on startup `_restore_context()` reloads the last `CONTEXT_TURNS` messages after the latest divider), `credentials.json` + `token.json` (Google OAuth, Desktop-app client, scope `calendar.events`).

## Model quirks (openai/gpt-oss-20b on Groq) and the conventions they drove

- **It miscounts and gets weekdays wrong.** Do counting and date math in Python and put the answer in the tool result: `list_tasks` leads with "N pending, M done"; calendar tools accept the user's words (`"friday"`, `"next tuesday"`, `"3pm"`) and resolve them in `_parse_day` / `_parse_time`; `list_holidays` computes long breaks itself. Don't ask the model to compute dates.
- **It guesses when data is thin** (web_search returns ~150-char snippets). Holidays come from Google's public holiday calendars (`en.<code>#holiday@group.v.calendar.google.com`), not web search. Home country is Bangladesh (`HOME_COUNTRY`, Friday–Saturday weekend `HOME_WEEKEND`).
- Replies are spoken aloud: the prompt demands 1–3 short sentences, no markdown or lists.

## Testing changes safely

The app runs against the user's real data and a connected Google Calendar. When exercising the agent from scripts: point `javid.HISTORY_FILE` at a temp file, back up `tasks.txt` / `memory.json` before tool tests and restore after, and never call `add_event` against the real calendar — stub `calendar_tools._service` with a fake instead. Restarting the server reloads recent chat history into context, so a bad past answer can resurface; `POST /reset` (the UI's "New chat") clears it.

**The voice is locked: `TTS_ENGINE="edge"`, `EDGE_VOICE="en-US-AndrewNeural"`, `LOCAL_FALLBACK=False` is the voice the user approved as "perfect". Never change the voice, engine or fallback behaviour — not even to fix latency or flakiness — unless the user explicitly tells you to.** Voice settings: `TTS_ENGINE` (`"edge"` default, free Microsoft neural voice `EDGE_VOICE`; `"groq"` uses Orpheus, which needs terms accepted in the Groq console). `"local"` uses macOS `say` (`LOCAL_VOICE`, instant). edge-tts is an unofficial endpoint and erratic here (normally ~1 s, but stalls of 3-5 s and `NoAudioReceived` in streaks), so `synthesize` abandons a try after `EDGE_ATTEMPT_SEC` and retries up to `EDGE_ATTEMPTS` times; the user prefers the natural voice over speed, so `LOCAL_FALLBACK` (speak a slow sentence with macOS `say`, plus a cooldown breaker) is **off** by default. The page only drops to the browser voice after 3 consecutive `/speak` failures. Short phrases like the "Yes?" ack are cached and pre-warmed at startup. Parallel/hedged edge-tts requests made it slower, not faster — don't add them back. Wake-word variants and Whisper noise phrases are `_WAKE` / `_NOISE` in `javid.py`. Hands-free clips are transcribed with `bias_name=False`: the `prompt="Javid"` hint makes Whisper answer silence/noise with "Javid", which the wake-word matcher then accepts (spurious "Yes?"). Keep the hint for push-to-talk only. `_WAKE` is deliberately phonetic (`_NAME`) because without the hint Whisper spells the name many ways (Jaaved, Jawid...).
