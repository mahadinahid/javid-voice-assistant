# Javid — request pipeline

How one request travels from Mahadi's voice (or the text box) to Javid's spoken answer: which file handles each
stage, what the latency trace records there, and how long each stage takes today. Design rules and model
quirks are in `CLAUDE.md`; this file is about the path and its timings.

## The pipeline

```
 BROWSER  index.html                           SERVER  server.py → javid.py                     OUTSIDE
 ───────────────────                           ────────────────────────────                     ───────
 mic ─┬─ push-to-talk  (MediaRecorder)
      └─ hands-free    (AudioWorklet + energy VAD, ends after 0.7 s of quiet)
          │                                ● speech_end, capture_end
          ▼
 submitAudio ── POST /talk (audio) ──────────► talk()                     ● request_received
 submitText ─── POST /type (text) ───────────► type_message()
                                                │
                                                ├─ transcribe_turn() ─────────────────────────► Groq Whisper
                                                │                          ● stt_done            (whisper-large-v3-turbo)
                                                ├─ wake word / noise / one-word gate  (voice only)
                                                ▼
                                              think_stream()
                                                ├─ memory yes/no, language switch   (decided in code)
                                                ├─ select_tools(), _request_messages()
                                                │                          ● routing_done
                                                │  ┌───────────────────────────────────────────┐
                                                │  │ _stream_completion() ─────────────────────┼─► LLM chain: gpt-oss-20b →
                                                │  │   ● llm[]: start, first chunk, end        │   gpt-oss-120b → Gemini → Qwen
                                                │  │ tool calls via TOOL_MAP ──────────────────┼─► Tavily, Google Calendar,
                                                │  │   ● tools[]: start, end                   │   exchange-rate APIs
                                                │  └── repeat: MAX_TOOL_ROUNDS + 4 rounds max ─┘
                                                │                          ● llm_done
                                                ├─ checks: required tools, reply language, holiday dates,
                                                │  web figures + source line, skill spoken/screen split
                                                ▼                          ● reply_out
 streamReply() ◄─── NDJSON: trace, transcript, tool, say, delta, done ─── _traced() / _reply_stream()
   │                                       ● server_done
   ├─ text shown                           ● first_text
   ├─ feedSpeech(): cut into sentences
   ├─ POST /speak, one per sentence ─────────► speak_text() → synthesize() ───────────────────► edge-tts
   │                                       ● tts_first_request                                  (en-US-AndrewNeural /
   └─ playNext(): play the clips in order  ● first_audio, first_answer_audio, tts_done           bn-BD-PradeepNeural)
                                                │
 POST /latency/<id>  (the page's stages) ──────► tracing.add_client() ──► data/traces.jsonl
```

The terminal version (`python javid.py`) runs the same core without the browser: `record()` (Enter to start
and stop) → `transcribe_turn()` → `think()` → `speak()` (edge-tts, played with `afplay`). Its turns are
traced too (`source: "cli"`).

## Stages

| # | Stage | What happens | Where | Trace stage |
|---|---|---|---|---|
| 1 | Capture | Push-to-talk records while the button or Space is held. Hands-free keeps the mic open: an energy VAD with an adaptive noise floor finds speech, keeps 0.45 s of pre-roll, and ends the clip after `VAD.endSilenceSec` (0.7 s) of quiet. The clip is encoded as 16 kHz WAV. Listening pauses while Javid speaks; barge-in can cut him off. | `index.html`: `startRecording`/`stopRecording`, `onFrame`, `encodeWav`, `bargeStep` | `speech_end` (release, or the last voiced frame), `capture_end` |
| 2 | Upload | Multipart audio to `/talk` (with `mode`: push, wake, followup), or JSON text to `/type`. Both carry the open chat's id. | `index.html`: `submitAudio`, `submitText`, `streamReply`; `server.py`: `talk`, `type_message` | `request_sent`, `request_received` |
| 3 | Speech-to-text | Whisper in the reply language. Push-to-talk adds a "Javid" spelling hint. In Bangla mode, a short clip gets a second English pass, used only for "speak English" or a bare wake word. | `javid.py`: `transcribe_turn`, `transcribe` | `stt_done` |
| 4 | Gate | Hands-free needs the wake word unless it's inside the follow-up window. Noise is ignored and not traced. A bare "Javid" gets "Yes?". A one-word push-to-talk clip is "I didn't catch that" unless it's yes/no or a greeting. | `server.py`: `talk`; `javid.py`: `strip_wake_word`, `is_noise` | — |
| 5 | Routing | A yes/no to a pending memory proposal, language switches, a self-check or "where did that come from?", questions about Javid itself, watches (set, list, cancel), exchange-rate questions, market briefs and news requests are settled in code, with no LLM request (the data ones still fetch their data: Binance, Yahoo, the ECB, Google News). Otherwise: pick tool groups by regex (`select_tools`), decide required tools and whether the turn is heavy, and build the request (static system prompt + memory block, last 6 turns, `[Context]` note with date, older-chat summary and recent tool facts). | `javid.py`: `think_stream`, `select_tools`, `_request_messages`, `context_block`, `_code_answer`; `memory_store.py`; `about.py`; `rates.py`: `request`/`answer`; `markets.py`: `brief_request`/`brief` | `routing_done` |
| 6 | LLM | Streaming completion on the first model in the chain the budget guard allows (`providers.py`: ready ones first, those past 80 % of an allowance only after them, none at 95 % or resting after a 429), sent what its provider may receive: a provider without `personal_ok` gets no memory, chat history or personal tools, and a request that needs them goes only to one with it (decision record 1). Provider clients for Groq, Gemini, DeepSeek and IFM. Each round's text is held until the round ends, and each request is a row in the usage ledger. | `javid.py`: `_stream_completion`, `_open_stream`, `_models_to_try`, `_public_request`, `_create`; `providers.py` | `llm[]` (start, first chunk, end, model) |
| 7 | Tools | Tasks and time (`javid.py`); calendar and public holidays (`calendar_tools.py`, Google Calendar API); headlines (`gnews.py`, Google News RSS, the search engines only as the fallback); web search and page reads (`web_tools.py` through the search router `search.py`: Tavily, Exa, Brave, Firecrawl, Cohere, by intent; decision record 2); skills (`skills.py`); agents (`agents.py`); exchange rates (`rates.py`); live prices (`markets.py`); weather, timers and sums (`weather.py`, `timers.py`, `calc.py`); memory proposals and past-chat search (`memory_store.py`). Results go back to the LLM for another round. A turn has at most `MAX_TOOL_ROUNDS` + 4 rounds (retries and rewrites use them up), and the last one is sent without tools so the model must answer. | `TOOL_MAP` in `javid.py` | `tools[]` (name, start, end) |
| 8 | Checks | One retry or rewrite each for: an empty, garbled or malformed reply; a missing required tool; the wrong language; holiday dates not in the tool result; web figures without an official or second source. Then the web answer is trimmed and gets its source line, and a skill reply is split into spoken and screen parts. | `javid.py`: `think_stream`; `calendar_tools.check_holiday_reply`; `web_tools.check_reply`/`finalize`; `skills.split_reply` | `llm_done` → `reply_out` |
| 9 | Stream out | NDJSON events. The reply text arrives as one `delta` once the final round is done (rounds are buffered, see 6). `say` ("Searching…") arrives as a web search starts. | `server.py`: `_traced`, `_reply_stream` | `reply_out`, `server_done` |
| 10 | Text-to-speech | The page cuts the reply into sentences (`.!?` or `।` followed by a space) and requests each from `/speak` straight away. The server synthesizes with edge-tts, the one locked voice per language, with a cache, at most 3 sentences at once, and up to 3 tries (6, 8 then 10 s for the first audio, 4 s between chunks after that). A failure shows "voice unavailable", never another voice. | `index.html`: `feedSpeech`, `speakChunk`, `fetchAudio`; `server.py`: `speak_text`; `javid.py`: `synthesize` | `tts_first_request` |
| 11 | Playback | Clips play in order. Stop, Esc, push-to-talk or barge-in cut them off. | `index.html`: `playNext`, `stopSpeech`, `bargeIn` | `first_audio`, `first_answer_audio`, `tts_done` |
| 12 | Records | The chat log, token usage, the usage ledger (every provider call in its own unit: `data/providers.db`), and the latency trace. | `javid.py`: `log_message`, `_record_usage`; `providers.py`: `record`; `tracing.py` | — |

## Current timings

Medians from 11 requests made on **2026-10-15, 21:45–21:52**, all answered by gpt-oss-20b. Each LLM request
was its first choice, with no fallbacks. The three web requests used live Tavily. *Time to first audio* starts
at the end of speech, or at Send for typed requests. With 3–5 requests per type, the 90th percentile is close
to the slowest request. Treat these as a first snapshot: the Latency panel computes the same figures from
real use over the latest 200 requests.

| Request type | n | First audio, median | p90 | First *answer* audio, median | p90 |
|---|---|---|---|---|---|
| command (a non-web tool ran) | 3 | **3.5 s** | 7.4 s | 3.5 s | 7.4 s |
| chat (no tools; 2 of 5 by voice) | 5 | **4.5 s** | 5.3 s | 4.5 s | 5.3 s |
| web (web search ran) | 3 | **0.75 s** ("Searching…") | 1.5 s | 9.1 s | 9.3 s |

Median per stage:

| Stage | Median | Notes |
|---|---|---|
| End-of-speech wait | 0 s push-to-talk · **0.7 s hands-free** | Hands-free waits `VAD.endSilenceSec` by design (not in the probe, which used push-to-talk timing) |
| Upload | 15 ms | Localhost |
| Speech-to-text | 0.39 s | Whisper on a 3.1 s clip; 0.30–0.42 s in separate runs |
| Routing | 5 ms | Regex tool selection and request building |
| One LLM request: first chunk | 0.50 s (p90 0.80 s) | 20 requests |
| One LLM request: total | 0.65 s (p90 0.97 s) | A turn with tools makes 2–3 (median 2) |
| LLM + tools, whole turn | command 1.3 s · chat 1.0 s · web 6.4 s | Web: `web_search` 3.9–4.9 s, `read_page` 0.3–0.6 s; `list_holidays` 0.9 s; local tools ~0 ms |
| Checks | 1–3 ms | |
| Reply to first audio | **2.3 s** (range 0.9–5.3 s) | edge-tts synthesis of the first sentence, the largest stage for commands and chat |
| Speaking | 2.6–4.1 s | Length of the spoken reply |

What these numbers say about where the time goes (nothing was changed in this session):

- **Text-to-speech starts only after the whole answer exists.** Each LLM round is buffered (stage 6), so the
  first sentence goes to `/speak` only at `reply_out`. For commands and chat, edge-tts on that sentence
  (median 2.3 s, highly variable) takes longer than the LLM work before it.
- **Web answers are dominated by Tavily.** One search takes about 4 s, then a page read and another LLM round
  follow. "Searching…" fills the gap at about 0.75 s, but the answer itself takes about 9 s.
- **Hands-free adds 0.7 s to every turn** before anything is sent: the silence that decides you've finished.

### How these were measured

The pipeline was driven in-process from a script: the real server code, model chain, Whisper, Tavily and
edge-tts, with scratch chat, task and memory files and calendar writes switched off. The page's stages were
posted to `/latency/<id>` as `index.html` does. Without a browser:

- *first audio* is when the clip is ready; the page adds tens of milliseconds to start playing it;
- *speaking* is the clips' total length;
- the test client delivered stream events only at the end of the stream, so a web request's "Searching…"
  is placed at the start of its first web search plus 16 ms. That 16 ms is the slowest of five cached
  `/speak` calls measured against the running server; the rest took 1–2 ms.

Real use replaces all of this: the page measures playback itself.

### 2026-10-25: the search router, before and after

`bench_web.py` ran six web questions (two facts, one news, one "companies like", one how-to, one "when did it open")
through the real pipeline on the code before the router and on the router, with Groq's gpt-oss-20b and Tavily only
(the only search key so far), scratch data and edge-tts for the first sentence, alternating which ran first:

| | first audio, median | first answer audio, median (range) | Tavily credits an answer | tokens an answer, median |
|---|---|---|---|---|
| before (`b7c156b`) | 0.78 s | 38.4 s (15.8-50.5) | 1.67 | 6,508 |
| router, first run | 0.99 s | 20.8 s (8.5-58.1) | 1.83 | 5,809 |
| router, after the fixes below | 0.76 s | 37.8 s (11.3-52.2) | 2.33 | 7,366 |

With Tavily as the only engine the router can't run anything in parallel, and the time went where the model took it:
2 to 5 model requests and 1 to 4 searches a question, run to run, which is why two runs of nearly the same code differ
by 17 s in median. Six questions can't separate that from the router. The first run found three things, fixed before
the second: a confirmed height answered "I couldn't confirm" (the new fact checks compared Everest with K2), a 365-day
window turning a how-to question into a news search, and "web search isn't set up" when Tavily simply found nothing.
The second run found the model limiting fact searches to news sites it picked itself; dropping those sites for facts
came after it and isn't measured yet: the runs had used Javid's own 40 Tavily credits for the day. Rerun
`python bench_web.py` (and the old code with `--code`) on a fresh day. In real use (traces 2026-10-15 to 10-24) web
answers had first audio at 1.5 s and the answer's first audio at 11.8 s (medians), with Tavily's search alone taking
3.8 s.

### Since 2026-10-24: news in code

The web row above was mostly news, and news no longer goes through the model: `news.py` answers a news request
from Google News (`gnews.py`) in **0.7–1.5 s** for one to six topics (measured live, 2026-10-24), against the
~9 s median (p90 16 s) the same requests took as web answers (real traces 2026-10-15 to 10-22, LLM done). The
first sentence still waits for edge-tts. Watches and the self-check add no request either; a watch's checks run
in a background thread (`watches.start()`), outside any turn.

## Latency traces

`tracing.py` records one trace per request (observation only; nothing in it changes what Javid does).

- **Server stages** are marked on the module's current trace. `/talk` and `/type` open it,
  `server._traced` wraps the reply stream (first event `trace` carries the id), and `javid.py` marks
  `routing_done`, each LLM request and each tool call. Without an open trace every call is a no-op, which
  covers `regression.py`, scripts and the greeting.
- **Page stages** (end of speech, first text, first `/speak` request, first audio, first answer audio, audio
  done) go to `POST /latency/<id>` when speech ends, is stopped or is interrupted. A trace the page never
  reports is written after 2 minutes as `no_client`.
- **`data/traces.jsonl`**: one line per request, with `type` (command / chat / web), `source` (voice / text / cli),
  `mode`, `outcome` (done / interrupted / stopped / superseded / no_client), `label` (the start of the
  message), `t0` (epoch ms), `stages` (ms after t0), `llm[]`, `tools[]`, and `ttfa`. Hands-free clips the
  server ignores aren't traced. The folder is gitignored, because labels quote your messages.
- **Latency panel** (left sidebar, under Providers). The card shows median / p90 time to first audio per
  type. **Details** opens the stage bars of the last 20 requests (one shared scale, starting at end of speech;
  the white tick marks the first audio), plus the per-type medians for first audio and first answer audio.
- To refresh the timings in this file, use the Latency panel's medians, or run `tracing.stats()` over
  `data/traces.jsonl` after a few days of normal use.

## Decision records

### 1. A provider registry, a usage ledger, a budget guard and a privacy rule (2026-10-25)

**Context.** Javid called four language-model providers and Tavily with keys from `.env`, and each limit was handled
on its own: a quota guard for Groq's daily tokens, a 40-credit daily cap for Tavily, cooldowns kept in memory and lost
on restart. Every request carried the memory block, the last six turns and any calendar or task results to whichever
model was next in the chain, Gemini's free tier included. More providers are planned (Exa, Brave, Firecrawl, Cohere,
OpenRouter, Cloudflare Workers AI, ElevenLabs), each with a free allowance in its own unit: credits, dollars, requests,
neurons, characters. Mahadi wants Javid to stay inside every free allowance, and to decide himself, after reading each
data policy, which providers may see personal data.

**Decision.**
1. **One registry**, `config/providers.yaml` (tracked, read again when it changes): per provider its kind, the `.env`
   variable(s) of its key, `enabled`, `priority`, the free allowance in its own unit (never converted to tokens; seeded
   from the pricing as given and marked `verified: false`), the reset rule, `timeout_ms` and `personal_ok` (default
   false). A provider is on only when enabled and every key it names is set; the start-up line lists names only.
2. **One usage ledger**, SQLite `data/providers.db`: a row per call (provider, model, op, units in the provider's unit,
   latency, status, an error's class name; never a query or a key). Usage is the sum since the start of the reset
   window (the 1st at 00:00 UTC, midnight Pacific for Gemini, 00:00 UTC for Cloudflare's neurons), so counters start
   again with their window and nothing has to reset them. Groq's quota is read from its rate-limit headers and 429s,
   ElevenLabs' balance from its API.
3. **A budget guard** (`providers.state`): from 80 % of an allowance a provider is in reserve, used only when the others
   fail; from 95 % it is off until its reset; a call whose known cost would pass the allowance isn't made; no free tier
   (DeepSeek) means off. This replaces the quota guard's exception for web and reasoning turns. Javid's own Tavily cap
   of 40 credits a day stays on top.
4. **Cooldowns** on a 429, a 402 or a quota error until the provider's retry-after (its header or its text, else the
   allowance's reset), stored with the ledger so a restart keeps them; per model where allowances are per model. A
   resting or exhausted model isn't tried as a last resort any more: when none is left, Javid says so.
5. **The privacy rule** in the model router: requests are built with tags, so a provider without `personal_ok` gets
   the request without the memory block, the chat history and the calendar facts, and isn't offered the personal tools;
   a turn that needs calendar, tasks, memory or past chats (by its words, its required tools, its agent, a personal
   tool's result, or a model reaching for one) goes only to providers with `personal_ok`, and with none nothing is
   sent. Web search queries carrying a task's, a memory line's or a calendar result's words don't go to a search
   provider without it.
6. **Tests that never call a real API by default**: the scripted cases and recorded or documented fixtures
   (`fixtures/providers/`), with every outbound socket refused; the live model cases need `--llm`, and `--live` makes
   one real call per provider that is on.

**Consequences.**
- As shipped (every `personal_ok` false), calendar, task, memory and past-chat requests are refused and other turns go
  out without memory or history, until Mahadi clears a provider. Code-answered requests (news, rates, market briefs,
  watches, the self-check, questions about Javid) are unaffected.
- A model that is past 80 % now waits behind the others on every turn, including web and reasoning turns.
- Speech recognition (Groq Whisper), speech (edge-tts) and Telegram aren't routed, so the rule doesn't cover them; the
  search check is best effort and doesn't look at chat history. `SECURITY.md` lists these as open.
- Groq's tokens are counted twice: `llm_usage.jsonl` still drives its token bucket and the per-turn averages, the
  ledger has the per-call rows. `tavily_usage.jsonl` is no longer written (the ledger imported it once).
- The seeded allowances may be wrong until checked against the accounts; the 5 % margin and each provider's own
  429/402 handling are the backstop. Cloudflare's neurons aren't in its responses, so they can't be counted yet.

**Alternatives considered.**
- Limits in code, one per provider (as before): rejected; nothing showed them together, and every new provider would
  add its own.
- Converting allowances to tokens: rejected; the units measure different things, and a converted limit hides when
  the real one runs out.
- Refusing every model request while no provider has `personal_ok`: rejected; Javid would answer only what code
  answers. Stripping the personal parts keeps general questions working.
- Deciding "personal" from the tool-group patterns: rejected; they err wide on purpose and matched 39 of Mahadi's 105
  logged messages, mostly news and weather "today" or "tomorrow". The narrower `_PERSONAL_ASK` matched 6, and a
  model reaching for a personal tool still makes the turn personal.
- Recording the live model cases' responses to replay them by default: not done; their checks depend on the date, and
  recording them costs a full live run (~210k tokens).

### 2. A search router across engines, by intent (2026-10-25)

**Context.** Web search was Tavily alone: one engine for news, facts and everything else, one index to confirm a figure
against, and nothing to fall back on when it was slow (median 3.8 s, p90 6.8 s) or down. The registry (decision 1) now
lists Exa (semantic search), Brave (an independent index with news), Firecrawl (reading pages that need JavaScript) and
Cohere (reranking), each with its own free allowance.

**Decision.** `search.py` routes each `web_search`:
1. **By intent, in code** (`classify`): news to Brave and Tavily at once; "companies or papers like X" to Exa; facts and
   figures to Tavily and Brave at once, whose answers need two independent publishers that agree (or an official
   source); everything else to Tavily, then Exa, then Brave. Mahadi's message and the model's query are read together.
2. **Engines in parallel with soft timeouts** (2.5 s, per engine in the registry): when one passes, the next engine
   starts too and the first usable answer wins. A slow engine isn't cut off unless another answered: with Tavily as the
   only engine with a key, a hard 2.5 s cut would fail 78 % of searches. Errors, nothing found, the budget guard, the
   daily cap, a cooldown or the privacy rule all count as a failed engine.
3. **Merge by URL, rerank, keep 3-5**: Cohere when it's on, keyword overlap otherwise.
4. **Reading**: at most two pages a turn, Tavily's extract or Firecrawl for a JavaScript page, and only the passages
   closest to the question (about 750 tokens a page).
5. **The source rules stay**, with "two sites" made two publishers, measures and counts checked after a fact search, and
   each source credited to the engines that found it, on screen too.
6. **Cache by normalized query**: 15 minutes for news and anything asked "today", 24 hours for facts.
7. **The model's habits are handled in code**: a 365-day window on a how-to isn't news; a fact isn't limited to sites the
   model picked (it needs two publishers); a restricted search that finds too little is run on all sites once.

**Consequences.**
- Until Exa, Brave, Cohere or Firecrawl has a key, every search runs on Tavily with the keyword rerank: the routing and
  the parallel runner are tested against documented response formats, not the engines themselves (`--live` runs one
  call per engine, and one routed search, once a key is there).
- Facts and figures are stricter: a figure only one publisher gives is "couldn't confirm"; the model is told that
  agreement allows for rounding. Fewer confident wrong numbers, some more hedged answers.
- A widened or parallel search costs more credits (1.67 to about 2 Tavily credits an answer in the benchmark), and an
  engine asked and then given up on is still billed (the ledger marks it "too late").
- When web search is unavailable, the model may still answer from memory: in the benchmark, with the daily cap reached,
  it said the Padma Bridge opened in December 2018 (it opened in June 2022). Not addressed here.

**Alternatives considered.**
- A hard 2.5 s timeout per engine: rejected while Tavily is alone (most of its searches take longer); 2.5 s decides when
  the next engine joins instead.
- Always reading the top pages in `web_search`: rejected; the model reads only when the snippets don't answer it, and an
  automatic read costs a round of tokens and time every time.
- Cohere for page passages too: not done; Tavily's extract already returns the chunks closest to the query, and keyword
  overlap picks among them without another call against Cohere's 1,000 a month.

