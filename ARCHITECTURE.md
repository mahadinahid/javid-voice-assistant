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
| 5 | Routing | A yes/no to a pending memory proposal, language switches, questions about Javid itself, exchange-rate questions and market briefs are settled in code, with no LLM request (the last two still fetch their data). Otherwise: pick tool groups by regex (`select_tools`), decide required tools and whether the turn is heavy, and build the request (static system prompt + memory block, last 6 turns, `[Context]` note with date, older-chat summary and recent tool facts). | `javid.py`: `think_stream`, `select_tools`, `_request_messages`, `context_block`, `_code_answer`; `memory_store.py`; `about.py`; `rates.py`: `request`/`answer`; `markets.py`: `brief_request`/`brief` | `routing_done` |
| 6 | LLM | Streaming completion on the first model in the chain that can serve (cooldowns, the quota guard holding back a model under 20 % of its daily tokens, provider clients for Groq, Gemini and DeepSeek). Each round's text is held until the round ends. | `javid.py`: `_stream_completion`, `_open_stream`, `_models_to_try`, `_create` | `llm[]` (start, first chunk, end, model) |
| 7 | Tools | Tasks and time (`javid.py`); calendar and public holidays (`calendar_tools.py`, Google Calendar API); web search and page reads (`web_tools.py`, Tavily); skills (`skills.py`); agents (`agents.py`); exchange rates (`rates.py`); live prices (`markets.py`); weather, timers and sums (`weather.py`, `timers.py`, `calc.py`); memory proposals and past-chat search (`memory_store.py`). Results go back to the LLM for another round. A turn has at most `MAX_TOOL_ROUNDS` + 4 rounds (retries and rewrites use them up), and the last one is sent without tools so the model must answer. | `TOOL_MAP` in `javid.py` | `tools[]` (name, start, end) |
| 8 | Checks | One retry or rewrite each for: an empty, garbled or malformed reply; a missing required tool; the wrong language; holiday dates not in the tool result; web figures without an official or second source. Then the web answer is trimmed and gets its source line, and a skill reply is split into spoken and screen parts. | `javid.py`: `think_stream`; `calendar_tools.check_holiday_reply`; `web_tools.check_reply`/`finalize`; `skills.split_reply` | `llm_done` → `reply_out` |
| 9 | Stream out | NDJSON events. The reply text arrives as one `delta` once the final round is done (rounds are buffered, see 6). `say` ("Searching…") arrives as a web search starts. | `server.py`: `_traced`, `_reply_stream` | `reply_out`, `server_done` |
| 10 | Text-to-speech | The page cuts the reply into sentences (`.!?` or `।` followed by a space) and requests each from `/speak` straight away. The server synthesizes with edge-tts, the one locked voice per language, with a cache and up to 3 tries of 6 s each. A failure shows "voice unavailable", never another voice. | `index.html`: `feedSpeech`, `speakChunk`, `fetchAudio`; `server.py`: `speak_text`; `javid.py`: `synthesize` | `tts_first_request` |
| 11 | Playback | Clips play in order. Stop, Esc, push-to-talk or barge-in cut them off. | `index.html`: `playNext`, `stopSpeech`, `bargeIn` | `first_audio`, `first_answer_audio`, `tts_done` |
| 12 | Records | The chat log, token and credit usage, and the latency trace. | `javid.py`: `log_message`, `_record_usage`; `web_tools.py`; `tracing.py` | — |

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
- **Latency panel** (left sidebar, under Models). The card shows median / p90 time to first audio per
  type. **Details** opens the stage bars of the last 20 requests (one shared scale, starting at end of speech;
  the white tick marks the first audio), plus the per-type medians for first audio and first answer audio.
- To refresh the timings in this file, use the Latency panel's medians, or run `tracing.stats()` over
  `data/traces.jsonl` after a few days of normal use.
