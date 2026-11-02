# Javid

A personal voice assistant that speaks English and Bangla. Talk to it, or type, in a web app on your Mac (or from
Telegram on your phone), and it answers aloud: your tasks and calendar, timers, weather, news, live prices,
exchange rates, research and more.

It began on 2026-10-11 as a 10-tool terminal loop (the day-by-day story is in [`LEARNING_LOG.md`](LEARNING_LOG.md))
and is now a web app with 25 tools, two agents, a Telegram bot and its own test suite.

## What it does

- **Talk naturally.** Hold Space to talk, or go hands-free: say "Javid" and keep the conversation going without the
  name for 30 seconds after each answer; hands-free, you can also cut in while it's speaking. Say "let's speak
  Bangla" and it answers in Bangla until you switch back.
- **Your day.** Tasks; Google Calendar (what's on, free time, adding and moving events, deleting only after you
  confirm); timers and reminders that ring in Javid's voice; a morning brief; a planner that lays out your day around
  your events.
- **The world.** Headlines from Google News, weather (Open-Meteo, with MET Norway as a backup), live crypto and stock
  prices (Binance, Yahoo Finance), exchange rates (the European Central Bank, ExchangeRate-API), public holidays, a
  calculator that knows Bangladesh's land units, web search across several engines, and a research agent that
  writes a cited report.
- **It keeps watch.** "Tell me when bitcoin goes above 90,000", "let me know when there's news about the IVAC
  holiday", "tell me when it starts raining": checked in the background, announced like a timer.
- **It remembers, with your say-so.** Nothing goes into its memory until you approve it, and secrets are refused.
  Past chats are searchable.
- **On your phone.** A Telegram bot, paired by scanning a QR code. Anything that changes your data waits for an
  Approve button.
- **Long answers, kept.** Reports, briefs and plans can be saved as Google Docs in a "Javid Reports" folder.
- **It knows itself.** "Is everything working?" runs a self-check; "where did you get that?" names the source of
  the last answer.
- **Your look.** Orbit (the default): a sci-fi look where the voice ball is a holographic globe, with satellites
  and a ring of letters orbiting it and light arcing out from Dhaka while Javid thinks. Or Glass, Overdrive, and
  colour themes you can make in the app with a live preview.

## How it works

```
 Browser: index.html ── /talk, /type ──►  server.py (FastAPI) ──►  javid.py: the agent loop
   microphone                              speech to text: Groq Whisper (whisper-large-v3-turbo)
   speaker   ◄── /speak ─────────────────  answered in code first: news, watches, exchange rates, market briefs,
                                           questions about Javid, the self-check
                                           otherwise a language model with 25 tools:
                                           Groq gpt-oss-20b, then the fallbacks in config/providers.yaml
                                           speech: edge-tts (en-US-AndrewNeural, bn-BD-PradeepNeural),
                                           ElevenLabs only as a backup
```

A few rules shape everything:

- **Code before the model.** Whatever can be answered exactly in Python is, before any model is asked: it's faster,
  free, and can't be made up. Counting, dates and sums are done in Python too, and the model only relays them.
- **Checked figures.** A rate, price or measure from the web must appear on an official site or with two
  independent publishers, or Javid says it couldn't confirm it.
- **Free allowances, guarded.** Every outside service is an entry in [`config/providers.yaml`](config/providers.yaml)
  with its free allowance. A usage ledger counts each call, and a budget guard keeps each service within its
  allowance.
- **Tools only for tested models.** A model changes your data only if it passed the tool tests
  (`python regression.py --certify <model>`). Every other model is chat-only.
- **Personal data stays with providers you trust.** Your calendar, tasks, memory and chats are sent only to providers
  you've marked `personal_ok: true` yourself.
- **One voice.** The voices are fixed; there's no browser speech and no other engine. If speech fails, the reply is
  shown with a "voice unavailable" note.

The request pipeline and its timings are in [`ARCHITECTURE.md`](ARCHITECTURE.md), with the reasoning behind the
bigger decisions.

## Quick start

**You need:** macOS, Python 3.14 (what Javid is developed and tested on), and a free
[Groq API key](https://console.groq.com). Optional: `ffmpeg` for voice notes on Telegram (`brew install ffmpeg`).

```bash
git clone https://github.com/mahadinahid/javid-voice-assistant.git
cd javid-voice-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your key: GROQ_API_KEY=gsk_...
python server.py              # open http://localhost:8000
```

The original terminal version still works: `python javid.py`, then Enter to start and stop recording. If the
microphone is blocked, allow your terminal in System Settings → Privacy & Security → Microphone.

### Optional services

Each one is off until its key is in `.env`; the server prints which are on when it starts, and the Providers panel
shows what each has used of its free allowance.

| In `.env` | Adds |
|-----------|------|
| `TAVILY_API_KEY` | Web search (`EXA_API_KEY`, `BRAVE_SEARCH_API_KEY`, `FIRECRAWL_API_KEY` and `COHERE_API_KEY` add more engines and reranking) |
| `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_AI_TOKEN`, `IFM_API_KEY` | Backup language models (OpenRouter: free models only) |
| `ELEVENLABS_API_KEY` | The backup voice |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_IDS` | The Telegram bot; the ✈ Telegram page in the app sets up both |

Google Calendar and Google Drive use your own OAuth client: put its `credentials.json` (a Desktop app client) in the
project folder, then connect each one from the sidebar.

## Privacy and security

Javid runs on your Mac, and its data stays there in plain files: `tasks.txt`, `memory/`, `chat_log.jsonl` and the
rest, all kept out of git. The server listens on `127.0.0.1` only. Keys, the model order, memory and the Telegram
lock can be changed only from Javid's own page; some other routes (typing a message, adding a task) don't check
yet, and that is the first open item in [`SECURITY.md`](SECURITY.md). What does leave the machine: each voice clip goes to Groq for transcription, each spoken
reply to Microsoft's edge-tts, and your requests to the language model and search providers that are on. Calendar,
tasks, memory and chats go only to providers with `personal_ok: true`.

[`SECURITY.md`](SECURITY.md) lists the threats, the defences and everything still open.

## Testing

```bash
python regression.py                         # 132 scripted checks: no network, no API tokens
python regression.py -k "news"               # only the cases whose names contain "news"
python regression.py --live                  # one tiny real call to each provider that's on
python regression.py --llm -k "task: add"    # a case through the real model (spends quota)
python regression.py --certify <model>       # the tool tests a model must pass to use tools
```

The scripted run uses scratch copies of every data file and fakes for the calendar, search, news, weather, prices,
Telegram and Drive, so it never touches your data or spends a credit. Run it before and after any change.

## Project layout

| Path | What it is |
|------|------------|
| `server.py` | The web server: routes for talking, typing, speech and the sidebar |
| `index.html` | The whole web app in one file: markup, styles and script |
| `javid.py` | The core: agent loop, model chain, speech in and out, the tools' wiring, and the terminal version |
| `providers.py`, `config/providers.yaml` | The provider registry, usage ledger, budget guard and privacy rule |
| `search.py`, `web_tools.py` | The web search router, page reading and the figure checks |
| `gnews.py`, `news.py` | Google News headlines, and news requests answered in code |
| `calendar_tools.py`, `timers.py`, `weather.py`, `markets.py`, `rates.py`, `calc.py` | Calendar, timers and reminders, weather, prices, exchange rates, the calculator |
| `memory_store.py` | Memory with approval, and the past-chat search |
| `skills/`, `agents/` (`skills.py`, `agents.py`) | The morning-brief and news-digest skills; the research and planner agents |
| `watches.py` | Alerts on prices, news and rain |
| `telegram_bot.py`, `approvals.py`, `telegram.html` | The Telegram bot, its Approve buttons, and its setup page |
| `drive_tools.py` | Saving long answers to Google Drive |
| `selfcheck.py`, `about.py` | The self-check, and answers to questions about Javid |
| `themes.py`, `themes/` | Colour themes ([how to make one](themes/README.md)) |
| `tracing.py` | Latency traces of every request |
| `regression.py`, `fixtures/` | The test suite and its recorded responses |
| `bench_web.py` | A benchmark of web answers (real requests) |

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): how a request travels, how long each stage takes, and the design decisions.
- [`SECURITY.md`](SECURITY.md): threats, protections and open items.
- [`CLAUDE.md`](CLAUDE.md): detailed notes on every part, for whoever works on the code next.
- [`themes/README.md`](themes/README.md): the colour theme format.
- [`LEARNING_LOG.md`](LEARNING_LOG.md): how Javid was built, day by day.
