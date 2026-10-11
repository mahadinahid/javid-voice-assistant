# Javid — Voice Assistant

A push-to-talk voice assistant built in 7 days, layer by layer. Speak a question or command, Javid transcribes it, thinks with an LLM, executes tools if needed, and speaks the answer back.

## How it works

```
Microphone → Groq Whisper (STT) → Groq LLM → Tools → macOS say (TTS)
```

1. **Record** — sounddevice captures audio between two Enter presses
2. **Transcribe** — Groq Whisper converts speech to text (whisper-large-v3-turbo)
3. **Think** — Groq LLM (openai/gpt-oss-20b) decides what to say or which tool to call
4. **Act** — if a tool is called, Python executes it and sends the result back to the LLM
5. **Speak** — macOS `say` reads the final reply aloud

Session memory is maintained by sending the full conversation history on every API call. Facts that need to survive restarts are stored in `memory.json` and injected into the system prompt at startup.

## Tools (10)

| Tool | What it does |
|------|-------------|
| `add_task` | Appends a timestamped task to `tasks.txt` |
| `list_tasks` | Returns all tasks as a numbered list |
| `delete_task` | Removes task N from the list |
| `complete_task` | Marks task N with ✓ |
| `web_search` | Searches DuckDuckGo for current info |
| `remember` | Saves a fact to `memory.json` for future sessions |
| `recall` | Reads all stored facts |
| `clear_memory` | Wipes `memory.json` |
| `get_datetime` | Returns current date, time, and day |
| `morning_briefing` | Time-appropriate greeting + pending task summary |

## Setup

**Requirements:** macOS, Python 3.8+, a [Groq API key](https://console.groq.com)

```bash
# 1. Clone and enter the project
git clone https://github.com/mahdynahid/javid-voice-assistant.git
cd javid-voice-assistant

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
open -e .env        # paste: GROQ_API_KEY=gsk_...
```

**Microphone access:** If the mic is blocked, go to  
System Settings → Privacy & Security → Microphone → enable your Terminal app.

## Run

```bash
source .venv/bin/activate
python javid.py
```

Javid greets you on startup. Press **Enter** to start recording, speak, press **Enter** again to stop. Say **"goodbye"** to quit cleanly, or press **Ctrl+C**.

## Project structure

```
javid.py          — main assistant (all logic in one file)
requirements.txt  — pinned dependencies
.env.example      — API key template (copy to .env)
.gitignore        — keeps .env, tasks.txt, memory.json out of git
LEARNING_LOG.md   — day-by-day build notes and lessons
```

## Built with

- [Groq](https://groq.com) — Whisper STT + LLM inference
- [sounddevice](https://python-sounddevice.readthedocs.io) — microphone capture
- [ddgs](https://github.com/deedy5/ddgs) — DuckDuckGo search
- macOS `say` — text-to-speech
