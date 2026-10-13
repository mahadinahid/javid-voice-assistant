"""
Javid v1.0 — push-to-talk voice assistant (7-day build, complete)
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> macOS `say` (voice)
Day 7 polish: startup greeting, graceful Ctrl+C, network-safe search, file-handle fix
"""

import asyncio
import io
import json
import os
import re
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timedelta

import edge_tts
import numpy as np
import sounddevice as sd
from ddgs import DDGS
from dotenv import load_dotenv
from groq import APIError, BadRequestError, Groq

import calendar_tools

# ---------- settings ----------
SAMPLE_RATE = 16000
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
VOICE = "Aman"                                  # macOS `say` voice (CLI)
TTS_ENGINE = "edge"                             # web UI voice: "edge" (free), "groq", or "local" (macOS say, instant)
EDGE_VOICE = "en-US-AndrewNeural"               # also: BrianNeural, ChristopherNeural, en-GB-RyanNeural
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"  # needs terms accepted in Groq console
GROQ_TTS_VOICE = "troy"                         # also: austin, daniel
TASKS_FILE = "tasks.txt"
MEMORY_FILE = "memory.json"
HISTORY_FILE = "chat_log.jsonl"
CONTEXT_TURNS = 20      # past messages Javid reloads into context on restart

load_dotenv()
client = Groq()


# ---------- persistent memory helpers ----------
def load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"facts": []}
    with open(MEMORY_FILE) as f:
        return json.load(f)


def build_system_prompt() -> str:
    base = (
        "You are Javid, Mahadi's sharp and reliable personal voice assistant. "
        "Speak like a calm, confident friend — not a corporate bot. "
        "Use Mahadi's name occasionally but not every sentence. "
        "Your replies are spoken aloud: 1-3 short sentences, no markdown, no lists, no emojis. "
        "Tool use: add_task / list_tasks / delete_task / complete_task for the task list; "
        "web_search for current events; remember / recall / clear_memory for long-term facts; "
        "get_datetime when the user asks about time or date; "
        "morning_briefing whenever greeted with 'good morning', 'good afternoon', or 'good evening', "
        "whatever the actual time. "
        "After adding a task, call list_tasks and flag any duplicates. "
        "Tool results are never shown to Mahadi, so say what matters from them in your reply. "
        "For any question about tasks (counts, what's open, what's done), always call list_tasks first "
        "and use its counts; never answer from memory or guess. "
        "Never state task counts or calendar contents unless a tool returned them in this turn. "
        "Holidays and days off: use list_holidays (official Google data), never web_search or memory, "
        "and always name the next upcoming public holiday with its weekday and date, even if it "
        "doesn't fall on a weekend; mention when holidays join up with the weekend into a longer break. "
        "Don't say you checked or double-checked unless you called a tool in this turn; if a "
        "search didn't give a clear answer, say you're not sure instead of guessing. "
        "Mahadi lives in Bangladesh, where the weekend is Friday and Saturday. "
        "Calendar: list_events / add_event / move_event / delete_event / find_free_time. Pass days "
        "as the user said them ('friday', 'tomorrow') and let the tool work out the date; when you "
        "confirm, repeat the day and date exactly as the tool result states. Say times like '3 PM', "
        "not '15:00'. To move or delete an event, pass part of its title; if the tool says several "
        "match, ask which. delete_event never deletes on the first call: tell the user what it "
        "found, ask yes or no, and only after they say yes call it again with confirm=true. "
        "Never say an event was moved or deleted unless the tool result says so."
    )
    # Rebuilt every turn, so this is always current. The day table lets the
    # model look weekdays up instead of computing them (it gets that wrong).
    now = datetime.now()
    days = ", ".join((now + timedelta(days=i)).strftime("%a %b %d") for i in range(14))
    base += (f"\n\nRight now it is {now.strftime('%A, %Y-%m-%d, %H:%M')}."
             f"\nThe next 14 days: {days}.")
    facts = load_memory()["facts"]
    if facts:
        fact_lines = "\n".join(f"- {f}" for f in facts)
        base += f"\n\nWhat you already know about Mahadi:\n{fact_lines}"
    return base


messages = [{"role": "system", "content": build_system_prompt()}]


# ---------- chat history (persists across restarts) ----------
# One JSON object per line: {"ts", "role": "user"|"assistant"|"divider", "text"}.
# A "divider" marks the start of a new conversation.
def log_message(role: str, text: str) -> None:
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "role": role, "text": text}
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    entries = []
    with open(HISTORY_FILE) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a half-written line rather than lose the whole log
    return entries


def reset_conversation() -> None:
    """Start a fresh conversation: clear short-term context, keep the log."""
    del messages[1:]
    log_message("divider", "")


def _restore_context() -> None:
    """Reload the tail of the current conversation so Javid remembers it after a restart."""
    history = read_history()
    start = max((i + 1 for i, e in enumerate(history) if e["role"] == "divider"), default=0)
    turns = [e for e in history[start:] if e["role"] in ("user", "assistant")]
    messages.extend({"role": e["role"], "content": e["text"]} for e in turns[-CONTEXT_TURNS:])


_restore_context()


# ---------- tool schemas ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a task or reminder to Mahadi's task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to add."}
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "Return all tasks as a numbered list.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task",
            "description": "Delete a task by its number in the list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "The task number to delete."}
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark a task as done by its number in the list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer", "description": "The task number to mark complete."}
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current news, events, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save an important fact about Mahadi to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember."}
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Read all facts stored in long-term memory about Mahadi.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_memory",
            "description": "Wipe all facts from long-term memory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the current date, time, and day of the week.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "morning_briefing",
            "description": "Return a morning briefing: current time, today's calendar, and a summary of pending and completed tasks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    *calendar_tools.TOOLS,
]


# ---------- task helpers ----------
def _read_lines() -> list[str]:
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE) as f:
        return [l for l in f.read().splitlines() if l.strip()]


def _write_lines(lines: list[str]) -> None:
    with open(TASKS_FILE, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


# ---------- tool implementations ----------
def add_task(task: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(TASKS_FILE, "a") as f:
        f.write(f"[{timestamp}] {task}\n")
    return f"Added: {task}"


def list_tasks() -> str:
    lines = _read_lines()
    if not lines:
        return "No tasks yet."
    # State the counts outright: the model miscounts ✓ marks if left to tally them.
    done = sum(line.startswith("✓") for line in lines)
    summary = f"{len(lines) - done} pending, {done} done ({len(lines)} total)."
    rows = []
    for i, line in enumerate(lines):
        is_done, prefix, text = _task_parts(line)
        added = f" (added {prefix.strip()[1:-1]})" if prefix else ""
        rows.append(f"{i + 1}. [{'done' if is_done else 'pending'}] {text}{added}")
    return summary + "\n" + "\n".join(rows)


def delete_task(number: int) -> str:
    lines = _read_lines()
    if not lines:
        return "No tasks to delete."
    if number < 1 or number > len(lines):
        return f"No task number {number}. You have {len(lines)} task(s)."
    removed = lines.pop(number - 1)
    _write_lines(lines)
    return f"Deleted: {removed}"


def complete_task(number: int) -> str:
    lines = _read_lines()
    if not lines:
        return "No tasks to complete."
    if number < 1 or number > len(lines):
        return f"No task number {number}. You have {len(lines)} task(s)."
    if lines[number - 1].startswith("✓"):
        return f"Task {number} is already marked complete."
    lines[number - 1] = "✓ " + lines[number - 1]
    _write_lines(lines)
    return f"Completed: {lines[number - 1]}"


def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception:
        return "Search unavailable right now. Try again in a moment."
    if not results:
        return "No results found."
    return "\n".join(f"{r['title']}: {r['body'][:150]}" for r in results)


def remember(fact: str) -> str:
    memory = load_memory()
    memory["facts"].append(fact)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    return f"Remembered: {fact}"


def recall() -> str:
    facts = load_memory()["facts"]
    if not facts:
        return "No facts stored yet."
    return "\n".join(f"- {f}" for f in facts)


def clear_memory() -> str:
    with open(MEMORY_FILE, "w") as f:
        json.dump({"facts": []}, f, indent=2)
    return "Memory cleared."


# ---------- direct edits (web UI sidebar, no LLM) ----------
def _task_parts(line: str) -> tuple[bool, str, str]:
    """Split a task line into (done, '[timestamp] ' prefix, text)."""
    done = line.startswith("✓ ")
    body = line.removeprefix("✓ ")
    prefix = body[: body.index("] ") + 2] if body.startswith("[") and "] " in body else ""
    return done, prefix, body[len(prefix):]


def _task_line(done: bool, prefix: str, text: str) -> str:
    return ("✓ " if done else "") + prefix + text


def set_task_done(number: int, done: bool) -> bool:
    lines = _read_lines()
    if not 1 <= number <= len(lines):
        return False
    _, prefix, text = _task_parts(lines[number - 1])
    lines[number - 1] = _task_line(done, prefix, text)
    _write_lines(lines)
    return True


def edit_task(number: int, text: str) -> bool:
    lines = _read_lines()
    if not 1 <= number <= len(lines):
        return False
    done, prefix, _ = _task_parts(lines[number - 1])
    lines[number - 1] = _task_line(done, prefix, text)
    _write_lines(lines)
    return True


def _save_facts(facts: list[str]) -> None:
    with open(MEMORY_FILE, "w") as f:
        json.dump({"facts": facts}, f, indent=2)


def edit_fact(index: int, text: str) -> bool:
    facts = load_memory()["facts"]
    if not 0 <= index < len(facts):
        return False
    facts[index] = text
    _save_facts(facts)
    return True


def delete_fact(index: int) -> bool:
    facts = load_memory()["facts"]
    if not 0 <= index < len(facts):
        return False
    facts.pop(index)
    _save_facts(facts)
    return True


def get_datetime() -> str:
    return datetime.now().strftime("It is %A, %B %d, %Y at %I:%M %p.")


def startup_greeting() -> str:
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"
    pending = [l for l in _read_lines() if not l.startswith("✓")]
    greeting = f"Good {time_of_day}, Mahadi. I'm online."
    if pending:
        count = len(pending)
        greeting += f" You have {count} pending task{'s' if count != 1 else ''}."
    return greeting


def morning_briefing() -> str:
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    lines = _read_lines()
    pending = [l for l in lines if not l.startswith("✓")]
    done = [l for l in lines if l.startswith("✓")]

    parts = [
        f"{greeting}! {now.strftime('%A, %B %d at %I:%M %p')}.",
        f"Pending tasks: {len(pending)}." if pending else "No pending tasks.",
    ]
    if pending:
        parts.append("\n".join(f"  - {l}" for l in pending))
    if done:
        parts.append(f"Completed tasks: {len(done)}.")
    calendar = calendar_tools.briefing_line()
    if calendar:
        parts.append(f"Calendar: {calendar}")
    return "\n".join(parts)


TOOL_MAP = {
    "add_task":      lambda args: add_task(**args),
    "list_tasks":    lambda args: list_tasks(),
    "delete_task":   lambda args: delete_task(**args),
    "complete_task": lambda args: complete_task(**args),
    "web_search":    lambda args: web_search(**args),
    "remember":      lambda args: remember(**args),
    "recall":        lambda args: recall(),
    "clear_memory":      lambda args: clear_memory(),
    "get_datetime":      lambda args: get_datetime(),
    "morning_briefing":  lambda args: morning_briefing(),
    **calendar_tools.TOOL_MAP,
}


# ---------- 1. EARS: record from the microphone ----------
def record() -> bytes:
    chunks = []

    def on_audio(indata, frames, time, status):
        chunks.append(indata.copy())

    input("\n[Enter] to talk...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("Listening... [Enter] to stop")

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    return buffer.getvalue()


# ---------- 2. TRANSCRIBE: audio -> text ----------
def transcribe(wav: bytes, filename: str = "speech.wav", bias_name: bool = True) -> str:
    """
    bias_name=True hints Whisper to spell "Javid" correctly, but on silence/noise it
    echoes the hint back as "Javid" — a false wake word. Hands-free must pass False.
    """
    kwargs = {"prompt": "Javid"} if bias_name else {}
    result = client.audio.transcriptions.create(
        file=(filename, wav),
        model=STT_MODEL,
        language="en",
        **kwargs,
    )
    return result.text.strip()


# ---------- wake word (hands-free mode) ----------
# "Javid, ...", "Hey Javid ...", plus how Whisper spells it without a prompt
# (Javid, Javed, Jaaved, Jawid, Jovid, Jarvis...). A bare "David" only counts
# after a greeting, since it's a common name in ordinary speech.
_NAME = r"[jz][aeo]{1,2}[vwb][aeiou]{1,2}[dt]h?|jarvis"
_WAKE = re.compile(
    rf"^\W*(?:(?:hey|hi|hello|okay|ok|yo)\W+(?:{_NAME}|david)|(?:{_NAME}))\b[\W]*",
    re.IGNORECASE,
)

# Whisper tends to "hear" these in silence or background noise.
_NOISE = {"", "you", "thank you", "thanks", "thank you for watching",
          "thanks for watching", "bye", "okay", "uh", "um", "hmm"}


def strip_wake_word(text: str) -> str | None:
    """Return the request after the wake word, or None if it doesn't start with one."""
    m = _WAKE.match(text)
    return text[m.end():].strip() if m else None


def is_noise(text: str) -> bool:
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in _NOISE


# ---------- 3. BRAIN: text -> reply ----------
def _stream_completion(tool_calls: dict):
    """Stream one completion. Yields content deltas; collects tool calls into `tool_calls`."""
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS,
        reasoning_effort="low",
        include_reasoning=False,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments


MAX_TOOL_ROUNDS = 4


def think_stream(user_text: str):
    """Yield ("tool", name) and ("delta", text) events as the reply is produced.

    Each round's text is held until the round ends: if the model also called
    tools, that text was a guess made before seeing the results (e.g. a made-up
    time before get_datetime), so it is dropped instead of spoken.
    """
    messages[0]["content"] = build_system_prompt()
    calendar_tools.new_turn()   # lets delete_event tell "next message" from "same turn"
    messages.append({"role": "user", "content": user_text})
    log_message("user", user_text)
    reply = ""
    try:
        retried_empty = False
        retried_error = False
        for _ in range(MAX_TOOL_ROUNDS + 1):
            tool_calls = {}
            try:
                text = "".join(_stream_completion(tool_calls))
            except APIError as e:
                # e.g. "Failed to parse tool call arguments as JSON" — the model
                # occasionally emits a malformed tool call; one retry usually fixes it.
                if retried_error or isinstance(e, BadRequestError):
                    raise
                retried_error = True
                print(f"[retry] {str(e)[:80]}")
                continue
            if not tool_calls and not text.strip() and not retried_empty:
                retried_empty = True  # model occasionally returns nothing at all; ask once more
                continue
            if not tool_calls:
                reply = text
                if reply:
                    yield ("delta", reply)
                break

            calls = [tool_calls[i] for i in sorted(tool_calls)]
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            })
            for call in calls:
                yield ("tool", call["name"])
                fn = TOOL_MAP.get(call["name"])
                args = json.loads(call["arguments"] or "{}")
                result = fn(args) if fn else f"Unknown tool: {call['name']}"
                print(f"[tool] {call['name']}({args}) → {result[:80]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })
        else:
            reply = "Sorry, I got stuck on that one. Try asking another way."
            yield ("delta", reply)

    except APIError:
        reply = "I couldn't complete that request. Try asking differently."
        yield ("delta", reply)

    finally:
        # Runs on normal finish and when the client aborts mid-stream,
        # so history always gets whatever was actually said.
        messages.append({"role": "assistant", "content": reply.strip()})
        if reply.strip():
            log_message("assistant", reply.strip())


def think(user_text: str) -> str:
    return "".join(text for kind, text in think_stream(user_text) if kind == "delta").strip()


# ---------- 4. MOUTH: text -> speech ----------
_tts_cache: dict[str, tuple[bytes, str]] = {}   # short, repeated phrases ("Yes?") skip the network
_TTS_CACHE_MAX_CHARS = 40
_TTS_CACHE_SIZE = 64

# edge-tts is a free, unofficial Microsoft endpoint. It normally answers in ~1 s
# but is erratic: stalls of 3-5 s and "no audio" failures come in streaks. By
# default Javid waits and retries so the voice is always the natural one.
EDGE_ATTEMPTS = 3
EDGE_ATTEMPT_SEC = 6.0       # abandon a stalled try after this long and retry
# True: trade the voice for speed — after one 2 s try (or when edge-tts keeps
# missing, skipped for EDGE_COOLDOWN_SEC) that sentence is spoken by macOS `say`.
LOCAL_FALLBACK = False
EDGE_COOLDOWN_SEC = 60
LOCAL_VOICE = "Daniel"                          # macOS `say` voice for the fallback / TTS_ENGINE="local"
_edge_strikes = 0
_edge_skip_until = 0.0


async def _edge_once(text: str) -> bytes:
    audio = b""
    async for chunk in edge_tts.Communicate(text, EDGE_VOICE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return audio


def _edge_or_none(text: str) -> bytes | None:
    """edge-tts audio, or None if it kept failing (or, with LOCAL_FALLBACK, is slow right now)."""
    global _edge_strikes, _edge_skip_until
    if LOCAL_FALLBACK and time.monotonic() < _edge_skip_until:
        return None
    attempts, budget = (1, 2.0) if LOCAL_FALLBACK else (EDGE_ATTEMPTS, EDGE_ATTEMPT_SEC)
    for n in range(1, attempts + 1):
        try:
            audio = asyncio.run(asyncio.wait_for(_edge_once(text), budget))
            _edge_strikes = 0
            return audio
        except Exception as e:
            print(f"[tts] edge-tts {type(e).__name__} (try {n}/{attempts})")
    if LOCAL_FALLBACK:
        _edge_strikes += 1
        if _edge_strikes >= 2:
            _edge_skip_until = time.monotonic() + EDGE_COOLDOWN_SEC
            _edge_strikes = 1   # after the cooldown, one more miss trips it again
    return None


def _say_wav(text: str) -> bytes:
    """Local macOS voice: always available, ~0.3 s."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "speech.wav")
        subprocess.run(["say", "-v", LOCAL_VOICE, "-o", path, "--data-format=LEI16@22050", text],
                       check=True, timeout=15)
        with open(path, "rb") as f:
            return f.read()


def synthesize(text: str) -> tuple[bytes, str]:
    """Neural TTS for the web UI. Returns (audio bytes, media type)."""
    if text in _tts_cache:
        return _tts_cache[text]

    if TTS_ENGINE == "local":
        return _say_wav(text), "audio/wav"

    if TTS_ENGINE == "groq":
        response = client.audio.speech.create(
            model=GROQ_TTS_MODEL,
            voice=GROQ_TTS_VOICE,
            input=text,
            response_format="wav",
        )
        result = (response.read(), "audio/wav")
    else:
        audio = _edge_or_none(text)
        if audio is None:
            if LOCAL_FALLBACK:
                return _say_wav(text), "audio/wav"   # not cached: edge quality next time it's healthy
            raise RuntimeError("edge-tts unavailable")
        result = (audio, "audio/mpeg")

    if len(text) <= _TTS_CACHE_MAX_CHARS:
        if len(_tts_cache) >= _TTS_CACHE_SIZE:
            _tts_cache.pop(next(iter(_tts_cache)))
        _tts_cache[text] = result
    return result


def speak(text: str) -> None:
    subprocess.run(["say", "-v", VOICE, text])


# ---------- main loop ----------
def main() -> None:
    greeting = startup_greeting()
    print(f"Javid: {greeting}")
    speak(greeting)
    print("(Say 'goodbye' to quit, Ctrl+C to force quit.)\n")

    try:
        while True:
            wav = record()
            if len(wav) < SAMPLE_RATE:
                print("(too short, try again)")
                continue

            text = transcribe(wav)
            if not text or len(text.split()) < 2:
                print("(didn't catch that, try again)")
                continue
            print(f"You:   {text}")

            reply = think(text)
            print(f"Javid: {reply}")
            speak(reply)

            if "goodbye" in text.lower():
                break

    except KeyboardInterrupt:
        farewell = "Goodbye, Mahadi. Talk soon."
        print(f"\nJavid: {farewell}")
        speak(farewell)


if __name__ == "__main__":
    main()
