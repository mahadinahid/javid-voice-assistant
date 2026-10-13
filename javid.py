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
import uuid
import wave
from datetime import datetime, timedelta

import edge_tts
import numpy as np
import sounddevice as sd
from ddgs import DDGS
from dotenv import load_dotenv
import groq as _groq
import openai as _openai
from groq import APIError, BadRequestError, Groq, InternalServerError, NotFoundError, RateLimitError

import calendar_tools

# Groq and OpenAI-compatible providers (Gemini) raise different exception classes for the same thing.
API_ERRORS   = (APIError, _openai.APIError)
BAD_REQUEST  = (BadRequestError, _openai.BadRequestError)
RATE_LIMITED = (RateLimitError, _openai.RateLimitError)
NOT_FOUND    = (NotFoundError, _openai.NotFoundError)
BAD_KEY      = (_groq.AuthenticationError, _groq.PermissionDeniedError,
                _openai.AuthenticationError, _openai.PermissionDeniedError)
OVERLOADED   = (InternalServerError, _openai.InternalServerError,
                _groq.APIConnectionError, _openai.APIConnectionError)   # 5xx, network, timeouts
UNAVAILABLE  = RATE_LIMITED + NOT_FOUND + BAD_KEY + OVERLOADED          # => try the next model

# ---------- settings ----------
SAMPLE_RATE = 16000
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
# Tried in order when LLM_MODEL is rate-limited (free tier: 200k tokens/day per model) or down.
# Quotas are per model, so each fallback has its own budget. Checked against this key's model list.
# Entries without a prefix are Groq models; "gemini:<model>" goes to Google's OpenAI-compatible
# endpoint (needs GEMINI_API_KEY) and "deepseek:<model>" to DeepSeek's (needs DEEPSEEK_API_KEY, paid).
# A provider without its key in .env is skipped.
GEMINI_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"
LLM_FALLBACKS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b",
                 f"gemini:{GEMINI_MODEL}", f"deepseek:{DEEPSEEK_MODEL}"]
# Daily token caps seen in Groq 429 messages; others are learned from the first 429 (llm_quota.json).
KNOWN_TPD_LIMITS = {"openai/gpt-oss-20b": 200000}
LLM_MIN_REQUEST_TOKENS = 2500   # one turn's request is ~2-3k tokens (prompt + tool schemas)
VOICE = "Aman"                                  # macOS `say` voice (CLI)
# LOCKED by the user (2026-10-13): this engine + voice is the one they approved ("perfect").
# Don't change TTS_ENGINE, EDGE_VOICE or LOCAL_FALLBACK unless they explicitly ask.
TTS_ENGINE = "edge"                             # web UI voice: "edge" (free), "groq", or "local" (macOS say, instant)
EDGE_VOICE = "en-US-AndrewNeural"               # also: BrianNeural, ChristopherNeural, en-GB-RyanNeural
GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"  # needs terms accepted in Groq console
GROQ_TTS_VOICE = "troy"                         # also: austin, daniel
TASKS_FILE = "tasks.txt"
MEMORY_FILE = "memory.json"
HISTORY_FILE = "chat_log.jsonl"
LLM_USAGE_FILE = "llm_usage.jsonl"   # one line per LLM request: tokens by model (for the Models panel)
LLM_QUOTA_FILE = "llm_quota.json"    # daily token limits Groq has told us about
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


def _is_garbled(text: str) -> bool:
    """Degenerate output: a run of ellipses/dots or one word on repeat (seen from gpt-oss-120b
    answering a question about itself). Real replies are never mostly punctuation."""
    s = text.strip()
    marks = sum(c in ".…" for c in s)
    non_space = sum(not c.isspace() for c in s)
    if non_space >= 12 and marks / non_space > 0.3:
        return True
    words = re.findall(r"[a-z']+", s.lower())
    return len(words) >= 8 and len(set(words)) / len(words) < 0.3


def _restore_context() -> None:
    """Reload the tail of the current conversation so Javid remembers it after a restart."""
    history = read_history()
    start = max((i + 1 for i, e in enumerate(history) if e["role"] == "divider"), default=0)
    turns = [e for e in history[start:] if e["role"] in ("user", "assistant")
             and not (e["role"] == "assistant" and _is_garbled(e["text"]))]
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
_model_cooldown: dict[str, float] = {}   # model -> time.time() after which it may be tried again
llm_usage: dict[str, int] = {}           # model -> requests served (handy for tests and logs)


_model_info: dict[str, dict] = {}        # model -> {"limits", "last_error", "quota"} learned at runtime


def _load_quota() -> None:
    try:
        with open(LLM_QUOTA_FILE) as f:
            for model, q in json.load(f).items():
                _model_info.setdefault(model, {})["quota"] = q
    except (OSError, ValueError):
        pass


def _save_quota() -> None:
    try:
        with open(LLM_QUOTA_FILE, "w") as f:
            json.dump({m: i["quota"] for m, i in _model_info.items() if "quota" in i}, f, indent=2)
    except OSError:
        pass


def _record_usage(model: str, usage) -> None:
    """Append one request's token counts (Groq's own numbers) to the usage log."""
    entry = {"ts": round(time.time(), 1), "model": model, "prompt": usage.prompt_tokens or 0,
             "completion": usage.completion_tokens or 0, "total": usage.total_tokens or 0}
    try:
        with open(LLM_USAGE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _read_usage(since: float) -> list[dict]:
    out = []
    try:
        with open(LLM_USAGE_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", 0) >= since:
                    out.append(e)
    except OSError:
        pass
    return out


def _note_headers(model: str, headers) -> None:
    """Groq reports per-minute token and per-day request headroom on every response."""
    def num(name):
        try:
            return int(headers.get(name))
        except (TypeError, ValueError):
            return None
    limits = {
        "tpm_limit": num("x-ratelimit-limit-tokens"), "tpm_remaining": num("x-ratelimit-remaining-tokens"),
        "rpd_limit": num("x-ratelimit-limit-requests"), "rpd_remaining": num("x-ratelimit-remaining-requests"),
        "rpd_reset": headers.get("x-ratelimit-reset-requests"), "at": time.time(),
    }
    if any(limits[k] is not None for k in ("tpm_limit", "rpd_limit")):
        _model_info.setdefault(model, {})["limits"] = limits


def _note_error(model: str, err: Exception) -> None:
    info = _model_info.setdefault(model, {})
    msg = str(err)
    kind = ("Insufficient balance: top up the account" if getattr(err, "status_code", None) == 402
            else type(err).__name__)
    info["last_error"] = {"kind": kind, "message": msg[:220], "at": time.time()}
    # "...on tokens per day (TPD): Limit 200000, Used 198720, Requested 2278..."
    m = re.search(r"\((TPD|TPM|RPD|RPM)\): Limit (\d+), Used (\d+)", msg)
    if m and m.group(1) == "TPD":
        info["quota"] = {"tpd_limit": int(m.group(2)), "reported_used": int(m.group(3)), "reported_at": time.time()}
        _save_quota()
    elif m:
        info["last_error"]["window"] = m.group(1)


_load_quota()


PROVIDERS = {
    "groq":   {"label": "Groq"},
    "gemini": {"label": "Google", "key_env": "GEMINI_API_KEY",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    "deepseek": {"label": "DeepSeek", "key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
}
_provider_clients: dict = {}


def _split(model_id: str) -> tuple[str, str]:
    """'gemini:gemini-2.5-flash' -> ('gemini', 'gemini-2.5-flash'); a bare id is a Groq model."""
    provider, sep, name = model_id.partition(":")
    return (provider, name) if sep and provider in PROVIDERS else ("groq", model_id)


def _provider_client(provider: str):
    if provider == "groq":
        return client
    if provider not in _provider_clients:
        key = os.environ.get(PROVIDERS[provider]["key_env"], "").strip()
        _provider_clients[provider] = _openai.OpenAI(
            api_key=key, base_url=PROVIDERS[provider]["base_url"], timeout=30.0, max_retries=0) if key else None
    return _provider_clients[provider]


def _has_key(model_id: str) -> bool:
    return _provider_client(_split(model_id)[0]) is not None


def _model_kwargs(model_id: str) -> dict:
    provider, name = _split(model_id)
    if provider == "gemini":
        return {"reasoning_effort": "low"}      # keep thinking short: replies are spoken
    # reasoning_effort is a gpt-oss knob. Qwen given reasoning_effort="none" skipped the
    # tool call and claimed the task was added, so other models get their defaults.
    if "gpt-oss" in name:
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}


def _messages_for(provider: str, msgs: list) -> list:
    """Gemini hands back an `extra_content` blob on tool calls that it wants echoed; other
    providers don't know the field, so it's stripped when failing over to them."""
    if provider == "gemini" or not any(
            c.get("extra_content") for m in msgs for c in m.get("tool_calls") or []):
        return msgs
    return [{**m, "tool_calls": [{k: v for k, v in c.items() if k != "extra_content"}
                                 for c in m["tool_calls"]]} if m.get("tool_calls") else m
            for m in msgs]


def _create(model_id: str, msgs: list, *, tools=None, stream=False, max_tokens=None):
    """One chat completion on whichever provider hosts `model_id`; returns the raw response."""
    provider, name = _split(model_id)
    kw = _model_kwargs(model_id)
    if tools:
        kw["tools"] = tools
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if stream:
        kw["stream"] = True
        if provider != "groq":
            kw["stream_options"] = {"include_usage": True}   # else no token counts in the stream
    return _provider_client(provider).chat.completions.with_raw_response.create(
        model=name, messages=_messages_for(provider, msgs), **kw)


def _is_unavailable(err: Exception) -> bool:
    # DeepSeek answers 402 when the account balance is empty: park it instead of failing the turn.
    return isinstance(err, UNAVAILABLE) or getattr(err, "status_code", None) == 402


def _cooldown_for(err: Exception) -> float:
    if getattr(err, "status_code", None) == 402 or isinstance(err, NOT_FOUND + BAD_KEY):
        return 3600.0       # model not on this key / key rejected: don't hammer it
    if isinstance(err, RATE_LIMITED):
        return _retry_after(err)
    return 30.0             # overloaded or unreachable


def _retry_after(err: Exception, default: float = 60.0) -> float:
    """Seconds until a rate-limited model is usable again (header, else 'try again in 7m11s')."""
    header = getattr(getattr(err, "response", None), "headers", {}).get("retry-after")
    try:
        if header:
            return max(1.0, float(header))
    except ValueError:
        pass
    g = re.search(r"retry(?: in|Delay['\"]?:\s*['\"]?)\s*([\d.]+)s", str(err), re.I)   # Google's wording
    if g:
        return max(1.0, float(g.group(1)))
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", str(err))
    if m and any(m.groups()):
        h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        return max(1.0, h * 3600 + mins * 60 + secs)
    return default


def _models_to_try() -> list[str]:
    chain = [m for m in (LLM_MODEL, *LLM_FALLBACKS) if _has_key(m)]
    now = time.time()
    ready = [m for m in chain if _model_cooldown.get(m, 0) <= now]
    # Everything is cooling down: still try the one that recovers first rather than give up.
    return ready or [min(chain, key=lambda m: _model_cooldown.get(m, 0))]


_NON_CHAT = ("whisper", "orpheus", "tts", "guard", "embed", "image", "imagen", "veo", "live", "audio")
_key_models: dict[str, dict] = {}


def _models_on_provider(provider: str) -> list[str] | None:
    """Chat-capable model ids (chain notation) this provider's key can use. Cached 10 min;
    None if the provider is unreachable or has no key."""
    c = _provider_client(provider)
    if c is None:
        return None
    cache = _key_models.setdefault(provider, {"at": 0.0, "ids": None})
    if time.time() - cache["at"] > 600:
        try:
            raw_ids = sorted(m.id.removeprefix("models/") for m in c.models.list().data)
            keep = [i for i in raw_ids if not any(k in i for k in _NON_CHAT)]
            if provider in ("gemini", "deepseek"):
                keep = [i for i in keep if provider in i]
            cache.update(at=time.time(), ids=[i if provider == "groq" else f"{provider}:{i}" for i in keep])
        except Exception:
            cache["at"] = time.time() - 540     # retry in a minute
    return cache["ids"]


def llm_status() -> dict:
    """Everything the Models panel shows: chain order, who can serve now, token usage, limits."""
    _load_quota()   # another process (CLI, regression run) may have seen a 429 since we started
    now = time.time()
    chain = [LLM_MODEL, *LLM_FALLBACKS]
    everything = _read_usage(0)
    recent = [e for e in everything if e.get("ts", 0) >= now - 86400]
    # Our tally only starts when logging began; before that, usage is simply unknown to us.
    tracking_since = min((e["ts"] for e in everything), default=now)
    on_key_by_provider = {p: _models_on_provider(p) for p in PROVIDERS}
    models = []
    for order, model in enumerate(chain):
        mine = [e for e in recent if e.get("model") == model]
        day = {"requests": len(mine), "prompt": sum(e["prompt"] for e in mine),
               "completion": sum(e["completion"] for e in mine), "total": sum(e["total"] for e in mine)}
        info = _model_info.get(model, {})
        quota = info.get("quota")
        if not quota and model in KNOWN_TPD_LIMITS:
            quota = {"tpd_limit": KNOWN_TPD_LIMITS[model], "reported_used": 0, "reported_at": 0}
        est_used = remaining = None
        if quota:
            # Groq's reported figure counts every client on the key; ours only counts Javid.
            # Right after a report, trust Groq's number plus what we've spent since.
            since_report = sum(e["total"] for e in mine if e["ts"] >= quota["reported_at"])
            fresh = quota["reported_at"] and now - quota["reported_at"] < 3600
            est_used = max(day["total"], quota["reported_used"] + since_report) if fresh else day["total"]
            remaining = max(0, quota["tpd_limit"] - est_used)
        cooling_for = max(0.0, _model_cooldown.get(model, 0) - now)
        provider = _split(model)[0]
        on_key = on_key_by_provider[provider]
        if not _has_key(model):
            state = "nokey"
        elif on_key is not None and model not in on_key:
            state = "missing"
        elif cooling_for > 0:
            state = "cooling"
        elif remaining is not None and remaining < LLM_MIN_REQUEST_TOKENS:
            state = "low"       # answers a tiny ping, but a real reply won't fit in what's left
        else:
            state = "ready"
        models.append({
            "id": model, "name": _split(model)[1].split("/")[-1], "order": order,
            "provider": provider, "provider_label": PROVIDERS[provider]["label"],
            "key_env": PROVIDERS[provider].get("key_env"),
            "role": "primary" if order == 0 else "fallback", "state": state,
            "retry_in_s": round(cooling_for), "tokens_24h": day,
            "session_requests": llm_usage.get(model, 0),
            "last_used": max((e["ts"] for e in mine), default=None),
            "limits": info.get("limits"),
            "tpd_limit": quota["tpd_limit"] if quota else None,
            "est_used_24h": est_used, "remaining_24h": remaining,
            "reported": ({"used": quota["reported_used"], "at": quota["reported_at"]}
                         if quota and quota["reported_at"] else None),
            "last_error": info.get("last_error"),
            "sends_reasoning_effort": "gpt-oss" in model,
        })
    serving = next((m for m in models if m["state"] == "ready"), None)
    if serving:
        serving["state"] = "serving"
    return {
        "now": now, "serving": serving["id"] if serving else None,
        "chain_size": len(chain), "tracking_since": tracking_since,
        "ready_count": sum(m["state"] in ("serving", "ready") for m in models),
        "min_request_tokens": LLM_MIN_REQUEST_TOKENS,
        "models": models,
        "on_key": [{"id": i, "in_chain": i in chain, "provider": p}
                   for p, ids in on_key_by_provider.items() for i in ids or []],
    }


def ping_models() -> list[dict]:
    """Send a tiny no-tools request to every model in the chain, in parallel.
    Shows which ones answer right now and how fast; a success clears that model's cooldown."""
    from concurrent.futures import ThreadPoolExecutor

    def ping(model: str) -> dict:
        t0 = time.time()
        ms = lambda: round((time.time() - t0) * 1000)
        if not _has_key(model):
            return {"id": model, "ok": False, "ms": 0,
                    "error": f"No API key: set {PROVIDERS[_split(model)[0]]['key_env']} in .env"}
        try:
            raw = _create(model, [{"role": "user", "content": "ping"}], max_tokens=8)
            _note_headers(model, raw.headers)
            done = raw.parse()
            if done.usage:
                _record_usage(model, done.usage)
            _model_cooldown.pop(model, None)
            return {"id": model, "ok": True, "ms": ms()}
        except Exception as e:
            if isinstance(e, API_ERRORS) and _is_unavailable(e):
                _model_cooldown[model] = time.time() + _cooldown_for(e)
                _note_error(model, e)
            return {"id": model, "ok": False, "ms": ms(), "error": f"{type(e).__name__}: {str(e)[:140]}"}

    chain = [LLM_MODEL, *LLM_FALLBACKS]
    with ThreadPoolExecutor(max_workers=len(chain)) as pool:
        return list(pool.map(ping, chain))


def _open_stream():
    """Start a streaming completion on the first model in the chain that accepts it.
    Returns (model, stream)."""
    last_error = None
    for model in _models_to_try():
        try:
            raw = _create(model, messages, tools=TOOLS, stream=True)
        except API_ERRORS as e:
            if not _is_unavailable(e):
                raise       # e.g. malformed tool call: think_stream retries those
            # 429: out of quota for now. 5xx/network: overloaded. 402/404/401/403: can't serve.
            wait = _cooldown_for(e)
            _model_cooldown[model] = time.time() + wait
            _note_error(model, e)
            print(f"[llm] {model} unavailable ({type(e).__name__}), skipping it for {wait:.0f}s")
            last_error = e
            continue
        _note_headers(model, raw.headers)
        if model != LLM_MODEL:
            print(f"[llm] answering with fallback {model}")
        llm_usage[model] = llm_usage.get(model, 0) + 1
        return model, raw.parse()
    raise last_error


def _stream_completion(tool_calls: dict):
    """Stream one completion. Yields content deltas; collects tool calls into `tool_calls`."""
    model, stream = _open_stream()
    counted = False
    for chunk in stream:
        if not counted:
            usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
            if usage:   # arrives on the final chunk
                _record_usage(model, usage)
                counted = True
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
        for tc in delta.tool_calls or []:
            idx = tc.index
            if idx is None:   # some OpenAI-compatible servers omit it: a named chunk starts a new call
                named = bool(tc.function and tc.function.name)
                idx = len(tool_calls) if named or not tool_calls else max(tool_calls)
            slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
            if extra:
                slot["extra"] = extra     # Gemini: must be echoed back with the tool result
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
        retried_garbled = False
        for _ in range(MAX_TOOL_ROUNDS + 1):
            tool_calls = {}
            try:
                text = "".join(_stream_completion(tool_calls))
            except API_ERRORS as e:
                # e.g. "Failed to parse tool call arguments as JSON" — the model
                # occasionally emits a malformed tool call; one retry usually fixes it.
                if retried_error or isinstance(e, BAD_REQUEST):
                    raise
                retried_error = True
                print(f"[retry] {str(e)[:80]}")
                continue
            if not tool_calls and not text.strip() and not retried_empty:
                retried_empty = True  # model occasionally returns nothing at all; ask once more
                continue
            if not tool_calls and _is_garbled(text):
                if not retried_garbled:
                    retried_garbled = True
                    print(f"[retry] garbled reply: {text[:60]!r}")
                    continue
                text = "Sorry, I lost my train of thought. Could you ask that again?"
            if not tool_calls:
                reply = text
                if reply:
                    yield ("delta", reply)
                break

            calls = [tool_calls[i] for i in sorted(tool_calls)]
            for c in calls:
                c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}"   # some providers send none
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]},
                     **({"extra_content": c["extra"]} if c.get("extra") else {})}
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

    except API_ERRORS:
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
