"""
Javid v1.0 — push-to-talk voice assistant (7-day build, complete)
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> macOS `say` (voice)
Day 7 polish: startup greeting, graceful Ctrl+C, network-safe search, file-handle fix
"""

import io
import json
import os
import subprocess
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from ddgs import DDGS
from dotenv import load_dotenv
from groq import BadRequestError, Groq

# ---------- settings ----------
SAMPLE_RATE = 16000
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
VOICE = "Aman"
TASKS_FILE = "tasks.txt"
MEMORY_FILE = "memory.json"

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
        "morning_briefing when greeted with 'good morning', 'good afternoon', or 'good evening'. "
        "After adding a task, call list_tasks and flag any duplicates."
    )
    facts = load_memory()["facts"]
    if facts:
        fact_lines = "\n".join(f"- {f}" for f in facts)
        base += f"\n\nWhat you already know about Mahadi:\n{fact_lines}"
    return base


messages = [{"role": "system", "content": build_system_prompt()}]


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
            "description": "Return a morning briefing: current time plus a summary of pending and completed tasks.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
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
    return "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))


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
def transcribe(wav: bytes, filename: str = "speech.wav") -> str:
    result = client.audio.transcriptions.create(
        file=(filename, wav),
        model=STT_MODEL,
        language="en",
    )
    return result.text.strip()


# ---------- 3. BRAIN: text -> reply ----------
def think(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            reasoning_effort="low",
            include_reasoning=False,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn = TOOL_MAP.get(tool_call.function.name)
                args = json.loads(tool_call.function.arguments)
                result = fn(args) if fn else f"Unknown tool: {tool_call.function.name}"
                print(f"[tool] {tool_call.function.name}({args}) → {result[:80]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            follow_up = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                reasoning_effort="low",
                include_reasoning=False,
            )
            reply = (follow_up.choices[0].message.content or "").strip()
        else:
            reply = (msg.content or "").strip()

    except BadRequestError:
        reply = "I couldn't complete that request. Try asking differently."

    messages.append({"role": "assistant", "content": reply})
    return reply


# ---------- 4. MOUTH: text -> speech ----------
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
