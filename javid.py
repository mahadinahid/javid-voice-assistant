"""
Javid v2 — push-to-talk voice loop with multiple tools
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> macOS `say` (voice)
Day 3 adds: list_tasks tool, web_search tool, TOOL_MAP dispatcher
"""

import io
import json
import os
import subprocess
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from ddgs import DDGS
from groq import BadRequestError, Groq

# ---------- settings ----------
SAMPLE_RATE = 16000                    # Whisper works at 16 kHz; small file = fast upload
STT_MODEL = "whisper-large-v3-turbo"   # fast + cheap speech-to-text
LLM_MODEL = "openai/gpt-oss-20b"       # fast model with tool-calling support
VOICE = "Aman"                         # list all voices: say -v '?'
TASKS_FILE = "tasks.txt"

load_dotenv()
client = Groq()

messages = [{
    "role": "system",
    "content": (
        "You are Javid, Mahadi's personal voice assistant. "
        "Your replies are spoken aloud, so answer in 1-3 short sentences. "
        "No markdown, no lists, no emojis. "
        "Use add_task to save tasks, list_tasks to read them back, "
        "and web_search for current events or facts you don't know."
    ),
}]

# ---------- tool schemas (tells the LLM what it can call) ----------
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
            "description": "Read and return all tasks from Mahadi's task list.",
            "parameters": {"type": "object", "properties": {}, "required": []},
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
]

# ---------- tool implementations ----------
def add_task(task: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(TASKS_FILE, "a") as f:
        f.write(f"[{timestamp}] {task}\n")
    return f"Added: {task}"


def list_tasks() -> str:
    if not os.path.exists(TASKS_FILE):
        return "No tasks yet."
    content = open(TASKS_FILE).read().strip()
    return content if content else "No tasks yet."


def web_search(query: str) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n".join(f"{r['title']}: {r['body'][:150]}" for r in results)


# TOOL_MAP: single place to register tool name → function
TOOL_MAP = {
    "add_task": lambda args: add_task(**args),
    "list_tasks": lambda args: list_tasks(),
    "web_search": lambda args: web_search(**args),
}


# ---------- 1. EARS: record from the microphone ----------
def record() -> bytes:
    """Record between two Enter presses. Returns WAV bytes."""
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
def transcribe(wav: bytes) -> str:
    result = client.audio.transcriptions.create(
        file=("speech.wav", wav),
        model=STT_MODEL,
        language="en",
    )
    return result.text.strip()


# ---------- 3. BRAIN: text -> reply (with tool dispatch) ----------
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
    print("Javid v2 ready. Say 'goodbye' to quit (Ctrl+C to force quit).")
    while True:
        wav = record()
        if len(wav) < SAMPLE_RATE:
            print("(too short, try again)")
            continue

        text = transcribe(wav)
        if not text:
            continue
        print(f"You:   {text}")

        reply = think(text)
        print(f"Javid: {reply}")
        speak(reply)

        if "goodbye" in text.lower():
            break


if __name__ == "__main__":
    main()
