"""
Javid v1 — push-to-talk voice loop with tool calling
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> macOS `say` (voice)
Day 2 adds: add_task tool that writes to tasks.txt
"""

import io
import json
import subprocess
import wave
from datetime import datetime

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import BadRequestError, Groq

# ---------- settings ----------
SAMPLE_RATE = 16000                    # Whisper works at 16 kHz; small file = fast upload
STT_MODEL = "whisper-large-v3-turbo"   # fast + cheap speech-to-text
LLM_MODEL = "openai/gpt-oss-20b"       # fast model with tool-calling support
VOICE = "Aman"                         # list all voices: say -v '?'
TASKS_FILE = "tasks.txt"

load_dotenv()
client = Groq()

# Conversation memory: the whole list is sent every turn, so Javid remembers the session.
messages = [{
    "role": "system",
    "content": (
        "You are Javid, Mahadi's personal voice assistant. "
        "Your replies are spoken aloud, so answer in 1-3 short sentences. "
        "No markdown, no lists, no emojis. "
        "When the user asks to add a task or reminder, use the add_task tool."
    ),
}]

# ---------- tools ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a task or reminder to Mahadi's task list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task description to add.",
                    }
                },
                "required": ["task"],
            },
        },
    }
]


# ---------- tool implementations ----------
def add_task(task: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(TASKS_FILE, "a") as f:
        f.write(f"[{timestamp}] {task}\n")
    return f"Added: {task}"


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


# ---------- 3. BRAIN: text -> reply (with tool calling) ----------
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

        # If the model called a tool, execute it and do a second API call
        if msg.tool_calls:
            messages.append(msg)  # save assistant message that contains the tool call
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "add_task":
                    args = json.loads(tool_call.function.arguments)
                    result = add_task(args["task"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            # Second call: model reads the tool result and forms a spoken reply
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
        reply = "I don't have internet access yet. I can answer from what I already know."

    messages.append({"role": "assistant", "content": reply})
    return reply


# ---------- 4. MOUTH: text -> speech ----------
def speak(text: str) -> None:
    subprocess.run(["say", "-v", VOICE, text])


# ---------- main loop ----------
def main() -> None:
    print("Javid v1 ready. Say 'goodbye' to quit (Ctrl+C to force quit).")
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
