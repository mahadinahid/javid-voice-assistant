"""
Javid v0 — push-to-talk voice loop
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> macOS `say` (voice)
"""

import io
import subprocess
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from groq import BadRequestError, Groq

# ---------- settings ----------
SAMPLE_RATE = 16000                    # Whisper works at 16 kHz; small file = fast upload
STT_MODEL = "whisper-large-v3-turbo"   # fast + cheap speech-to-text
LLM_MODEL = "openai/gpt-oss-20b"       # fast model with tool-calling support (needed on day 2)
VOICE = "Aman"                         # list all voices: say -v '?'

load_dotenv()    # loads GROQ_API_KEY from the .env file
client = Groq()

# Conversation memory: the whole list is sent every turn, so Javid remembers the session.
messages = [{
    "role": "system",
    "content": (
        "You are Javid, Mahadi's personal voice assistant. "
        "Your replies are spoken aloud, so answer in 1-3 short sentences. "
        "No markdown, no lists, no emojis."
    ),
}]


# ---------- 1. EARS: record from the microphone ----------
def record() -> bytes:
    """Record between two Enter presses. Returns WAV bytes."""
    chunks = []

    def on_audio(indata, frames, time, status):
        chunks.append(indata.copy())   # called by sounddevice for every small audio block

    input("\n[Enter] to talk...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input("Listening... [Enter] to stop")

    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:   # wrap raw samples into a .wav file in memory
        w.setnchannels(1)
        w.setsampwidth(2)                # int16 = 2 bytes per sample
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


# ---------- 3. BRAIN: text -> reply ----------
def think(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            reasoning_effort="low",    # less thinking = faster spoken reply
            include_reasoning=False,   # only return the final answer
        )
        reply = (response.choices[0].message.content or "").strip()
    except BadRequestError:
        reply = "I don't have internet access yet. I can answer from what I already know."
    messages.append({"role": "assistant", "content": reply})
    return reply


# ---------- 4. MOUTH: text -> speech ----------
def speak(text: str) -> None:
    subprocess.run(["say", "-v", VOICE, text])


# ---------- main loop ----------
def main() -> None:
    print("Javid v0 ready. Say 'goodbye' to quit (Ctrl+C to force quit).")
    while True:
        wav = record()
        if len(wav) < SAMPLE_RATE:       # under ~0.5 s of audio -> ignore
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
