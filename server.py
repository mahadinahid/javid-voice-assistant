"""
Javid web server — wraps javid.py in a FastAPI backend.
Run: python server.py  →  open http://localhost:8000
"""

import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from javid import (
    _read_lines, load_memory, speak, startup_greeting, think, transcribe,
)


# ---------- startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    greeting = startup_greeting()
    print(f"\nJavid: {greeting}")
    print("Open http://localhost:8000\n")
    threading.Thread(target=speak, args=(greeting,), daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


# ---------- helpers ----------
def _tasks_json():
    lines = _read_lines()
    return [
        {
            "number": i + 1,
            "text": line.removeprefix("✓ "),
            "done": line.startswith("✓"),
        }
        for i, line in enumerate(lines)
    ]


def _memory_json():
    return load_memory()["facts"]


class TypeBody(BaseModel):
    text: str


# ---------- routes ----------
@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/talk")
def talk(audio: UploadFile = File(...)):
    audio_bytes = audio.file.read()

    content_type = audio.content_type or "audio/webm"
    ext = content_type.split("/")[-1].split(";")[0]
    if ext not in {"webm", "ogg", "mp4", "wav", "mp3", "m4a", "mpeg", "flac"}:
        ext = "webm"

    transcript = transcribe(audio_bytes, f"audio.{ext}")

    if not transcript or len(transcript.split()) < 2:
        return {
            "transcript": transcript,
            "reply": "I didn't catch that clearly. Try again.",
            "tasks": _tasks_json(),
            "memory": _memory_json(),
        }

    reply = think(transcript)
    threading.Thread(target=speak, args=(reply,), daemon=True).start()

    return {
        "transcript": transcript,
        "reply": reply,
        "tasks": _tasks_json(),
        "memory": _memory_json(),
    }


@app.post("/type")
def type_message(body: TypeBody):
    reply = think(body.text)
    threading.Thread(target=speak, args=(reply,), daemon=True).start()
    return {
        "reply": reply,
        "tasks": _tasks_json(),
        "memory": _memory_json(),
    }


@app.get("/tasks")
def tasks():
    return _tasks_json()


@app.get("/memory")
def memory():
    return _memory_json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
