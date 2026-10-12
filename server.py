"""
Javid web server — wraps javid.py in a FastAPI backend.
Run: python server.py  →  open http://localhost:8000
"""

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from javid import (
    _read_lines, load_memory, startup_greeting, think_stream, transcribe,
)


# ---------- startup ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nJavid is up. Open http://localhost:8000\n")
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


def _event(**data) -> str:
    return json.dumps(data) + "\n"


def _reply_stream(user_text: str):
    """Stream Javid's reply as newline-delimited JSON events."""
    try:
        for kind, text in think_stream(user_text):
            if kind == "tool":
                yield _event(type="tool", name=text)
            else:
                yield _event(type="delta", text=text)
    except Exception as e:
        print(f"[error] {e!r}")
        yield _event(type="error", message="Something went wrong on my end.")
    yield _event(type="done", tasks=_tasks_json(), memory=_memory_json())


def _ndjson(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="application/x-ndjson")


@app.get("/greeting")
def greeting():
    return {"text": startup_greeting()}


@app.post("/talk")
def talk(audio: UploadFile = File(...)):
    audio_bytes = audio.file.read()

    content_type = audio.content_type or "audio/webm"
    ext = content_type.split("/")[-1].split(";")[0]
    if ext not in {"webm", "ogg", "mp4", "wav", "mp3", "m4a", "mpeg", "flac"}:
        ext = "webm"

    transcript = transcribe(audio_bytes, f"audio.{ext}")

    def events():
        yield _event(type="transcript", text=transcript)
        if not transcript or len(transcript.split()) < 2:
            yield _event(type="delta", text="I didn't catch that clearly. Try again.")
            yield _event(type="done", tasks=_tasks_json(), memory=_memory_json())
            return
        yield from _reply_stream(transcript)

    return _ndjson(events())


@app.post("/type")
def type_message(body: TypeBody):
    return _ndjson(_reply_stream(body.text))


@app.get("/tasks")
def tasks():
    return _tasks_json()


@app.get("/memory")
def memory():
    return _memory_json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
