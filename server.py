"""
Javid web server — wraps javid.py in a FastAPI backend.
Run: python server.py  →  open http://localhost:8000
"""

import json
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import calendar_tools
from javid import (
    _read_lines, add_task, delete_fact, delete_task, edit_fact, edit_task,
    is_noise, load_memory, read_history, reset_conversation, set_task_done,
    startup_greeting, strip_wake_word, synthesize, think_stream, transcribe,
)


# ---------- startup ----------
def _warm_tts():
    try:
        synthesize("Yes?")
    except Exception as e:
        print(f"[tts] warm-up failed: {str(e)[:80]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nJavid is up. Open http://localhost:8000\n")
    # The first TTS call after startup is slow (cold connection); warm it and the
    # cache with the hands-free acknowledgement so the first "Hey Javid" is quick.
    threading.Thread(target=_warm_tts, daemon=True).start()
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
def talk(audio: UploadFile = File(...), mode: str = Form("push")):
    """
    mode = "push"     push-to-talk: always answer
           "wake"     hands-free, idle: only answer if it starts with "Javid"
           "followup" hands-free, right after Javid spoke: no wake word needed
    """
    audio_bytes = audio.file.read()

    content_type = audio.content_type or "audio/webm"
    ext = content_type.split("/")[-1].split(";")[0]
    if ext not in {"webm", "ogg", "mp4", "wav", "mp3", "m4a", "mpeg", "flac"}:
        ext = "webm"

    # Hands-free clips are often just noise; the name hint would make Whisper say "Javid" for them.
    transcript = transcribe(audio_bytes, f"audio.{ext}", bias_name=(mode == "push"))
    request = transcript

    def ignored():
        yield _event(type="ignored", text=transcript)

    if mode in ("wake", "followup"):
        request = strip_wake_word(transcript)
        if request is None:
            if mode == "wake" or is_noise(transcript):
                print(f"[hands-free] ignored: {transcript!r}")
                return _ndjson(ignored())
            request = transcript  # follow-up: no wake word needed
        elif not request or is_noise(request):
            # Just "Hey Javid" — acknowledge and open a follow-up window.
            def ack():
                yield _event(type="transcript", text=transcript)
                yield _event(type="delta", text="Yes?")
                yield _event(type="done", tasks=None, memory=None)
            return _ndjson(ack())

    def events():
        yield _event(type="transcript", text=transcript)
        too_short = len(request.split()) < 2 if mode == "push" else not request
        if too_short:
            yield _event(type="delta", text="I didn't catch that clearly. Try again.")
            yield _event(type="done", tasks=_tasks_json(), memory=_memory_json())
            return
        yield from _reply_stream(request)

    return _ndjson(events())


@app.post("/type")
def type_message(body: TypeBody):
    return _ndjson(_reply_stream(body.text))


@app.get("/history")
def history(before: int | None = None, limit: int = 60):
    """Chat log, newest last. Page back with ?before=<index of oldest shown>."""
    entries = read_history()
    end = len(entries) if before is None else max(0, min(before, len(entries)))
    start = max(0, end - limit)
    return {
        "messages": [{"i": i, **entries[i]} for i in range(start, end)],
        "has_more": start > 0,
    }


@app.post("/reset")
def reset():
    reset_conversation()
    return {"ok": True}


@app.get("/calendar")
def calendar_agenda():
    return calendar_tools.agenda("today")


@app.post("/calendar/connect")
def calendar_connect():
    """Opens Google's sign-in page in the browser and waits (up to 5 min) for consent."""
    try:
        calendar_tools.connect()
    except Exception as e:
        print(f"[calendar] connect failed: {e!r}")
        raise HTTPException(500, "Could not connect Google Calendar")
    return calendar_tools.agenda("today")


@app.post("/speak")
def speak_text(body: TypeBody):
    """Neural voice for one sentence. 503 tells the browser to use its own voice."""
    try:
        audio, media_type = synthesize(body.text)
    except Exception as e:
        print(f"[tts] unavailable: {str(e)[:120]}")
        raise HTTPException(503, "TTS unavailable")
    return Response(audio, media_type=media_type)


# ---------- sidebar: direct edits, no LLM ----------
class TaskPatch(BaseModel):
    text: str | None = None
    done: bool | None = None


def _ok_or_404(ok: bool):
    if not ok:
        raise HTTPException(404, "Not found")


@app.get("/tasks")
def tasks():
    return _tasks_json()


@app.post("/tasks")
def create_task(body: TypeBody):
    if body.text.strip():
        add_task(body.text.strip())
    return _tasks_json()


@app.patch("/tasks/{number}")
def update_task(number: int, body: TaskPatch):
    if body.text is not None and body.text.strip():
        _ok_or_404(edit_task(number, body.text.strip()))
    if body.done is not None:
        _ok_or_404(set_task_done(number, body.done))
    return _tasks_json()


@app.delete("/tasks/{number}")
def remove_task(number: int):
    _ok_or_404(not delete_task(number).startswith(("No task", "No tasks")))
    return _tasks_json()


@app.get("/memory")
def memory():
    return _memory_json()


@app.patch("/memory/{index}")
def update_fact(index: int, body: TypeBody):
    if body.text.strip():
        _ok_or_404(edit_fact(index, body.text.strip()))
    return _memory_json()


@app.delete("/memory/{index}")
def remove_fact(index: int):
    _ok_or_404(delete_fact(index))
    return _memory_json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
