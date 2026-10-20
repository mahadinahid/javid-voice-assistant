"""
Javid web server — wraps javid.py in a FastAPI backend.
Run: python server.py  →  open http://localhost:8000
"""

import json
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import agents
import calendar_tools
import memory_store
import skills
import timers
import tracing
import web_tools
from javid import (
    _SMALL_TALK, _read_lines, add_task, delete_task, edit_task,
    clear_provider_key, get_language, is_noise, llm_status, phrase,
    ping_models, read_history, reset_conversation, set_chain, set_language, set_provider_key,
    _restore_context, current_session, delete_session, list_sessions, new_session, open_session, update_session,
    set_task_done, startup_greeting, strip_wake_word, synthesize, think_stream, transcribe_turn, warm_phrases,
)


# ---------- startup ----------
def _warm_tts():
    try:
        warm_phrases()
    except Exception as e:
        print(f"[tts] warm-up failed: {str(e)[:80]}")


def _check_models():
    """Tiny ping to each model at startup, so the Models panel starts out truthful."""
    try:
        ping_models()
    except Exception as e:
        print(f"[llm] startup check failed: {str(e)[:80]}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _restore_context()      # reopen the chat that was open (and split an old single log into chats, once)
    print("\nJavid is up. Open http://localhost:8000\n")
    # The first TTS call after startup is slow (cold connection); warm it and regenerate the
    # cache of fixed phrases ("Yes?" first) so the first "Hey Javid" is quick.
    threading.Thread(target=_warm_tts, daemon=True).start()
    threading.Thread(target=_check_models, daemon=True).start()
    yield
    tracing.flush_all()     # traces still waiting for the page's stages


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
    """Both memory files plus proposals waiting for approval (the sidebar shows all three)."""
    return {**memory_store.all_entries(),
            "pending": [{k: p[k] for k in ("id", "text", "file")} for p in memory_store.pending()]}


class TypeBody(BaseModel):
    text: str


class ChatBody(BaseModel):
    text: str
    session: str | None = None    # the chat this message belongs to (the page's open chat)


def _use_session(sid: str | None) -> None:
    """Messages go to the chat the page has open, even if another tab switched the server's."""
    if sid and sid != current_session() and not open_session(sid):
        raise HTTPException(404, "No such chat")


class SpeakBody(BaseModel):
    text: str     # the voice comes from the server's language setting, never the page's copy of it


class LanguageBody(BaseModel):
    lang: str


# ---------- routes ----------
@app.get("/")
def root():
    return FileResponse("index.html")


def _event(**data) -> str:
    return json.dumps(data) + "\n"


def _reply_stream(user_text: str):
    """Stream Javid's reply as newline-delimited JSON events."""
    try:
        for kind, value in think_stream(user_text):
            if kind == "tool":
                yield _event(type="tool", name=value)
            elif kind == "say":
                yield _event(type="say", text=value)
            elif kind == "sources":
                yield _event(type="sources", items=value)
            elif kind == "lang":
                yield _event(type="lang", lang=value)
            elif kind == "skill":
                yield _event(type="skill", name=value)
            elif kind == "screen":
                yield _event(type="screen", text=value)
            elif kind == "memory_proposal":
                yield _event(type="memory_proposal", id=value["id"], text=value["text"], file=value["file"])
            elif kind == "memory":
                yield _event(type="memory", outcome=value["outcome"], id=value["id"], text=value["text"])
            elif kind == "agent":
                yield _event(type="agent", name=value)
            elif kind == "delta":
                yield _event(type="delta", text=value)
            # any other kind is internal: never sent, and never spoken as reply text
    except Exception as e:
        print(f"[error] {e!r}")
        yield _event(type="error", message="Something went wrong on my end.")
    yield _event(type="done", tasks=_tasks_json(), memory=_memory_json(), lang=get_language(),
                 timers=timers.active_json())


def _ndjson(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="application/x-ndjson")


def _traced(trace: tracing.Trace, gen):
    """The reply stream, passed through unchanged apart from a first "trace" event (its id, so the
    page can add its own stages); javid.py marks the stages in between on this, the current trace."""
    tracing.activate(trace)
    try:
        yield _event(type="trace", id=trace.id)
        for line in gen:
            if line.startswith('{"type": "delta"'):
                trace.mark("reply_out")
            yield line
    finally:
        tracing.finish(trace)


@app.get("/greeting")
def greeting():
    return {"text": startup_greeting()}


@app.post("/talk")
def talk(audio: UploadFile = File(...), mode: str = Form("push"), session: str | None = Form(None)):
    """
    mode = "push"     push-to-talk: always answer
           "wake"     hands-free, idle: only answer if it starts with "Javid"
           "followup" hands-free, right after Javid spoke: no wake word needed
    """
    _use_session(session)
    trace = tracing.start("voice", mode)
    audio_bytes = audio.file.read()

    content_type = audio.content_type or "audio/webm"
    ext = content_type.split("/")[-1].split(";")[0]
    if ext not in {"webm", "ogg", "mp4", "wav", "mp3", "m4a", "mpeg", "flac"}:
        ext = "webm"

    # Hands-free clips are often just noise; the name hint would make Whisper say "Javid" for them.
    transcript = transcribe_turn(audio_bytes, f"audio.{ext}", bias_name=(mode == "push"))
    trace.mark("stt_done")
    trace.label = transcript[:60]
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
                yield _event(type="delta", text=phrase("ack"))
                yield _event(type="done", tasks=None, memory=None, lang=get_language())
            return _ndjson(_traced(trace, ack()))

    if mode == "push" and strip_wake_word(request) == "":
        # Just "Javid": acknowledge, as in hands-free.
        def ack_push():
            yield _event(type="transcript", text=transcript)
            yield _event(type="delta", text=phrase("ack"))
            yield _event(type="done", tasks=None, memory=None, lang=get_language())
        return _ndjson(_traced(trace, ack_push()))

    def events():
        yield _event(type="transcript", text=transcript)
        # One word is usually a clipped recording, but a plain yes/no (memory approval) or a
        # greeting ("Hello") is a real message.
        answer = (memory_store._YES.match(request) or memory_store._NO.match(request)
                  or _SMALL_TALK.match(request.strip()))
        too_short = (len(request.split()) < 2 and not answer) if mode == "push" else not request
        if too_short:
            yield _event(type="delta", text=phrase("not_caught"))
            yield _event(type="done", tasks=_tasks_json(), memory=_memory_json(), lang=get_language())
            return
        yield from _reply_stream(request)

    return _ndjson(_traced(trace, events()))


@app.post("/type")
def type_message(body: ChatBody):
    _use_session(body.session)
    return _ndjson(_traced(tracing.start("text", "type", body.text), _reply_stream(body.text)))


@app.get("/history")
def history(session: str | None = None, before: int | None = None, limit: int = 60):
    """One chat's messages (default: the open one), newest last. Page back with ?before=<index of oldest shown>."""
    entries = read_history(session)
    end = len(entries) if before is None else max(0, min(before, len(entries)))
    start = max(0, end - limit)
    return {
        "messages": [{"i": i, **entries[i]} for i in range(start, end)],
        "has_more": start > 0,
        "session": session or current_session(),
    }


@app.post("/reset")
def reset():
    reset_conversation()
    return {"ok": True, "session": current_session()}


# ---------- chat sessions (the Chats panel) ----------
class SessionPatch(BaseModel):
    title: str | None = None
    pinned: bool | None = None


@app.get("/sessions")
def sessions(q: str = ""):
    return {"active": current_session(), "sessions": list_sessions(q)}


@app.post("/sessions")
def sessions_new():
    return new_session()


@app.post("/sessions/{sid}/open")
def sessions_open(sid: str):
    _ok_or_404(open_session(sid))
    return {"ok": True, "session": sid}


@app.patch("/sessions/{sid}")
def sessions_update(sid: str, body: SessionPatch, request: Request):
    _local_page_only(request)
    meta = update_session(sid, body.title, body.pinned)
    _ok_or_404(meta is not None)
    return meta


@app.delete("/sessions/{sid}")
def sessions_delete(sid: str, request: Request):
    """Deletes the chat and its messages for good (asked twice in the page)."""
    _local_page_only(request)
    _ok_or_404(delete_session(sid))
    return {"ok": True, "active": current_session()}


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
def speak_text(body: SpeakBody):
    """The locked voice for one sentence, chosen here from the language setting: a tab's own copy of the
    language can be stale (another tab or the CLI switched it), and English sent with "bn" was read by
    Pradeep. 503 = edge-tts failed every try: the page shows the text with a "voice unavailable" notice
    (there is no other voice)."""
    try:
        audio, media_type = synthesize(body.text)
    except Exception as e:
        print(f"[tts] unavailable: {str(e)[:120]}")
        raise HTTPException(503, "TTS unavailable")
    return Response(audio, media_type=media_type)


# ---------- barge-in log (index.html posts every trigger and whether speech followed) ----------
BARGEIN_LOG = "bargein_log.jsonl"


@app.post("/bargein/log")
async def bargein_log(request: Request):
    """False triggers (Javid cut himself off on his own echo or a cough) are what to look for here."""
    body = await request.body()
    if len(body) > 4000:
        raise HTTPException(413, "Too large")
    try:
        entry = json.loads(body)
    except ValueError:
        raise HTTPException(400, "Not JSON")
    with open(BARGEIN_LOG, "a") as f:
        f.write(json.dumps({k: entry.get(k) for k in ("at", "outcome", "transcript", "micRms", "playRms", "thr", "echoK")},
                           ensure_ascii=False) + "\n")
    if entry.get("outcome") == "false":
        print(f"[barge-in] false trigger: mic {entry.get('micRms')}, playback {entry.get('playRms')}, threshold {entry.get('thr')}")
    return {"ok": True}


# ---------- latency traces (Latency panel) ----------
class TraceBody(BaseModel):
    stages: dict[str, float] = {}
    outcome: str = "done"


@app.post("/latency/{trace_id}")
def latency_report(trace_id: str, body: TraceBody, request: Request):
    """The page's stages for one request (end of speech, first audio, audio done…), epoch ms."""
    _local_page_only(request)
    outcome = body.outcome if body.outcome in ("done", "interrupted", "stopped", "superseded") else "done"
    return {"ok": tracing.add_client(trace_id[:32], body.stages, outcome)}


@app.get("/latency")
def latency(limit: int = 20):
    traces = tracing.read(tracing.STATS_WINDOW)
    return {"recent": traces[-max(1, min(limit, 100)):], "stats": tracing.stats(traces),
            "segments": [name for name, _, _ in tracing.SEGMENTS], "window": tracing.STATS_WINDOW}


# ---------- skills (sidebar list) ----------
@app.get("/agents")
def agents_list():
    return agents.all_agents()


# Timers and reminders: the page rings them and reports back; only the first tab to report announces one.
@app.get("/timers")
def timers_list():
    return timers.active_json()


@app.post("/timers/{tid}/fired")
def timers_fired(tid: str, request: Request):
    _local_page_only(request)
    return {"first": timers.mark_fired(tid)}


@app.delete("/timers/{tid}")
def timers_cancel(tid: str, request: Request):
    _local_page_only(request)
    if not timers.cancel(tid):
        raise HTTPException(404, "No such timer")
    return {"ok": True}


@app.get("/skills")
def skills_list():
    return skills.all_skills()


# ---------- reply language (English / Bangla) ----------
@app.get("/language")
def language():
    return {"lang": get_language()}


@app.put("/language")
def language_set(body: LanguageBody, request: Request):
    _local_page_only(request)
    try:
        set_language(body.lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"lang": get_language()}


# ---------- model chain status (Models panel) ----------
@app.get("/llm")
def llm():
    return llm_status()


@app.post("/llm/ping")
def llm_ping():
    """Tiny live request to each model (a few tokens each) to see who answers right now."""
    pings = ping_models()
    return {"pings": pings, "status": llm_status()}


def _local_page_only(request: Request):
    """Changing keys or the model order must come from this page, not from another website
    that happens to POST to localhost (browsers always send Origin on such requests)."""
    host = request.headers.get("host", "")
    if host not in ("localhost:8000", "127.0.0.1:8000") or request.headers.get("origin") != f"http://{host}":
        raise HTTPException(403, "Only the Javid page may change this.")


class ChainBody(BaseModel):
    chain: list[str] | None = None      # None = back to the default order


class KeyBody(BaseModel):
    key: str


@app.put("/llm/chain")
def llm_set_chain(body: ChainBody, request: Request):
    _local_page_only(request)
    try:
        set_chain(body.chain)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return llm_status()


@app.put("/llm/key/{provider}")
def llm_set_key(provider: str, body: KeyBody, request: Request):
    """Verify the key with the provider, save it to .env, and test that provider's models."""
    _local_page_only(request)
    try:
        set_provider_key(provider, body.key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"pings": ping_models(only_provider=provider), "status": llm_status()}


@app.delete("/llm/key/{provider}")
def llm_clear_key(provider: str, request: Request):
    _local_page_only(request)
    try:
        clear_provider_key(provider)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return llm_status()


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


def _memory_error(e: Exception):
    raise HTTPException(400, str(e))


@app.patch("/memory/{file}/{index}")
def update_memory(file: str, index: int, body: TypeBody, request: Request):
    """Edit a line by hand. Secrets are refused here too (400)."""
    _local_page_only(request)   # only this page may change memory (no cross-site approvals)
    try:
        if body.text.strip():
            _ok_or_404(memory_store.edit(file, index, body.text))
    except ValueError as e:
        _memory_error(e)
    return _memory_json()


@app.delete("/memory/{file}/{index}")
def remove_memory(file: str, index: int, request: Request):
    _local_page_only(request)   # only this page may change memory (no cross-site approvals)
    try:
        _ok_or_404(memory_store.delete(file, index))
    except ValueError as e:
        _memory_error(e)
    return _memory_json()


class ApproveBody(BaseModel):
    text: str | None = None     # edited text; None = as proposed


@app.post("/memory/proposals/{pid}/approve")
def approve_memory(pid: str, body: ApproveBody, request: Request):
    _local_page_only(request)   # only this page may change memory (no cross-site approvals)
    try:
        memory_store.approve(pid, body.text)
    except KeyError:
        raise HTTPException(404, "No such proposal")
    except ValueError as e:
        _memory_error(e)
    return _memory_json()


@app.post("/memory/proposals/{pid}/reject")
def reject_memory(pid: str, request: Request):
    _local_page_only(request)   # only this page may change memory (no cross-site approvals)
    _ok_or_404(memory_store.reject(pid))
    return _memory_json()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
