"""
Per-request latency traces -> data/traces.jsonl (the page's Latency panel, ARCHITECTURE.md timings).

Observation only: nothing here changes what Javid does or when it does it.

A trace starts when /talk or /type receives a request (or the CLI finishes recording). javid.py
marks stages on the *current* trace — one turn at a time, like javid.messages — and every call
here is a no-op when no trace is active (regression.py, scripts, the startup greeting). The page
adds what only it can see (end of speech, first audio, audio done) with POST /latency/<id>; the
line is written then, or after CLIENT_WAIT_SEC without it (closed tab), marked "no_client".

Times are epoch milliseconds while a trace is open (page and server share the clock: the page is
served from localhost) and offsets from t0 in the file. t0 is the end of speech for voice (the
last voiced frame, so hands-free's end-of-utterance silence counts), the moment Send was
pressed for typed messages.
"""

import json
import math
import os
import threading
import time
import uuid
from datetime import datetime

TRACES_FILE = os.path.join("data", "traces.jsonl")
CLIENT_WAIT_SEC = 120          # how long a finished server part waits for the page's part
STATS_WINDOW = 200             # latest traces the medians are computed over
WEB_TOOLS = {"web_search", "read_page", "news_headlines"}
TYPES = ("command", "chat", "web")

# Stages, in pipeline order. The page reports the starred ones.
STAGES = [
    "speech_end",        # * last voiced frame (push-to-talk: key/button released)
    "capture_end",       # * clip finalized (hands-free: after the end-of-utterance silence)
    "request_sent",      # * fetch started
    "request_received",  # /talk or /type entered (upload done)
    "stt_done",          # Whisper transcript back (voice only)
    "routing_done",      # tools chosen, code-decided replies settled; next is the first LLM request
    "llm_first_token",   # first chunk (text or tool call) of the first LLM request
    "llm_done",          # last LLM request finished
    "reply_out",         # reply text sent to the page (after the web/holiday/language checks)
    "server_done",       # stream closed
    "first_text",        # * reply text shown
    "tts_first_request", # * first sentence sent to /speak (it may be "Searching…")
    "first_audio",       # * first audio started playing
    "first_answer_audio",# * first sentence of the reply itself started playing
    "tts_done",          # * last sentence finished (or speech was stopped)
]
CLIENT_STAGES = {"speech_end", "capture_end", "request_sent", "first_text", "tts_first_request",
                 "first_audio", "first_answer_audio", "tts_done"}

_current = None
_pending: dict[str, "Trace"] = {}
_lock = threading.Lock()


def _ms() -> float:
    return time.time() * 1000


class Trace:
    def __init__(self, source: str, mode: str, label: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.source, self.mode, self.label = source, mode, label[:60]
        self.stages = {"request_received": _ms()}
        self.llm: list[dict] = []
        self.tools: list[dict] = []
        self.outcome = None
        self.finished_at = None

    def mark(self, name: str, at: float | None = None) -> None:
        self.stages.setdefault(name, at if at is not None else _ms())   # first time counts

    @property
    def type(self) -> str:
        names = {t["name"] for t in self.tools}
        if names & WEB_TOOLS:
            return "web"
        return "command" if names or not self.llm else "chat"   # no LLM request: a reply decided in code

    def record(self) -> dict:
        st = dict(self.stages)
        if self.llm:
            st.setdefault("llm_first_token", next((s["first"] for s in self.llm if s.get("first")), None))
            st["llm_done"] = self.llm[-1].get("end")
        st = {k: v for k, v in st.items() if v is not None}
        t0 = st.get("speech_end") or st.get("request_sent") or st["request_received"]
        rel = lambda v: round(v - t0) if v is not None else None
        out = {
            "id": self.id,
            "ts": datetime.fromtimestamp(t0 / 1000).isoformat(timespec="seconds"),
            "source": self.source, "mode": self.mode, "type": self.type,
            "outcome": self.outcome or "no_client", "label": self.label,
            "t0": round(t0),
            "stages": {k: rel(st[k]) for k in STAGES if k in st},
            "llm": [{"model": s.get("model"), "start": rel(s["start"]), "first": rel(s.get("first")),
                     "end": rel(s.get("end"))} for s in self.llm],
            "tools": [{"name": t["name"], "start": rel(t["start"]), "end": rel(t.get("end"))} for t in self.tools],
        }
        out["ttfa"] = out["stages"].get("first_audio")
        return out


# ---------- recording (no-ops without a current trace) ----------
def start(source: str, mode: str, label: str = "") -> Trace:
    return Trace(source, mode, label)


def activate(trace: Trace | None) -> None:
    global _current
    _current = trace


def current() -> Trace | None:
    return _current


def mark(name: str) -> None:
    if _current:
        _current.mark(name)


def llm_start() -> dict | None:
    if not _current:
        return None
    span = {"start": _ms()}
    _current.llm.append(span)
    return span


def llm_first(span: dict | None) -> None:
    if span is not None and "first" not in span:
        span["first"] = _ms()


def llm_end(span: dict | None, model: str | None) -> None:
    if span is not None:
        span["end"] = _ms()
        span["model"] = model


def tool_start(name: str) -> dict | None:
    if not _current:
        return None
    span = {"name": name, "start": _ms()}
    _current.tools.append(span)
    return span


def tool_end(span: dict | None) -> None:
    if span is not None:
        span["end"] = _ms()


# ---------- finishing ----------
def finish(trace: Trace) -> None:
    """The server's part is done: wait for the page's stages (or write now if it already sent them)."""
    global _current
    if _current is trace:
        _current = None
    trace.mark("server_done")
    trace.finished_at = time.time()
    with _lock:
        if trace.outcome:                       # the page reported before the stream closed
            _write(trace)
        else:
            _pending[trace.id] = trace
        _flush_stale()


def add_client(trace_id: str, stages: dict, outcome: str) -> bool:
    """The page's stages (epoch ms) for a trace. False if it's unknown (already written, or a restart)."""
    with _lock:
        trace = _pending.pop(trace_id, None) or (_current if _current and _current.id == trace_id else None)
        if not trace:
            return False
        for k, v in stages.items():
            if k in CLIENT_STAGES and isinstance(v, (int, float)):
                trace.stages[k] = float(v)
        trace.outcome = outcome
        if trace.finished_at:
            _write(trace)
        return True


def finish_local(trace: Trace, outcome: str = "done") -> None:
    """CLI: the caller measured the client stages itself, so write straight away."""
    trace.outcome = outcome
    finish(trace)


def flush_all() -> None:
    with _lock:
        for t in list(_pending.values()):
            _write(t)
        _pending.clear()


def _flush_stale() -> None:
    now = time.time()
    for tid, t in list(_pending.items()):
        if now - t.finished_at > CLIENT_WAIT_SEC:
            _write(_pending.pop(tid))


def _write(trace: Trace) -> None:
    try:
        os.makedirs(os.path.dirname(TRACES_FILE), exist_ok=True)
        with open(TRACES_FILE, "a") as f:
            f.write(json.dumps(trace.record(), ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[trace] not saved: {e}")


# ---------- reading (Latency panel, ARCHITECTURE.md) ----------
def read(limit: int | None = None) -> list[dict]:
    try:
        with open(TRACES_FILE) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines[-limit:] if limit else lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100) — a value that actually occurred."""
    if not values:
        return None
    v = sorted(values)
    return v[max(0, min(len(v) - 1, math.ceil(len(v) * p / 100) - 1))]


# Segments shown as bar colours and summarised per type: (name, start stage(s), end stage). A tuple
# of starts means the first one present (voice has stt_done / capture_end, typed requests don't).
SEGMENTS = [
    ("end-of-speech wait", ("speech_end",), "capture_end"),
    ("upload", ("capture_end", "request_sent"), "request_received"),
    ("speech-to-text", ("request_received",), "stt_done"),
    ("routing", ("stt_done", "request_received"), "routing_done"),
    ("LLM + tools", ("routing_done",), "llm_done"),
    ("checks", ("llm_done",), "reply_out"),
    ("reply to audio", ("reply_out",), "first_answer_audio"),   # first sentence's TTS + playback start
    ("speaking", ("first_answer_audio",), "tts_done"),
]


def segment(t: dict, name: str) -> float | None:
    s = t["stages"]
    for n, starts, end in SEGMENTS:
        if n == name:
            start = next((a for a in starts if a in s), None)
            return s[end] - s[start] if start and end in s else None
    return None


def stats(traces: list[dict] | None = None) -> dict:
    """Median and 90th percentile time to first audio per request type, over the latest traces."""
    traces = read(STATS_WINDOW) if traces is None else traces
    out = {}
    for kind in TYPES:
        ts = [t for t in traces if t.get("type") == kind]
        ttfa = [t["ttfa"] for t in ts if t.get("ttfa") is not None]
        answer = [t["stages"]["first_answer_audio"] for t in ts if "first_answer_audio" in t["stages"]]
        out[kind] = {
            "n": len(ts), "n_audio": len(ttfa),
            "ttfa_median": percentile(ttfa, 50), "ttfa_p90": percentile(ttfa, 90),
            "answer_median": percentile(answer, 50), "answer_p90": percentile(answer, 90),
            "segments": {name: percentile([v for t in ts if (v := segment(t, name)) is not None], 50)
                         for name, _, _ in SEGMENTS},
        }
    return out
