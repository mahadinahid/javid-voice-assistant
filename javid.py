"""
Javid v1.0 — push-to-talk voice assistant (7-day build, complete)
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> edge-tts en-US-AndrewNeural (voice)
Day 7 polish: startup greeting, graceful Ctrl+C, network-safe search, file-handle fix
"""

import asyncio
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import types
import uuid
import wave
from datetime import datetime, timedelta

import edge_tts
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
import groq as _groq
import openai as _openai
from groq import APIError, BadRequestError, Groq, InternalServerError, NotFoundError, RateLimitError

import about
import agents
import calc
import calendar_tools
import markets
import memory_store
import rates
import skills
import timers
import tracing
import weather
import web_tools

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
# endpoint (needs GEMINI_API_KEY), "deepseek:<model>" to DeepSeek's (needs DEEPSEEK_API_KEY, paid) and
# "ifm:<model>" to IFM's (MBZUAI) K2 Horizon API at api.ifm.ai (needs IFM_API_KEY).
# A provider without its key in .env is skipped.
# Highest model that answered on the free tier on 2026-10-13 (tool calls OK). gemini-3.8-flash is newer
# but returned 503 "high demand" on every try; gemini-3-flash-preview is much faster (~2 s vs 6-8 s).
GEMINI_MODEL = "gemini-3.7-flash"
DEEPSEEK_MODEL = "deepseek-chat"
# The only K2 Horizon size api.ifm.ai served on 2026-10-16 (the others are open weights only).
IFM_MODEL = "IFM/K2-Horizon-375B-A23B"
LLM_FALLBACKS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b",
                 f"gemini:{GEMINI_MODEL}", f"deepseek:{DEEPSEEK_MODEL}", f"ifm:{IFM_MODEL}"]
# Daily token caps seen in Groq 429 messages; others are learned from the first 429 (llm_quota.json).
KNOWN_TPD_LIMITS = {"openai/gpt-oss-20b": 200000}
LLM_MIN_REQUEST_TOKENS = 1500   # one request is ~0.6-1.7k tokens (prompt + the tools sent); a turn ~2 requests
# LOCKED by the user: approved as "perfect" on 2026-10-13, made permanent on 2026-10-14 ("set this
# voice for everything, don't change it again"). Every spoken word, web UI and CLI, is this voice
# via edge-tts. There is deliberately no other engine and no fallback voice: if edge-tts fails,
# the reply is shown as text, not spoken by something else. Never change this unless told to.
EDGE_VOICE = "en-US-AndrewNeural"
# Bangla mode's voice, LOCKED the same way (added 2026-10-14 when the user asked for Bangla mode):
# AndrewNeural turns Bengali script into near-silence, so Bengali needs its own voice. Male bn-BD, to
# match Andrew. Never change it or add a substitute unless told to.
EDGE_VOICE_BN = "bn-BD-PradeepNeural"
VOICES = {"en": EDGE_VOICE, "bn": EDGE_VOICE_BN}
LANGUAGE_FILE = "language.json"      # {"lang": "en" | "bn"}: the reply language, kept across reloads
TASKS_FILE = "tasks.txt"
HISTORY_FILE = "chat_log.jsonl"
LLM_USAGE_FILE = "llm_usage.jsonl"   # one line per LLM request: tokens by model (for the Models panel)
LLM_TURNS_FILE = "llm_turns.jsonl"   # one line per conversation turn: tokens summed over its requests
LLM_QUOTA_FILE = "llm_quota.json"    # daily token limits Groq has told us about
LLM_CHAIN_FILE = "llm_chain.json"    # model order chosen in the Models panel (absent = defaults below)
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
CONTEXT_TURNS = 20      # past messages Javid reloads into context on restart
HISTORY_TURNS = 6       # past turns sent in full with each request; older ones only as a short summary

load_dotenv()
client = Groq()


# ---------- reply language (English / Bangla) ----------
# Set in code, by a voice command or the page's toggle, and re-read every turn, so the language
# never depends on the model remembering an earlier "speak Bangla".
LANGUAGES = ("en", "bn")


def get_language() -> str:
    try:
        with open(LANGUAGE_FILE) as f:
            lang = json.load(f).get("lang")
    except (OSError, ValueError, AttributeError):
        return "en"
    return lang if lang in LANGUAGES else "en"


def set_language(lang: str) -> None:
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}")
    with open(LANGUAGE_FILE, "w") as f:
        json.dump({"lang": lang}, f)


# Fixed lines Javid says without the model, in each language.
PHRASES = {
    "mem_saved":  {"en": "Saved. I'll remember that.", "bn": "সেভ করেছি। মনে রাখব।"},
    "mem_dropped": {"en": "Okay, I won't save that.", "bn": "ঠিক আছে, ওটা সেভ করব না।"},
    "switched":   {"en": "Sure, Mahadi. I'll speak English from now on.",
                   "bn": "ঠিক আছে মাহাদি, এখন থেকে আমি বাংলায় কথা বলব।"},
    "ack":        {"en": "Yes?", "bn": "হ্যাঁ, বলো?"},
    "searching":  {"en": web_tools.SEARCHING, "bn": "খুঁজে দেখছি…"},
    "working":    {"en": "On it. Give me a moment.", "bn": "করছি, একটু সময় দাও।"},   # while an agent works
    "not_caught": {"en": "I didn't catch that clearly. Try again.", "bn": "ঠিক বুঝতে পারিনি। আবার বলো।"},
    "stuck":      {"en": "Sorry, I got stuck on that one. Try asking another way.",
                   "bn": "দুঃখিত, এটায় আটকে গেলাম। অন্যভাবে বলে দেখো।"},
    "api_failed": {"en": "I couldn't complete that request. Try asking differently.",
                   "bn": "অনুরোধটা শেষ করতে পারলাম না। অন্যভাবে জিজ্ঞেস করে দেখো।"},
    "garbled":    {"en": "Sorry, I lost my train of thought. Could you ask that again?",
                   "bn": "দুঃখিত, গুলিয়ে ফেলেছি। আরেকবার বলবে?"},
    "holiday_unsure": {"en": "I couldn't confirm those holiday dates. Ask me again.",
                       "bn": "ছুটির তারিখগুলো নিশ্চিত করতে পারিনি। আবার জিজ্ঞেস করো।"},
}


def phrase(key: str, lang: str | None = None) -> str:
    return PHRASES[key][lang or get_language()]


# "Bangla te kotha bolo" arrives romanized from English Whisper, in Bengali script from Bangla
# Whisper or the keyboard, or as plain English ("let's speak Bangla", "switch back to English").
_LANG_WORD = {"bn": r"(?:bangla|bengali|bangali|বাংলা)",
              "en": r"(?:english|ইংলি[শস]|ই[ংন]\u09cd?রে[জসযদ][িী])"}   # ইংরেজি as Bangla Whisper hears it: ইন্রেসি, ইনরেদি...
_SPEAK = r"(?:speak|talk|reply|answer|respond|chat|switch|change)"
_SPEAK_FILLER = r"(?:\s+(?:back|to|with|me|us|in|into|only|now|please|just|always|again))*"
# English Whisper writes "kotha bolo" any way it likes: "kotha bolo", "Cotabioello"...
_KOTHA = r"[ck][oa]u?t+h?[ao]\s*-?\s*b\w{0,3}l"


def _switch_patterns(lang: str) -> list[re.Pattern]:
    word = _LANG_WORD[lang]
    return [re.compile(p, re.I) for p in (
        rf"\b{_SPEAK}{_SPEAK_FILLER}\s+{word}",                    # speak Bangla, talk to me in English
        rf"\bswitch\b.{{0,30}}?\bto\s+{word}",                   # switch from English to Bangla
        rf"{word}\s+mode\b",
        rf"{word}\W*(?:(?:te|tay|tey|t|y|e|ey|a)\b\W*)?{_KOTHA}",   # Bangla te kotha bolo, "Bangla T. Cotabioello"
        rf"{word}(?:\u09df|\u09af\u09bc|তে|য|ে|এ)?\s*(?:কথা\s*)?বল",              # বাংলায় কথা বলো, ইংরেজিতে বলো
        rf"{word}\S{{0,3}}\s+\S{{1,6}}\s+[বপ]\S{{0,2}}ল",                  # misheard কথা বলো: "ইন্রেসি ত্সা পালো"
        rf"স্পি?ী?ক\s*{word}",                                     # স্পিক ইংলিশ (English heard as Bangla)
    )]


_SWITCHES = {lang: _switch_patterns(lang) for lang in LANGUAGES}
_SWITCH_FILLER = {"let's", "lets", "let", "us", "please", "ok", "okay", "now", "from", "on", "javid", "hey",
                  "hi", "can", "could", "you", "will", "with", "me", "only", "just", "always", "again", "back",
                  "the", "in", "to", "and", "so", "then", "mode", "করে", "এখন", "থেকে", "প্লিজ", "ঠিক", "আছে",
                  "আমার", "সাথে", "দয়া", "তুমি", "আপনি", "একটু", "জাভেদ", "জাভিদ", "জাবেদ"}


def language_switch(text: str) -> tuple[str, bool] | None:
    """(language, nothing else asked) if the message switches the reply language, else None.
    When both languages match ("switch from English to Bangla"), the one named last wins."""
    hits = [(m.end(), lang, m) for lang in LANGUAGES for p in _SWITCHES[lang] if (m := p.search(text))]
    if not hits:
        return None
    _, lang, m = max(hits, key=lambda h: h[0])
    rest = text[:m.start()] + " " + text[m.end():]
    for _, _, other in hits:
        rest = rest.replace(other.group(0), " ")
    words = [w for w in re.findall(r"[\w\u0980-\u09ff']+", rest.lower()) if w not in _SWITCH_FILLER]
    return lang, len(words) <= 2


_BN_CHARS = re.compile(r"[\u0980-\u09ff]")
_LATIN = re.compile(r"[A-Za-z]")


def wrong_language(text: str, lang: str) -> bool:
    """A reply mostly in the other script. Names like "Google Calendar" or "3 PM" in a Bangla reply
    are fine; a whole English reply is not (nor Bengali text while English is set)."""
    bn, latin = len(_BN_CHARS.findall(text)), len(_LATIN.findall(text))
    if not bn + latin:
        return False
    share = bn / (bn + latin)
    return share < 0.5 if lang == "bn" else share > 0.2


def language_note(lang: str) -> str:
    if lang == "bn":
        return ("That reply wasn't in Bangla. Mahadi's language is set to Bangla: rewrite it in Bangla "
                "(Bengali script), at most 2 sentences.")
    return "That reply wasn't in English. The language is set to English: rewrite it in English, at most 2 sentences."


# Static on purpose: identical bytes on every request let the provider cache this prefix.
# Anything that changes (time, facts, older-chat summary) goes in context_block(), which rides on
# the newest user message; rules for one tool live in that tool's description, which is only sent
# when the tool is (see select_tools).
SYSTEM_PROMPT = (
    "You are Javid, Mahadi's personal voice assistant: a calm, confident friend, not a corporate bot. "
    "Your replies are spoken aloud: short sentences, no markdown, lists or emojis. "
    "Reply in the current language named in the [Context] note (Bangla means Bengali script), in at most 2 spoken sentences. "
    "Use Mahadi's name occasionally. Say times like '3 PM', not '15:00'. "
    "For the current time, always call get_datetime. "
    "Tool results are never shown to Mahadi, so say what matters from them. "
    "Never state task counts, calendar contents, or that something was added, saved, remembered, "
    "moved or deleted unless a tool result in this turn says so (to keep a fact, call propose_memory: it's saved only once Mahadi approves), "
    "and don't say you checked unless you called a tool. "
    "If you're not sure, say so instead of guessing. "
    "Weather, prices and sums come only from get_weather, market_price and calculate, and a timer exists only "
    "once set_timer says so. "
    "Mahadi lives in Bangladesh, where the weekend is Friday and Saturday. "
    "Calendar: when the day and time are given, add the event straight away. To move or delete one, "
    "call move_event or delete_event with part of its title; the tool finds it, so don't ask for its "
    "current day or list events first. delete_event never deletes on the first call: tell Mahadi what "
    "it found and ask yes or no; after a yes, call it again with confirm=true. "
    "The newest message starts with a [Context] note: the reply language, today's date, a table of the "
    "next days (look weekdays up there, don't compute them), and older chat. What you know about Mahadi "
    "is in the [Memory] block below; for what was said in earlier chats, call search_history."
) + skills.prompt_index()   # skill names and descriptions only (static until restart)


def context_block(older_summary: str = "", lang: str = "en", facts: str = "") -> str:
    """The per-turn facts the model needs, kept out of the cached system prompt."""
    now = datetime.now()
    days = ", ".join((now + timedelta(days=i)).strftime("%a %b %d") for i in range(14))
    # Date only: the clock comes from get_datetime, and a date changes once a day, not every minute.
    language = ("Bangla (write Bengali script, even if earlier replies were English; in Mahadi's questions "
                "কাল/আগামীকাল = tomorrow, আজ = today, পরশু = the day after tomorrow)") if lang == "bn" else "English"
    parts = [f"Reply language: {language}.", f"Today: {now.strftime('%A, %Y-%m-%d')}.", f"Next 14 days: {days}."]
    if older_summary:
        parts.append(older_summary)
    if facts:
        parts.append(facts)
    return "[Context] " + " ".join(parts) + "\n\n"


messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# ---------- chat sessions and history (persist across restarts) ----------
# chat_log.jsonl: one JSON object per line, {"ts", "session", "role": "user"|"assistant", "text", ...}.
# sessions.json: {"active": id, "sessions": {id: {"title", "created", "updated", "pinned", "named"}}}.
# Each session is its own conversation: `messages` holds only the open one, so Javid's context
# (and the 6-turn window) never mixes chats. search_history still searches all of them.
SESSIONS_FILE = "sessions.json"
TITLE_WORDS = 7
_active = {"id": None}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    return f"s{datetime.now():%Y%m%d%H%M%S}{uuid.uuid4().hex[:4]}"


# One turn at a time: the page's tabs and Telegram share `messages` and the open chat. A plain Lock, not an RLock:
# a web reply is a generator the server may resume on another thread. A turn that never lets go (a stream left
# hanging) is waited out for TURN_WAIT_SEC, then ignored rather than locking Javid up.
TURN_LOCK = threading.Lock()
TURN_WAIT_SEC = 120


@contextlib.contextmanager
def hold_turn():
    got = TURN_LOCK.acquire(timeout=TURN_WAIT_SEC)
    if not got:
        print(f"[turn] the last turn held on for {TURN_WAIT_SEC}s; going ahead without it")
    try:
        yield
    finally:
        if got:
            TURN_LOCK.release()


def active_session() -> str | None:
    """The open chat's id, without creating one (current_session() does)."""
    return _active["id"]


def has_session(sid: str) -> bool:
    return sid in _load_sessions()["sessions"]


def create_session(title: str) -> str:
    """A new chat with this title, not opened (Telegram's own chat)."""
    data = _load_sessions()
    sid = _new_id()
    data["sessions"][sid] = {"title": title, "created": _now(), "updated": _now(), "pinned": False, "named": True}
    _save_sessions(data)
    return sid


def _read_log() -> list[dict]:
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


def _write_log(entries: list[dict]) -> None:
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    os.replace(tmp, HISTORY_FILE)


def _load_sessions() -> dict:
    try:
        with open(SESSIONS_FILE) as f:
            data = json.load(f)
        if isinstance(data.get("sessions"), dict):
            return data
    except (OSError, ValueError, AttributeError):
        pass
    return {"active": None, "sessions": {}}


def _save_sessions(data: dict) -> None:
    tmp = SESSIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, SESSIONS_FILE)


def auto_title(text: str) -> str:
    """A title from the first message, in Python (no LLM request): its first few words."""
    text = strip_wake_word(text) or text
    first = re.split(r"(?<=[.!?\u0964])\s", " ".join(text.split()), maxsplit=1)[0].rstrip(".!?\u0964 ")
    words = first.split()
    title = " ".join(words[:TITLE_WORDS]) + ("…" if len(words) > TITLE_WORDS else "")
    return (title[:1].upper() + title[1:])[:60] or "New chat"


def _migrate_sessions() -> None:
    """v1 logs had no sessions, just "divider" lines between conversations: split the log there, once.
    The old file is kept as chat_log.jsonl.bak-<time>."""
    entries = _read_log()
    if not any("session" not in e for e in entries):
        return
    data = _load_sessions()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(HISTORY_FILE) as src, open(f"{HISTORY_FILE}.bak-{stamp}", "w") as dst:
        dst.write(src.read())
    out, sid = [], None
    for e in entries:
        if "session" in e:
            out.append(e)
            continue
        if e.get("role") == "divider":
            sid = None
            continue
        if sid is None:
            sid = f"s{e.get('ts', _now()).replace('-', '').replace(':', '').replace('T', '')[:14]}{uuid.uuid4().hex[:4]}"
            data["sessions"][sid] = {"title": "", "created": e.get("ts", _now()), "updated": e.get("ts", _now()),
                                     "pinned": False, "named": False}
        meta = data["sessions"][sid]
        meta["updated"] = e.get("ts", meta["updated"])
        if e.get("role") == "user" and not meta["title"]:
            meta["title"] = auto_title(e.get("text", ""))
        out.append({**e, "session": sid})
    _write_log(out)
    if entries and entries[-1].get("role") != "divider" and sid:
        data["active"] = sid          # carry on where the last conversation left off
    _save_sessions(data)
    print(f"[sessions] split chat_log into {len(data['sessions'])} chats (backup: {HISTORY_FILE}.bak-{stamp})")


def list_sessions(query: str = "") -> list[dict]:
    """Newest first; with a query, the chats whose title or messages match it."""
    data = _load_sessions()
    ids = set(data["sessions"])
    if query.strip():
        q = query.strip().lower()
        hits = {sid for sid, m in data["sessions"].items() if q in (m.get("title") or "").lower()}
        hits |= set(memory_store.search_sessions(query)) & ids
        ids = hits
    rows = [{"id": sid, **data["sessions"][sid], "active": sid == _active["id"]} for sid in ids
            if data["sessions"][sid].get("title") or sid == _active["id"]]   # empty chats only while open
    return sorted(rows, key=lambda r: r["updated"], reverse=True)


def _session_has_messages(sid: str | None) -> bool:
    return bool(sid) and any(e.get("session") == sid and e.get("role") in ("user", "assistant") for e in _read_log())


def new_session() -> dict:
    """Open a new, empty chat (the open one is reused if nothing was said in it yet)."""
    data = _load_sessions()
    sid = _active["id"]
    if not (sid and sid in data["sessions"] and not _session_has_messages(sid)):
        sid = _new_id()
        data["sessions"][sid] = {"title": "", "created": _now(), "updated": _now(), "pinned": False, "named": False}
    data["active"] = sid
    _save_sessions(data)
    _active["id"] = sid
    del messages[1:]
    return {"id": sid, **data["sessions"][sid]}


def open_session(sid: str) -> bool:
    """Make `sid` the open chat and load its recent turns as Javid's context."""
    data = _load_sessions()
    if sid not in data["sessions"]:
        return False
    if data.get("active") != sid:
        data["active"] = sid
        _save_sessions(data)
    _active["id"] = sid
    del messages[1:]
    turns = [e for e in _read_log() if e.get("session") == sid and e.get("role") in ("user", "assistant")
             and not (e["role"] == "assistant" and _is_garbled(e["text"]))]
    messages.extend({"role": e["role"], "content": e["text"]} for e in turns[-CONTEXT_TURNS:])
    return True


def update_session(sid: str, title: str | None = None, pinned: bool | None = None) -> dict | None:
    data = _load_sessions()
    meta = data["sessions"].get(sid)
    if meta is None:
        return None
    if title is not None and title.strip():
        meta["title"], meta["named"] = " ".join(title.split())[:80], True
    if pinned is not None:
        meta["pinned"] = bool(pinned)
    _save_sessions(data)
    return {"id": sid, **meta}


def delete_session(sid: str) -> bool:
    """Delete a chat and its messages for good (the user's own action, from the Chats panel)."""
    data = _load_sessions()
    if sid not in data["sessions"]:
        return False
    del data["sessions"][sid]
    _write_log([e for e in _read_log() if e.get("session") != sid])
    _save_sessions(data)
    if _active["id"] == sid:
        _active["id"] = None
        rest = sorted(data["sessions"], key=lambda k: data["sessions"][k]["updated"], reverse=True)
        open_session(rest[0]) if rest else new_session()
    return True


def current_session() -> str:
    """The open chat. If none is open yet, a new one is created for the conversation already in
    `messages` (so, unlike new_session, nothing is cleared)."""
    if not _active["id"]:
        data = _load_sessions()
        sid = _new_id()
        data["sessions"][sid] = {"title": "", "created": _now(), "updated": _now(), "pinned": False, "named": False}
        data["active"] = sid
        _save_sessions(data)
        _active["id"] = sid
    return _active["id"]


def log_message(role: str, text: str, **extra) -> None:
    sid = current_session()
    entry = {"ts": _now(), "session": sid, "role": role, "text": text, **extra}
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    data = _load_sessions()
    meta = data["sessions"].setdefault(sid, {"title": "", "created": entry["ts"], "updated": entry["ts"],
                                             "pinned": False, "named": False})
    meta["updated"] = entry["ts"]
    if role == "user" and not meta.get("named") and not meta.get("title"):
        meta["title"] = auto_title(text)
    _save_sessions(data)


def read_history(session: str | None = None) -> list[dict]:
    """One chat's messages (the open one by default)."""
    sid = session or current_session()
    return [e for e in _read_log() if e.get("session") == sid]


def reset_conversation() -> None:
    """The page's "+ New": a new, empty chat. Old chats stay in the Chats panel."""
    new_session()


_PAUSE = re.compile(r"…+|\.{2,}|(?:\.\s){2,}\.?")      # an ellipsis, however it was typed
_NO_WORD = re.compile(r"^[^\w\u0980-\u09ff]+$")       # a token with no letters or digits: "…", "—", "!!!"
_FRAGMENT_END = re.compile(r"(?:…+|\.{2,}|[.!?।]+(?=\s|$)|\n)+")
_WORD_EDGES = "\"'“”‘’()[]{}.,;:!?…—–-*·•"


def _words(text: str) -> list[str]:
    return [w for w in (t.strip(_WORD_EDGES).lower() for t in text.split()) if w and not _NO_WORD.match(w)]


def _degenerate(text: str) -> str | None:
    """Why one piece of text is degenerate model output, or None. The first check alone (mostly dots) caught the
    2026-10-13 reply, but not a normal opening that trails off into "... … …", ellipsis-chopped fragments that
    restart themselves ("I'm... I'm not... I'm not sure..."), other punctuation piling up, or a phrase on repeat."""
    s = " ".join(text.replace("\u00a0", " ").split())
    non_space = sum(not c.isspace() for c in s)
    if non_space < 12:
        return None
    if sum(len(m) for m in re.findall(r"(?<!\d)\.(?!\d)|…", s)) / non_space > 0.3:   # dots in "27.3" don't count
        return "mostly dots"
    run = longest = 0
    for token in s.split():
        run = run + 1 if _NO_WORD.match(token) else 0
        longest = max(longest, run)
    if longest >= 3:
        return "punctuation piling up"
    words = _words(s)
    pauses = len(_PAUSE.findall(s))
    if pauses >= 4 and pauses * 6 >= len(words):
        return "chopped into ellipsis fragments"
    frags = [f for f in (tuple(_words(f)) for f in _FRAGMENT_END.split(s))
             if f and not all(re.fullmatch(r"[\d.,:]+", w) for w in f)]      # "1." of a numbered list isn't one
    if len(frags) >= 4:
        seen, loops = set(), 0
        for prev, f in zip([()] + frags, frags):
            loops += f in seen or bool(prev and f[:len(prev)] == prev)
            seen.add(f)
        if loops * 2 >= len(frags):
            return "fragments repeating"
    joined = " " + " ".join(words) + " "
    if re.search(r"\s((?:\S+\s){2,6}?)\1{2,}", joined) or re.search(r"\s(\S+\s)\1{3,}", joined):
        return "a phrase on repeat"     # 2-6 words three times running, or one word four ("no, no, no" is speech)
    if len(words) >= 8 and len(set(words)) / len(words) < 0.3:
        return "one word on repeat"
    return None


def _is_garbled(text: str) -> bool:
    """Degenerate output (seen from gpt-oss-120b and gpt-oss-20b): mostly dots, fragments trailing off into
    ellipses, or words on repeat. A skill's or agent's SPOKEN and SCREEN parts are checked on their own too: a
    long screen list used to dilute a garbled spoken part, which was then read aloud."""
    spoken, screen = skills.split_reply(text)
    return any(_degenerate(part) for part in dict.fromkeys((text, spoken, screen)) if part)


def _restore_context() -> None:
    """At startup (server lifespan, CLI main; never on import, so scripts and tests can't touch the real
    chats): split a v1 log into chats once, then reopen the chat that was open with its recent turns."""
    _migrate_sessions()
    active = _load_sessions().get("active")
    if not (active and open_session(active)):
        new_session()


# ---------- tool schemas ----------
def _tool(name: str, description: str, properties: dict | None = None, required: tuple = ()) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": list(required)}}}


_TASK_REF = {"number": {"type": "integer", "description": "The task's number in the list."},
             "text": {"type": "string", "description": "Or words from the task, as the user said them."}}

TOOLS = [
    _tool("add_task", "Add a task or reminder to Mahadi's task list. The result says if a similar task "
          "already exists; if so, tell Mahadi.",
          {"task": {"type": "string", "description": "The task to add."}}, ("task",)),
    _tool("list_tasks", "All tasks with pending/done counts. Call it for any question about tasks and use "
          "its counts; never answer from memory."),
    _tool("delete_task", "Delete a task, by number or by words from it.", _TASK_REF),
    _tool("complete_task", "Mark a task done, by number or by words from it.", _TASK_REF),
    _tool("propose_memory", "Propose one line for long-term memory. It is NOT saved until Mahadi approves "
          "(card on screen, or a yes to your question), so ask him. Never passwords, keys or ID numbers. "
          "To forget something, tell him to delete it in the Memory panel.",
          {"text": {"type": "string", "description": "One short line, e.g. 'Sister's name is Nadia'."},
           "file": {"type": "string", "enum": ["user", "memory"],
                    "description": "user = a fact about Mahadi; memory = how he likes things done."}}, ("text",)),
    _tool("search_history", "Search past chats (all earlier conversations) for what was said.",
          {"query": {"type": "string", "description": "Key words, e.g. 'dentist insurance'."}}, ("query",)),
    _tool("get_datetime", "Current date, time and weekday. Call it whenever asked the time or date."),
    _tool("morning_briefing", "Time, today's calendar and a task summary. Call it whenever greeted with "
          "good morning, good afternoon or good evening, whatever the actual time; always say how many "
          "tasks are pending and what's on today."),
    *web_tools.TOOLS,
    *calendar_tools.TOOLS,
    *rates.TOOLS,
    *skills.TOOLS,
    *weather.TOOLS,
    *timers.TOOLS,
    *calc.TOOLS,
    *markets.TOOLS,
    *agents.TOOLS,
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
_FILLER = {"a", "an", "the", "to", "my", "for", "some", "task", "please"}


def _task_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower())) - _FILLER


def add_task(task: str) -> str:
    # Duplicate check and numbering happen here, not in a follow-up list_tasks round.
    lines = _read_lines()
    new = _task_words(task)
    similar = [f"#{i + 1} '{_task_parts(l)[2]}'" + (" (done)" if l.startswith("✓") else "")
               for i, l in enumerate(lines)
               if new and (old := _task_words(_task_parts(l)[2]))
               and len(new & old) / len(new | old) >= 0.5]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(TASKS_FILE, "a") as f:
        f.write(f"[{timestamp}] {task}\n")
    pending = sum(not l.startswith("✓") for l in lines) + 1
    if similar:
        return (f"Added as task #{len(lines) + 1}: {task}, but the list already had {', '.join(similar)}. "
                f"Say you added it and that it may be a duplicate. {pending} pending now.")
    return f"Added as task #{len(lines) + 1}: {task}. {pending} pending now."


def _find_task(lines: list[str], number: int | None, text: str | None) -> int | str:
    """Index of the task meant by number or words, else a sentence for the model."""
    if number is not None:
        if 1 <= number <= len(lines):
            return number - 1
        return f"No task number {number}. You have {len(lines)} task(s)."
    words = _task_words(text or "")
    hits = [i for i, l in enumerate(lines) if words and words <= _task_words(_task_parts(l)[2])]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return "Several tasks match; ask which: " + "; ".join(f"#{i + 1} {_task_parts(lines[i])[2]}" for i in hits)
    return f"No task matches '{text}'. Tasks: " + "; ".join(
        f"#{i + 1} {_task_parts(l)[2]}" for i, l in enumerate(lines))


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


def delete_task(number: int | None = None, text: str | None = None) -> str:
    lines = _read_lines()
    if not lines:
        return "No tasks to delete."
    i = _find_task(lines, number, text)
    if isinstance(i, str):
        return i
    removed = lines.pop(i)
    _write_lines(lines)
    return f"Deleted task #{i + 1}: {_task_parts(removed)[2]}. {len(lines)} task(s) left."


def complete_task(number: int | None = None, text: str | None = None) -> str:
    lines = _read_lines()
    if not lines:
        return "No tasks to complete."
    i = _find_task(lines, number, text)
    if isinstance(i, str):
        return i
    if lines[i].startswith("✓"):
        return f"Task #{i + 1} is already marked complete."
    lines[i] = "✓ " + lines[i]
    _write_lines(lines)
    return f"Completed task #{i + 1}: {_task_parts(lines[i])[2]}."




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
    if get_language() == "bn":
        part = {"morning": "শুভ সকাল", "afternoon": "শুভ বিকেল", "evening": "শুভ সন্ধ্যা"}[time_of_day]
        greeting = f"{part}, মাহাদি। আমি অনলাইনে আছি।"
        if pending:
            greeting += f" তোমার {len(pending)}টি কাজ বাকি আছে।"
        return greeting
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
    # Real weather, so it's never made up ("sunny, 28°C in India" in a greeting, 2026-10-17) - but after the
    # tasks and calendar: with it, a "good morning" reply once spent its two sentences on the weather.
    parts.append(weather.brief_line() or "Weather: unavailable right now; don't mention or guess it.")
    parts.append("Say the pending count and today's calendar first; the weather only if it fits.")
    return "\n".join(parts)


_new_proposals: list[dict] = []   # proposals made this turn, sent to the page as approval cards
# "Remember that..." must end in a proposal: models like to answer "Got it, I'll remember" and save nothing.
_MEMORY_ASK = re.compile(r"^\W*(?:(?:please|ok(?:ay)?|hey javid|javid)\W+)?(?:remember(?! to\b)|don'?t forget(?! to\b)|"
                         r"keep in mind|note that|make a note|save (?:this|that))\b|মনে রাখ", re.I)   # "remember to X" is a task
_PAST_ASK = re.compile(r"\bdid (i|we|you) (say|tell|mention|ask|talk|discuss)|\bwe (talked|discussed|spoke)\b|"
                       r"\bwhat did i (say|tell)|\bremind me what|বলেছিলাম", re.I)
# An explicit "look it up" / "search online" must search: K2 answered "Look up how tall Mount Everest is"
# from memory, with no source. Not when it's about Mahadi's own data ("look up my calendar").
_WEB_ASK = re.compile(r"^(?!.*\b(?:my|our)\s+(?:calendar|schedule|tasks?|to-?do|list|notes?|memory|chats?|history|"
                      r"events?|meetings?|appointments?)\b|.*\b(?:tasks?|to-?do|remind(?:er)?|remember to)\b)"
                      r".*?(?:\blook\s+(?:it\s+|that\s+|this\s+)?up\b|\bsearch\s+(?:online|the\s+(?:web|internet)|for\b)|"
                      r"\bgoogle\s+(?:it|that|this|for|how|what|who|when|where|why)\b|\b(?:check|find(?:\s+out)?)\s+online\b|"
                      r"\bon\s+the\s+(?:web|internet)\b|সার্চ|অনলাইনে|ইন্টারনেটে)", re.I | re.S)
_WEATHER_ASK = re.compile(r"\bweather\b|\bforecast\b|\bwill it (?:rain|be (?:hot|cold|sunny))|\b(?:is it|it'?s) "
                          r"(?:going to )?(?:rain|hot|cold)\b|\btemperature (?:today|tomorrow|now|outside)|"
                          r"\bhow (?:hot|cold) is it\b|\bumbrella\b|আবহাওয়া|বৃষ্টি হবে", re.I)
_TIMER_ASK = re.compile(r"\b(?:set|start|put on|make)\b.{0,20}\b(?:timer|alarm|countdown)\b|\btimer for\b|"
                        r"\bremind me (?:in|at) |\bremind me\b.{0,60}\b(?:in \d+|at \d|at (?:noon|midnight)|"
                        r"in (?:an?|one|two|five|ten|half an?) (?:hour|minute))|\bwake me\b|টাইমার", re.I)
_COINS = (r"(?:bitcoin|\bbtc\b|ethereum|\beth\b|\bsolana\b|\bsol\b|\bbnb\b|\bxrp\b|dogecoin|\bdoge\b|"
          r"\bpepe\b|cardano|\bton\b|crypto|\bcoin\b|nvidia|tesla|apple stock|\bstock\b|\bshares?\b|s&p 500|"
          r"nasdaq|dow jones|বিটকয়েন|ক্রিপ্টো|শেয়ার)")
# Bengali price words sit outside \b: Python's \b breaks at Bengali vowel signs. A price word alone ("আলুর দাম")
# isn't a market question: it needs a coin or stock word next to it.
_MARKET_ASK = re.compile(rf"(?:\b(?:price|prices|worth|value|trading at|how much|rate)\b|দাম|মূল্য).{{0,40}}{_COINS}|"
                         rf"{_COINS}.{{0,30}}(?:\b(?:price|prices|worth|trading|now|today)\b|দাম|মূল্য)", re.I)


class _Calc:
    """A message that is a sum or a conversion ("what's 17.5% of 2,340", "how many miles is 42 km"):
    gpt-oss-20b gets arithmetic wrong, so its answer must come from calculate. Not "open 24/7" or a date."""
    _ONLY_MATH = re.compile(r"^\W*(?:what'?s|what is|calculate|compute|how much is|work out|solve)?\s*"
                            r"[\d\s.,()]+(?:\s*(?:[-+*/×÷^x]|times|plus|minus|divided by|multiplied by)\s*[\d\s.,()]+)+"
                            r"(?:squared|cubed)?\s*\??\W*$", re.I)
    _WORDED = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent|per cent)\s*of\s*\d|\bsquare root of\s*\d|"
                         r"\bconvert\s+-?\d|\bhow many \w+(?: \w+)? (?:is|are|in|make) -?\d|"
                         r"\d\s*(?:km|kg|miles?|lbs?|pounds|feet|ft|c|f|celsius|fahrenheit|katha|bigha|acres?)\s+"
                         r"(?:to|in|into) (?:km|kg|miles?|lbs?|pounds|feet|ft|meters?|celsius|fahrenheit|sq ft|square feet|acres?|decimal)\b",
                         re.I)

    def search(self, text: str):
        return self._ONLY_MATH.match(text) or self._WORDED.search(text)


_CALC_ASK = _Calc()
# (request pattern, tool the answer needs, note for the one retry when the model answered without it)
REQUIRED_TOOLS = [
    (_MEMORY_ASK, "propose_memory", "Mahadi asked you to remember that, and you can't save memory yourself: call "
                                    "propose_memory with it now, then ask him to approve. Don't say it's saved."),
    (_PAST_ASK, "search_history", "Mahadi is asking about an earlier chat, which you can't see: call search_history "
                                  "with the key words, then answer from what it finds."),
    (_WEB_ASK, "web_search", "Mahadi asked you to look this up, so don't answer from memory: call web_search now, "
                             "then answer from the results and name the source."),
    (_WEATHER_ASK, "get_weather", "Mahadi asked about the weather, which you can't know: call get_weather now and "
                                  "answer only from its figures."),
    (_TIMER_ASK, "set_timer", "Mahadi asked for a timer or a timed reminder: call set_timer now with his words. "
                              "Don't say it's set until the tool says so."),
    (_CALC_ASK, "calculate", "That's a sum or a conversion: call calculate with it now and say its result; don't "
                             "work it out yourself."),
    (_MARKET_ASK, "market_price", "Mahadi asked for a live price: call market_price now and use its figures."),
    # Rates questions and market briefs are answered in code; these reach the model only with something else asked.
    (rates.ASK, "exchange_rates", "Mahadi asked about exchange rates: call exchange_rates now and use its figures "
                                  "exactly, naming each one's source."),
    (markets.BRIEF_ASK, "market_price", "Mahadi asked how the market is doing: call market_price now and use its figures."),
]
# A required agent's request goes to that agent (delegate), with nothing else to call first.
for _a in agents.all_agents():
    if (_meta := agents._agents[_a["name"]]["metadata"]).get("required") == "true" and _meta.get("triggers"):
        REQUIRED_TOOLS.append((re.compile(_meta["triggers"], re.I), "delegate",
                               f"This is a job for the {_a['name']} agent: call delegate with agent='{_a['name']}' and "
                               "Mahadi's request as the task."))
# A required skill's request also goes without the web tools: with them, gpt-oss-120b ran three searches
# and a page read (~4 Tavily credits) before the retry note sent it to the skill.
_REQUIRED_SKILL = (re.compile(skills.trigger_pattern(required_only=True), re.I)
                   if skills.trigger_pattern(required_only=True) else None)
WEB_TOOLS = {"web_search", "read_page", "news_headlines"}
if _REQUIRED_SKILL:
    REQUIRED_TOOLS.append((_REQUIRED_SKILL, "use_skill",
                           "One of your skills covers this request: call use_skill with it now and follow its "
                           "steps."))


def _propose_memory(text: str, file: str = "user") -> str:
    result, proposal = memory_store.propose(text, file)
    if proposal:
        _new_proposals.append(proposal)
    return result


TOOL_MAP = {
    "add_task":      lambda args: add_task(**args),
    "list_tasks":    lambda args: list_tasks(),
    "delete_task":   lambda args: delete_task(**args),
    "complete_task": lambda args: complete_task(**args),
    "propose_memory":    lambda args: _propose_memory(**args),
    "search_history":    lambda args: memory_store.search_history(**args),
    "get_datetime":      lambda args: get_datetime(),
    "morning_briefing":  lambda args: morning_briefing(),
    **web_tools.TOOL_MAP,
    **calendar_tools.TOOL_MAP,
    **rates.TOOL_MAP,
    **skills.TOOL_MAP,
    **weather.TOOL_MAP,
    **timers.TOOL_MAP,
    **calc.TOOL_MAP,
    **markets.TOOL_MAP,
    **agents.TOOL_MAP,
}
# Agents run their own requests through the same model chain and tools. The lambdas look the names up when
# called, so regression.py's request counter (it replaces _stream_completion) and its tool spies see them too.
agents.configure(complete=lambda *a: _stream_completion(*a), tools=TOOLS, tool_map=TOOL_MAP,
                 context=lambda: context_block("", get_language()), memory=memory_store.block,
                 api_errors=API_ERRORS, garbled=_is_garbled)


# ---------- which tools a request gets ----------
# Every schema costs prompt tokens on every request, so a turn only gets the groups its words
# point at (plus whatever the previous turn used, for follow-ups like "yes" or "that one").
# Erring wide is cheap; a message that matches nothing gets every tool.
_DAYS = r"today|tonight|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|weekend"
_HOLIDAY_ASK = re.compile(r"holiday|day off|days off|vacation|long weekend|\bbreak\b|\beid\b|puja|festival|ছুটি|ঈদ|পূজা")
TOOL_GROUPS = [
    (["add_task", "list_tasks", "delete_task", "complete_task"],
     r"tasks?\b|to-?dos?|remind|\blist|\bdone\b|complet|finish|\bmark|pending|\badd|delete|remove|"
     r"\bdrop|tick|check (it )?off|that one|টাস্ক|কাজ"),
    (["propose_memory"],
     r"remember|memor|keep in mind|note that|don'?t forget|from now on|i prefer|i like|i hate|i don'?t like|"
     r"always|never|call me|my \w+('s)? (name|birthday) is|মনে রাখ|আমার নাম|আমি পছন্দ"),
    (["search_history"],
     r"did (i|we|you) (say|tell|mention|ask|talk|discuss)|we (talked|discussed|spoke)|last (time|week|month)|"
     r"earlier|previous(ly)?|a while ago|days? ago|conversation|past chat|history|what was (it|that)|"
     r"বলেছিলাম|আগে|গত (সপ্তাহ|মাস)"),
    (["web_search", "read_page"], r"search|look (it )?up|google|news|headline|latest|"
     r"price|stock|score|\bwho (is|was|won)|capital of|population|exchange|\brates?\b|currenc|dollar|"
     r"taka|euro|gold|inflation|how (tall|high|big|far|old|long)|when (was|did|is)|website|online|"
     r"official|\bopen\b|\bclosed\b|opening|office hours|visa|embassy|consulate|ivac|খোলা|বন্ধ|"
     r"খবর|দাম|সার্চ|খুঁজ|টাকা|ডলার"),
    (["get_weather"], r"weather|forecast|\brain|umbrella|temperature|\bhot\b|\bcold\b|humid|\bwindy?\b|sunny|cloudy|"
     r"storm|thunder|\bdegrees\b|আবহাওয়া|বৃষ্টি|তাপমাত্রা|গরম|ঠান্ডা|ঝড়"),
    (["set_timer", "timers"], r"timer|alarm|countdown|remind me|\bwake me\b|minutes? from now|টাইমার|অ্যালার্ম|মনে করিয়ে"),
    (["calculate"], r"\d\s*[-+*/x×÷^%]\s*\d|\d\s*(%|percent)|calculat|\bsum\b|how much is \d|what'?s \d|square root|"
     r"how many \w+( \w+)? (is|are|in|make) -?\d|"
     r"sqrt|\bconvert|\b(to|in) (miles|km|kilometers?|kg|pounds|lbs|feet|meters?|celsius|fahrenheit|ounces|grams|"
     r"liters?|gallons|mph|acres?|katha|bigha|decimal|sq ?ft|square feet)\b|times \d|divided|multipl|\bplus\b|"
     r"\bminus\b|হিসাব|যোগফল|গুণ|ভাগ"),
    (["exchange_rates"], rates.ASK.pattern),
    (["market_price"], r"bitcoin|\bbtc\b|ethereum|\beth\b|crypto|binance|solana|\bsol\b|\bbnb\b|\bxrp\b|dogecoin|"
     r"\bdoge\b|\bpepe\b|\busdt\b|altcoin|\bcoins?\b|\bstocks?\b|\bshares?\b|nasdaq|s&p|dow jones|ticker|nvidia|"
     r"tesla|apple stock|বিটকয়েন|ক্রিপ্টো|শেয়ার"),
    (["get_datetime"], r"\btime\b|\bdate\b|what day|which day|today|clock|o'?clock|সময়|কয়টা|কটা বাজে|তারিখ|বার আজ"),
    (["morning_briefing"], r"good (morning|afternoon|evening)|briefing"),
    # Calendar is split: reading is most turns and costs ~270 tokens; the write tools add ~570.
    (["list_events", "find_free_time"],
     r"calendar|schedul|event|meeting|appointment|\bfree\b|busy|availab|\bplans?\b|anything on|lunch|"
     r"dinner|class|" + _DAYS + "|"
     r"ক্যালেন্ডার|মিটিং|ইভেন্ট|অ্যাপয়েন্টমেন্ট|শিডিউল|আজ|কাল|পরশু|সপ্তাহ|সোমবার|মঙ্গলবার|বুধবার|"
     r"বৃহস্পতিবার|শুক্রবার|শনিবার|রবিবার|ফ্রি|ব্যস্ত"),
    (["add_event", "move_event", "delete_event"],
     r"schedul|\bbook|\bmove|postpone|push|reschedul|cancel|delete|remove|\d\s*(am|pm)\b|"
     r"\b(add|put|set up|create|make)\b.*\b(calendar|event|meeting|appointment|lunch|dinner|class|call)\b|"
     r"শিডিউল|যোগ|সরাও|বাতিল|মুছ|ডিলিট|রাখো"),
    (["list_holidays"], _HOLIDAY_ASK.pattern),
]
if skills.trigger_pattern():
    TOOL_GROUPS.append((["use_skill"], skills.trigger_pattern()))
if agents.trigger_pattern():
    TOOL_GROUPS.append((["delegate"], agents.trigger_pattern()))
_SMALL_TALK = re.compile(r"^(hi|hey|hello|yo|thanks|thank you|thx|ok(ay)?|cool|great|nice|got it|bye|"
                         r"good ?night|see you)\b[\w\s,.!']{0,20}$", re.I)


# Left out when nothing matched (~660 tokens on every such turn): each has its own trigger words, and an agent's
# job goes to it in code. If the model still reaches for one, the "not in request" retry sends every tool.
_NOT_IN_FALLBACK = {"delegate", "get_weather", "set_timer", "timers", "calculate", "market_price"}


def select_tools(user_text: str, previous_tools: set[str] = frozenset()) -> list[dict]:
    text = user_text.lower()
    names = set()
    for group, pattern in TOOL_GROUPS:
        if re.search(pattern, text) or previous_tools & set(group):
            names.update(group)
    if not names:
        return [] if _SMALL_TALK.match(user_text.strip()) else [t for t in TOOLS if t["function"]["name"] not in _NOT_IN_FALLBACK]
    return [t for t in TOOLS if t["function"]["name"] in names]


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
def transcribe(wav: bytes, filename: str = "speech.wav", bias_name: bool = True, lang: str | None = None) -> str:
    """
    In the current reply language (Whisper set to "en" turns Bangla speech into English).
    bias_name=True hints Whisper to spell "Javid" correctly, but on silence/noise it
    echoes the hint back as "Javid" — a false wake word. Hands-free must pass False.
    In Bangla the hint is Bengali script: a Latin "Javid" made Whisper romanize words ("kāl").
    """
    lang = lang or get_language()
    kwargs = {"prompt": "জাভেদ" if lang == "bn" else "Javid"} if bias_name else {}
    result = client.audio.transcriptions.create(
        file=(filename, wav),
        model=STT_MODEL,
        language=lang,
        **kwargs,
    )
    return _IVAC.sub("IVAC", result.text.strip())


# Whisper spells the Indian Visa Application Centre "IVEC", "IVEX", "IBEX"... ("IBEX" alone is also a
# stock index, so only next to "center"/"centre"/"visa").
_IVAC = re.compile(r"\b(?:i\.?v\.?e\.?c\.?|ivex|ibex|ivack?|i\.v\.a\.c\.?)(?=\s+(?:cent(?:er|re)|visa))", re.I)


def transcribe_turn(wav: bytes, filename: str = "speech.wav", bias_name: bool = True) -> str:
    """transcribe() in the current language, plus a rescue for English said in Bangla mode: Bangla
    Whisper turns "speak English" or "Hey Javid" into noise. A short clip that isn't understood gets
    a second, English pass, used only if it is the switch to English or a bare wake word."""
    text = transcribe(wav, filename, bias_name)
    if get_language() != "bn" or len(text.split()) > 5 or language_switch(text):
        return text
    if (rest := strip_wake_word(text)) is not None and not rest:
        return text                               # "জাভেদ": already a bare wake word
    english = transcribe(wav, filename, bias_name, "en")
    rest = strip_wake_word(english)
    switch = language_switch(rest if rest is not None else english)
    if (switch and switch[0] == "en") or (rest is not None and (not rest or is_noise(rest))):
        print(f"[stt] Bangla heard {text!r}; English pass {english!r} used")
        return english
    return text


# ---------- wake word (hands-free mode) ----------
# "Javid, ...", "Hey Javid ...", plus how Whisper spells it without a prompt
# (Javid, Javed, Jaaved, Jawid, Jovid, Jarvis...). A bare "David" only counts
# after a greeting, since it's a common name in ordinary speech.
_NAME = r"[jz][aeo]{1,2}[vwb][aeiou]{1,2}[dt]h?|jarvis"
# Bangla Whisper writes the name in Bengali script: জাভেদ, জাভিদ, জাবেদ, যাবেদ...
_NAME_BN = r"[জযঝ][\u09be-\u09cc]{0,2}[ভবওৱ][\u09be-\u09cc]{0,2}[দডটধ]"
_WAKE = re.compile(
    rf"^\W*(?:(?:hey|hi|hello|okay|ok|yo|হেই|হাই|হ্যালো|ওকে)\W+(?:{_NAME}|david|{_NAME_BN})|(?:{_NAME}|{_NAME_BN}))\b[\W]*",
    re.IGNORECASE,
)

# Whisper tends to "hear" these in silence or background noise.
_NOISE = {"", "you", "thank you", "thanks", "thank you for watching",
          "thanks for watching", "bye", "okay", "uh", "um", "hmm",
          "ধন্যবাদ", "হুম", "উম", "আহ"}


def strip_wake_word(text: str) -> str | None:
    """Return the request after the wake word, or None if it doesn't start with one."""
    m = _WAKE.match(text)
    return text[m.end():].strip() if m else None


def is_noise(text: str) -> bool:
    return re.sub(r"[^a-z\u0980-\u09ff ]", "", text.lower()).strip() in _NOISE   # keeps Bengali letters


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


def _cached_tokens(usage) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        return details.get("cached_tokens") or 0
    return getattr(details, "cached_tokens", None) or 0


_turn_usage: dict[str, dict] | None = None   # model -> token sums for the turn in progress
last_turn_usage: dict | None = None          # the finished turn's record (regression reads it)


def _record_usage(model: str, usage) -> None:
    """Append one request's token counts (the provider's own numbers) to the usage log."""
    entry = {"ts": round(time.time(), 1), "model": model, "prompt": usage.prompt_tokens or 0,
             "completion": usage.completion_tokens or 0, "total": usage.total_tokens or 0,
             "cached": _cached_tokens(usage)}
    if _turn_usage is not None:
        t = _turn_usage.setdefault(model, {"requests": 0, "prompt": 0, "completion": 0, "total": 0, "cached": 0})
        t["requests"] += 1
        for k in ("prompt", "completion", "total", "cached"):
            t[k] += entry[k]
    try:
        with open(LLM_USAGE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _start_turn_usage() -> None:
    global _turn_usage
    _turn_usage = {}


def _finish_turn_usage() -> None:
    """Log the turn's tokens, credited to the model that served most of its requests."""
    global _turn_usage, last_turn_usage
    per_model, _turn_usage = _turn_usage, None
    if not per_model:
        last_turn_usage = None
        return
    entry = {"ts": round(time.time(), 1), "model": max(per_model, key=lambda m: per_model[m]["requests"])}
    for k in ("requests", "prompt", "completion", "total", "cached"):
        entry[k] = sum(t[k] for t in per_model.values())
    last_turn_usage = entry
    try:
        with open(LLM_TURNS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


TURN_AVG_WINDOW = 20     # the Models panel averages each model's last N turns


def _read_usage(since: float, path: str | None = None) -> list[dict]:
    out = []
    try:
        with open(path or LLM_USAGE_FILE) as f:
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


_KEY_SHAPE = re.compile(r"(?:gsk_|AIza|sk-|ya29\.)[A-Za-z0-9_\-.*\u2022]{6,}")


def _redact(text: str) -> str:
    """Strip API keys from provider error text before it is stored, logged or sent to the browser.
    Covers the configured keys (whole, and the prefix/suffix a provider might echo in a masked
    form) and anything shaped like a Groq/Google/OpenAI-style key."""
    for name, value in os.environ.items():
        value = value.strip()
        if len(value) >= 12 and any(t in name for t in ("KEY", "SECRET", "TOKEN")):
            for fragment in (value, value[:8], value[-8:]):
                text = text.replace(fragment, "[key hidden]")
    return _KEY_SHAPE.sub("[key hidden]", text)


def _note_error(model: str, err: Exception) -> None:
    info = _model_info.setdefault(model, {})
    msg = _redact(str(err))
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
    "ifm":    {"label": "IFM (K2 Horizon)", "key_env": "IFM_API_KEY", "base_url": "https://api.ifm.ai/v1"},
}
_provider_clients: dict = {}


def default_chain() -> list[str]:
    return [LLM_MODEL, *LLM_FALLBACKS]


def current_chain() -> list[str]:
    """Models in the order they're tried: the user's choice from the panel, else the defaults."""
    try:
        with open(LLM_CHAIN_FILE) as f:
            ids = [m for m in json.load(f).get("chain", []) if isinstance(m, str) and m]
        if ids:
            return ids
    except (OSError, ValueError, AttributeError):
        pass
    return default_chain()


def set_chain(ids: list[str] | None) -> None:
    """Save a new model order (None restores the defaults). Every entry must be a default model
    or one the provider's key actually lists, so typos can't end up in the chain."""
    if ids is None:
        try:
            os.remove(LLM_CHAIN_FILE)
        except OSError:
            pass
        return
    if not 1 <= len(ids) <= 10 or len(set(ids)) != len(ids):
        raise ValueError("Pick between 1 and 10 different models.")
    for m in ids:
        known = m in default_chain() or m in (_models_on_provider(_split(m)[0]) or [])
        if not known:
            raise ValueError(f"{m} isn't a model your keys can use.")
    with open(LLM_CHAIN_FILE, "w") as f:
        json.dump({"chain": ids}, f, indent=2)


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
    providers don't know the field, so it's stripped when failing over to them. IFM 400s
    on a follow-up turn unless every assistant message has a thinking field; "" is accepted
    (checked 2026-10-17), and history replies (other models, chat_log) have none to replay."""
    if provider == "ifm":
        msgs = [{**m, "reasoning_content": m.get("reasoning_content") or ""}
                if m["role"] == "assistant" else m for m in msgs]
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
    # Groq: "try again in 7m11.1s"; Google: "Please retry in 6h21m37.1s" or retryDelay: '41s'
    m = re.search(r"(?:try again in|retry in|retryDelay['\"]?:\s*['\"]?)\s*"
                  r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", str(err), re.I)
    if m and any(m.groups()):
        h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        return max(1.0, h * 3600 + mins * 60 + secs)
    return default


# ---------- daily token budget and the quota guard ----------
# Groq's daily limit is a bucket that refills continuously (limit/24h per second: a 429 for 800 tokens
# says "try again in 5m45s"), not a midnight reset. daily_left() replays our usage log through that bucket,
# starting from Groq's last "Used N" report when there is one (that also covers other clients on the key).
QUOTA_RESERVE = 0.20     # below this share left, a model is kept for web and reasoning turns


def _tpd_limit(model: str) -> tuple[int, dict | None] | None:
    quota = _model_info.get(model, {}).get("quota")
    if quota:
        return quota["tpd_limit"], quota
    return (KNOWN_TPD_LIMITS[model], None) if model in KNOWN_TPD_LIMITS else None


def daily_left(model: str, now: float | None = None) -> int | None:
    """Estimated tokens left in the model's daily bucket; None when no daily token limit is known."""
    known = _tpd_limit(model)
    if not known:
        return None
    limit, quota = known
    now = now or time.time()
    rate = limit / 86400
    start = now - 86400
    bucket = float(limit)          # a bucket untouched for 24 h is full
    if quota and quota.get("reported_at", 0) > start:
        start, bucket = quota["reported_at"], float(limit - quota["reported_used"])
    t = start
    for e in _read_usage(start):
        if e.get("model") != model:
            continue
        bucket = min(limit, bucket + rate * (e["ts"] - t)) - e.get("total", 0)
        t = e["ts"]
    return max(0, int(min(limit, bucket + rate * (now - t))))


def _reserved(model: str) -> bool:
    known, left = _tpd_limit(model), daily_left(model)
    return bool(known) and left is not None and left < QUOTA_RESERVE * known[0]


_REASONING = re.compile(r"\b(why|explain|compare|difference|analy[sz]|calculate|plan|pros and cons|"
                        r"how (do|does|can|should|would)|step by step|summari[sz]e|translate|write)\b|"
                        r"কেন|ব্যাখ্যা|তুলনা|হিসাব|পরিকল্পনা", re.I)
_turn_heavy = False     # set per turn by think_stream: web or reasoning, which may use reserved models


def is_heavy_turn(user_text: str, tools: list) -> bool:
    names = {t["function"]["name"] for t in tools}
    return bool(names & {"web_search", "read_page"}) or bool(_REASONING.search(user_text)) \
        or len(user_text.split()) > 40


def _models_to_try() -> list[str]:
    chain = [m for m in current_chain() if _has_key(m)] or default_chain()[:1]
    now = time.time()
    ready = [m for m in chain if _model_cooldown.get(m, 0) <= now]
    if not _turn_heavy:
        # Quota guard: a model under QUOTA_RESERVE of its daily tokens is saved for web and reasoning
        # turns. Simple turns go to the others first; it is still tried last rather than failing the turn.
        spare = [m for m in ready if not _reserved(m)]
        if spare and len(spare) < len(ready):
            print(f"[llm] reserving {[m for m in ready if m not in spare]} for web/reasoning turns")
            ready = spare + [m for m in ready if m not in spare]
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
            elif provider == "ifm":
                keep = [i for i in keep if "K2-" in i]   # also lists Jais (Arabic) and Jais-ASR
            cache.update(at=time.time(), ids=[i if provider == "groq" else f"{provider}:{i}" for i in keep])
        except Exception:
            cache["at"] = time.time() - 540     # retry in a minute
    return cache["ids"]


def llm_status() -> dict:
    """Everything the Models panel shows: chain order, who can serve now, token usage, limits."""
    _load_quota()   # another process (CLI, regression run) may have seen a 429 since we started
    now = time.time()
    chain = current_chain()
    everything = _read_usage(0)
    recent = [e for e in everything if e.get("ts", 0) >= now - 86400]
    # Our tally only starts when logging began; before that, usage is simply unknown to us.
    tracking_since = min((e["ts"] for e in everything), default=now)
    turns = _read_usage(now - 7 * 86400, LLM_TURNS_FILE)
    on_key_by_provider = {p: _models_on_provider(p) for p in PROVIDERS}
    models = []
    for order, model in enumerate(chain):
        mine = [e for e in recent if e.get("model") == model]
        day = {"requests": len(mine), "prompt": sum(e["prompt"] for e in mine),
               "completion": sum(e["completion"] for e in mine), "total": sum(e["total"] for e in mine)}
        my_turns = [t for t in turns if t.get("model") == model][-TURN_AVG_WINDOW:]
        per_turn = ({k: round(sum(t.get(k, 0) for t in my_turns) / len(my_turns), 1 if k == "requests" else 0)
                     for k in ("requests", "prompt", "completion", "total", "cached")}
                    | {"turns": len(my_turns)}) if my_turns else None
        info = _model_info.get(model, {})
        quota = info.get("quota")
        if not quota and model in KNOWN_TPD_LIMITS:
            quota = {"tpd_limit": KNOWN_TPD_LIMITS[model], "reported_used": 0, "reported_at": 0}
        est_used = remaining = None
        if quota:
            remaining = daily_left(model, now)       # refilling bucket, see daily_left
            est_used = quota["tpd_limit"] - remaining
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
        elif remaining is not None and remaining < QUOTA_RESERVE * quota["tpd_limit"]:
            state = "reserved"  # quota guard: only web and reasoning turns use it while others can serve
        else:
            state = "ready"
        models.append({
            "id": model, "name": _split(model)[1].split("/")[-1], "order": order,
            "provider": provider, "provider_label": PROVIDERS[provider]["label"],
            "key_env": PROVIDERS[provider].get("key_env"),
            "role": "primary" if order == 0 else "fallback", "state": state,
            "retry_in_s": round(cooling_for), "tokens_24h": day, "per_turn": per_turn,
            "session_requests": llm_usage.get(model, 0),
            "last_used": max((e["ts"] for e in mine), default=None),
            "limits": info.get("limits"),
            "tpd_limit": quota["tpd_limit"] if quota else None,
            "est_used_24h": est_used, "remaining_24h": remaining, "reserve_below": QUOTA_RESERVE,
            "reported": ({"used": quota["reported_used"], "at": quota["reported_at"]}
                         if quota and quota["reported_at"] else None),
            "last_error": info.get("last_error"),
            "sends_reasoning_effort": "gpt-oss" in model,
        })
    serving = next((m for m in models if m["state"] == "ready"), None)
    if serving:
        serving["state"] = "serving"
    providers = {}
    for name, cfg in PROVIDERS.items():
        key = os.environ.get(cfg.get("key_env", "GROQ_API_KEY"), "").strip()
        providers[name] = {"label": cfg["label"], "key_env": cfg.get("key_env", "GROQ_API_KEY"),
                           "has_key": bool(key), "key_hint": key[-4:] if len(key) >= 8 else "",
                           "editable": name != "groq"}   # the Groq key also drives speech, set it in .env
    return {
        "providers": providers, "is_default_chain": chain == default_chain(),
        "now": now, "serving": serving["id"] if serving else None,
        "chain_size": len(chain), "tracking_since": tracking_since,
        "ready_count": sum(m["state"] in ("serving", "ready") for m in models),
        "min_request_tokens": LLM_MIN_REQUEST_TOKENS,
        "models": models,
        "web": web_tools.status(),
        "on_key": [{"id": i, "in_chain": i in chain, "provider": p}
                   for p, ids in on_key_by_provider.items() for i in ids or []],
    }


_KEY_FORMAT = re.compile(r"^[A-Za-z0-9._\-]{8,300}$")


def _write_env(name: str, value: str | None) -> None:
    """Set (or with None, remove) one NAME=value line in .env, leaving everything else as it was."""
    try:
        with open(ENV_FILE) as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    kept = [l for l in lines if not pat.match(l)]
    if value is not None:
        kept.append(f"{name}={value}")
    tmp = ENV_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(kept) + "\n")
    os.chmod(tmp, 0o600)     # a secrets file is owner-only, whatever the old file's mode was
    os.replace(tmp, ENV_FILE)


def _forget_provider(provider: str) -> None:
    """After a key change: rebuild the client and drop cooldowns/errors from the old key."""
    _provider_clients.pop(provider, None)
    _key_models.pop(provider, None)
    for m in set(current_chain()) | set(default_chain()):
        if _split(m)[0] == provider:
            _model_cooldown.pop(m, None)
            _model_info.get(m, {}).pop("last_error", None)


def _verify_key(provider: str, c) -> None:
    """Raise the provider's API error if it rejects the key; spends no tokens. api.ifm.ai lists
    its models without checking the key at all, so IFM gets a chat request with no messages:
    a bad key is a 401, a good one the 400 for the empty request (checked 2026-10-16)."""
    if provider != "ifm":
        c.models.list()
        return
    try:
        c.chat.completions.create(model=IFM_MODEL, messages=[], max_tokens=1)
    except API_ERRORS as e:
        if getattr(e, "status_code", None) not in (400, 422):   # key accepted, request refused
            raise


def set_provider_key(provider: str, key: str) -> None:
    """Check `key` against the provider (a model-list call: no tokens spent), then save it to
    .env and use it straight away. Raises ValueError with a message safe to show the user."""
    cfg = PROVIDERS.get(provider, {})
    if not cfg.get("key_env"):
        raise ValueError("That provider's key can't be changed here.")
    key = key.strip()
    if not _KEY_FORMAT.match(key):
        raise ValueError("That doesn't look like an API key (letters, digits, - _ . only, no spaces).")
    try:
        _verify_key(provider, _openai.OpenAI(api_key=key, base_url=cfg["base_url"], timeout=15.0, max_retries=0))
    except API_ERRORS as e:
        if getattr(e, "status_code", None) in (400, 401, 403):
            raise ValueError(f"{cfg['label']} rejected this key. Nothing was saved.") from None
        raise ValueError(f"Couldn't reach {cfg['label']} to check the key ({type(e).__name__}). "
                         "Nothing was saved; try again.") from None
    _write_env(cfg["key_env"], key)
    os.environ[cfg["key_env"]] = key
    _forget_provider(provider)


def clear_provider_key(provider: str) -> None:
    cfg = PROVIDERS.get(provider, {})
    if not cfg.get("key_env"):
        raise ValueError("That provider's key can't be changed here.")
    _write_env(cfg["key_env"], None)
    os.environ.pop(cfg["key_env"], None)
    _forget_provider(provider)


def ping_models(only_provider: str | None = None) -> list[dict]:
    """Send a tiny no-tools request to every model in the chain (or one provider's), in parallel.
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
            return {"id": model, "ok": False, "ms": ms(), "error": f"{type(e).__name__}: {_redact(str(e))[:140]}"}

    chain = [m for m in current_chain() if only_provider in (None, _split(m)[0])]
    with ThreadPoolExecutor(max_workers=max(1, len(chain))) as pool:
        return list(pool.map(ping, chain))


def _open_stream(msgs: list, tools: list):
    """Start a streaming completion on the first model in the chain that accepts it.
    Returns (model, stream)."""
    last_error = None
    for model in _models_to_try():
        try:
            raw = _create(model, msgs, tools=tools, stream=True)
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
        if model != current_chain()[0]:
            print(f"[llm] answering with fallback {model}")
        llm_usage[model] = llm_usage.get(model, 0) + 1
        return model, raw.parse()
    raise last_error


def _stream_completion(tool_calls: dict, msgs: list, tools: list):
    """Stream one completion. Yields content deltas; collects tool calls into `tool_calls`."""
    span = tracing.llm_start()
    model = None
    try:
        model, stream = _open_stream(msgs, tools)
        yield from _read_stream(model, stream, tool_calls, span)
    finally:
        tracing.llm_end(span, model)


def _read_stream(model: str, stream, tool_calls: dict, span):
    counted = False
    for chunk in stream:
        tracing.llm_first(span)
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


def _past_turns(msgs: list) -> list[tuple[str, str]]:
    """(user text, final reply) per earlier turn. Their tool calls and results are left out:
    the reply already says what mattered, and old tool output is the bulk of a long chat."""
    turns = []
    for m in msgs:
        if m["role"] == "user":
            turns.append([m["content"], ""])
        elif m["role"] == "assistant" and m.get("content") and not m.get("tool_calls"):
            if not turns:
                turns.append(["", ""])
            turns[-1][1] = m["content"]
    return [tuple(t) for t in turns]


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _summarize(older: list[tuple[str, str]]) -> str:
    """A few words per older turn, built in Python: no extra LLM request."""
    if not older:
        return ""
    asks = "; ".join(_clip(u, 50) for u, _ in older[-8:] if u)
    return f"Earlier in this chat ({len(older)} older turns), Mahadi asked: {asks}."


# Earlier turns travel as plain user/reply text, so a follow-up ("so it's open Sunday?") only saw the
# model's own earlier reply, and it repeated its own wrong date over the tool's. The latest result of
# these tools from the last few turns rides in the [Context] note instead (~150 tokens).
FACT_TOOLS = ("list_holidays", "list_events", "find_free_time")
SCREEN_CHARS = 900       # the previous reply's screen part, in the next request only
FACT_TURNS = 3
FACT_CHARS = 700
_CALENDAR_WRITES = {"add_event", "move_event", "delete_event"}   # make earlier calendar reads stale
# Pushing back on an answer ("how come?", "are you sure?") gets a fresh tool call, not just agreement.
_CHALLENGE = re.compile(r"\bhow come\b|\bare you sure\b|\b(?:that'?s|that is|you'?re|you are) (?:wrong|not right|"
                        r"incorrect|mistaken)|\byou (?:just )?said\b|\bbut you\b|\bdouble.?check|\bcheck again\b|"
                        r"\bre-?check\b|\bcontradict|\bi thought\b|\bwrong\b|\breally\?|ভুল|নিশ্চিত\?", re.I)
RECHECK_TOOLS = FACT_TOOLS + ("list_tasks", "web_search")


def _recent_tool_results(turn_start: int, names: tuple, turns: int = FACT_TURNS) -> dict[str, str]:
    """{tool: its latest result} from the last `turns` turns before this one, newest call last."""
    start, seen = turn_start, 0
    while start > 1 and seen < turns:
        start -= 1
        seen += messages[start]["role"] == "user"
    called, out = {}, {}
    for m in messages[start:turn_start]:
        for c in m.get("tool_calls") or []:
            called[c["id"]] = c["function"]["name"]
        name = called.get(m.get("tool_call_id"))
        if m["role"] == "tool" and name:
            if name in _CALENDAR_WRITES:
                out.pop("list_events", None)
                out.pop("find_free_time", None)
            if name in names:
                out.pop(name, None)
                out[name] = m["content"]
    return out


def _results_since(start: int, names: tuple) -> list[str]:
    called = {c["id"]: c["function"]["name"] for m in messages[start:] for c in m.get("tool_calls") or []}
    return [m["content"] for m in messages[start:] if m["role"] == "tool" and called.get(m["tool_call_id"]) in names]


def _facts_note(turn_start: int) -> str:
    facts = _recent_tool_results(turn_start, FACT_TOOLS)
    if not facts:
        return ""
    body = " | ".join(f"{name}: {_clip(text, FACT_CHARS)}" for name, text in facts.items())
    return ("Tool results from earlier in this chat (still current; they outrank your earlier replies, "
            f"so correct any earlier reply they contradict): {body}")


def _request_messages(turn_start: int, lang: str = "en") -> list:
    """What one request sends: static system prompt, the last HISTORY_TURNS turns, then this
    turn (its user message prefixed with the context block, and any tool rounds so far)."""
    past = _past_turns(messages[1:turn_start])
    recent, older = past[-HISTORY_TURNS:], past[:-HISTORY_TURNS]
    # The memory files ride in the system message as a fixed block: it only changes when an entry does.
    out = [{"role": "system", "content": messages[0]["content"] + "\n\n" + memory_store.block()}]
    # The previous reply's screen part (a skill's or agent's full version) rides along for one turn only, so a
    # follow-up can use it: "yes, add those blocks" after the planner, "tell me more about the third one".
    last = next((m for m in reversed(messages[1:turn_start]) if m["role"] == "assistant" and not m.get("tool_calls")), None)
    for i, (user, reply) in enumerate(recent):
        if i == len(recent) - 1 and last and last.get("screen") and reply == last["content"]:
            reply += f"\n[Shown on screen, not spoken: {_clip(last['screen'], SCREEN_CHARS)}]"
        if user:
            out.append({"role": "user", "content": user})
        if reply:
            out.append({"role": "assistant", "content": reply})
    current = messages[turn_start:]
    out.append({**current[0], "content": context_block(_summarize(older), lang, _facts_note(turn_start))
                + current[0]["content"]})
    return out + current[1:]


def _agent_task(user_text: str, turn_start: int) -> str:
    """The task for an agent handed a request in code: the request plus the last two turns (the agent can't
    see the chat, and "research that" needs them)."""
    recent = [(u, r) for u, r in _past_turns(messages[1:turn_start])[-2:] if u or r]
    if not recent:
        return user_text
    earlier = " | ".join(f"Mahadi: {_clip(u, 200)} / Javid: {_clip(r, 200)}" for u, r in recent)
    return f"{user_text}\n(Earlier in this chat: {earlier})"


def _tools_used_in_previous_turn(turn_start: int) -> set[str]:
    names = set()
    for m in reversed(messages[1:turn_start]):
        if m["role"] == "user":
            break
        for c in m.get("tool_calls") or []:
            if c["function"]["name"] == "delegate":      # a follow-up gets the agent's tools, not a second run
                try:
                    names.update(agents.followup_tools(json.loads(c["function"]["arguments"]).get("agent", "")))
                except (ValueError, AttributeError):
                    pass
            else:
                names.add(c["function"]["name"])
    return names


def _model_name(model_id: str) -> str:
    provider, name = _split(model_id)
    return f"{name.split('/')[-1]} on {PROVIDERS[provider]['label']}"


def next_model() -> str:
    """The model that would answer now ("gpt-oss-20b on Groq"): chain, keys, cooldowns and the quota guard, no
    network (llm_status lists every provider's models)."""
    return _model_name(_models_to_try()[0])


def _self_facts() -> dict:
    """What about.py says about the running Javid: read from the code and settings, never from the model."""
    chain = [m for m in current_chain() if _has_key(m)] or default_chain()[:1]
    return {"model": _model_name(chain[0]), "backups": [_model_name(m) for m in chain[1:]],
            "provider": PROVIDERS[_split(chain[0])[0]]["label"], "stt": STT_MODEL,
            "voice_en": EDGE_VOICE, "voice_bn": EDGE_VOICE_BN,
            "skills": [s["name"] for s in skills.all_skills()], "agents": [a["name"] for a in agents.all_agents()]}


def _code_answer(user_text: str):
    """A rates question or a market brief that code answers whole (rates.request, markets.brief_request): a
    function of the reply language giving the answer's generator. None leaves the message to the model."""
    if (req := rates.request(user_text)):
        return lambda lang: rates.answer(req, lang)
    if (req := markets.brief_request(user_text)):
        return lambda lang: markets.brief(req, lang)
    return None


def _relay(answer, lang: str):
    """Pass a code answer's tool events on, with "Searching…" before its web search as for the model's, and
    return its (spoken, screen)."""
    said_searching = False
    while True:
        try:
            kind, value = next(answer)
        except StopIteration as done:
            return done.value
        if kind == "tool":
            print(f"[tool] {value} (answered in code)")
            if value in WEB_TOOLS and not said_searching:
                said_searching = True
                yield ("say", phrase("searching", lang))
        yield (kind, value)


def think_stream(user_text: str):
    """Yield ("tool", name), ("say", text), ("sources", links), ("lang", code) and ("delta", text)
    events as the reply is produced. "say" is spoken but isn't part of the reply ("Searching…");
    "lang" means the message switched the reply language.

    Each round's text is held until the round ends: if the model also called
    tools, that text was a guess made before seeing the results (e.g. a made-up
    time before get_datetime), so it is dropped instead of spoken.
    """
    _start_turn_usage()
    calendar_tools.new_turn()   # lets delete_event tell "next message" from "same turn"
    web_tools.new_turn()
    skills.new_turn()
    memory_store.new_turn()
    agents.new_turn()
    _new_proposals.clear()
    global _turn_heavy
    turn_start = len(messages)
    tools = select_tools(user_text, _tools_used_in_previous_turn(turn_start))
    # A job for a required agent goes straight to it: the turn gets only delegate (the agent has the tools).
    agent_first = agents.required_agent(user_text)
    skill_first = not agent_first and bool(_REQUIRED_SKILL and _REQUIRED_SKILL.search(user_text))
    if agent_first:
        tools = [t for t in TOOLS if t["function"]["name"] == "delegate"]
    elif skill_first:
        tools = [t for t in tools if t["function"]["name"] not in WEB_TOOLS]
    elif _MARKET_ASK.search(user_text):      # a live price: market_price, no web search (no Tavily credits)
        tools = [t for t in tools if t["function"]["name"] not in WEB_TOOLS]
    widened = False
    _turn_heavy = is_heavy_turn(user_text, tools) or bool(agent_first)
    messages.append({"role": "user", "content": user_text})
    log_message("user", user_text)
    reply = ""
    links = []
    screen = ""
    skill_results = []    # what the active skill's tools returned (the screen part if the model leaves it out)
    via_agent = None      # an agent's answer ({agent, spoken, screen}) when one handled the turn
    lang = get_language()
    try:
        # A yes/no right after a memory proposal decides it here: the model can't save memory.
        decided = memory_store.answer_from_reply(user_text)
        if decided:
            outcome, proposal = decided
            tracing.mark("routing_done")
            yield ("memory", {"outcome": outcome, **proposal})
            reply = phrase("mem_saved" if outcome == "approved" else "mem_dropped", lang)
            yield ("delta", reply)
            return
        # The language switch is done here, not by the model: it would forget it a turn later.
        switch = language_switch(user_text)
        if switch:
            lang, only_switch = switch
            set_language(lang)
            yield ("lang", lang)
            if only_switch:      # nothing else asked: confirm in the new language, no LLM request
                tracing.mark("routing_done")
                reply = phrase("switched", lang)
                yield ("delta", reply)
                return
        # Answered in code, no LLM request: a question about Javid itself (the model only guesses), a rates
        # question or a market brief (the model only copied the figures, and got their sources wrong).
        own = about.answer(user_text, lang, _self_facts())
        in_code = None if own else _code_answer(user_text)
        if own or in_code:
            tracing.mark("routing_done")
        if in_code:
            try:
                own = yield from _relay(in_code(lang), lang)
            except Exception as e:      # a bug in a code answer must not lose the turn: the model answers it
                print(f"[code] {type(e).__name__}: {e}; the model answers instead")
        if own:
            reply, screen = own
            if web_tools.used():
                links = web_tools.sources()
                yield ("sources", links)
            if screen:
                yield ("screen", screen)
            yield ("delta", reply)
            return
        tracing.mark("routing_done")    # tools chosen; the next step is the first LLM request
        retried_empty = False
        retried_error = False
        retried_garbled = False
        retried_web = False
        retried_lang = False
        retried_required = False
        retried_holiday = False
        required = [(tool, note) for pattern, tool, note in REQUIRED_TOOLS if pattern.search(user_text)
                    and not (skill_first and tool in WEB_TOOLS)   # the skill answers without the web
                    and not (agent_first and tool != "delegate" and tool not in agents.agent_tools(agent_first))]
        if _CHALLENGE.search(user_text) and (earlier := _recent_tool_results(turn_start, RECHECK_TOOLS)):
            tool = list(earlier)[-1]
            required.append((tool, f"Mahadi is questioning your earlier answer. Don't just agree or repeat it: "
                                   f"call {tool} again, answer from its result, and if you were wrong earlier, "
                                   f"say so plainly."))
        said_searching = False
        # A reply in the wrong language, or a web answer that fails web_tools.check_reply, gets one
        # rewrite each. The rejected draft and the note go only into requests (never `messages`), at
        # the point in the turn they were made: (len(messages) then, [draft, note]).
        notes = []
        handoff = None      # an agent's answer waiting to be used as this round's reply
        # A required agent's request is handed over in code: gpt-oss-20b offered only delegate still reached for
        # other tools (Groq: "not in request"), and once widened it called use_skill("planner").
        forced = None
        if agent_first:
            forced = {0: {"id": f"call_{uuid.uuid4().hex[:12]}", "name": "delegate",
                          "arguments": json.dumps({"agent": agent_first, "task": _agent_task(user_text, turn_start)},
                                                  ensure_ascii=False)}}
        rounds = MAX_TOOL_ROUNDS + 4
        for round_no in range(rounds):
            tool_calls = {}
            # Last round: no tools, so the model has to answer with what it has instead of
            # ending the turn on "I got stuck".
            round_tools = tools if round_no < rounds - 1 else []
            req = _request_messages(turn_start, lang)
            for at, extra in reversed(notes):   # later notes first, so earlier cut points stay put
                cut = len(req) - (len(messages) - at)
                req = req[:cut] + extra + req[cut:]
            try:
                if forced:      # this round's call was decided in code
                    tool_calls, text, forced = forced, "", None
                elif handoff:   # the agent already answered Mahadi: no request to reword it
                    text, handoff = handoff["spoken"], None
                else:
                    text = "".join(_stream_completion(tool_calls, req, round_tools))
            except API_ERRORS as e:
                # The model reached for a tool select_tools didn't send (Groq rejects the call):
                # widen to every tool and ask again.
                if not widened and "not in request" in str(e):
                    print(f"[retry] needed a tool outside {[t['function']['name'] for t in tools]}")
                    widened = True
                    tools = [t for t in TOOLS if not (skill_first and t["function"]["name"] in WEB_TOOLS)]
                    continue
                # e.g. "Failed to parse tool call arguments as JSON" — the model
                # occasionally emits a malformed tool call; one retry usually fixes it.
                if retried_error or isinstance(e, BAD_REQUEST):
                    raise
                retried_error = True
                print(f"[retry] {_redact(str(e))[:80]}")
                continue
            if not tool_calls and not text.strip() and not retried_empty:
                retried_empty = True  # model occasionally returns nothing at all; ask once more
                continue
            if not tool_calls and _is_garbled(text):
                if not retried_garbled:
                    retried_garbled = True
                    print(f"[retry] garbled reply: {text[:60]!r}")
                    continue
                text = phrase("garbled", lang)
            called = ({c["function"]["name"] for m in messages[turn_start:] for c in m.get("tool_calls") or []}
                      | agents.tools_used())
            missing = [(tool, note) for tool, note in required if tool not in called]
            if not tool_calls and missing and not retried_required:
                retried_required = True
                tool, note = missing[0]
                print(f"[required] answered without {tool}: {text[:60]!r}")
                notes.append((len(messages), [{"role": "assistant", "content": text}, {"role": "user", "content": note}]))
                if not any(t["function"]["name"] == tool for t in tools):
                    tools = tools + [t for t in TOOLS if t["function"]["name"] == tool]
                continue
            if not tool_calls and not retried_lang and wrong_language(text, lang):
                retried_lang = True
                print(f"[lang] not in {lang}, asking for a rewrite: {text[:60]!r}")
                notes.append((len(messages), [{"role": "assistant", "content": text},
                                              {"role": "user", "content": language_note(lang)}]))
                continue
            if not tool_calls and web_tools.used():
                problems = web_tools.check_reply(text + ("\n" + via_agent["screen"] if via_agent else ""))
                if problems and not retried_web and not via_agent:
                    retried_web = True
                    print(f"[web] answer rejected: {problems}")
                    notes.append((len(messages), [{"role": "assistant", "content": text},
                                                  {"role": "user", "content": web_tools.correction(problems, bool(skills.active()))}]))
                    continue
            if not tool_calls:
                held = _recent_tool_results(len(messages), ("list_holidays",), turns=FACT_TURNS + 1)
                web = _results_since(turn_start, ("web_search", "read_page"))
                on_holidays = bool(_HOLIDAY_ASK.search(user_text.lower()) or _results_since(turn_start, ("list_holidays",)))
                bad = calendar_tools.check_holiday_reply(text, list(held.values()), web, on_holidays)
                if bad and not retried_holiday:
                    retried_holiday = True
                    print(f"[holiday] dates not in list_holidays: {bad}: {text[:60]!r}")
                    notes.append((len(messages), [{"role": "assistant", "content": text},
                                                  {"role": "user", "content": calendar_tools.holiday_correction(bad)}]))
                    continue
                if bad:
                    print(f"[holiday] rewrite still has {bad}; using the tool's list")
                    text = (lang == "en" and calendar_tools.holiday_fallback(list(held.values()))) or phrase("holiday_unsure", lang)
            skill = skills.active()
            if not tool_calls and via_agent and not screen:
                screen = via_agent["screen"]
            if not tool_calls and skill:
                # A skill reply has a spoken part (cut to the skill's time budget) and a screen-only part.
                text, screen = skills.split_reply(text)
                text = skills.trim_spoken(text, skills.spoken_seconds(skill))
                screen = screen or skills.screen_from_results(skill_results)
            if not tool_calls and web_tools.used():
                # (problems were checked on the whole reply, screen part included, above)
                # A skill's own spoken budget replaces the plain web answer's sentence cap.
                text = web_tools.finalize(text, problems,
                                          max_sentences=None if skill or via_agent else web_tools.MAX_SENTENCES)
                links = web_tools.sources()
                yield ("sources", links)
            if not tool_calls and screen:
                yield ("screen", screen)
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
                if call["name"] in ("web_search", "news_headlines") and not said_searching:
                    said_searching = True
                    yield ("say", phrase("searching", lang))   # spoken while the search runs
                elif call["name"] == "delegate" and not said_searching:
                    said_searching = True
                    yield ("say", phrase("working", lang))     # an agent takes a while
                yield ("tool", call["name"])
                fn = TOOL_MAP.get(call["name"])
                args = json.loads(call["arguments"] or "{}")
                span = tracing.tool_start(call["name"])
                try:
                    result = fn(args) if fn else f"Unknown tool: {call['name']}"
                    if isinstance(result, types.GeneratorType):   # an agent: relay its progress, then its result
                        result = yield from result
                except TypeError as e:   # a made-up argument name (propose_memory(user=...)): let it retry
                    result = f"Wrong arguments for {call['name']}: {e}. Use only the parameters in its schema."
                finally:
                    tracing.tool_end(span)
                print(f"[tool] {call['name']}({args}) → {result[:80]}")
                if skills.active() and call["name"] in skills.active()["tools"]:
                    skill_results.append(result)
                while _new_proposals:
                    yield ("memory_proposal", _new_proposals.pop(0))
                if call["name"] == "delegate":
                    # One agent run per turn: if it failed, the model answers instead of starting it again.
                    tools = [t for t in tools if t["function"]["name"] != "delegate"]
                    if h := agents.take_handoff():
                        handoff = via_agent = h
                if call["name"] == "use_skill" and skills.active():
                    yield ("skill", skills.active()["name"])
                    # Unlock the skill's tools for the rest of the turn (and its follow-up turn).
                    have = {t["function"]["name"] for t in tools}
                    tools = tools + [t for t in TOOLS if t["function"]["name"] in skills.active()["tools"]
                                     and t["function"]["name"] not in have]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })
        else:
            reply = phrase("stuck", lang)
            yield ("delta", reply)

    except API_ERRORS:
        reply = phrase("api_failed", lang)
        yield ("delta", reply)

    finally:
        # Runs on normal finish and when the client aborts mid-stream,
        # so history always gets whatever was actually said.
        messages.append({"role": "assistant", "content": reply.strip(), **({"screen": screen} if screen else {})})
        if reply.strip():
            log_message("assistant", reply.strip(), **({"sources": links} if links else {}),
                        **({"screen": screen} if screen else {}), **({"skill": skills.active()["name"]} if skills.active() else {}),
                        **({"agent": via_agent["agent"]} if via_agent else {}))
        _finish_turn_usage()


def think(user_text: str) -> str:
    return "".join(text for kind, text in think_stream(user_text) if kind == "delta").strip()


# ---------- 4. MOUTH: text -> speech ----------
_tts_cache: dict[tuple[str, str], tuple[bytes, str]] = {}   # short, repeated phrases ("Yes?") skip the network
_TTS_CACHE_MAX_CHARS = 40
_TTS_CACHE_SIZE = 64

# edge-tts is a free, unofficial Microsoft endpoint. It normally answers in ~1 s
# but is erratic: stalls of 3-5 s and "no audio" failures come in streaks. Javid
# waits and retries so the voice is always the locked one, never a substitute.
EDGE_ATTEMPTS = 3
EDGE_ATTEMPT_SEC = 6.0       # abandon a stalled try after this long and retry


# Bengali letters, not the whole block: the taka sign and Bengali digits are in it too, and "৳122.50" in an
# English reply would have switched that one sentence to Pradeep (Andrew reads "৳" fine).
_BN_LETTERS = re.compile(r"[\u0985-\u09b9\u09ce\u09dc-\u09e1\u09f0\u09f1]")


def voice_for(text: str, lang: str | None = None) -> str:
    """The reply language's locked voice; the server's setting unless a caller names the language (the
    warm-up does). Bengali letters always get EDGE_VOICE_BN: AndrewNeural can't say them (it returns a
    near-silent clip, which is how Bangla replies went unheard)."""
    if _BN_LETTERS.search(text):
        return EDGE_VOICE_BN
    return VOICES[lang if lang in VOICES else get_language()]


async def _edge_once(text: str, voice: str) -> bytes:
    audio = b""
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return audio


def _edge_or_none(text: str, voice: str) -> bytes | None:
    """edge-tts audio, or None if every try failed."""
    for n in range(1, EDGE_ATTEMPTS + 1):
        try:
            return asyncio.run(asyncio.wait_for(_edge_once(text, voice), EDGE_ATTEMPT_SEC))
        except Exception as e:
            print(f"[tts] edge-tts {type(e).__name__} (try {n}/{EDGE_ATTEMPTS})")
    return None


def synthesize(text: str, lang: str | None = None) -> tuple[bytes, str]:
    """The language's voice (EDGE_VOICE / EDGE_VOICE_BN) as MP3 bytes. Raises if edge-tts is down
    after EDGE_ATTEMPTS tries: there is no fallback voice."""
    voice = voice_for(text, lang)
    if (voice, text) in _tts_cache:
        return _tts_cache[voice, text]
    audio = _edge_or_none(text, voice)
    if audio is None:
        raise RuntimeError("edge-tts unavailable")
    result = (audio, "audio/mpeg")
    if len(text) <= _TTS_CACHE_MAX_CHARS:
        if len(_tts_cache) >= _TTS_CACHE_SIZE:
            _tts_cache.pop(next(iter(_tts_cache)))
        _tts_cache[voice, text] = result
    return result


_WARM_FIRST = ("ack", "searching", "working")     # the ones said while Mahadi waits


def warm_phrases() -> list[tuple[str, str]]:
    """Regenerate the cached audio of every fixed phrase in each language, sentence by sentence as the page
    asks /speak for them, so a cached clip is always the locked voice's. Returns (sentence, voice) pairs."""
    _tts_cache.clear()
    done = []
    for key in sorted(PHRASES, key=lambda k: k not in _WARM_FIRST):
        for lang, text in PHRASES[key].items():
            for sentence in web_tools.sentences(text):
                if len(sentence) <= _TTS_CACHE_MAX_CHARS:
                    synthesize(sentence, lang)
                    done.append((sentence, voice_for(sentence, lang)))
    return done


def speak(text: str, trace: tracing.Trace | None = None) -> None:
    """CLI: the same voices as the web UI, played with macOS afplay."""
    if trace:
        trace.mark("tts_first_request")
    try:
        audio, _ = synthesize(text)
    except RuntimeError:
        print("(voice unavailable right now; reply shown as text only)")
        return
    if trace:
        trace.mark("first_audio")
        trace.mark("first_answer_audio")
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio)
        f.flush()
        subprocess.run(["afplay", f.name])
    if trace:
        trace.mark("tts_done")


# ---------- main loop ----------
def _ring_timers() -> None:
    """CLI only: announce timers and reminders as they come due (in the web UI the page rings them)."""
    while True:
        for t in timers.take_due():
            text = t["announce"].get(get_language()) or t["announce"]["en"]
            print(f"\n⏰ {text}")
            try:
                speak(text)
            except Exception as e:
                print(f"[tts] {str(e)[:80]}")
        time.sleep(1)


def main() -> None:
    _restore_context()
    threading.Thread(target=_ring_timers, daemon=True).start()
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

            trace = tracing.start("cli", "push")       # Enter pressed: end of speech
            trace.mark("speech_end", trace.stages["request_received"])
            text = transcribe_turn(wav)
            trace.mark("stt_done")
            if not text or len(text.split()) < 2:
                print("(didn't catch that, try again)")
                continue
            print(f"You:   {text}")
            trace.label = text[:60]

            tracing.activate(trace)
            reply = think(text)
            trace.mark("reply_out")
            print(f"Javid: {reply}")
            trace.mark("first_text")
            speak(reply, trace)
            tracing.finish_local(trace)

            if "goodbye" in text.lower():
                break

    except KeyboardInterrupt:
        farewell = "Goodbye, Mahadi. Talk soon."
        print(f"\nJavid: {farewell}")
        speak(farewell)


if __name__ == "__main__":
    main()
