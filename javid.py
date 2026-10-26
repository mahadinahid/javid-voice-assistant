"""
Javid v1.0 — push-to-talk voice assistant (7-day build, complete)
Mic -> Groq Whisper (speech-to-text) -> Groq GPT-OSS 20B (brain) -> edge-tts en-US-AndrewNeural (voice)
Day 7 polish: startup greeting, graceful Ctrl+C, network-safe search, file-handle fix
"""

import asyncio
import collections
import contextlib
import hashlib
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
import news
import providers
import rates
import selfcheck
import skills
import timers
import tracing
import watches
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
# The primary model. The whole chain comes from config/providers.yaml: each language model provider's `models`, provider
# by provider in priority order (the order saved from the Providers panel replaces it). Groq models are bare ids, the
# others "<provider>:<model>" ("gemini:gemini-3.7-flash", "openrouter:google/gemma-4-31b-it:free",
# "cloudflare:@cf/openai/gpt-oss-120b"). A provider without its key in .env is skipped; a model that hasn't passed its
# tool tests (config/model_tests.json) is chat-only.
LLM_MODEL = "openai/gpt-oss-20b"
# Daily token caps seen in Groq 429 messages; others are learned from the first 429 (llm_quota.json).
KNOWN_TPD_LIMITS = {"openai/gpt-oss-20b": 200000}
LLM_MIN_REQUEST_TOKENS = 1500   # one request is ~0.6-1.7k tokens (prompt + the tools sent); a turn ~2 requests
# LOCKED by the user: approved as "perfect" on 2026-10-13, made permanent on 2026-10-14 ("set this
# voice for everything, don't change it again"). Every spoken word, web UI and CLI, is this voice
# via edge-tts. The one fallback, since Mahadi asked for it on 2026-10-26, is ElevenLabs' fixed backup voice after
# every edge-tts try has failed (synthesize, _eleven); then the reply is shown as text. No other engine, ever (no
# browser speech, no macOS say). Never change this unless told to.
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
client = Groq(timeout=providers.timeout_s("groq", 60.0))


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
    "stt_failed": {"en": "I couldn't make that out: the speech recognition service didn't answer. Try again in a moment.",
                   "bn": "কথাটা বুঝতে পারিনি: কথা চেনার সার্ভিস সাড়া দেয়নি। একটু পরে আবার বলো।"},
    "no_personal": {"en": "That needs your calendar, tasks, memory or past chats, and none of my language models is "
                          "cleared for personal data yet, so I haven't sent it anywhere.",
                    "bn": "এর জন্য তোমার ক্যালেন্ডার, কাজ, মেমরি বা আগের চ্যাট লাগবে, কিন্তু আমার কোনো ল্যাঙ্গুয়েজ "
                          "মডেলকে এখনো ব্যক্তিগত তথ্য দেওয়ার অনুমতি নেই, তাই কোথাও পাঠাইনি।"},
    "no_model":   {"en": "None of my language models can answer right now: they're resting after a limit or at their "
                         "free allowance. Try again in a few minutes.",
                   "bn": "এখন কোনো ল্যাঙ্গুয়েজ মডেল উত্তর দিতে পারছে না: সবগুলো বিশ্রামে বা ফ্রি সীমায় পৌঁছেছে। "
                         "কয়েক মিনিট পরে আবার চেষ্টা করো।"},
    "no_tool_model": {"en": "I can't do that right now: the models I let use tools are resting or at their free limits. "
                            "I can still chat; try that again in a few minutes.",
                      "bn": "এখন এটা করতে পারছি না: যে মডেলগুলোকে টুল ব্যবহার করতে দিই, সেগুলো বিশ্রামে বা ফ্রি সীমায় "
                            "আছে। কথা বলতে পারি; কয়েক মিনিট পরে আবার চেষ্টা করো।"},
}
# Shown under "no_personal" (never spoken: a file name read aloud helps nobody).
NO_PERSONAL_SCREEN = ("Personal data (calendar, tasks, memory, past chats) only goes to providers marked personal_ok: true "
                      "in config/providers.yaml. Set it for a model provider once you've read its data policy; Javid "
                      "picks the change up within a few seconds.")


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
    # A reply keeps what it was based on (its tools and links), for "where did you get that?" after a restart.
    messages.extend({"role": e["role"], "content": e["text"],
                     **{k: e[k] for k in ("tools", "sources") if e["role"] == "assistant" and e.get(k)}}
                    for e in turns[-CONTEXT_TURNS:])
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


def record_turn(user_text: str, reply: str) -> None:
    """A turn answered without think_stream (Telegram's reply to a forwarded message), kept in the open chat and
    its log like any other, so the next turn can refer to it. Call under hold_turn."""
    messages.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}])
    log_message("user", user_text)
    log_message("assistant", reply)


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
# A question whose answer is a figure: when a web answer's figure can't be confirmed, the whole answer is replaced by
# "couldn't confirm"; for any other question only the sentences with that figure go (web_tools.finalize).
_FIGURE_ASK = re.compile(r"\b(?:rates?|prices?|costs?|how (?:much|many)|percent(?:age)?|inflation|gdp|growth|figures?|"
                         r"numbers?|amount|value|worth|salary|fees?|fares?|tariffs?|scores?|population|interest)\b|%|"
                         r"দাম|রেট|কত", re.I)
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
# Tools that change Mahadi's data. On a channel whose writes need a button press (Telegram), think_stream gets a
# `gate`: for that turn these don't run, and gate(name, args) records the exact action and says what the model is
# told instead (approvals.py). Wrapped here in TOOL_MAP itself, so an agent's or a skill's call is gated too.
WRITE_TOOLS = ("add_task", "complete_task", "delete_task", "add_event", "move_event", "delete_event", "propose_memory")
# Tools that read or write Mahadi's personal data: calendar, memory, chat history, tasks. Privacy rule (the model
# router, _open_stream): a request that carries their results, or a turn that needs them, goes only to providers with
# personal_ok: true in config/providers.yaml; any other provider gets the request without them (_public_request).
PERSONAL_TOOLS = frozenset({"add_task", "list_tasks", "delete_task", "complete_task", "propose_memory",
                            "search_history", "morning_briefing", "list_events", "find_free_time", "add_event",
                            "move_event", "delete_event"})
_write_gate = None


def _gated(name: str, run):
    return lambda args: _write_gate(name, args) if _write_gate else run(args)


for _name in WRITE_TOOLS:
    TOOL_MAP[_name] = _gated(_name, TOOL_MAP[_name])
# Agents run their own requests through the same model chain and tools. The lambdas look the names up when
# called, so regression.py's request counter (it replaces _stream_completion) and its tool spies see them too.
agents.configure(complete=lambda *a: _stream_completion(*a), tools=TOOLS, tool_map=TOOL_MAP,
                 personal_tools=PERSONAL_TOOLS,
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


def _matched_tools(user_text: str, previous_tools: set[str] = frozenset()) -> set[str]:
    """The tools of every group the words point at, plus the groups the previous turn used (no fallback)."""
    text = user_text.lower()
    names = set()
    for group, pattern in TOOL_GROUPS:
        if re.search(pattern, text) or previous_tools & set(group):
            names.update(group)
    return names


def select_tools(user_text: str, previous_tools: set[str] = frozenset()) -> list[dict]:
    names = _matched_tools(user_text, previous_tools)
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
    t0 = time.time()
    try:
        result = client.audio.transcriptions.create(
            file=(filename, wav),
            model=STT_MODEL,
            language=lang,
            **kwargs,
        )
    except API_ERRORS as e:
        providers.record("groq", 0, model=STT_MODEL, op="stt", unit="requests", latency_ms=(time.time() - t0) * 1000,
                         status=providers.error_kind(e), detail=type(e).__name__)
        raise
    providers.record("groq", 1, model=STT_MODEL, op="stt", unit="requests", latency_ms=(time.time() - t0) * 1000)
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


_KEY_SHAPE = re.compile(r"(?<![A-Za-z0-9])(?:gsk_|AIza|sk-|ya29\.)[A-Za-z0-9_\-.*\u2022]{6,}")   # not "risk-free"


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


_provider_clients: dict = {}
# OpenRouter serves paid models through the same key and endpoint: only its free variants, ids ending ":free", may ever
# be called. Enforced here in code (_create refuses anything else before sending; _usable, set_chain, the model list and
# the pings skip it), never left to config.
FREE_ONLY = {"openrouter": ":free"}
# Request fields that would let OpenRouter route to another model (paid ones included): never sent, whatever the
# registry's options say.
_ROUTING_FIELDS = ("models", "route", "fallbacks", "preset")
_PRICED_UNITS = ("neurons", "usd")     # allowances counted per token: a model needs its price in the registry


class PaidModelRefused(ValueError):
    """A request for an OpenRouter model whose id doesn't end in ":free": refused before anything is sent."""


def _chain_id(provider: str, name: str) -> str:
    return name if provider == "groq" else f"{provider}:{name}"


def default_chain() -> list[str]:
    """Every language model in config/providers.yaml, provider by provider in priority order."""
    return [_chain_id(e["name"], m["id"]) for e in providers.entries("llm") for m in e["models"]] or [LLM_MODEL]


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
    or one the provider's key actually lists, so typos can't end up in the chain; never a paid OpenRouter model."""
    if ids is None:
        try:
            os.remove(LLM_CHAIN_FILE)
        except OSError:
            pass
        return
    if not 1 <= len(ids) <= 10 or len(set(ids)) != len(ids):
        raise ValueError("Pick between 1 and 10 different models.")
    for m in ids:
        if _paid(m):
            raise ValueError(f"{m} isn't free: only OpenRouter models whose id ends in ':free' can be used.")
        known = m in default_chain() or m in (_models_on_provider(_split(m)[0]) or [])
        if not known:
            raise ValueError(f"{m} isn't a model your keys can use.")
    with open(LLM_CHAIN_FILE, "w") as f:
        json.dump({"chain": ids}, f, indent=2)


def _split(model_id: str) -> tuple[str, str]:
    """'gemini:gemini-3.7-flash' -> ('gemini', 'gemini-3.7-flash'); a bare id is a Groq model. The prefix must be a
    language model provider in config/providers.yaml (an OpenRouter id keeps its own ":free")."""
    provider, sep, name = model_id.partition(":")
    entry = providers.get(provider) if sep else None
    return (provider, name) if entry and entry["kind"] == "llm" else ("groq", model_id)


def _paid(model_id: str) -> bool:
    """An OpenRouter model that isn't one of its free variants (FREE_ONLY): never called."""
    provider, name = _split(model_id)
    return provider in FREE_ONLY and not name.endswith(FREE_ONLY[provider])


def _label(provider: str) -> str:
    return (providers.get(provider) or {}).get("label") or provider


def _price(model_id: str) -> dict | None:
    """A million tokens in the provider's own unit ({input, output}), from config/providers.yaml."""
    provider, name = _split(model_id)
    return (providers.model_entry(provider, name) or {}).get("price")


def _key_env(provider: str) -> str:
    """The .env variable holding a model provider's API key (config/providers.yaml)."""
    return providers.key_env(provider) or "GROQ_API_KEY"


def _provider_client(provider: str):
    """Groq's own SDK client, or an OpenAI-compatible one on the provider's base_url (None without a key or URL)."""
    if provider == "groq":
        return client
    key, url = os.environ.get(_key_env(provider), "").strip(), providers.base_url(provider)
    cached = _provider_clients.get(provider)
    if cached is None or cached[0] != (key, url):
        _provider_clients[provider] = ((key, url), _openai.OpenAI(
            api_key=key, base_url=url, timeout=providers.timeout_s(provider, 30.0), max_retries=0) if key and url else None)
    return _provider_clients[provider][1]


def _has_key(model_id: str) -> bool:
    provider = _split(model_id)[0]
    return providers.has_key(provider) and _provider_client(provider) is not None


def _usable(model_id: str) -> bool:
    """Its provider is on (enabled in config/providers.yaml, key in .env) and has a client. Never a paid OpenRouter
    model, nor one whose usage couldn't be counted (a provider counted per token, Cloudflare's neurons, without the
    model's price)."""
    provider = _split(model_id)[0]
    return (providers.is_on(provider) and _provider_client(provider) is not None and not _paid(model_id)
            and (providers.unit_of(provider) not in _PRICED_UNITS or bool(_price(model_id))))


def _streams(model_id: str) -> bool:
    provider, name = _split(model_id)
    return (providers.model_entry(provider, name) or {}).get("stream", True)


def _model_kwargs(model_id: str) -> dict:
    provider, name = _split(model_id)
    if provider == "gemini":
        kw = {"reasoning_effort": "low"}      # keep thinking short: replies are spoken
    # reasoning_effort is a gpt-oss knob (include_reasoning is Groq's own). Qwen given reasoning_effort="none" skipped
    # the tool call and claimed the task was added, so other models get their defaults.
    elif provider == "groq" and "gpt-oss" in name:
        kw = {"reasoning_effort": "low", "include_reasoning": False}
    else:
        kw = {}
    options = {k: v for k, v in ((providers.model_entry(provider, name) or {}).get("options") or {}).items()
               if not (provider in FREE_ONLY and k in _ROUTING_FIELDS)}
    if options:
        kw["extra_body"] = options
    return kw


def _messages_for(provider: str, msgs: list) -> list:
    """Gemini hands back an `extra_content` blob on tool calls that it wants echoed; other
    providers don't know the field, so it's stripped when failing over to them. IFM 400s
    on a follow-up turn unless every assistant message has a thinking field; "" is accepted
    (checked 2026-10-17), and history replies (other models, chat_log) have none to replay. The privacy tags
    (_public, _personal, _needs_personal, _needs_tools) never leave Javid."""
    msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in msgs]
    if provider == "ifm":
        msgs = [{**m, "reasoning_content": m.get("reasoning_content") or ""}
                if m["role"] == "assistant" else m for m in msgs]
    if provider == "gemini" or not any(
            c.get("extra_content") for m in msgs for c in m.get("tool_calls") or []):
        return msgs
    return [{**m, "tool_calls": [{k: v for k, v in c.items() if k != "extra_content"}
                                 for c in m["tool_calls"]]} if m.get("tool_calls") else m
            for m in msgs]


def _as_stream(completion):
    """A whole completion as the chunks a stream would have sent (its text and tool calls, then its usage): for a
    model set to stream: false in config/providers.yaml."""
    ns = types.SimpleNamespace
    msg = completion.choices[0].message if completion.choices else None
    calls = [ns(index=i, id=c.id, function=ns(name=c.function.name, arguments=c.function.arguments), model_extra={})
             for i, c in enumerate(getattr(msg, "tool_calls", None) or [])]
    yield ns(choices=[ns(delta=ns(content=getattr(msg, "content", None), tool_calls=calls or None))], usage=None,
             x_groq=None)
    yield ns(choices=[], usage=completion.usage, x_groq=None)


def _create(model_id: str, msgs: list, *, tools=None, stream=False, max_tokens=None):
    """One chat completion on whichever provider hosts `model_id`; returns the raw response. A paid OpenRouter model is
    refused here, before anything is sent."""
    if _paid(model_id):
        raise PaidModelRefused(f"{model_id} isn't one of OpenRouter's free models (ids ending ':free'): not called")
    provider, name = _split(model_id)
    kw = _model_kwargs(model_id)
    if tools:
        kw["tools"] = tools
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if provider in FREE_ONLY:
        body = kw.setdefault("extra_body", {})
        body["usage"] = {"include": True}     # the call's cost comes back with it: 0 for a free model (_check_cost)
        if tools:
            body["provider"] = {"require_parameters": True}   # only an endpoint that really takes the tools
    whole = stream and not _streams(model_id)
    if stream and not whole:
        kw["stream"] = True
        if provider != "groq":
            kw["stream_options"] = {"include_usage": True}   # else no token counts in the stream
    raw = _provider_client(provider).chat.completions.with_raw_response.create(
        model=name, messages=_messages_for(provider, msgs), **kw)
    if whole:
        return types.SimpleNamespace(headers=raw.headers, parse=lambda: _as_stream(raw.parse()))
    return raw


def _is_unavailable(err: Exception) -> bool:
    # DeepSeek answers 402 when the account balance is empty: park it instead of failing the turn.
    return isinstance(err, UNAVAILABLE) or getattr(err, "status_code", None) == 402


def _cooldown_for(err: Exception, provider: str = "groq") -> float:
    """How long a model rests after an error (providers.cooldown_seconds): a 429 until its retry-after, a 402 or a
    quota error until its retry-after or its allowance's reset, a rejected key or a missing model an hour, an outage
    30 s."""
    return providers.cooldown_seconds(provider, err) or 30.0


def _retry_after(err: Exception, default: float = 60.0) -> float:
    """Seconds until a rate-limited model is usable again (header, else Groq's 'try again in 7m11s', Google's 'retry
    in 6h21m37s')."""
    return providers.explicit_wait(err) or default


_DAILY_LIMIT = re.compile(r"per.?day|daily", re.I)


def _cool(model_id: str, err: Exception) -> float:
    """Rest a model after an error that makes it unavailable, in providers.py (so a restart respects it): only that
    model when the error is the model's (a 429, a missing model, an outage; for Groq and Gemini, whose allowances are
    per model, a quota too); the whole provider for a rejected key, an empty balance or an account's daily limit.
    Returns the seconds."""
    provider = _split(model_id)[0]
    wait = _cooldown_for(err, provider)
    kind = providers.error_kind(err)
    if providers.per_model(provider):
        # a 403 for one model is that model's (access to it); a 401 is the key's, so the whole provider rests
        own = kind in ("rate_limited", "quota", "missing", "unavailable") or (
            kind == "auth" and getattr(err, "status_code", None) == 403)
    else:
        # One allowance for the account (OpenRouter's free requests a day, Cloudflare's neurons): its daily limit rests
        # the provider; a busy free model ("rate-limited upstream"), a missing one or an outage only that model.
        own = kind in ("rate_limited", "missing", "unavailable") and not _DAILY_LIMIT.search(str(err))
    providers.cool(provider, wait, kind, model=model_id if own else "")
    return wait


# ---------- daily token budget and the budget guard ----------
# Groq's daily limit is a bucket that refills continuously (limit/24h per second: a 429 for 800 tokens
# says "try again in 5m45s"), not a midnight reset. daily_left() replays our usage log through that bucket,
# starting from Groq's last "Used N" report when there is one (that also covers other clients on the key).
# The guard itself is providers.py's: at 80 % of an allowance a model is "reserve" (tried only when the others
# fail), at 95 % it is off until it refills, and a model resting after a 429 isn't tried until its retry-after.


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


def _duration(text) -> float | None:
    """Groq's reset durations ("2m59.56s", "7.66s", "1h2m3s", "450ms") in seconds."""
    m = re.fullmatch(r"\s*(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?(?:(\d+)ms)?\s*", str(text or ""))
    if not m or not any(m.groups()):
        return None
    h, mins, secs, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + secs + ms / 1000


def _groq_usage(model: str) -> dict | None:
    """Groq's own numbers for one model, for the budget guard: tokens per day (its 429s and our log, through
    daily_left) and requests per day (its x-ratelimit headers; a reading older than its reset time counts as full
    again). The fuller of the two is the model's share."""
    if not model:
        return None
    now = time.time()
    parts = []
    known, left = _tpd_limit(model), daily_left(model, now)
    if known and left is not None:
        parts.append({"unit": "tokens", "used": known[0] - left, "amount": known[0]})
    lim = _model_info.get(model, {}).get("limits") or {}
    if lim.get("rpd_limit") and lim.get("rpd_remaining") is not None:
        reset = _duration(lim.get("rpd_reset"))
        if reset is None or now - lim.get("at", now) < reset:
            parts.append({"unit": "requests", "used": lim["rpd_limit"] - lim["rpd_remaining"], "amount": lim["rpd_limit"],
                          "next_reset": lim.get("at", now) + reset if reset else None})
    if not parts:
        return None
    worst = max(parts, key=lambda p: p["used"] / p["amount"])
    return {**worst, "per": "day", "parts": parts}


providers.set_reporter("groq", _groq_usage)


def _model_state(model_id: str, now: float | None = None) -> dict:
    """The budget guard's view of one model (providers.state): its provider's switch and key, cooldowns, and its
    allowance (Groq's and Gemini's are per model)."""
    return providers.state(_split(model_id)[0], model_id, now=now)


def _models_to_try(personal: bool = False, tools: bool = False) -> list[str]:
    """The models to try, in chain order: the ready ones, then those in reserve (at 80 % of an allowance: used only
    when the others fail). A model resting after an error, or off (95 % of an allowance, no free tier, no key, turned
    off in config/providers.yaml), isn't tried. personal: the request carries calendar, memory, chat history or tasks,
    so only providers with personal_ok may take it (PrivacyBlocked when no model in the chain has it). tools: the
    request needs tools, so only models that passed their tool tests may take it (NoToolModel when only chat-only ones
    could answer now)."""
    chain = [m for m in current_chain() if _usable(m)]
    if personal:
        chain = [m for m in chain if providers.personal_ok(_split(m)[0])]
        if not chain:
            raise providers.PrivacyBlocked("no model provider in the chain has personal_ok: true")
    states = {m: _model_state(m)["state"] for m in chain}
    ready = [m for m in chain if states[m] == "ready"]
    reserve = [m for m in chain if states[m] == "reserve"]
    if tools and (ready or reserve):
        able = [m for m in ready + reserve if tools_allowed(m)]
        if not able:
            raise NoToolModel("only chat-only models can answer now")
        ready, reserve = [m for m in ready if m in able], [m for m in reserve if m in able]
    if ready and reserve:
        print(f"[llm] {', '.join(reserve)} in reserve: used only if the others fail")
    return ready + reserve


# ---------- the tool gate: only models that passed their tool tests use tools ----------
# config/model_tests.json is written by regression.py --certify: each model's latest result on a fixed set of live cases
# (writes, a must-not-write, reads, a plain chat). A model whose verdict is "tools" gets the turn's tools; any other model,
# one never tested included, is chat-only: its requests go without tools, and a request that needs one skips it. Tracked
# in git, so the evidence for every model that may change Mahadi's data is reviewed with the code.
MODEL_TESTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "model_tests.json")
CERTIFYING: set[str] = set()      # regression.py --certify: the model under test has tools for its own tests
_tests_cache = {"path": None, "mtime": None, "models": {}}
_round_model = {"id": None}       # the model that answered the latest request (the write tools check it)
refused_calls: list = []          # this turn's tool calls that weren't run: (tool, why) (regression.py --certify reads it)


def model_tests() -> dict:
    """{model id: its latest tool tests} from config/model_tests.json (read again when it changes)."""
    try:
        mtime = os.stat(MODEL_TESTS_FILE).st_mtime
    except OSError:
        return {}
    if (_tests_cache["path"], _tests_cache["mtime"]) != (MODEL_TESTS_FILE, mtime):
        try:
            with open(MODEL_TESTS_FILE) as f:
                found = json.load(f).get("models")
        except (OSError, ValueError, AttributeError):
            found = None
        _tests_cache.update(path=MODEL_TESTS_FILE, mtime=mtime, models=found if isinstance(found, dict) else {})
    return _tests_cache["models"]


def tools_allowed(model_id: str) -> bool:
    """May this model use tools, and so change Mahadi's data? Only if it passed its tool tests."""
    return model_id in CERTIFYING or (model_tests().get(model_id) or {}).get("verdict") == "tools"


def _needs_tools(msgs: list) -> bool:
    """The request is part of a turn that needs tools: the turn's words point at one or one is required (think_stream
    tags it), it already called one, or it's an agent's."""
    return any(m.get("_needs_tools") or m.get("tool_calls") or m.get("role") == "tool" for m in msgs)


_NO_TOOLS_NOTE = ("\n[No tools for this reply: answer from what you know. Don't say you looked anything up, set "
                  "anything or changed anything.]")


def _without_tools(msgs: list) -> list:
    """A chat-only model's version of a request: the last user message says there are no tools."""
    i = max((n for n, m in enumerate(msgs) if m["role"] == "user"), default=None)
    if i is None or not isinstance(msgs[i].get("content"), str):
        return msgs
    return msgs[:i] + [{**msgs[i], "content": msgs[i]["content"] + _NO_TOOLS_NOTE}] + msgs[i + 1:]


def _cleared(name: str, run):
    """A write tool runs only for a model that passed its tool tests (the request routing already keeps chat-only
    models away from tools; this is the second lock). No model recorded: the call was made in code."""
    def call(args):
        model = _round_model["id"]
        if model and not tools_allowed(model):
            refused_calls.append((name, "chat-only model"))
            return f"Not done: {model} hasn't passed Javid's tool tests, so it can't change anything. Say so."
        return run(args)
    return call


for _name in WRITE_TOOLS:
    TOOL_MAP[_name] = _cleared(_name, TOOL_MAP[_name])


_NON_CHAT = ("whisper", "orpheus", "tts", "guard", "embed", "image", "imagen", "veo", "live", "audio")
_key_models: dict[str, dict] = {}


def _models_on_provider(provider: str) -> list[str] | None:
    """Chat-capable model ids (chain notation) this provider's key can use. Cached 10 min;
    None if the provider is unreachable, has no key or is turned off."""
    c = _provider_client(provider) if providers.is_on(provider) else None
    if c is None:
        return None
    if providers.unit_of(provider) in _PRICED_UNITS:   # counted per token (Cloudflare): only models with a price
        return [_chain_id(provider, m["id"]) for m in providers.models(provider) if m["price"]]
    cache = _key_models.setdefault(provider, {"at": 0.0, "ids": None})
    if time.time() - cache["at"] > 600:
        t0 = time.time()
        try:
            listed = c.models.list().data
            providers.record(provider, 0, op="models", latency_ms=(time.time() - t0) * 1000)
            raw_ids = sorted(m.id.removeprefix("models/") for m in listed
                             if provider not in FREE_ONLY or "tools" in ((m.model_extra or {}).get(
                                 "supported_parameters") or ["tools"]))
            keep = [i for i in raw_ids if not any(k in i for k in _NON_CHAT)]
            if provider in FREE_ONLY:
                keep = [i for i in keep if i.endswith(FREE_ONLY[provider])]
            if provider in ("gemini", "deepseek"):
                keep = [i for i in keep if provider in i]
            elif provider == "ifm":
                keep = [i for i in keep if "K2-" in i]   # also lists Jais (Arabic) and Jais-ASR
            cache.update(at=time.time(), ids=[i if provider == "groq" else f"{provider}:{i}" for i in keep])
        except Exception as e:
            providers.record(provider, 0, op="models", latency_ms=(time.time() - t0) * 1000,
                             status=providers.error_kind(e) if isinstance(e, API_ERRORS) else "error",
                             detail=type(e).__name__)
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
    on_key_by_provider = {e["name"]: _models_on_provider(e["name"]) for e in providers.entries("llm")}
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
        provider = _split(model)[0]
        on_key = on_key_by_provider.get(provider)
        guard = _model_state(model, now)
        cooling_for = max(0.0, (guard["until"] or now) - now) if guard["state"] == "cooldown" else 0.0
        if _paid(model):
            state, guard["why"] = "disabled", "not one of OpenRouter's free models (ids ending :free): never called"
        elif guard["state"] in ("disabled", "unknown"):
            state = "disabled"  # turned off (or missing) in config/providers.yaml
        elif not _has_key(model):
            state = "nokey"
        elif on_key is not None and model not in on_key:
            state = "missing"
        elif guard["state"] == "cooldown":
            state = "cooling"
        elif guard["state"] in ("off", "reserve"):
            state = guard["state"]  # budget guard: 95 % of an allowance (or no free tier) / 80 %: only if others fail
        else:
            state = "ready"
        share = (guard["usage"] or {}).get("share")
        models.append({
            "id": model, "name": _split(model)[1].split("/")[-1], "order": order,
            "provider": provider, "provider_label": _label(provider),
            "key_env": _key_env(provider), "personal_ok": providers.personal_ok(provider),
            "tools": tools_allowed(model), "tests": _test_summary(model), "free_only": provider in FREE_ONLY,
            "role": "primary" if order == 0 else "fallback", "state": state,
            "guard_why": guard["why"], "share": share, "off_until": guard["until"] if state == "off" else None,
            "retry_in_s": round(cooling_for), "tokens_24h": day, "per_turn": per_turn,
            "session_requests": llm_usage.get(model, 0),
            "last_used": max((e["ts"] for e in mine), default=None),
            "limits": info.get("limits"),
            "tpd_limit": quota["tpd_limit"] if quota else None,
            "est_used_24h": est_used, "remaining_24h": remaining,
            "reported": ({"used": quota["reported_used"], "at": quota["reported_at"]}
                         if quota and quota["reported_at"] else None),
            "last_error": info.get("last_error"),
            "sends_reasoning_effort": "gpt-oss" in model,
        })
    serving = (next((m for m in models if m["state"] == "ready"), None)
               or next((m for m in models if m["state"] == "reserve"), None))
    if serving:
        serving["state"] = "serving"
    keys = {}
    for e in providers.entries("llm"):
        name = e["name"]
        key = os.environ.get(_key_env(name), "").strip()
        keys[name] = {"label": e["label"], "key_env": _key_env(name),
                      "has_key": bool(key), "key_hint": key[-4:] if len(key) >= 8 else "",
                      "editable": _key_editable(name)}
    return {
        "providers": keys, "is_default_chain": chain == default_chain(),
        "now": now, "serving": serving["id"] if serving else None,
        "chain_size": len(chain), "tracking_since": tracking_since,
        "ready_count": sum(m["state"] in ("serving", "ready", "reserve") for m in models),
        "min_request_tokens": LLM_MIN_REQUEST_TOKENS,
        "reserve_at": providers.RESERVE_AT, "off_at": providers.OFF_AT,
        "models": models,
        "web": web_tools.status(),
        "on_key": [{"id": i, "in_chain": i in chain, "provider": p}
                   for p, ids in on_key_by_provider.items() for i in ids or []],
    }


def _test_summary(model: str) -> dict | None:
    """What the Providers panel shows of a model's tool tests (config/model_tests.json)."""
    t = model_tests().get(model)
    if not t:
        return None
    return {k: t.get(k) for k in ("verdict", "tested", "cases", "pass_rate", "tool_accuracy", "unsafe", "median_s",
                                  "why")}


def _key_editable(provider: str) -> bool:
    """A key that can be pasted in the Providers dialog: one variable (Cloudflare needs its account id too), and not
    Groq's (it also drives speech recognition: set it in .env)."""
    e = providers.get(provider)
    return bool(e) and e["kind"] == "llm" and provider != "groq" and len(e["env"]) == 1 and bool(e["base_url"])


def providers_status() -> dict:
    """The Providers panel: every provider in config/providers.yaml (status, used / allowance in its own unit, reset
    time, last error), the language models of the chain (llm_status) and Tavily's day."""
    llm = llm_status()
    out = providers.status()
    for row in out["providers"]:
        row["models"] = [m["id"] for m in llm["models"] if m["provider"] == row["name"]]
    return {**out, "llm": llm, "next_model": next_model()}


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
    # Owner-only from the moment it exists (open() then chmod left the keys readable by anyone in between).
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write("\n".join(kept) + "\n")
    os.chmod(tmp, 0o600)     # O_CREAT keeps the mode of a .tmp left behind by a crash
    os.replace(tmp, ENV_FILE)


def _forget_provider(provider: str) -> None:
    """After a key change: rebuild the client and drop cooldowns/errors from the old key."""
    _provider_clients.pop(provider, None)
    _key_models.pop(provider, None)
    providers.clear_cooldown(provider)
    for m in set(current_chain()) | set(default_chain()):
        if _split(m)[0] == provider:
            _model_info.get(m, {}).pop("last_error", None)


class KeyRejected(ValueError):
    """The provider said no to the key (OpenRouter's key check)."""


def _verify_key(provider: str, c) -> None:
    """Raise the provider's API error (or KeyRejected) if it rejects the key; spends no tokens. api.ifm.ai lists
    its models without checking the key at all, so IFM gets a chat request with no messages:
    a bad key is a 401, a good one the 400 for the empty request (checked 2026-10-16). OpenRouter's model list is
    public too: its key endpoint checks the key (GET /key, which uses none of the free requests)."""
    if provider == "openrouter":
        import httpx
        try:
            r = httpx.get(str(c.base_url).rstrip("/") + "/key", headers={"Authorization": f"Bearer {c.api_key}"},
                          timeout=15.0)
        except httpx.HTTPError as e:
            raise ConnectionError(type(e).__name__) from None
        if r.status_code in (401, 403):
            raise KeyRejected("rejected")
        if r.status_code >= 400:
            raise ConnectionError(f"HTTP {r.status_code}")
        return
    if provider != "ifm":
        c.models.list()
        return
    try:
        c.chat.completions.create(model=(providers.models("ifm") or [{"id": "IFM/K2-Horizon-375B-A23B"}])[0]["id"],
                                  messages=[], max_tokens=1)
    except API_ERRORS as e:
        if getattr(e, "status_code", None) not in (400, 422):   # key accepted, request refused
            raise


def set_provider_key(provider: str, key: str) -> None:
    """Check `key` against the provider (a model-list call: no tokens spent), then save it to
    .env and use it straight away. Raises ValueError with a message safe to show the user."""
    if not _key_editable(provider):
        raise ValueError("That provider's key can't be changed here.")
    cfg = {"label": _label(provider)}
    key = key.strip()
    if not _KEY_FORMAT.match(key):
        raise ValueError("That doesn't look like an API key (letters, digits, - _ . only, no spaces).")
    t0 = time.time()
    try:
        _verify_key(provider, _openai.OpenAI(api_key=key, base_url=providers.base_url(provider), timeout=15.0,
                                             max_retries=0))
    except KeyRejected:
        providers.record(provider, 0, op="verify", latency_ms=(time.time() - t0) * 1000, status="auth",
                         detail="KeyRejected")
        raise ValueError(f"{cfg['label']} rejected this key. Nothing was saved.") from None
    except (*API_ERRORS, OSError) as e:
        providers.record(provider, 0, op="verify", latency_ms=(time.time() - t0) * 1000,
                         status=providers.error_kind(e), detail=type(e).__name__)
        if getattr(e, "status_code", None) in (400, 401, 403):
            raise ValueError(f"{cfg['label']} rejected this key. Nothing was saved.") from None
        raise ValueError(f"Couldn't reach {cfg['label']} to check the key ({type(e).__name__}). "
                         "Nothing was saved; try again.") from None
    providers.record(provider, 0, op="verify", latency_ms=(time.time() - t0) * 1000)
    _write_env(_key_env(provider), key)
    os.environ[_key_env(provider)] = key
    _forget_provider(provider)


def clear_provider_key(provider: str) -> None:
    if not _key_editable(provider):
        raise ValueError("That provider's key can't be changed here.")
    _write_env(_key_env(provider), None)
    os.environ.pop(_key_env(provider), None)
    _forget_provider(provider)


def ping_models(only_provider: str | None = None) -> list[dict]:
    """Send a tiny no-tools request to every model in the chain (or one provider's), in parallel.
    Shows which ones answer right now and how fast; a success clears that model's cooldown. A model the budget
    guard keeps off (95 % of an allowance) or that is resting after a 429 isn't pinged: a ping is a real request."""
    from concurrent.futures import ThreadPoolExecutor

    def ping(model: str) -> dict:
        t0 = time.time()
        ms = lambda: round((time.time() - t0) * 1000)
        provider = _split(model)[0]
        if _paid(model):
            return {"id": model, "ok": False, "ms": 0, "skipped": True, "error": "Not tested: not a free model"}
        if not _has_key(model):
            return {"id": model, "ok": False, "ms": 0, "error": f"No API key: set {_key_env(provider)} in .env"}
        guard = _model_state(model)
        if guard["state"] in ("disabled", "unknown", "off", "cooldown"):
            return {"id": model, "ok": False, "ms": 0, "skipped": True,
                    "error": f"Not tested: {guard['why'] or guard['state']}"}
        try:
            raw = _create(model, [{"role": "user", "content": "ping"}], max_tokens=8)
            _note_headers(model, raw.headers)
            done = raw.parse()
            if done.usage:
                _record_usage(model, done.usage)
            providers.clear_cooldown(provider, model)
            providers.record(provider, _call_units(provider, done.usage, model, (12, 8)), model=model, op="ping",
                             latency_ms=ms())
            return {"id": model, "ok": True, "ms": ms()}
        except Exception as e:
            kind = providers.error_kind(e) if isinstance(e, API_ERRORS) else "error"
            if isinstance(e, API_ERRORS) and _is_unavailable(e):
                _cool(model, e)
                _note_error(model, e)
            providers.record(provider, 0, model=model, op="ping", latency_ms=ms(), status=kind, detail=type(e).__name__)
            return {"id": model, "ok": False, "ms": ms(), "error": f"{type(e).__name__}: {_redact(str(e))[:140]}"}

    chain = [m for m in current_chain() if only_provider in (None, _split(m)[0])]
    with ThreadPoolExecutor(max_workers=max(1, len(chain))) as pool:
        return list(pool.map(ping, chain))


class NoModelReady(Exception):
    """No model in the chain may take a request now: all resting after an error, off at their free limit, without a
    key or turned off. Nothing is sent (think_stream says so)."""


class NoToolModel(NoModelReady):
    """The request needs tools and only chat-only models (no pass in config/model_tests.json) could answer now."""


def _needs_personal(msgs: list) -> bool:
    """The request carries (or its turn needs) Mahadi's calendar, memory, chat history or tasks."""
    return any(m.get("_needs_personal") for m in msgs)


def _public_request(msgs: list, tools: list) -> tuple[list, list]:
    """The request as a provider without personal_ok gets it: chat history and earlier drafts dropped ("_personal"),
    each message's personal-free text where it has one ("_public": the system prompt without the memory block, the
    [Context] note without the older-chat summary and the calendar facts), and no personal tools."""
    out = [({**m, "content": m["_public"]} if "_public" in m else m) for m in msgs if not m.get("_personal")]
    return out, [t for t in tools if t["function"]["name"] not in PERSONAL_TOOLS]


def _call_units(provider: str, usage, model: str = "", estimate: tuple | None = None) -> float:
    """One request in its provider's own unit: tokens for Groq, one for per-request allowances, and for a provider
    counted per token (Cloudflare's neurons) its tokens at the model's price, from `estimate` ((input, output) tokens)
    when the response gave no usage."""
    unit = providers.unit_of(provider)
    if unit == "tokens":
        return float(getattr(usage, "total_tokens", 0) or 0) if usage else 0.0
    price = _price(model) if model else None
    if price:
        tin = getattr(usage, "prompt_tokens", None) if usage else None
        tout = getattr(usage, "completion_tokens", None) if usage else None
        if tin is None and estimate:
            tin, tout = estimate
        return round(((tin or 0) * price["input"] + (tout or 0) * price["output"]) / 1e6, 3)
    return 1.0 if unit in ("requests", "calls") else 0.0


def _estimate(msgs: list, tools: list | None, out_tokens: int = 600) -> tuple[int, int]:
    """(input, output) tokens a request takes, on the high side (3.5 characters a token): for the budget guard of a
    provider counted per token, before a call and when a response gives no usage."""
    chars = sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in msgs)
    return int((chars + len(json.dumps(tools or []))) / 3.5) + 1, out_tokens


def _check_cost(model: str, usage) -> None:
    """OpenRouter reports what each call cost, and a free model's is 0. Anything more means a paid model was reached
    somehow: OpenRouter is switched off (a 30-day rest, until someone checks the account) and it's said loudly."""
    provider = _split(model)[0]
    if provider not in FREE_ONLY or usage is None:
        return
    cost = getattr(usage, "cost", None)
    if cost is None:
        cost = (getattr(usage, "model_extra", None) or {}).get("cost")
    if isinstance(cost, (int, float)) and cost > 0:
        providers.cool(provider, 30 * 86400, "a call was billed: check the OpenRouter account")
        print(f"[llm] {model} cost {cost}: OpenRouter is off until someone checks the account")


def _open_stream(msgs: list, tools: list):
    """Start a streaming completion on the first model in the chain that may take this request (the budget guard:
    ready ones first, reserve ones only after them). The privacy rule: a request that needs personal data goes only to
    providers with personal_ok; every other provider gets it without its personal parts (_public_request). The tool
    gate: a model without a pass in config/model_tests.json gets the request without tools, and a request of a turn that
    needs tools skips it. A provider counted per token (Cloudflare) isn't asked when this request could take it past its
    free allowance. Returns (model, stream, start time)."""
    candidates = _models_to_try(personal=_needs_personal(msgs), tools=bool(tools) and _needs_tools(msgs))
    if not candidates:
        raise NoModelReady("no model in the chain can take a request now")
    last_error = None
    for model in candidates:
        provider = _split(model)[0]
        send, send_tools = (msgs, tools) if providers.personal_ok(provider) else _public_request(msgs, tools)
        if not tools_allowed(model):
            send, send_tools = _without_tools(send), []
        if _price(model) and not providers.allow(provider, model, _call_units(provider, None, model,
                                                                              _estimate(send, send_tools)))[0]:
            print(f"[llm] {model}: this request could go past its free allowance, skipping it")
            continue
        t0 = time.time()
        try:
            raw = _create(model, send, tools=send_tools, stream=True)
        except API_ERRORS as e:
            providers.record(provider, 0, model=model, op="chat", latency_ms=(time.time() - t0) * 1000,
                             status=providers.error_kind(e), detail=type(e).__name__)
            if not _is_unavailable(e):
                raise       # e.g. malformed tool call: think_stream retries those
            # 429: out of quota for now. 5xx/network: overloaded. 402/404/401/403: can't serve.
            wait = _cool(model, e)
            _note_error(model, e)
            print(f"[llm] {model} unavailable ({type(e).__name__}), skipping it for {wait:.0f}s")
            last_error = e
            continue
        _note_headers(model, raw.headers)
        if model != current_chain()[0]:
            print(f"[llm] answering with fallback {model}")
        llm_usage[model] = llm_usage.get(model, 0) + 1
        return model, raw.parse(), t0
    if last_error is None:
        raise NoModelReady("every model that could take this request is at its free allowance")
    raise last_error


def _stream_completion(tool_calls: dict, msgs: list, tools: list):
    """Stream one completion. Yields content deltas; collects tool calls into `tool_calls`. The request goes in the
    usage ledger in its provider's own unit, with its latency."""
    span = tracing.llm_start()
    model, t0, done = None, time.time(), {}
    _round_model["id"] = None
    try:
        model, stream, t0 = _open_stream(msgs, tools)
        _round_model["id"] = model
        yield from _read_stream(model, stream, tool_calls, span, done)
    finally:
        tracing.llm_end(span, model)
        if model:
            provider = _split(model)[0]
            units = _call_units(provider, done.get("usage"), model,
                                _estimate(msgs, tools, int(done.get("out_chars", 0) / 3.5) + 1))
            providers.record(provider, units, model=model, op="chat",
                             latency_ms=(time.time() - t0) * 1000, status="ok" if done.get("complete") else "aborted")
            _check_cost(model, done.get("usage"))


def _read_stream(model: str, stream, tool_calls: dict, span, done: dict | None = None):
    done = {} if done is None else done
    counted = False
    for chunk in stream:
        tracing.llm_first(span)
        if not counted:
            usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
            if usage:   # arrives on the final chunk
                _record_usage(model, usage)
                done["usage"] = usage
                counted = True
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            done["out_chars"] = done.get("out_chars", 0) + len(delta.content)
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
                done["out_chars"] = done.get("out_chars", 0) + len(tc.function.arguments)
    done["complete"] = True


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
    # Privacy tags (see _public_request): "_public" is the text a provider without personal_ok gets instead, and
    # "_personal" messages (chat history) it doesn't get at all.
    out = [{"role": "system", "content": messages[0]["content"] + "\n\n" + memory_store.block(),
            "_public": messages[0]["content"]}]
    # The previous reply's screen part (a skill's or agent's full version) rides along for one turn only, so a
    # follow-up can use it: "yes, add those blocks" after the planner, "tell me more about the third one".
    last = next((m for m in reversed(messages[1:turn_start]) if m["role"] == "assistant" and not m.get("tool_calls")), None)
    for i, (user, reply) in enumerate(recent):
        if i == len(recent) - 1 and last and last.get("screen") and reply == last["content"]:
            reply += f"\n[Shown on screen, not spoken: {_clip(last['screen'], SCREEN_CHARS)}]"
        if user:
            out.append({"role": "user", "content": user, "_personal": True})
        if reply:
            out.append({"role": "assistant", "content": reply, "_personal": True})
    current = messages[turn_start:]
    out.append({**current[0], "content": context_block(_summarize(older), lang, _facts_note(turn_start))
                + current[0]["content"], "_public": context_block("", lang) + current[0]["content"]})
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
    return f"{name.split('/')[-1]} on {_label(provider)}"


def next_model() -> str:
    """The model that would answer now ("gpt-oss-20b on Groq"): chain, keys, cooldowns and the budget guard, no
    network (llm_status lists every provider's models)."""
    ready = _models_to_try()
    return _model_name(ready[0]) if ready else "none (every model is resting or at its free limit)"


def _self_facts() -> dict:
    """What about.py says about the running Javid: read from the code and settings, never from the model."""
    chain = [m for m in current_chain() if _usable(m)] or default_chain()[:1]
    cleared = [e["label"] for e in providers.entries() if e["personal_ok"] and providers.is_on(e["name"])]
    return {"model": _model_name(chain[0]), "backups": [_model_name(m) for m in chain[1:]],
            "provider": _label(_split(chain[0])[0]), "stt": STT_MODEL,
            "personal_en": ", ".join(cleared) or "none yet", "personal_bn": ", ".join(cleared) or "এখনো কেউ না",
            "voice_en": EDGE_VOICE, "voice_bn": EDGE_VOICE_BN,
            "skills": [s["name"] for s in skills.all_skills()], "agents": [a["name"] for a in agents.all_agents()]}


_turn_text = {"text": ""}   # the message being answered (think_stream)

# ---------- privacy: what a web search may carry ----------
_STOP = frozenset("""the and for with from that this what when where which who how why about into onto over under near
are was were has have had not but you your our his her its they them their can could would should will just than then
there here also some any all more most less very much many one two three today tonight tomorrow yesterday now new
news latest best top time times open opening hours please find search look show tell give bring get make take""".split())
_PUBLIC_WORDS = frozenset(re.findall(r"[a-z0-9]+", SYSTEM_PROMPT.lower())) | {"mahadi", "javid", "bangladesh", "dhaka"}


def _words_of(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9ঀ-৿]+", (text or "").lower()) if len(w) >= 3 and w not in _STOP]


def _names_in(text: str) -> set[str]:
    """Capitalized words that don't start a sentence: names ("Nadia", "Rafi", "Gulshan")."""
    out, first = set(), True
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]*|[.!?:;]", text or ""):
        token = m.group()
        if token in ".!?:;":
            first = True
            continue
        if not first and token[0].isupper() and len(token) >= 3 and not token.isupper():
            out.add(token.lower().strip("'-"))
        first = False
    return out - _STOP


def _personal_items() -> list[tuple[str, str]]:
    """(where it's from, text) for each piece of Mahadi's personal data a search query could carry: his tasks, his
    memory lines, and the calendar and task results in Javid's context."""
    items = [("your tasks", _task_parts(line)[2]) for line in _read_lines()]
    entries = memory_store.all_entries()
    items += [("your memory", e) for f in entries for e in entries[f]]
    for m in messages:
        if m.get("role") == "tool" and m.get("_needs_personal"):
            items += [("your calendar or tasks", line) for line in str(m.get("content") or "").splitlines() if line.strip()]
    return items


def _personal_leak(text: str) -> str | None:
    """Where the personal data in `text` (a web search query) comes from, or None: two words in a row from a task, a
    memory line or a calendar/task result, or a name from one. Best effort, and chat history isn't checked. Words
    Mahadi used in this message don't count (he said them to be searched), nor Javid's own (the system prompt)."""
    query = _words_of(text)
    if not query:
        return None
    said = set(_words_of(_turn_text["text"])) | _PUBLIC_WORDS
    pairs, asked = set(zip(query, query[1:])), set(query)
    for where, item in _personal_items():
        words = _words_of(item)
        if any((a, b) in pairs and not {a, b} <= said for a, b in zip(words, words[1:])):
            return where
        if (_names_in(item) & asked) - said:
            return where
    return None


web_tools.privacy_check = _personal_leak
web_tools.user_text = lambda: _turn_text["text"]      # search.py reads the intent from the message too


def _personal_models() -> list[str]:
    """Models in the chain whose provider is on and has personal_ok: true."""
    return [m for m in current_chain() if _usable(m) and providers.personal_ok(_split(m)[0])]


# A message that asks for Mahadi's own calendar, tasks, memory or past chats (the privacy rule). Much narrower than the
# tool groups, which err wide on purpose: "will it rain tomorrow?" or "add 5 and 7" mustn't count. What this misses is
# caught later: a provider without personal_ok isn't offered the personal tools, and a model that reaches for one makes
# the turn personal (think_stream).
_PERSONAL_ASK = re.compile(
    r"\bmy\s+(?:calendar|schedule|agenda|diary|tasks?|to-?dos?|list|reminders?|meetings?|appointments?|events?|plans?|"
    r"day|week|weekend|morning|afternoon|evening|memory|notes?|chats?|history)\b|"
    r"\b(?:am i|are we)\s+(?:free|busy|available)\b|"
    r"\bwhat(?:'s| is| do i have)\s+on\s+(?:my|for|today|tonight|tomorrow|this|next|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)\b|\bdo i have\s+(?:any|anything|a|an)\b|"
    r"\b(?:add|put|schedule|book|move|push|reschedule|cancel|delete|remove|complete|finish|tick off|mark)\b.{0,40}?"
    r"\b(?:tasks?|to-?dos?|calendar|events?|meetings?|appointments?|reminders?)\b|"
    r"\bremind me to\b|\b(?:remember|don'?t forget|keep in mind|note that)\b|"
    r"\b(?:what(?:'s| is| are| was)|who(?:'s| is| are)|when(?:'s| is| was)|where(?:'s| is| do i))\s+my\b|"
    r"\bdo you (?:know|remember)\s+(?:my|me|what i)\b|\babout me\b|"
    r"\b(?:did|have) (?:i|we|you) (?:say|tell|mention|ask|talk|discuss)\b|\bwe (?:talked|discussed|spoke)\b|"
    r"\bgood (?:morning|afternoon|evening)\b|\bmorning brief|"
    r"ক্যালেন্ডার|মিটিং|অ্যাপয়েন্টমেন্ট|টাস্ক|আমার কাজ|মনে রাখ|বলেছিলাম", re.I)


def _turn_needs_personal(user_text: str, turn_start: int, agent_first: str | None, required: list) -> bool:
    """Does answering need Mahadi's calendar, memory, chat history or tasks? His words (_PERSONAL_ASK), the tools the
    turn must call (remember that..., what did I say...), the agent it goes to (the planner reads the calendar), or a
    short follow-up to a turn that used them ("yes", "the second one", "and on Friday?")."""
    if _PERSONAL_ASK.search(user_text) or PERSONAL_TOOLS & {tool for tool, _ in required}:
        return True
    if agent_first and PERSONAL_TOOLS & set(agents.agent_tools(agent_first)):
        return True
    return len(user_text.split()) <= 8 and bool(PERSONAL_TOOLS & _tools_used_in_previous_turn(turn_start))


def _personal_call(call: dict) -> bool:
    """A tool call whose result is personal data: a personal tool, or an agent that uses one (the planner)."""
    if call["name"] in PERSONAL_TOOLS:
        return True
    if call["name"] == "delegate":
        try:
            agent = json.loads(call.get("arguments") or "{}").get("agent", "")
        except (ValueError, AttributeError):
            return True
        return bool(PERSONAL_TOOLS & set(agents.agent_tools(str(agent).strip().lower())))
    return False


def _code_answer(user_text: str):
    """A watch to set, list or cancel, a rates question, a market brief or a news request that code answers whole
    (watches.request, rates.request, markets.brief_request, news.request): a function of the reply language giving
    the answer's generator. None leaves the message to the model."""
    if (req := watches.request(user_text)):     # first: "tell me when bitcoin goes above 70k" isn't a market brief
        return lambda lang: watches.answer(req, lang)
    if (req := rates.request(user_text)):
        return lambda lang: rates.answer(req, lang)
    if (req := markets.brief_request(user_text)):
        return lambda lang: markets.brief(req, lang)
    if (req := news.request(user_text)):
        return lambda lang: news.answer(req, lang)
    return None


def _relay(answer, lang: str, used: list):
    """Pass a code answer's tool events on, with "Searching…" before its web search as for the model's, and
    return its (spoken, screen). The tools it used go in `used` (the reply records them)."""
    said_searching = False
    while True:
        try:
            kind, value = next(answer)
        except StopIteration as done:
            return done.value
        if kind == "tool":
            print(f"[tool] {value} (answered in code)")
            used.append(value)
            if value in WEB_TOOLS and not said_searching:
                said_searching = True
                yield ("say", phrase("searching", lang))
        yield (kind, value)


def think_stream(user_text: str, gate=None, blocked: frozenset = frozenset()):
    """Yield ("tool", name), ("say", text), ("sources", links), ("lang", code) and ("delta", text)
    events as the reply is produced. "say" is spoken but isn't part of the reply ("Searching…");
    "lang" means the message switched the reply language.

    Each round's text is held until the round ends: if the model also called
    tools, that text was a guess made before seeing the results (e.g. a made-up
    time before get_datetime), so it is dropped instead of spoken.

    gate: for this turn, WRITE_TOOLS go to gate(name, args) instead of running (Telegram's Approve buttons), and a
    typed yes doesn't decide a memory proposal. blocked: tools this turn never gets or runs, whatever the model
    asks for (Telegram: the web tools for a message with a link).
    """
    global _write_gate
    _start_turn_usage()
    calendar_tools.new_turn()   # lets delete_event tell "next message" from "same turn"
    web_tools.new_turn()
    skills.new_turn()
    memory_store.new_turn()
    agents.new_turn()
    _new_proposals.clear()
    refused_calls.clear()
    _round_model["id"] = None
    _turn_text["text"] = user_text      # web_tools' privacy check lets through the words Mahadi said himself
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
    messages.append({"role": "user", "content": user_text})
    log_message("user", user_text)
    reply = ""
    links = []
    screen = ""
    skill_results = []    # what the active skill's tools returned (the screen part if the model leaves it out)
    via_agent = None      # an agent's answer ({agent, spoken, screen}) when one handled the turn
    code_tools = []       # tools a code answer used (the model's are in its tool calls)
    previous = next((m for m in reversed(messages[1:turn_start]) if m["role"] == "assistant" and not m.get("tool_calls")), None)
    lang = get_language()
    try:
        _write_gate = gate
        # A yes/no right after a memory proposal decides it here: the model can't save memory. Not on a gated
        # channel: there only a button saves, and a typed yes there must not approve a card on the web page.
        decided = None if gate else memory_store.answer_from_reply(user_text)
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
        # Answered in code, no LLM request: a self-check or where the last answer came from (the model can't see
        # either), a question about Javid itself (the model only guesses), a rates question, a market brief or a
        # news request (the model only copied the figures and headlines, and got their sources wrong).
        check = selfcheck.request(user_text)
        own = None if check else about.answer(user_text, lang, _self_facts())
        in_code = ((lambda lang: selfcheck.answer(check, user_text, lang, previous)) if check else
                   None if own else _code_answer(user_text))
        if own or in_code:
            tracing.mark("routing_done")
        if in_code:
            try:
                own = yield from _relay(in_code(lang), lang, code_tools)
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
        # Privacy rule: a turn that needs calendar, tasks, memory or past chats goes only to providers with
        # personal_ok: true (config/providers.yaml). With none in the chain, nothing is sent and Javid says so.
        if _turn_needs_personal(user_text, turn_start, agent_first, required):
            if not _personal_models():
                print("[privacy] this needs personal data and no model provider has personal_ok: true")
                raise providers.PrivacyBlocked("no model provider has personal_ok: true")
            messages[turn_start]["_needs_personal"] = True
        # The tool gate (_open_stream): a turn whose words point at a tool, or that must call one, goes only to models
        # that passed their tool tests; a chat-only model may answer the rest, without tools.
        if _matched_tools(user_text, _tools_used_in_previous_turn(turn_start)) or required or agent_first or skill_first:
            messages[turn_start]["_needs_tools"] = True
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
            round_tools = [t for t in tools if t["function"]["name"] not in blocked] if round_no < rounds - 1 else []
            req = _request_messages(turn_start, lang)
            for at, extra in reversed(notes):   # later notes first, so earlier cut points stay put
                cut = len(req) - (len(messages) - at)
                req = req[:cut] + extra + req[cut:]
            _round_model["id"] = None      # set by the request (_stream_completion); none for a call made in code
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
                    if (wanted := re.search(r"tool '([^']+)'", str(e))) and wanted.group(1) in PERSONAL_TOOLS:
                        messages[turn_start]["_needs_personal"] = True   # it reached for personal data
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
                notes.append((len(messages), [{"role": "assistant", "content": text, "_personal": True}, {"role": "user", "content": note}]))
                if not any(t["function"]["name"] == tool for t in tools):
                    tools = tools + [t for t in TOOLS if t["function"]["name"] == tool]
                continue
            if not tool_calls and not retried_lang and wrong_language(text, lang):
                retried_lang = True
                print(f"[lang] not in {lang}, asking for a rewrite: {text[:60]!r}")
                notes.append((len(messages), [{"role": "assistant", "content": text, "_personal": True},
                                              {"role": "user", "content": language_note(lang)}]))
                continue
            if not tool_calls and web_tools.used():
                problems = web_tools.check_reply(text + ("\n" + via_agent["screen"] if via_agent else ""))
                if problems and not retried_web and not via_agent:
                    retried_web = True
                    print(f"[web] answer rejected: {problems}")
                    notes.append((len(messages), [{"role": "assistant", "content": text, "_personal": True},
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
                    notes.append((len(messages), [{"role": "assistant", "content": text, "_personal": True},
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
                                          max_sentences=None if skill or via_agent else web_tools.MAX_SENTENCES,
                                          figure_asked=bool(_FIGURE_ASK.search(user_text)))
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
            # Only tools this round offered run. Groq refuses a call to any other ("not in request", handled above by
            # widening to every tool); other providers may pass one through, so the same happens here. A chat-only
            # model was offered none. A call made in code (an agent's handover; regression.py's scripts) isn't checked.
            served = _round_model["id"]
            offered = (None if served is None else
                       {t["function"]["name"] for t in round_tools} if tools_allowed(served) else set())
            stray = [] if offered is None else [c["name"] for c in calls
                                                if c["name"] not in offered and c["name"] not in blocked]
            if stray and not widened and offered and round_no < rounds - 2:
                print(f"[retry] needed a tool outside {sorted(offered)}: {stray}")
                widened = True
                tools = [t for t in TOOLS if not (skill_first and t["function"]["name"] in WEB_TOOLS)]
                if PERSONAL_TOOLS & set(stray):
                    messages[turn_start]["_needs_personal"] = True   # it reached for personal data
                continue
            for c in calls:
                c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}"   # some providers send none
            messages.append({
                "role": "assistant",
                "content": text or None,
                **({"_needs_personal": True} if any(_personal_call(c) for c in calls) else {}),
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]},
                     **({"extra_content": c["extra"]} if c.get("extra") else {})}
                    for c in calls
                ],
            })
            for call in calls:
                if call["name"] in blocked or (offered is not None and call["name"] not in offered):   # never run
                    if call["name"] not in blocked:
                        refused_calls.append((call["name"], "not offered"))
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": f"{call['name']} isn't available for this message; answer without it."})
                    continue
                try:
                    args = json.loads(call["arguments"] or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("not an object")
                except ValueError:     # Groq refuses malformed JSON itself; another provider may hand it over
                    refused_calls.append((call["name"], "invalid arguments"))
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": f"Those {call['name']} arguments weren't valid JSON: call it again "
                                                "with a JSON object."})
                    continue
                if call["name"] in ("web_search", "news_headlines") and not said_searching:
                    said_searching = True
                    yield ("say", phrase("searching", lang))   # spoken while the search runs
                elif call["name"] == "delegate" and not said_searching:
                    said_searching = True
                    yield ("say", phrase("working", lang))     # an agent takes a while
                yield ("tool", call["name"])
                fn = TOOL_MAP.get(call["name"])
                span = tracing.tool_start(call["name"])
                try:
                    result = fn(args) if fn else f"Unknown tool: {call['name']}"
                    if isinstance(result, types.GeneratorType):   # an agent: relay its progress, then its result
                        result = yield from result
                except TypeError as e:   # a made-up argument name (propose_memory(user=...)): let it retry
                    refused_calls.append((call["name"], "wrong arguments"))
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
                    if PERSONAL_TOOLS & set(skills.active()["tools"]):   # morning-brief: tasks and calendar
                        messages[turn_start]["_needs_personal"] = True
                    # Unlock the skill's tools for the rest of the turn (and its follow-up turn).
                    have = {t["function"]["name"] for t in tools}
                    tools = tools + [t for t in TOOLS if t["function"]["name"] in skills.active()["tools"]
                                     and t["function"]["name"] not in have]
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                    **({"_needs_personal": True} if _personal_call(call) else {}),
                })
        else:
            reply = phrase("stuck", lang)
            yield ("delta", reply)

    except API_ERRORS:
        reply = phrase("api_failed", lang)
        yield ("delta", reply)
    except providers.PrivacyBlocked:
        screen = NO_PERSONAL_SCREEN
        yield ("screen", screen)
        reply = phrase("no_personal", lang)
        yield ("delta", reply)
    except NoToolModel:
        print("[llm] this turn needs tools and only chat-only models can answer now")
        reply = phrase("no_tool_model", lang)
        yield ("delta", reply)
    except NoModelReady:
        print("[llm] no model in the chain can take a request now")
        reply = phrase("no_model", lang)
        yield ("delta", reply)

    finally:
        _write_gate = None
        # Runs on normal finish and when the client aborts mid-stream,
        # so history always gets whatever was actually said.
        # What the reply was based on (its tools and links) stays with it: selfcheck answers "where did you get that?".
        used = list(dict.fromkeys([c["function"]["name"] for m in messages[turn_start:] for c in m.get("tool_calls") or []]
                                  + code_tools + sorted(agents.tools_used())))
        basis = {**({"tools": used} if used else {}), **({"sources": links} if links else {})}
        messages.append({"role": "assistant", "content": reply.strip(), **({"screen": screen} if screen else {}), **basis})
        if messages[turn_start].get("_needs_personal") and reply.strip():
            _personal_speech.append(" ".join(reply.split()))     # the backup voice may not say it (_eleven)
        if reply.strip():
            log_message("assistant", reply.strip(), **basis,
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
# waits and retries so the voice is the locked one whenever edge-tts can give it; only after every try fails does the
# backup voice speak (ElevenLabs, below), and without that, nothing: the reply is text with a "voice unavailable" note.
EDGE_ATTEMPTS = 3
# A try waits this long for its first audio (longer on each retry), then at most EDGE_STALL_SEC between chunks: a
# slow sentence that is still arriving is kept. A fixed 6 s for the whole sentence threw those away during a slow
# spell (2026-10-24: every try of a news answer timed out while the endpoint answered a single request in 2-3.5 s).
EDGE_FIRST_SEC = (6.0, 8.0, 10.0)
EDGE_STALL_SEC = 4.0
# At most this many syntheses at once. A news answer is 6-10 sentences and the page asks for all of them together:
# ten at once took up to 4.7 s each (2-3 at once: ~1 s), so they ran into the time limit together. Not hedging: one
# request per sentence, as before; the rest wait their turn (the waiting doesn't count against a try's time).
EDGE_PARALLEL = 3
_edge_slots = threading.BoundedSemaphore(EDGE_PARALLEL)


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


async def _edge_once(text: str, voice: str, first_sec: float) -> bytes:
    """One try: up to first_sec for the first chunk, then EDGE_STALL_SEC from each chunk to the next."""
    audio = b""
    async with asyncio.timeout(first_sec) as limit:
        async for chunk in edge_tts.Communicate(text, voice).stream():
            limit.reschedule(asyncio.get_running_loop().time() + EDGE_STALL_SEC)    # still arriving: keep going
            if chunk["type"] == "audio":
                audio += chunk["data"]
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return audio


def _edge_or_none(text: str, voice: str) -> bytes | None:
    """edge-tts audio, or None if every try failed. Each try is noted for the self-check."""
    for n in range(1, EDGE_ATTEMPTS + 1):
        try:
            with _edge_slots:
                audio = asyncio.run(_edge_once(text, voice, EDGE_FIRST_SEC[min(n, len(EDGE_FIRST_SEC)) - 1]))
            selfcheck.note_tts(True)
            return audio
        except Exception as e:
            print(f"[tts] edge-tts {type(e).__name__} (try {n}/{EDGE_ATTEMPTS})")
            selfcheck.note_tts(False, type(e).__name__)
    return None


# ---------- the backup voice: ElevenLabs (Mahadi, 2026-10-26: "edge-tts first, then ElevenLabs, then the notice") ----------
# One fixed voice per language from config/providers.yaml (elevenlabs → voices); a voice is used only once Mahadi has
# approved it after listening (Bangla isn't yet). Never a browser voice, macOS say or any other engine. Before each use
# the character balance is known (asked from its API, cached providers.BALANCE_SEC), and it stops at 90 % of the month
# (off_at). The privacy rule holds: a reply to a turn that needed personal data isn't sent to it without personal_ok.
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVEN_FORMAT = "mp3_44100_128"       # the same MP3 the page plays from edge-tts
ELEVEN_PARALLEL = 2                   # the free tier's concurrent requests
EDGE_DOWN_SEC = 60.0                  # after edge-tts failed every try, the next sentences skip it this long
TTS_DISK_CACHE = os.path.join("data", "tts_cache")   # the backup voice's fixed phrases: they cost characters
_eleven_slots = threading.BoundedSemaphore(ELEVEN_PARALLEL)
_edge_down = {"until": 0.0}
_personal_speech: collections.deque = collections.deque(maxlen=20)   # replies to personal turns (think_stream)


def _eleven_post(url: str, headers: dict, params: dict, body: dict, timeout: float) -> bytes:
    """One ElevenLabs text-to-speech request (regression.py swaps this for a fake)."""
    import httpx
    resp = httpx.post(url, headers=headers, params=params, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _eleven_voice(text: str, lang: str | None) -> dict | None:
    """The backup voice for this sentence, or None: ElevenLabs off (no key, turned off) or that language's voice not
    approved. Bengali letters take the Bangla voice, as they take Pradeep's."""
    if not providers.is_on("elevenlabs"):
        return None
    lang = "bn" if _BN_LETTERS.search(text) else (lang if lang in VOICES else get_language())
    v = providers.voices("elevenlabs").get(lang)
    return v if v and v["approved"] else None


def _fixed_phrases() -> set[str]:
    """The sentences of Javid's fixed phrases (both languages): the only clips kept on disk."""
    return {" ".join(x.split()) for p in PHRASES.values() for t in p.values() for x in web_tools.sentences(t)}


def _personal(text: str) -> bool:
    """Part of a reply to a turn that needed calendar, tasks, memory or past chats."""
    flat = " ".join(text.split())
    return bool(flat) and any(flat in reply for reply in _personal_speech)


def _eleven_budget(chars: float) -> tuple[bool, str]:
    """May ElevenLabs speak this many characters now? Its balance must be known and fresh (asked again after
    providers.BALANCE_SEC; if that check fails, no), and the month must stay under its 90 % stop."""
    b = providers._balance
    if time.time() - b["at"] > providers.BALANCE_SEC:
        providers.refresh_balance(wait=True)
    if b["error"] or not b["data"]:
        return False, "its character balance couldn't be checked"
    ok, st = providers.allow("elevenlabs", cost=chars)
    return ok, st["why"]


def _eleven(text: str, v: dict, check: bool = True) -> bytes | None:
    """The backup voice `v` saying `text`, or None (not allowed now, or it failed). Fixed phrases come from the disk
    cache once made, short sentences from memory, so a repeated phrase costs its characters once. check=False: the
    listening test (regression.py --voice-sample) only; the budget still applies."""
    key = (f"elevenlabs:{v['voice_id']}:{v['model']}", text)
    if key in _tts_cache:
        return _tts_cache[key][0]
    flat = " ".join(text.split())
    path = (os.path.join(TTS_DISK_CACHE, hashlib.sha256(f"{v['voice_id']}|{v['model']}|{flat}".encode()).hexdigest()
                         + ".mp3") if flat in _fixed_phrases() else None)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            audio = f.read()
        _remember_clip(key, text, audio)
        return audio
    if check and _personal(text) and not providers.personal_ok("elevenlabs"):
        print("[tts] backup voice: not for a reply with personal data (ElevenLabs has no personal_ok)")
        return None
    cost = len(text) * v["cost_per_char"]
    ok, why = _eleven_budget(cost)
    if not ok:
        print(f"[tts] backup voice not used: {why}")
        return None
    body = {"text": text, "model_id": v["model"], **({"language_code": v["language_code"]} if v["language_code"] else {})}
    headers = {"xi-api-key": os.environ.get(providers.key_env("elevenlabs"), "").strip(), "Accept": "audio/mpeg"}
    t0 = time.time()
    try:
        with _eleven_slots:
            audio = _eleven_post(ELEVEN_URL.format(voice_id=v["voice_id"]), headers, {"output_format": ELEVEN_FORMAT},
                                 body, providers.timeout_s("elevenlabs", 10.0))
        if not audio:
            raise RuntimeError("ElevenLabs returned no audio")
    except Exception as e:
        kind = providers.error_kind(e)
        providers.record("elevenlabs", 0, model=v["model"], op="tts", latency_ms=(time.time() - t0) * 1000,
                         status=kind, detail=type(e).__name__)
        if kind != "error" and (rest := providers.cooldown_seconds("elevenlabs", e)):
            providers.cool("elevenlabs", rest, kind)
        print(f"[tts] backup voice failed: {type(e).__name__} {_redact(str(e))[:100]}")
        selfcheck.note_tts(False, type(e).__name__, engine="elevenlabs")
        return None
    providers.record("elevenlabs", cost, model=v["model"], op="tts", latency_ms=(time.time() - t0) * 1000)
    selfcheck.note_tts(True, engine="elevenlabs")
    if path:
        os.makedirs(TTS_DISK_CACHE, mode=0o700, exist_ok=True)
        with os.fdopen(os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
            f.write(audio)
        os.replace(path + ".tmp", path)
    _remember_clip(key, text, audio)
    return audio


def _remember_clip(key: tuple, text: str, audio: bytes) -> None:
    if len(text) <= _TTS_CACHE_MAX_CHARS:
        if len(_tts_cache) >= _TTS_CACHE_SIZE:
            _tts_cache.pop(next(iter(_tts_cache)))
        _tts_cache[key] = (audio, "audio/mpeg")


def synthesize(text: str, lang: str | None = None, backup: bool = True) -> tuple[bytes, str]:
    """The language's locked voice (EDGE_VOICE / EDGE_VOICE_BN) as MP3 bytes. If edge-tts fails every try, the backup
    voice (ElevenLabs, when its voice for the language is approved and it may take the sentence), and for
    EDGE_DOWN_SEC after that the backup speaks first. Raises when neither can: the page shows "voice unavailable".
    backup=False: edge-tts only (the warm-up: the backup costs characters)."""
    voice = voice_for(text, lang)
    if (voice, text) in _tts_cache:
        return _tts_cache[voice, text]
    alt = _eleven_voice(text, lang) if backup else None
    if alt and time.time() < _edge_down["until"] and (audio := _eleven(text, alt)):
        return audio, "audio/mpeg"
    audio = _edge_or_none(text, voice)
    if audio is not None:
        _remember_clip((voice, text), text, audio)
        return audio, "audio/mpeg"
    if alt:
        _edge_down["until"] = time.time() + EDGE_DOWN_SEC
        if (audio := _eleven(text, alt)):
            return audio, "audio/mpeg"
    raise RuntimeError("edge-tts unavailable" + (" and the backup voice couldn't take it" if alt else ""))


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
                    synthesize(sentence, lang, backup=False)      # never the backup's characters for a warm-up
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
    print(providers.startup_line())
    _restore_context()
    threading.Thread(target=_ring_timers, daemon=True).start()
    watches.start()     # a watch that fires becomes a timer-style alert, which _ring_timers says
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
