"""
Timers and reminders: the set_timer and timers tools, the Timers panel and its routes (server.py).

They're kept in timers.json, so a page reload or a server restart keeps them. The page rings them - a chime,
then the announcement in Javid's own voice through /speak - and reports POST /timers/{id}/fired; only the
first tab to report a timer announces it. The CLI rings them from a background thread (javid.main), and
Telegram gets each one as a message (take_due_for, telegram_bot.push_due), whether or not a page rang it. All the
time math is here: the model passes Mahadi's words ("10 minutes", "half an hour", "7pm", "tomorrow").
"""
import json
import os
import re
import threading
import time
import uuid
from datetime import date, datetime, timedelta

import calendar_tools   # _parse_day / _parse_time: the same day and time words as the calendar

TIMERS_FILE = "timers.json"
MAX_SECONDS = 7 * 24 * 3600
MAX_ACTIVE = 20
KEEP_FIRED_SEC = 3600       # a rung timer stays in the file this long (so a second tab doesn't ring it again)
MISSED_SEC = 12 * 3600      # one that came due while no page was open still rings if it's this recent

_lock = threading.Lock()
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
              "twenty five": 25, "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60, "ninety": 90}
_UNITS = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "m": 60, "min": 60, "mins": 60, "minute": 60,
          "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "d": 86400, "day": 86400,
          "days": 86400, "সেকেন্ড": 1, "মিনিট": 60, "মিনিটের": 60, "ঘণ্টা": 3600, "ঘন্টা": 3600, "ঘণ্টার": 3600,
          "ঘন্টার": 3600, "দিন": 86400}
_BN_FRACTIONS = {"আধা ঘণ্টা": "30 মিনিট", "আধা ঘন্টা": "30 মিনিট", "আধ ঘণ্টা": "30 মিনিট", "আধ ঘন্টা": "30 মিনিট",
                 "দেড় ঘণ্টা": "90 মিনিট", "দেড় ঘন্টা": "90 মিনিট", "আড়াই ঘণ্টা": "150 মিনিট", "আড়াই ঘন্টা": "150 মিনিট"}


# ---------- storage ----------
def _load() -> list[dict]:
    try:
        with open(TIMERS_FILE) as f:
            items = json.load(f)
        return items if isinstance(items, list) else []
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    tmp = TIMERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, TIMERS_FILE)


def _prune(items: list[dict], now: float) -> list[dict]:
    return [t for t in items if (t.get("fired") and now - t["fired"] < KEEP_FIRED_SEC)
            or (not t.get("fired") and now - t["due"] < MISSED_SEC)]


def _pending(items: list[dict]) -> list[dict]:
    return sorted((t for t in items if not t.get("fired")), key=lambda t: t["due"])


# ---------- parsing ----------
def parse_duration(text: str) -> int:
    """'10 minutes', '1 hour 30 minutes', '1h30m', 'half an hour', '90 sec', '১০ মিনিট', '15' (minutes) -> seconds."""
    t = " " + (text or "").lower().translate(_BN_DIGITS) + " "
    for bn, plain in _BN_FRACTIONS.items():
        t = t.replace(bn, plain)
    t = re.sub(r"\b(an? hour and a half|one and a half hours?|1\.5 hours?)\b", "90 minutes", t)
    t = re.sub(r"\bhalf (an )?hour\b", "30 minutes", t)
    t = re.sub(r"\b(a )?quarter (of an |an )?hour\b", "15 minutes", t)
    for word, n in sorted(_NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{word}\b", str(n), t)
    total, found = 0.0, False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-zঀ-৿]+)", t):
        unit = _UNITS.get(m[2]) or _UNITS.get(m[2].rstrip("s"))
        if unit:
            total, found = total + float(m[1]) * unit, True
    if not found and (m := re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", t)):
        total, found = float(m[1]) * 60, True        # a bare number: minutes
    if not found or total < 1:
        raise ValueError(f"no duration in '{text}'")
    return int(round(total))


_DAY_WORDS = r"today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|আজ|কাল|আগামীকাল|পরশু"


def parse_at(at: str, day: str = "", now: datetime | None = None) -> datetime:
    """A clock time ('7pm', '19:30', 'noon', '7' = the next 7 o'clock), on `day` or else the next time it comes."""
    now = now or datetime.now()
    text = " ".join((at or "").lower().translate(_BN_DIGITS).split())
    day = (day or "").strip().lower()
    if not day and (m := re.search(rf"\b(?:next |this |on )?({_DAY_WORDS})\b", text)):
        day, text = m[1], text.replace(m[0], " ")
    text = re.sub(r"\b(at|by|on|around|about|o'?clock|sharp)\b", " ", text)
    text = " ".join(text.split())
    if text in ("noon", "midday"):
        text = "12pm"
    elif text == "midnight":
        text = "12am"
    elif text in ("morning", "this morning"):
        text = "9am"
    elif text in ("evening", "this evening"):
        text = "7pm"
    has_ampm = bool(re.search(r"\d\s*[ap]\.?\s*m\b|\d\s*[ap]\b", text))
    hour, minute = calendar_tools._parse_time(text)
    tonight = day == "tonight"
    d = calendar_tools._parse_day("today" if tonight else day) if day else now.date()
    if tonight and not has_ampm and hour < 12:
        hour += 12
    candidates = [hour] if has_ampm or hour >= 12 or hour == 0 else [hour, hour + 12]   # "at 7" = 7 AM or 7 PM
    for h in candidates:
        dt = datetime.combine(d, datetime.min.time()).replace(hour=h % 24, minute=minute)
        if dt > now:
            return dt
    if day:
        raise ValueError("that time has already passed")
    return datetime.combine(d + timedelta(days=1), datetime.min.time()).replace(hour=candidates[0] % 24, minute=minute)


# ---------- wording ----------
def _human(secs: int, lang: str = "en") -> str:
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if lang == "bn":
        parts = [f"{h} ঘণ্টা" if h else "", f"{m} মিনিট" if m else "", f"{s} সেকেন্ড" if s and not h else ""]
        return " ".join(p for p in parts if p).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
    parts = [f"{h} hour{'s' if h != 1 else ''}" if h else "", f"{m} minute{'s' if m != 1 else ''}" if m else "",
             f"{s} second{'s' if s != 1 else ''}" if s and not h else ""]
    return " ".join(p for p in parts if p)


def _length_adj(secs: int) -> str:
    """'10-minute', '2-hour', or '' for mixed lengths."""
    for unit, name in ((3600, "hour"), (60, "minute"), (1, "second")):
        if secs % unit == 0:
            return f"{secs // unit}-{name}"
    return ""


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")


def _when(due: float, now: datetime | None = None) -> str:
    now = now or datetime.now()
    dt = datetime.fromtimestamp(due)
    if dt.date() == now.date():
        return f"today at {_clock(dt)}"
    if dt.date() == now.date() + timedelta(days=1):
        return f"tomorrow ({dt:%A}) at {_clock(dt)}"
    return f"{dt:%A, %B} {dt.day} at {_clock(dt)}"


def _left(due: float, now: float) -> str:
    secs = max(0, int(due - now))
    return _human(secs - secs % 60 if secs >= 120 else secs) or "under a second"


def _announce(kind: str, label: str, secs: int) -> dict:
    if kind == "reminder" or len(label.split()) > 3:      # "call mom", "check the oven" read as a reminder
        what = label or "your reminder"
        return {"en": f"Mahadi, reminder: {what}.", "bn": f"মাহাদি, মনে করিয়ে দিচ্ছি: {what}।"}
    if label:
        return {"en": f"Mahadi, your {label} timer is done.", "bn": f"মাহাদি, তোমার {label} টাইমার শেষ।"}
    adj = _length_adj(secs)
    return {"en": f"Mahadi, your {adj} timer is done." if adj else f"Mahadi, your timer for {_human(secs)} is done.",
            "bn": f"মাহাদি, তোমার {_bn_of(_human(secs, 'bn'))} টাইমার শেষ।"}


def _bn_of(words: str) -> str:
    """Bangla possessive: ১০ মিনিট -> ১০ মিনিটের, ১ ঘণ্টা -> ১ ঘণ্টার."""
    return words + ("র" if words.endswith("া") else "ের")


def _describe(t: dict, now: float) -> str:
    name = (f"{t['label']} timer" if t["label"] else "timer") if t["kind"] == "timer" else f"reminder: {t['label']}"
    if t["kind"] == "timer":
        return f"{name}, rings at {_clock(datetime.fromtimestamp(t['due']))} ({_left(t['due'], now)} left)"
    return f"{name}, {_when(t['due'])}"


# ---------- tools ----------
def set_timer(duration: str = "", at: str = "", day: str = "", label: str = "") -> str:
    label = " ".join((label or "").split())[:80].strip(" .")
    now = time.time()
    try:
        if duration and not at:
            secs = parse_duration(str(duration))
            if secs > MAX_SECONDS:
                return "That's longer than a week; set a calendar event or a reminder for a day instead."
            kind, due = "timer", now + secs
        elif at:
            kind, due = "reminder", parse_at(str(at), day).timestamp()
            secs = int(due - now)
            if secs > MAX_SECONDS:
                return "That's more than a week away; add it to the calendar instead."
        else:
            return "Say how long (duration, e.g. '10 minutes') or when (at, e.g. '7pm')."
    except ValueError as e:
        if "passed" in str(e):
            return f"{(day or 'That day').capitalize()} at {at} has already passed. Ask Mahadi for another time."
        return (f"Couldn't understand '{duration or at}'. Give a length like '10 minutes' or '1 hour 30 minutes', "
                "or a time like '7:30 pm'.")
    with _lock:
        items = _prune(_load(), now)
        if len(_pending(items)) >= MAX_ACTIVE:
            return f"There are already {MAX_ACTIVE} timers and reminders running; cancel one first."
        item = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": label, "due": due, "seconds": secs,
                "created": now, "fired": None, "announce": _announce(kind, label, secs)}
        items.append(item)
        _save(items)
        running = len(_pending(items))
    if kind == "timer":
        what = f"{label} timer" if label else "timer"
        return (f"Timer set: {what} for {_human(secs)}. It rings at {_clock(datetime.fromtimestamp(due))}. "
                f"{running} running now. Javid announces it out loud while the page is open.")
    return f"Reminder set for {_when(due)}: {label or 'reminder'}. {running} timer(s)/reminder(s) running now."


def timers_tool(action: str = "list", which: str = "") -> str:
    now = time.time()
    with _lock:
        items = _prune(_load(), now)
        pending = _pending(items)
        if action != "cancel":
            if not pending:
                return "No timers or reminders running."
            return f"{len(pending)} running:\n" + "\n".join(f"{i + 1}. {_describe(t, now)}" for i, t in enumerate(pending))
        if not pending:
            return "No timers or reminders to cancel."
        w = " ".join((which or "").lower().split())
        if w in ("all", "everything", "every timer", "all timers", "all reminders", "সব"):
            hits = pending
        elif re.fullmatch(r"#?\d+", w) and 1 <= int(w.lstrip("#")) <= len(pending):
            hits = [pending[int(w.lstrip("#")) - 1]]
        else:
            words = set(re.findall(r"[a-z0-9ঀ-৿']+", w)) - {"the", "my", "a", "timer", "reminder", "for", "to"}
            hits = [t for t in pending if words and words <= set(re.findall(r"[a-z0-9ঀ-৿']+", t["label"].lower()))]
            if not hits and not words:     # "cancel the timer" with one running
                kind = "reminder" if "reminder" in w else "timer" if "timer" in w else None
                hits = [t for t in pending if kind in (None, t["kind"])]
        if not hits:
            return f"Nothing matches '{which}'. Running: " + "; ".join(f"{i + 1}. {_describe(t, now)}" for i, t in enumerate(pending))
        if len(hits) > 1 and w not in ("all", "everything", "every timer", "all timers", "all reminders", "সব"):
            return "Several match; ask which: " + "; ".join(f"{pending.index(t) + 1}. {_describe(t, now)}" for t in hits)
        ids = {t["id"] for t in hits}
        items = [t for t in items if t["id"] not in ids]
        _save(items)
    left = len(pending) - len(hits)
    return f"Cancelled {', '.join(_describe(t, now).split(',')[0] for t in hits)}. {left} still running."


# ---------- page and CLI ----------
def active_json() -> list[dict]:
    now = time.time()
    with _lock:
        items = _prune(_load(), now)
    return [{"id": t["id"], "kind": t["kind"], "label": t["label"], "due": int(t["due"] * 1000),
             "seconds": t["seconds"], "announce": t["announce"]} for t in _pending(items)]


def mark_fired(tid: str) -> bool:
    """True for the first report of a timer (that tab announces it), False if it was already rung or is gone."""
    with _lock:
        items = _load()
        for t in items:
            if t["id"] == tid and not t.get("fired"):
                t["fired"] = time.time()
                _save(items)
                return True
    return False


def cancel(tid: str) -> bool:
    with _lock:
        items = _load()
        kept = [t for t in items if t["id"] != tid]
        if len(kept) == len(items):
            return False
        _save(kept)
    return True


def take_due() -> list[dict]:
    """Timers due now, marked rung (the CLI's ring thread)."""
    now = time.time()
    with _lock:
        items = _prune(_load(), now)
        due = [t for t in items if not t.get("fired") and t["due"] <= now]
        for t in due:
            t["fired"] = now
        if due:
            _save(items)
    return due


def take_due_for(channel: str, baseline: bool = False) -> list[dict]:
    """Timers due now that haven't been sent to `channel` (Telegram) yet, marked sent there; the page's and the
    CLI's ringing (`fired`) is separate. baseline=True (the channel starting up) marks the ones that already rang
    without returning them, so old timers aren't re-sent; one that came due unrung is still sent."""
    now = time.time()
    with _lock:
        items = _prune(_load(), now)
        due = [t for t in items if t["due"] <= now and channel not in t.get("sent", [])
               and (t.get("fired") or not baseline)]
        for t in due:
            t.setdefault("sent", []).append(channel)
        if due:
            _save(items)
    return [] if baseline else due


TOOLS = [
    {"type": "function", "function": {
        "name": "set_timer",
        "description": "Start a countdown timer (duration) or a reminder at a clock time (at, day). Javid rings it "
                       "out loud. Use it for 'set a timer', 'remind me in 20 minutes' or 'remind me at 7pm to...'; "
                       "a reminder with no time is a task instead. Pass Mahadi's words; never compute times.",
        "parameters": {"type": "object", "properties": {
            "duration": {"type": "string", "description": "How long, e.g. '10 minutes', '1 hour 30 minutes'."},
            "at": {"type": "string", "description": "Or a clock time, e.g. '7pm', '19:30'."},
            "day": {"type": "string", "description": "With 'at': 'today', 'tomorrow' or a weekday (default: next time it comes)."},
            "label": {"type": "string", "description": "What it's for, e.g. 'tea' or 'call mom'."}}}}},
    {"type": "function", "function": {
        "name": "timers",
        "description": "List the running timers and reminders, or cancel one (by its words, its number, or 'all').",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["list", "cancel"]},
            "which": {"type": "string", "description": "For cancel: words from its label, its number, or 'all'."}}}}},
]
TOOL_MAP = {"set_timer": lambda args: set_timer(**args), "timers": lambda args: timers_tool(**args)}
