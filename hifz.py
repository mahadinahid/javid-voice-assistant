"""
The hifz revision tracker (2026-11-22). Mahadi is a Hafiz: this keeps his whole Quran strong with a steady rotation,
because revision slips when life gets busy and nothing tracked it for him.

Logging, the rotation, the questions and the reminders are all code: no model request, like the news and the rates.
- The log (data/hifz_log.jsonl, mode 0600, never in git) is his: one line a portion he revised, with the day, what
  (juz, manzil, a surah or some of its ayahs, as ayah ranges: quran.py), a strength (strong, okay, weak; okay unless
  he says) and where it came from. "I revised juz 5", "did para 12", "revised Surah Kahf", "finished manzil 3",
  "revised juz 7, weak", "aaj 5 para revise korechi", "আজ ৫ পারা রিভাইস করেছি" (request/answer). A message is taken
  only if all of it is a revision log or question; a mixed one goes to the model, whose `hifz` tool does the same in
  code (and is a personal tool: only providers with personal_ok see its results).
- The rotation (config/hifz.yaml: a juz or one of the 7 manzils as the unit, the daily amount, how soon a weak
  portion comes back): today's portion is the stalest one, in rotation order from where he started (his first log, or
  "I'm on juz 12"); a portion is behind once it's older than one cycle (or, before the first round is through, once
  its day in that round has passed); a weak one comes back weak_days after it was logged, until it's logged again.
  Each ayah keeps the day it was last revised, so a surah counts toward the juz it's in.
- "undo that" right after a log (or "undo my last revision", within UNDO_SEC) takes it back with no card; deleting
  logs or the whole history waits for an Approve card (approvals.py: hifz_delete, hifz_reset).
- Reminders ride the watches thread like prayers.py's (tick): one after Fajr with today's exact portion, the streak
  and the days to Ramadan, and while today's portion isn't logged, a nudge after Asr and a last call after Isha; on
  the page (its own chime and banner, with buttons that log it) and on Telegram (buttons too). The morning brief gets
  a line (brief_line).
- The Hifz panel and view (GET /hifz: status_json) show each juz (or manzil) by days since it was last revised.
"""
import json
import math
import os
import re
import secrets
import threading
import time
from datetime import date, datetime, timedelta

import yaml

import prayers
import quran
import timers

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "hifz.yaml")   # beside the code
LOG_FILE = os.path.join("data", "hifz_log.jsonl")
STATE_FILE = os.path.join("data", "hifz_state.json")
RELOAD_SEC = 5
UNDO_SEC = 24 * 3600        # "undo my last revision" reaches this far back
LEAD_SEC = 20               # a reminder goes into timers.json this early, so the page rings it on its minute
NUDGE_SEC = 60 * 60         # a nudge is still rung this long after its time
MORNING_LATE_SEC = 5 * 3600  # the morning reminder, until about Dhuhr
KEEP_DAYS = 3               # what was raised is remembered this long
STRENGTHS = ("strong", "okay", "weak")
_time = time.time           # the clock (regression.py moves it with timers._now)
_lock = threading.RLock()

# ---------- settings: config/hifz.yaml ----------
ANCHORS = ("Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha")
FALLBACK = {"Fajr": "05:00", "Sunrise": "06:15", "Dhuhr": "11:45", "Asr": "15:30", "Maghrib": "17:15",
            "Isha": "18:30"}     # if the prayer times can't be had, the reminders still go out near these


def _defaults() -> dict:
    return {"unit": "juz", "daily": 1, "weak_days": 2, "reciter": "husary", "start": 1, "brief": True,
            "ramadan": True,
            "reminders": {"page": True, "telegram": True, "morning": {"after": "Fajr", "minutes": 25},
                          "nudges": [{"after": "Asr", "minutes": 45}, {"after": "Isha", "minutes": 45}]}}


def _when(v, where: str) -> dict | None:
    if v in (None, False, "off"):
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{where}: give after: a prayer with minutes:, or at: \"HH:MM\", or off")
    if "at" in v:
        if set(v) - {"at"} or not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", str(v["at"]).strip()):
            raise ValueError(f"{where}: at: is a clock time like \"05:30\"")
        return {"at": str(v["at"]).strip()}
    if set(v) - {"after", "minutes"} or v.get("after") not in ANCHORS:
        raise ValueError(f"{where}: after: is one of {', '.join(ANCHORS)}")
    m = v.get("minutes", 0)
    if not isinstance(m, int) or isinstance(m, bool) or not 0 <= m <= 300:
        raise ValueError(f"{where}: minutes: is 0 to 300")
    return {"after": v["after"], "minutes": m}


def _validate(raw) -> tuple[dict, list[str]]:
    cfg, problems = _defaults(), []
    if raw is None:
        return cfg, problems
    if not isinstance(raw, dict):
        return cfg, ["it isn't a list of settings"]
    for k, v in raw.items():
        try:
            if k == "unit":
                if v not in ("juz", "manzil"):
                    raise ValueError("juz or manzil")
            elif k in ("daily", "weak_days", "start"):
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 30:
                    raise ValueError("a whole number from 1 to 30")
            elif k in ("brief", "ramadan"):
                if not isinstance(v, bool):
                    raise ValueError("true or false")
            elif k == "reciter":
                if v not in quran.RECITERS:
                    raise ValueError(f"one of {', '.join(quran.RECITERS)}")
            elif k == "reminders":
                if not isinstance(v, dict) or set(v) - {"page", "telegram", "morning", "nudges"}:
                    raise ValueError("page:, telegram:, morning: and nudges:")
                rem = dict(_defaults()["reminders"])
                for flag in ("page", "telegram"):
                    if flag in v:
                        if not isinstance(v[flag], bool):
                            raise ValueError(f"{flag}: is true or false")
                        rem[flag] = v[flag]
                if "morning" in v:
                    rem["morning"] = _when(v["morning"], "morning")
                if "nudges" in v:
                    nudges = v["nudges"] or []
                    if not isinstance(nudges, list) or len(nudges) > 3:
                        raise ValueError("nudges: is a list of up to three")
                    rem["nudges"] = [w for i, n in enumerate(nudges) if (w := _when(n, f"nudge {i + 1}"))]
                v = rem
            else:
                raise ValueError("not a setting")
            cfg[k] = v
        except ValueError as e:
            problems.append(f"{k}: {e}")
    if cfg["start"] > quran.COUNT[cfg["unit"]] or cfg["daily"] > quran.COUNT[cfg["unit"]]:
        problems.append(f"start and daily are at most {quran.COUNT[cfg['unit']]} for a {cfg['unit']}")
    return cfg, problems


_cfg = {"cfg": None, "mtime": None, "checked": 0.0, "problems": []}


def config() -> dict:
    """The settings, read again when the file changes; a file with a mistake keeps the last good ones (printed once)."""
    with _lock:
        now = time.time()
        if _cfg["cfg"] is not None and now - _cfg["checked"] < RELOAD_SEC:
            return _cfg["cfg"]
        _cfg["checked"] = now
        try:
            mtime = os.path.getmtime(CONFIG_FILE)
        except OSError:
            mtime = None
        if _cfg["cfg"] is not None and mtime == _cfg["mtime"]:
            return _cfg["cfg"]
        _cfg["mtime"] = mtime
        try:
            raw = yaml.safe_load(open(CONFIG_FILE, encoding="utf-8")) if mtime is not None else None
            cfg, problems = _validate(raw)
        except (OSError, yaml.YAMLError) as e:
            cfg, problems = None, [f"it can't be read: {type(e).__name__}"]
        if problems:
            if problems != _cfg["problems"]:
                print(f"[hifz] {CONFIG_FILE} isn't used ({'; '.join(problems)}); keeping the last good settings")
            _cfg["problems"] = problems
            if _cfg["cfg"] is None:
                _cfg["cfg"] = _defaults()
            return _cfg["cfg"]
        _cfg.update(cfg=cfg, problems=[])
        return cfg


def reload() -> None:
    with _lock:
        _cfg.update(cfg=None, mtime=None, checked=0.0)


def problems() -> list[str]:
    config()
    return list(_cfg["problems"])


# ---------- the log: data/hifz_log.jsonl ----------
def _read() -> list[dict]:
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("kind") in ("log", "start") and isinstance(e.get("date"), str):
            out.append(e)
    return out


def _write(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    tmp = LOG_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    os.replace(tmp, LOG_FILE)


def _append(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    try:
        os.chmod(LOG_FILE, 0o600)
    except OSError:
        pass


def _today() -> date:
    return datetime.fromtimestamp(_time(), prayers._zone()).date()


def log(portions: list[dict], strengths: list[str] | str = "okay", day: date | None = None, via: str = "page") -> list[dict]:
    """Keep what he revised: one entry per portion (a shared batch, so "undo that" takes the message back whole)."""
    if isinstance(strengths, str):
        strengths = [strengths] * len(portions)
    day = day or _today()
    batch = secrets.token_hex(4)
    entries = [{"id": secrets.token_hex(5), "batch": batch, "kind": "log", "at": round(_time(), 3),
                "date": day.isoformat(), "portion": {k: p[k] for k in ("kind", "n", "a1", "a2") if k in p},
                "ranges": [list(r) for r in p["ranges"]], "strength": st if st in STRENGTHS else "okay", "via": via}
               for p, st in zip(portions, strengths)]
    with _lock:
        _append(entries)
    return entries


def set_start(n: int, via: str = "page") -> dict:
    """The rotation starts again here today: "I'm on juz 12"."""
    cfg = config()
    e = {"id": secrets.token_hex(5), "batch": secrets.token_hex(4), "kind": "start", "at": round(_time(), 3),
         "date": _today().isoformat(), "unit": cfg["unit"], "n": n, "via": via}
    with _lock:
        _append([e])
    return e


def undo_last(within: float = UNDO_SEC) -> list[dict]:
    """Take back the last message's entries (its batch) if it was made within `within` seconds; what was removed."""
    with _lock:
        entries = _read()
        if not entries:
            return []
        last = entries[-1]          # the log is append-only: its last line is the newest (two may share a second)
        if _time() - last["at"] > within:
            return []
        gone = [e for e in entries if e.get("batch") == last.get("batch")] if last.get("batch") else [last]
        _write([e for e in entries if e not in gone])
    return gone


def delete(ids: list[str]) -> int:
    with _lock:
        entries = _read()
        kept = [e for e in entries if e["id"] not in set(ids)]
        if len(kept) != len(entries):
            _write(kept)
    return len(entries) - len(kept)


def reset() -> tuple[int, str]:
    """The whole history goes; one copy is kept beside it (data/hifz_log.jsonl.bak-<time>, 0600), replaced at the
    next reset. (logs removed, the copy's path)."""
    with _lock:
        entries = _read()
        if not entries:
            return 0, ""
        folder = os.path.dirname(LOG_FILE) or "."
        for old in os.listdir(folder):
            if old.startswith(os.path.basename(LOG_FILE) + ".bak-"):
                os.remove(os.path.join(folder, old))
        copy = f"{LOG_FILE}.bak-{datetime.fromtimestamp(_time()):%Y%m%d-%H%M%S}"
        os.replace(LOG_FILE, copy)
        os.chmod(copy, 0o600)
    return len([e for e in entries if e["kind"] == "log"]), copy


# ---------- where he stands ----------
def _label(e: dict, lang: str = "en", spoken: bool = False) -> str:
    p = e["portion"]
    return quran.portion_label({**p, "ranges": e["ranges"]}, lang, spoken)


def _portion_of_unit(kind: str, n: int) -> dict:
    return {"kind": kind, "n": n, "ranges": [quran.unit_range(kind, n)]}


def _state(today: date | None = None, entries: list[dict] | None = None, cfg: dict | None = None) -> dict:
    """Everything the answers, the reminders and the panel need, computed from the log."""
    cfg = cfg or config()
    today = today or _today()
    entries = _read() if entries is None else entries
    kind, d = cfg["unit"], cfg["daily"]
    count = quran.COUNT[kind]
    cycle = math.ceil(count / d)
    logs = sorted((e for e in entries if e["kind"] == "log" and e["date"] <= today.isoformat()),
                  key=lambda e: (e["date"], e["at"]))
    last = [None] * quran.TOTAL           # per ayah: (date ordinal, logged at) of its latest log
    strength = [None] * quran.TOTAL
    by = [None] * quran.TOTAL
    for e in logs:
        stamp = (date.fromisoformat(e["date"]).toordinal(), e["at"])
        for lo, hi in e["ranges"]:
            for i in range(lo, hi + 1):
                last[i], strength[i], by[i] = stamp, e["strength"], e["id"]
    # where the rotation starts: his "I'm on juz 12", else his first log of a whole unit, else config's start
    starts = [e for e in entries if e["kind"] == "start" and e.get("unit") == kind and e["date"] <= today.isoformat()]
    if starts:
        s = max(starts, key=lambda e: e["at"])
        start_n, start_stamp = s["n"], (date.fromisoformat(s["date"]).toordinal(), s["at"])
    else:
        first = next(((e, quran.units_in(kind, e["ranges"])) for e in logs if quran.units_in(kind, e["ranges"])), None)
        if first:
            start_n = first[1][0]
            start_stamp = (date.fromisoformat(first[0]["date"]).toordinal(), first[0]["at"])
        else:
            start_n, start_stamp = cfg["start"], (today.toordinal(), 0)
    t = today.toordinal()
    units = []
    for n in range(1, count + 1):
        lo, hi = quran.unit_range(kind, n)
        span = last[lo:hi + 1]
        full = None if any(x is None for x in span) else min(span)
        newest = max((x for x in span if x is not None), default=None)
        offset = (n - start_n) % count
        since = full is not None and full >= start_stamp
        due = (full[0] + cycle) if since else (start_stamp[0] + offset // d)
        days = None if full is None else t - full[0]
        units.append({"n": n, "lo": lo, "hi": hi, "full": full, "last": full and date.fromordinal(full[0]),
                      "days": days, "partial": newest and newest != full and date.fromordinal(newest[0]),
                      "weak": any(x == "weak" for x in strength[lo:hi + 1]), "offset": offset, "since": since,
                      "due": date.fromordinal(due), "behind": max(0, t - due), "today_done": bool(full and full[0] == t)})
    pending = sorted((u for u in units if not u["since"]), key=lambda u: u["offset"])
    revised = sorted((u for u in units if u["since"]), key=lambda u: (u["full"], u["offset"]))
    order = [u for u in pending + revised if not u["today_done"]]
    done_today = [u for u in units if u["today_done"]]
    left = max(0, d - len(done_today))
    today_units = order[:left]
    tomorrow = order[left:left + d] if left else order[:d]
    # weak runs: ayahs whose latest log was weak, grouped by that log; due weak_days after it
    weak, i = [], 0
    while i < quran.TOTAL:
        if strength[i] == "weak":
            j = i
            while j + 1 < quran.TOTAL and strength[j + 1] == "weak" and by[j + 1] == by[i]:
                j += 1
            e = next(x for x in logs if x["id"] == by[i])
            due = date.fromisoformat(e["date"]) + timedelta(days=cfg["weak_days"])
            whole = (i, j) == tuple(e["ranges"][0]) and len(e["ranges"]) == 1
            weak.append({"lo": i, "hi": j, "id": e["id"], "date": date.fromisoformat(e["date"]), "due": due,
                         "label": _label(e) if whole else quran.span_text(i, j),
                         "label_bn": _label(e, "bn") if whole else quran.span_text(i, j, "bn"), "entry": e, "whole": whole})
            i = j + 1
        else:
            i += 1
    whole = {}                            # how many times each unit was logged whole since the start
    for e in logs:
        if (date.fromisoformat(e["date"]).toordinal(), e["at"]) >= start_stamp:
            for n in quran.units_in(kind, e["ranges"]):
                whole[n] = whole.get(n, 0) + 1
    rounds = min(whole.get(n, 0) for n in range(1, count + 1))
    days_logged = sorted({e["date"] for e in logs})
    streak, best, run, prev = 0, 0, 0, None
    for iso in days_logged:
        o = date.fromisoformat(iso).toordinal()
        run = run + 1 if prev is not None and o == prev + 1 else 1
        best, prev = max(best, run), o
    if prev is not None and prev >= t - 1:
        streak = run
    return {"cfg": cfg, "kind": kind, "count": count, "daily": d, "cycle": cycle, "today": today, "units": units,
            "today_units": today_units, "done_today": done_today, "tomorrow": tomorrow, "weak": weak,
            "weak_due": [w for w in weak if w["due"] <= today],
            "behind": sorted((u for u in units if u["behind"] > 0 and not u["today_done"]), key=lambda u: -u["behind"]),
            "fresh": sum(1 for u in units if u["days"] is not None and u["days"] < cycle), "rounds": rounds,
            "streak": streak, "best": best, "logs": logs, "start_n": start_n, "started": bool(logs or starts),
            "entries": entries}


def today_portions() -> list[dict]:
    """Today's rotation portion(s) still to do, plus the weak ones due (the player's "play today's revision")."""
    s = _state()
    out = [_portion_of_unit(s["kind"], u["n"]) for u in s["today_units"]]
    for w in s["weak_due"]:
        if not any(p["ranges"][0][0] <= w["lo"] and w["hi"] <= p["ranges"][0][1] for p in out):
            out.append({"kind": "surah", "n": quran.verse(w["lo"])[0], "ranges": [(w["lo"], w["hi"])]}
                       if not w["whole"] else {**w["entry"]["portion"], "ranges": [tuple(r) for r in w["entry"]["ranges"]]})
    return out


def ramadan() -> dict | None:
    """{date, days} to the next Ramadan (prayers.ramadan_start, saved; None when it can't be had yet)."""
    if not config()["ramadan"]:
        return None
    try:
        start = prayers.ramadan_start(fetch=False)
    except Exception:
        return None
    if not start:
        return None
    days = (start - _today()).days
    return {"date": start, "days": days} if days >= 0 else None


# ---------- words ----------
_N = quran._nfc


def _unit_word(kind: str, lang: str, n: int | None = None) -> str:
    if lang == "bn":
        w = "পারা" if kind == "juz" else "মঞ্জিল"
        return f"{w} {quran.bn_digits(n)}" if n else w
    return f"{kind} {n}" if n else kind


def _units_text(kind: str, ns: list[int], lang: str) -> str:
    if not ns:
        return ""
    if lang == "bn":
        return f"{'পারা' if kind == 'juz' else 'মঞ্জিল'} {quran.join([quran.bn_digits(n) for n in ns], 'bn')}"
    return f"{kind} {quran.join([str(n) for n in ns])}"


def _day(d: date, today: date, lang: str) -> str:
    if d == today:
        return "আজ" if lang == "bn" else "today"
    if d == today - timedelta(days=1):
        return "গতকাল" if lang == "bn" else "yesterday"
    if d == today + timedelta(days=1):
        return "কাল" if lang == "bn" else "tomorrow"
    if abs((d - today).days) < 7:
        return prayers.DAYS_BN[d.weekday()] if lang == "bn" else d.strftime("%A")
    return prayers._date_text(d, lang)


def _ago(days: int, lang: str) -> str:
    if lang == "bn":
        return "আজ" if days == 0 else "গতকাল" if days == 1 else f"{quran.bn_digits(days)} দিন আগে"
    return "today" if days == 0 else "yesterday" if days == 1 else f"{days} days ago"


def _span_of(s: dict, n: int, lang: str, spoken: bool = True) -> str:
    lo, hi = quran.unit_range(s["kind"], n)
    return quran.span_text(lo, hi, lang, spoken)


def _streak_text(s: dict, lang: str) -> str:
    n = s["streak"]
    if lang == "bn":
        return f"টানা {quran.bn_digits(n)} দিন" if n > 1 else ""
    return f"day {n} in a row" if n > 1 else ""


def _ramadan_text(lang: str, short: bool = False) -> str:
    r = ramadan()
    if not r:
        return ""
    if r["days"] == 0:
        return "আজ রমজানের প্রথম দিন" if lang == "bn" else "Ramadan begins today"
    if lang == "bn":
        return f"রমজান আর {quran.bn_digits(r['days'])} দিন" if short else f"রমজানের আর {quran.bn_digits(r['days'])} দিন বাকি"
    return f"{r['days']} days to Ramadan"


LINES = {
    "en": ("A juz a day keeps the whole Quran with you.", "Steady beats heavy: a little, every single day.",
           "What you revise today stays with you in Ramadan.", "Weak portions come back sooner, until they're strong.",
           "Keep the chain unbroken.", "Your Quran, kept fresh one portion at a time."),
    "bn": ("দিনে এক পারা, পুরো কুরআন সাথে থাকবে।", "ভারী নয়, নিয়মিত: প্রতিদিন একটু করে।",
           "আজ যা রিভাইস করবে, রমজানে তা সাথে থাকবে।", "কাঁচা অংশ আগে ফেরে, যতক্ষণ না পাকা হয়।",
           "শিকলটা ভেঙো না।", "একেক অংশ করে, পুরো কুরআন তাজা।"),
}


def line_of_the_day(lang: str, today: date | None = None) -> str:
    today = today or _today()
    return LINES["bn" if lang == "bn" else "en"][today.toordinal() % len(LINES["en"])]


# ---------- requests answered in code ----------
_STRENGTH = [(phrase, st) for st, phrases in (
    ("weak", ("a lot of mistakes", "lots of mistakes", "many mistakes", "with mistakes", "not strong", "not solid",
              "not good", "not great", "not well", "not perfect", "not fluent", "weakly", "weak", "shaky", "rough",
              "rusty", "struggled", "struggling", "stuck", "mistakes", "forgot", "forgetting", "kacha", "kancha",
              "durbol", "norbore", "valo hoyni", "bhalo hoyni", "ভালো হয়নি", "ভালো হয় নি", "ভালো না", "অনেক ভুল",
              "ভুল হয়েছে", "আটকে গেছি", "আটকে গেছে", "আটকেছি", "ভুলে গেছি", "কাঁচা", "কাচা", "দুর্বল", "নড়বড়ে")),
    ("strong", ("without mistakes", "no mistakes", "very good", "very well", "strongly", "strong", "solid",
                "perfectly", "perfect", "excellent", "flawless", "fluently", "fluent", "smoothly", "smooth", "great",
                "paka", "pakka", "mojbut", "khub valo", "khub bhalo", "কোনো ভুল নেই", "কোন ভুল হয়নি", "ভুল ছাড়া",
                "খুব ভালো", "পাকা", "মজবুত", "শক্ত", "পারফেক্ট", "নির্ভুল")),
    ("okay", ("not bad", "so so", "so-so", "all right", "alright", "okay", "ok", "fine", "average", "decent", "good",
              "medium", "moderate", "motamuti", "thik ache", "valo", "bhalo", "মোটামুটি", "ঠিক আছে", "ঠিকঠাক", "ভালো",
              "চলনসই"))) for phrase in phrases]
_STRENGTH = sorted(((tuple(quran.tokens(p)), st) for p, st in _STRENGTH), key=lambda x: -len(x[0]))
_PAST = _N({"revised", "reviewed", "did", "done", "finished", "finish", "completed", "complete", "read", "recited",
            "covered", "practiced", "practised", "went", "gone", "got", "log", "logged", "mark", "marked", "record",
            "recorded", "note", "noted", "add", "added", "revise", "revision",
            "করেছি", "করলাম", "করেছিলাম", "দিয়েছি", "দিলাম", "দিছি", "পড়েছি", "পড়লাম", "শেষ", "হয়ে", "গেছে", "হয়েছে",
            "শুনিয়েছি", "শোনালাম", "শুনালাম", "করসি", "দিসি", "পড়সি", "রিভাইস", "রিভিশন", "আমুখতা", "তিলাওয়াত",
            "korechi", "korlam", "korsi", "korchi", "diyechi", "dilam", "disi", "porechi", "porlam", "porsi", "shesh",
            "hoye", "geche", "hoyeche", "hoise", "shunalam", "shuniyechi", "amukhta", "tilawat"})
_VERB_PAST = _N({"revised", "reviewed", "did", "done", "finished", "completed", "complete", "read", "recited",
                 "covered", "practiced", "practised", "went", "gone", "got", "log", "logged", "mark", "marked",
                 "record", "recorded", "note", "noted", "add", "added", "করেছি", "করলাম", "করেছিলাম", "দিয়েছি",
                 "দিলাম", "দিছি", "পড়েছি", "পড়লাম", "শেষ", "গেছে", "হয়েছে", "শুনিয়েছি", "শোনালাম", "শুনালাম",
                 "করসি", "দিসি", "পড়সি", "korechi", "korlam", "korsi", "diyechi", "dilam", "disi", "porechi",
                 "porlam", "porsi", "shesh", "geche", "hoyeche", "hoise", "shunalam", "shuniyechi"})
_NEG = quran._re(r"\b(?:not|didn'?t|did not|haven'?t|have not|hasn'?t|couldn'?t|could not|wasn'?t|never|won'?t)\b|"
                 r"করিনি|পারিনি|হয়নি|দিইনি|দেইনি|পড়িনি|করি নি|দিই নি|\b(?:korini|korinai|parini|hoyni|porini|dei ni)\b")
_FUTURE = quran._re(r"\b(?:will|i'll|going to|gonna|need to|needs to|have to|has to|should|must|want to|wanna|plan to|"
                    r"planning|remind|tomorrow|later|let's|lets|please revise|help me)\b|"
                    r"করব|করবো|দিব|দেব|পড়ব|পড়বো|মনে করিয়ে|\b(?:korbo|dibo|porbo)\b")
_QUESTIONISH = quran._re(r"^(?:did|have|has) (?:i|you|we)\b|^(?:when|what|how|which|should|does|is|are|can|could|"
                         r"where|why)\b|"
                         rf"(?<![{quran.BN}])(?:কবে|কখন|কী|কি|কোন|কত|কয়টা|কয়টি)(?![{quran.BN}])")
_LOG_FILLER = _N({
    "i", "i've", "ive", "i'd", "i'm", "just", "have", "has", "had", "the", "a", "an", "my", "of", "today", "this",
    "morning", "afternoon", "evening", "also", "and", "it", "was", "is", "were", "its", "it's", "now", "please",
    "javid", "jabid", "javed", "hey", "so", "well", "all", "whole", "entire", "full", "with", "for", "as", "both",
    "them", "me", "earlier", "alhamdulillah", "mashallah", "mashaallah", "quran", "today's", "from", "in", "at", "on",
    "by", "over", "through", "again", "once", "twice", "times", "time", "too", "already", "finally", "yes", "yeah",
    "ok", "okay", "right", "and", "then", "after", "before", "fajr", "isha", "asr", "maghrib", "dhuhr", "zuhr",
    "prayer", "salah", "namaz", "be", "to", "that", "today", "tonight", "night", "last", "yesterday", "day",
    "আমি", "আজ", "আজকে", "মাত্র", "এইমাত্র", "এখন", "পুরো", "সম্পূর্ণ", "টা", "টি", "এর", "ও", "আর", "এবং", "তো",
    "জাভিদ", "হিসেবে", "হিসাবে", "আলহামদুলিল্লাহ", "মাশাআল্লাহ", "সকালে", "বিকেলে", "রাতে", "দুপুরে", "ভাই", "একবার",
    "দুইবার", "বার", "গতকাল", "কাল", "পরশু", "গত", "ফজরের", "পর", "নামাজের", "আগে", "করে", "দিয়ে", "সবগুলো",
    "ami", "aaj", "aj", "ajke", "matro", "ekhon", "pura", "puro", "ta", "ti", "er", "o", "ar", "ebong", "to",
    "revision", "revise", "shokale", "rate", "bhai", "kal", "kalke", "gotokal", "porshu", "kore", "diye", "number",
    "no", "নম্বর", "ম", "তম", "th", "st", "nd", "rd", "kora", "korlam", "it"})
_DAY_WORDS = [(("day", "before", "yesterday"), -2), (("yesterday",), -1), (("last", "night"), -1), (("গতকাল",), -1),
              (("গত", "রাতে"), -1), (("কাল",), -1), (("পরশু",), -2), (("gotokal",), -1), (("kal",), -1),
              (("kalke",), -1), (("porshu",), -2)]
_DAY_WORDS = [(tuple(_N(w) for w in ws), off) for ws, off in _DAY_WORDS]

_RESET = quran._re(
    r"\b(?:reset|clear|wipe|erase|delete|remove|start over)\b.{0,25}\b(?:hifz|hifdh|revision|revisions)\b.{0,15}"
    r"\b(?:history|data|progress|records|tracker|everything)\b|\breset (?:my |the )?(?:hifz|revision)\b|"
    r"\b(?:delete|remove|clear|erase|wipe)\b.{0,20}\b(?:all|whole|entire|every)\b.{0,20}"
    r"\b(?:hifz|revision|revisions|logs?|history|entries)\b|"
    r"(?:হিফজ|হিফয|রিভিশন).{0,25}(?:সব|পুরো|ইতিহাস|লগ|হিস্টরি).{0,15}(?:মুছে|ডিলিট|রিসেট)|"
    r"(?:রিসেট|মুছে ফেলো|ডিলিট).{0,20}(?:হিফজ|হিফয|রিভিশন)")
_DELETE = quran._re(r"\b(?:delete|remove|erase|drop|cancel)\b|মুছে|ডিলিট|বাদ দাও|\b(?:muche dao|delete koro)\b")
_DELETE_FILLER = _N({
    "delete", "remove", "erase", "drop", "cancel", "the", "my", "that", "this", "last", "latest", "today's", "todays",
    "today", "yesterday's", "yesterdays", "log", "logs", "entry", "entries", "revision", "revisions", "record",
    "records", "hifz", "from", "of", "for", "on", "please", "javid", "it", "one", "i", "made", "logged", "a", "an",
    "মুছে", "দাও", "ডিলিট", "করো", "বাদ", "লগ", "এন্ট্রি", "রিভিশন", "শেষ", "আজকের", "গতকালের", "আমার", "টা", "টি",
    "এর", "জাভিদ", "muche", "dao", "koro", "delete"})
_LOG_WORD = quran._re(r"\b(?:log|logs|entry|entries|revision|revisions|record|records)\b|লগ|এন্ট্রি|রিভিশন")
_UNDO = quran._re(r"^(?:please |javid )?(?:undo|undo it|undo that|undo this|cancel that|cancel it|scratch that|"
                  r"take that back|that was (?:a )?(?:mistake|wrong)|আনডু|আনডু করো|বাতিল করো|undo koro|batil koro)"
                  r"(?: please)?$")
_UNDO_EXPLICIT = quran._re(r"\bundo (?:the |my )?(?:last |latest )?(?:hifz |revision )?(?:log|entry|revision)\b|"
                           r"\bundo (?:my )?last (?:hifz )?(?:log|entry|revision|one)\b|শেষ (?:লগ|এন্ট্রি) বাতিল")
_START = quran._re(r"\b(?:i'?m|i am|we'?re|i'?ve reached|i reached) (?:on|at|doing|up to|upto|starting) |"
                   r"\b(?:start|begin|set|move|put|restart)(?: my| the)? (?:rotation|cycle|revision|schedule|plan|hifz)"
                   r"(?: plan)? (?:at|from|with|to|on)\b|\bstart (?:from|at|with) |"
                   r"থেকে শুরু|(?:রোটেশন|চক্র)|আমি (?:এখন|আজ) .{0,15}পারায়|\b(?:theke shuru)\b")
_DUE = quran._re(
    r"\bwhat(?:'s| is| are)\b.{0,20}\bdue\b|\bwhat(?:'s| is)\b.{0,15}\b(?:today'?s|tomorrow'?s|my|the) "
    r"(?:revision|portion|juz|para|manzil|sabaq|amukhta)\b|\bwhat (?:should|do|shall|must|can) i "
    r"(?:revise|do|read|recite|review|cover)\b|\bwhich (?:juz|para|manzil|portion)\b.{0,20}"
    r"\b(?:today|now|next|tomorrow|due|should i|do i)\b|^(?:today'?s|tomorrow'?s) (?:revision|portion|juz|para)$|"
    r"\bwhat'?s next (?:in|for) (?:my )?(?:revision|rotation|hifz)\b|\bdid i (?:revise|do my revision|finish my "
    r"revision)\b(?: today)?$|\bhave i (?:revised|done my revision)(?: today)?$|"
    r"আজ (?:কী|কি|কোন|কোনটা|কোনগুলো).{0,25}(?:রিভাইস|রিভিশন|পারা|পড়ব|পড়বো|দেব|দিব|মঞ্জিল)|"
    r"আজকের (?:রিভিশন|পারা|পড়া|অংশ|সবক|আমুখতা)|কাল (?:কী|কি|কোন).{0,20}(?:পারা|রিভাইস|রিভিশন)|"
    r"\baaj (?:ki|kon|konta).{0,25}(?:revise|para|porbo|dibo)\b|\bajker (?:revision|para)\b|"
    r"\bkal (?:ki|kon).{0,20}(?:para|revise)\b")
_TOMORROW = quran._re(r"\btomorrow\b|কাল|আগামীকাল|\bkal\b")
_BEHIND = quran._re(r"\bwhat(?:'s| is| are)\b.{0,15}\b(?:behind|overdue|late|pending)\b|\bam i (?:behind|late|on track)\b|"
                    r"\bwhat am i behind\b|\bhow far behind\b|\b(?:what|anything) (?:have i|did i) (?:missed|skipped|"
                    r"miss|skip)\b|পিছিয়ে|বকেয়া|বাকি (?:আছে|পড়ে)|\bpichiye\b")
_LAST = quran._re(r"\bwhen did i (?:last )?(?:revise|do|read|finish|review|recite|go over|cover|log)\b|"
                  r"\b(?:last time|when was the last time) i (?:revised|did|read|finished|reviewed|recited|logged)\b|"
                  r"\bwhen was\b.{0,40}\b(?:last )?(?:revised|done|reviewed|read)\b|"
                  r"\b(?:did|have) i (?:revise|revised|do|done|finish|finished|read|review|reviewed)\b|"
                  r"কবে (?:শেষ )?.{0,30}(?:রিভাইস|পড়েছি|দিয়েছি|শেষ করেছি|পড়লাম|দিলাম|করেছি)|শেষ কবে|"
                  r"\bkobe (?:shesh )?.{0,30}(?:revise|korechi|porechi|diyechi)\b")
_WEEK = quran._re(r"\b(?:this|last|past) (?:week|7 days|seven days)\b|\bweekly\b|এই সপ্তাহে|এ সপ্তাহে|গত সপ্তাহে|"
                  r"সপ্তাহে|\b(?:ei|e) shoptahe\b")
_MONTH = quran._re(r"\b(?:this|last|past) (?:month|30 days|thirty days)\b|এই মাসে|গত মাসে|\bei mashe\b")
_HOW_MUCH = quran._re(r"\bhow (?:much|many)\b|কত|কয়টা|কয়টি|\b(?:koto|koyta)\b|\b(?:summary|total|report)\b")
_STATUS = quran._re(
    r"\b(?:show|give|tell|open|display|see)(?: me)?(?: my| the)? (?:hifz|hifdh|hifz revision|revision|quran revision|"
    r"memori[sz]ation)(?: status| progress| map| chart| panel| tracker| dashboard| summary| report)?\b|"
    r"\b(?:hifz|hifdh|revision) (?:status|progress|map|panel|tracker|summary|report)\b|\bhow'?s my "
    r"(?:hifz|revision|hifdh)\b|^my (?:hifz|revision)$|\bhow am i doing (?:with|on|in) (?:my )?(?:hifz|revision|quran)\b|"
    r"\bwhat'?s my (?:hifz |revision )?streak\b|\bhow many days in a row\b|"
    r"(?:হিফজ|হিফয|হেফজ|হিফজের|হিফযের|রিভিশনের).{0,15}(?:অবস্থা|স্ট্যাটাস|খবর|দেখাও|কেমন|প্রোগ্রেস)|"
    r"কত দিন (?:একটানা|টানা)|\bhifz(?:er)? (?:obostha|status)\b")
_QFILLER = _N({
    "what", "what's", "whats", "is", "are", "the", "my", "today", "today's", "todays", "tomorrow", "tomorrow's", "due",
    "revision", "portion", "juz", "para", "manzil", "should", "do", "i", "revise", "read", "recite", "review", "cover",
    "which", "next", "in", "for", "rotation", "hifz", "hifdh", "now", "shall", "must", "can", "behind", "overdue",
    "late", "pending", "am", "on", "track", "how", "far", "anything", "have", "did", "missed", "skipped", "miss",
    "skip", "when", "last", "was", "time", "the", "finish", "finished", "go", "over", "done", "reviewed", "revised",
    "much", "many", "this", "week", "past", "7", "days", "seven", "weekly", "month", "30", "thirty", "show", "give",
    "tell", "me", "open", "display", "see", "status", "progress", "map", "chart", "panel", "tracker", "dashboard",
    "summary", "report", "quran", "memorization", "memorisation", "how's", "hows", "doing", "with", "streak", "in",
    "a", "row", "javid", "please", "hey", "so", "far", "of", "it", "to", "total", "log", "logged", "up", "there",
    "and", "or", "am", "been", "far", "yet", "already", "all", "any", "do", "does", "you", "know", "it's", "at",
    "আজ", "আজকের", "কী", "কি", "কোন", "কোনটা", "কোনগুলো", "রিভাইস", "রিভিশন", "পারা", "পড়ব", "পড়বো", "দেব", "দিব",
    "মঞ্জিল", "পড়া", "অংশ", "সবক", "আমুখতা", "কাল", "আগামীকাল", "পিছিয়ে", "আছি", "আছে", "বকেয়া", "বাকি", "পড়ে",
    "কবে", "শেষ", "পড়েছি", "দিয়েছি", "শেষ", "করেছি", "পড়লাম", "দিলাম", "এই", "সপ্তাহে", "গত", "মাসে", "কত",
    "কয়টা", "কয়টি", "হিফজ", "হিফয", "হেফজ", "হিফজের", "হিফযের", "রিভিশনের", "অবস্থা", "স্ট্যাটাস", "খবর", "দেখাও",
    "কেমন", "প্রোগ্রেস", "দিন", "একটানা", "টানা", "আমার", "আমি", "জাভিদ", "তো", "এখন", "হবে", "করব", "করবো", "এ",
    "aaj", "ki", "kon", "konta", "revise", "para", "porbo", "dibo", "ajker", "kal", "pichiye", "kobe", "shesh",
    "korechi", "porechi", "diyechi", "ei", "e", "shoptahe", "mashe", "koto", "koyta", "hifzer", "obostha", "ami",
    "korbo", "ache", "er"})
ASK = quran._re(
    r"\b(?:revised|reviewed|finished|completed|recited|did|done)\b.{0,30}\b(?:juz|para|manzil|surah|sura)\b|"
    r"\b(?:juz|para|manzil)\b.{0,15}\b(?:done|revised|finished|weak|strong)\b|\bhifz\b|\bhifdh\b|"
    r"\b(?:my )?revision (?:log|status|streak|rotation|cycle|schedule)\b|\bwhat'?s due\b|\bam i behind\b|"
    r"রিভাইস করেছি|রিভাইস করলাম|রিভিশন দিয়েছি|রিভিশন দিলাম|হিফজ|হিফয|আমুখতা|\brevise korechi\b")
WORDS = quran._re(r"\b(?:juz|para|paara|manzil|hifz|hifdh|revis\w*|surah|sura|ayah|ayat|quran|qur'an|recit\w*)\b|"
                  r"পারা|মঞ্জিল|হিফজ|হিফয|রিভাইস|রিভিশন|সূরা|সুরা|আয়াত|কুরআন|কোরআন|তিলাওয়াত")


def _strengths(words: list[str]) -> tuple[list[tuple[int, str]], set[int]]:
    """Strength words with their positions: [(position, strength)], and every position they took."""
    out, used, i = [], set(), 0
    while i < len(words):
        for phrase, st in _STRENGTH:
            if tuple(words[i:i + len(phrase)]) == phrase:
                out.append((i, st))
                used.update(range(i, i + len(phrase)))
                i += len(phrase)
                break
        else:
            i += 1
    return out, used


def _day_offset(words: list[str]) -> tuple[int, set[int]]:
    for i in range(len(words)):
        for ws, off in _DAY_WORDS:
            if tuple(words[i:i + len(ws)]) == ws:
                return off, set(range(i, i + len(ws)))
    return 0, set()


def request(text: str, previous: dict | None = None) -> dict | None:
    """What a message asks of the tracker, when code can answer all of it: a log, undo, a delete or a reset (both
    wait for an Approve card), the rotation's start, or a question (due, behind, when he last revised something,
    this week's total, the status). None leaves it to the model. previous: the reply before (its tools say whether
    "undo that" means the log just made)."""
    t = quran.norm(text)
    words = quran.tokens(text)
    if not words:
        return None
    raw = text.strip()
    if _RESET.search(t):
        return {"action": "reset"}
    after_log = "hifz_log" in ((previous or {}).get("tools") or [])
    if _UNDO_EXPLICIT.search(t) or (after_log and _UNDO.search(t)):
        return {"action": "undo"}
    found, used = quran.portions(text, words)
    found = [p for p in found if p["kind"] != "bad_ayah"]
    if _DELETE.search(t) and (found or _LOG_WORD.search(t)):
        last = bool(re.search(r"\b(?:last|latest)\b|শেষ", t)) and not found
        off, d_used = _day_offset(words)
        today_ref = bool(re.search(r"\btoday'?s?\b|আজকের|আজ", t))
        if (found or last or off or today_ref) and not quran.leftover(words, used | d_used, _DELETE_FILLER):
            return {"action": "delete", "portions": found, "day": off if (off or today_ref) else None, "last": last}
    kind_words = {"juz", "para", "manzil", "পারা", "মঞ্জিল", "sipara"}
    if _START.search(t) and len(found) == 1 and found[0]["kind"] in ("juz", "manzil"):
        if not quran.leftover(words, used, _QFILLER | _LOG_FILLER | _N({"i'm", "im", "start", "begin", "set",
                                                                         "move", "put", "restart", "rotation",
                                                                         "cycle", "schedule", "plan", "from",
                                                                         "with", "starting", "reached", "upto",
                                                                         "শুরু", "থেকে", "রোটেশন", "চক্র",
                                                                         "পারায়", "theke", "shuru", "koro",
                                                                         "করো", "করুন", "we're", "we"}) | kind_words):
            return {"action": "start", "portion": found[0]}
    # questions
    tomorrow = bool(_TOMORROW.search(t)) and not re.search(r"\byesterday\b|গতকাল", t)
    if _STATUS.search(t) and not quran.leftover(words, used, _QFILLER):
        return {"action": "status"}
    if _LAST.search(t) and found and not quran.leftover(words, used, _QFILLER):
        return {"action": "last", "portions": found}
    if _HOW_MUCH.search(t) and (_WEEK.search(t) or _MONTH.search(t)) and not quran.leftover(words, used, _QFILLER):
        return {"action": "week", "days": 30 if _MONTH.search(t) else 7}
    if _BEHIND.search(t) and not quran.leftover(words, used, _QFILLER):
        return {"action": "behind"}
    if _DUE.search(t) and not quran.leftover(words, used, _QFILLER):
        return {"action": "due", "tomorrow": tomorrow}
    # a log
    return _log_request(text, t, raw, words, found, used)


def _log_request(text: str, t: str, raw: str, words: list[str], found: list[dict], used: set[int]) -> dict | None:
    today_words = bool(re.search(r"\btoday'?s (?:revision|portion|juz|para|manzil|sabaq|amukhta)\b|"
                                 r"আজকের (?:পারা|রিভিশন|পড়া|অংশ)|\bajker (?:para|revision)\b", t))
    if not found and not today_words:
        return None
    strengths, s_used = _strengths(words)
    verbs = {i for i, w in enumerate(words) if w in _VERB_PAST}
    if not verbs:
        return None
    rest = " ".join(w for i, w in enumerate(words) if i not in s_used)
    if _NEG.search(rest) or _FUTURE.search(rest) or raw.endswith("?") or _QUESTIONISH.search(rest):
        return None
    off, d_used = _day_offset(words)
    if today_words:
        used = used | {i for i, w in enumerate(words) if w in _N({"today's", "todays", "revision", "portion", "juz",
                                                                    "para", "manzil", "sabaq", "amukhta", "আজকের",
                                                                    "পারা", "রিভিশন", "পড়া", "অংশ", "ajker"})}
    left = quran.leftover(words, used | s_used | verbs | d_used, _LOG_FILLER | _PAST | quran.AND |
                          quran.ORD_TAIL)
    if left:
        return None
    # each portion's strength: the first strength word after it, else the nearest before it, else okay
    marks = []
    for a, b in (_portion_spans(words, found, used) if found else []):
        after = [st for i, st in strengths if i > b]
        before = [st for i, st in strengths if i < a]
        marks.append(after[0] if after else before[-1] if before else "okay")
    return {"action": "log", "portions": found, "today": today_words and not found,
            "strengths": marks or ([strengths[0][1]] if strengths else ["okay"]), "day": off}


def _portion_spans(words: list[str], found: list[dict], used: set[int]) -> list[tuple[int, int]]:
    """Where each found portion's words are (in order), for placing the strength words: runs of used positions
    split among the portions; a run that named several ("juz 5 to 7") is shared by them."""
    runs, run = [], []
    for i in sorted(used):
        if run and i != run[-1] + 1:
            runs.append((run[0], run[-1]))
            run = []
        run.append(i)
    if run:
        runs.append((run[0], run[-1]))
    if len(runs) >= len(found):
        return runs[:len(found)]
    out = []
    per = max(1, len(found) // max(1, len(runs)))
    for k in range(len(found)):
        out.append(runs[min(k // per, len(runs) - 1)])
    return out


# ---------- answers ----------
def answer(req: dict, lang: str, via: str = "page"):
    """A generator: ("tool", "hifz" or "hifz_log"), ("hifz", {...}) for the page's panel, then (spoken, screen)."""
    act = req["action"]
    yield ("tool", "hifz_log" if act in ("log", "undo", "start") else "hifz")
    if act == "log":
        return (yield from _log_answer(req, lang, via))
    if act == "undo":
        gone = undo_last()
        yield ("hifz", {"refresh": True})
        if not gone:
            return ("গত ২৪ ঘণ্টায় বাতিল করার মতো কিছু নেই।" if lang == "bn" else
                    "There's nothing from the last 24 hours to undo."), ""
        labels = [(_label(e, lang, True) + (f" ({_strength_word(e['strength'], lang)})" if e["strength"] != "okay" else ""))
                  if e["kind"] == "log" else (f"রোটেশনের শুরু, {_unit_word(e['unit'], 'bn', e['n'])}" if lang == "bn"
                                              else f"the rotation's start at {e['unit']} {e['n']}") for e in gone]
        return (f"বাতিল করা হলো: {quran.join(labels, 'bn')}।" if lang == "bn" else
                f"Undone: {quran.join(labels)}."), ""
    if act == "start":
        p = req["portion"]
        cfg = config()
        if p["kind"] != cfg["unit"]:
            return (f"তোমার রোটেশন {_unit_word(cfg['unit'], 'bn')} ধরে চলে; config/hifz.yaml-এ unit বদলানো যায়।"
                    if lang == "bn" else
                    f"Your rotation goes by {cfg['unit']}; the unit can be changed in config/hifz.yaml."), ""
        set_start(p["n"], via)
        yield ("hifz", {"refresh": True})
        s = _state()
        today_ns, nxt = [u["n"] for u in s["today_units"]], [u["n"] for u in s["tomorrow"]]
        if lang == "bn":
            head = f"ঠিক আছে: রোটেশন এখন {_unit_word(p['kind'], 'bn', p['n'])} থেকে"
            rest = (f"আজ {_units_text(p['kind'], today_ns, 'bn')}, কাল {_units_text(p['kind'], nxt, 'bn')}" if today_ns
                    else f"আজকেরটা হয়ে গেছে, কাল {_units_text(p['kind'], nxt, 'bn')}")
            return f"{head}: {rest}।", ""
        rest = (f"today's revision is {_units_text(p['kind'], today_ns, 'en')} and tomorrow "
                f"{_units_text(p['kind'], nxt, 'en')}" if today_ns else
                f"today's is already done, and tomorrow is {_units_text(p['kind'], nxt, 'en')}")
        return f"Done: your rotation now runs from {p['kind']} {p['n']}, so {rest}.", ""
    if act == "delete":
        return _delete_answer(req, lang)
    if act == "reset":
        return _reset_answer(lang)
    s = _state()
    if act == "due":
        return _due_said(s, lang, req.get("tomorrow")), _due_screen(s, lang)
    if act == "behind":
        return _behind_said(s, lang), ""
    if act == "last":
        return _last_said(s, req["portions"], lang), ""
    if act == "week":
        return _week_said(s, lang, req.get("days", 7)), ""
    if act == "status":
        yield ("hifz", {"open": True})
        return _status_said(s, lang), _status_screen(s, lang)
    return "", ""


def _strength_word(st: str, lang: str) -> str:
    return {"en": {"weak": "weak", "strong": "strong", "okay": "okay"},
            "bn": {"weak": "কাঁচা", "strong": "পাকা", "okay": "মোটামুটি"}}["bn" if lang == "bn" else "en"][st]


def _log_answer(req: dict, lang: str, via: str):
    before = _state()
    found = req["portions"] or (today_portions() if req.get("today") else [])
    if not found:
        return ("আজকের রিভিশন আগেই লগ করা আছে, মাশাআল্লাহ।" if lang == "bn" else
                "Today's revision is already logged, MashaAllah."), ""
    strengths = req["strengths"] if len(req["strengths"]) == len(found) else [req["strengths"][0]] * len(found)
    day = before["today"] + timedelta(days=req.get("day", 0))
    entries = log(found, strengths, day, via)
    yield ("hifz", {"refresh": True})
    return _logged_said(before, entries, lang), ""


def logged_said(entries: list[dict], before: dict, lang: str) -> str:
    return _logged_said(before, entries, lang)


def _logged_said(before: dict, entries: list[dict], lang: str) -> str:
    """What Javid says to a log: what he logged, today's revision done (and the streak), a weak one's return day, a
    full round completed, tomorrow's portion."""
    after = _state()
    today = after["today"]
    bn = lang == "bn"
    day = date.fromisoformat(entries[0]["date"])
    labels = []
    for e in entries:
        lab = _label(e, lang, True)
        if e["strength"] != "okay":
            lab += f", {_strength_word(e['strength'], lang)}" if not bn else f" ({_strength_word(e['strength'], lang)})"
        labels.append(lab)
    when = "" if day == today else (f" ({_day(day, today, lang)})" if bn else f" for {_day(day, today, lang)}")
    said = [f"লগ করা হলো: {quran.join(labels, 'bn')}{when}।" if bn else f"Logged {quran.join(labels)}{when}."]
    was_done = len(before["done_today"]) >= before["daily"]
    now_done = len(after["done_today"]) >= after["daily"]
    if now_done and not was_done:
        st = _streak_text(after, lang)
        said.append(("আজকের রিভিশন শেষ, মাশাআল্লাহ" + (f": {st}।" if st else "।")) if bn else
                    ("That's today's revision done, MashaAllah" + (f": {st}." if st else ".")))
    weak = [e for e in entries if e["strength"] == "weak"]
    if weak:
        back = date.fromisoformat(weak[0]["date"]) + timedelta(days=after["cfg"]["weak_days"])
        said.append(f"কাঁচা অংশটা {_day(back, today, lang)} আবার আসবে।" if bn else
                    f"It comes back {_on(back, today)} for another go.")
    if after["rounds"] > before["rounds"]:
        r = after["rounds"]
        said.append((f"এতে পুরো কুরআনের {quran.bn_digits(r)} নম্বর চক্র শেষ হলো, আলহামদুলিল্লাহ।" if r > 1 else
                     "এতে পুরো কুরআনের এক চক্র শেষ হলো, আলহামদুলিল্লাহ।") if bn else
                    (f"That completes round {r} of the whole Quran, Alhamdulillah." if r > 1 else
                     "That completes a full round of the Quran, Alhamdulillah."))
    if now_done and not was_done and after["tomorrow"]:
        nxt = _units_text(after["kind"], [u["n"] for u in after["tomorrow"]], lang)
        said.append(f"কাল: {nxt}।" if bn else f"Tomorrow: {nxt}.")
    return " ".join(said)


def _on(d: date, today: date) -> str:
    w = _day(d, today, "en")
    return w if w in ("today", "tomorrow", "yesterday") else f"on {w}"


def _due_said(s: dict, lang: str, tomorrow: bool = False) -> str:
    bn = lang == "bn"
    kind = s["kind"]
    if tomorrow:
        ns = [u["n"] for u in s["tomorrow"]]
        if not ns:
            return ""
        spans = "; ".join(_span_of(s, n, lang) for n in ns)
        return f"কাল: {_units_text(kind, ns, 'bn')}, {spans}।" if bn else f"Tomorrow: {_units_text(kind, ns, 'en')}, {spans}."
    if not s["started"]:
        n = s["start_n"]
        return (f"শুরু করতে: {_unit_word(kind, 'bn', n)}, {_span_of(s, n, lang)}। অথবা বলো তুমি কোথায় আছ, যেমন "
                f"“আমি আজ ১২ নম্বর পারায়”।" if bn else
                f"To begin: {kind} {n}, {_span_of(s, n, lang)}. Or tell me where you are, like “I'm on juz 12”.")
    said = []
    if not s["today_units"]:
        nxt = [u["n"] for u in s["tomorrow"]]
        said.append(f"আজকের রিভিশন শেষ, মাশাআল্লাহ। কাল: {_units_text(kind, nxt, 'bn')}।" if bn else
                    f"Today's revision is done, MashaAllah. Tomorrow: {_units_text(kind, nxt, 'en')}.")
    else:
        parts = []
        for u in s["today_units"]:
            late = (f" ({quran.bn_digits(u['behind'])} দিন দেরি)" if bn else
                    f" ({u['behind']} day{'s' if u['behind'] != 1 else ''} late)") if u["behind"] else ""
            parts.append(f"{_unit_word(kind, lang, u['n'])}, {_span_of(s, u['n'], lang)}{late}")
        said.append(f"আজকের রিভিশন: {'; '.join(parts)}।" if bn else f"Today's revision: {'; '.join(parts)}.")
    weak = [w for w in s["weak_due"] if not any(u["lo"] <= w["lo"] and w["hi"] <= u["hi"] for u in s["today_units"])]
    if weak:
        labs = quran.join([w["label_bn"] if bn else quran.cap(w["label"]) for w in weak[:3]], lang)
        said.append(f"কাঁচা হিসেবে ফেরত: {labs}।" if bn else f"Back as weak: {labs}.")
    others = [u for u in s["behind"] if u not in s["today_units"]]
    if others:
        ns = [u["n"] for u in others[:4]]
        said.append(f"আরও পিছিয়ে: {_units_text(kind, ns, 'bn')}।" if bn else
                    f"Also behind: {_units_text(kind, ns, 'en')}.")
    return " ".join(said)


def _due_screen(s: dict, lang: str) -> str:
    if not s["started"]:
        return ""
    bn = lang == "bn"
    rows = ["আজ:" if bn else "Today:"]
    for u in s["today_units"]:
        pg = quran.pages([(u["lo"], u["hi"])])
        rows.append(f"- {_unit_word(s['kind'], lang, u['n'])} · {_span_of(s, u['n'], lang, False)} · "
                    f"{quran.bn_digits(round(pg)) if bn else round(pg)} {'পৃষ্ঠা' if bn else 'pages'}")
    if not s["today_units"]:
        rows.append("- " + ("শেষ ✓" if bn else "done ✓"))
    for w in s["weak_due"][:4]:
        rows.append(f"- {w['label_bn'] if bn else w['label']} · " +
                    (f"কাঁচা, {_day(w['date'], s['today'], lang)} থেকে" if bn else f"weak since {_day(w['date'], s['today'], lang)}"))
    if s["behind"]:
        rows.append("পিছিয়ে:" if bn else "Behind:")
        rows += [f"- {_unit_word(s['kind'], lang, u['n'])} · " +
                 (f"{quran.bn_digits(u['behind'])} দিন" if bn else f"{u['behind']} day{'s' if u['behind'] != 1 else ''}")
                 for u in s["behind"][:6]]
    extra = [x for x in (_streak_text(s, lang), _ramadan_text(lang, True)) if x]
    if extra:
        rows.append(" · ".join(quran.cap(x) for x in extra))
    return "\n".join(rows)


def _behind_said(s: dict, lang: str) -> str:
    bn = lang == "bn"
    kind, cycle = s["kind"], s["cycle"]
    if not s["started"]:
        return "এখনো কিছু লগ করা হয়নি, তাই কিছু পিছিয়ে নেই।" if bn else "Nothing's logged yet, so nothing is behind."
    weak_late = [w for w in s["weak_due"] if w["due"] < s["today"]]
    if not s["behind"]:
        said = (f"কিছুই পিছিয়ে নেই: সব {_unit_word(kind, 'bn')} {quran.bn_digits(cycle)} দিনের চক্রের ভেতরে, মাশাআল্লাহ।"
                if bn else f"Nothing's behind: every {kind} is within your {cycle}-day cycle, MashaAllah.")
    else:
        items = [f"{_unit_word(kind, lang, u['n'])} " + (f"{quran.bn_digits(u['behind'])} দিন" if bn else
                                                        f"by {u['behind']} day{'s' if u['behind'] != 1 else ''}")
                 for u in s["behind"][:5]]
        n = len(s["behind"])
        if bn:
            said = f"{quran.bn_digits(n)}টি পিছিয়ে: {quran.join(items, 'bn')}।"
            if n > 1:
                said += f" কয়েক দিন দিনে {quran.bn_digits(s['daily'] + 1)}টি করে করলে চক্রে ফিরে আসবে।"
        else:
            said = f"{n} behind: {quran.join(items)}."
            if n > 1:
                said += f" {s['daily'] + 1} a day for a few days brings you back to your {cycle}-day cycle."
    if weak_late:
        labs = quran.join([w["label_bn" if bn else "label"] for w in weak_late[:3]], lang)
        said += f" কাঁচা অংশ, সময় পেরিয়ে গেছে: {labs}।" if bn else f" Weak and past their day: {labs}."
    return said


def _last_said(s: dict, found: list[dict], lang: str) -> str:
    bn = lang == "bn"
    today = s["today"]
    said = []
    for p in found:
        lo, hi = p["ranges"][0]
        cover = [e for e in s["logs"] if any(a <= hi and lo <= b for a, b in e["ranges"])]
        lab = quran.portion_label(p, lang, True)
        if not cover:
            said.append(f"{lab}: এখনো লগ করা হয়নি।" if bn else f"There's no log for {lab} yet.")
            continue
        # the day all of it had been revised by (its stalest ayah), and the logs that did it
        last = {}
        for e in cover:
            for a, b in e["ranges"]:
                for i in range(max(a, lo), min(b, hi) + 1):
                    last[i] = e
        if len(last) < hi - lo + 1:
            e = max(cover, key=lambda x: (x["date"], x["at"]))
            part = f"{_label(e, lang, True)}, {_day(date.fromisoformat(e['date']), today, lang)}"
            said.append(f"পুরো {lab} এখনো একসাথে লগ হয়নি; শেষ অংশ: {part}।" if bn else
                        f"All of {lab} hasn't been revised since you started logging; the latest part was {part}.")
            continue
        stalest = min(last.values(), key=lambda x: (x["date"], x["at"]))
        newest = max(last.values(), key=lambda x: (x["date"], x["at"]))
        logs = sorted({x["id"]: x for x in last.values()}.values(), key=lambda x: (x["date"], x["at"]))
        d1 = date.fromisoformat(stalest["date"])
        if len(logs) == 1:
            e = logs[0]
            st = f", {_strength_word(e['strength'], lang)}" if e["strength"] != "okay" else ""
            if bn:
                said.append(f"শেষবার {lab}: {_day(d1, today, lang)}, {_label(e, lang)} হিসেবে{st}।")
            else:
                via = "" if _label(e) == quran.portion_label(p) else f", as {_label(e, lang, True)}"
                said.append(f"You last revised {lab} {_on(d1, today)}{via}{st}.")
        else:
            parts = quran.join([f"{_label(x, lang, True)} {_on(date.fromisoformat(x['date']), today) if not bn else _day(date.fromisoformat(x['date']), today, lang)}"
                                for x in logs[:3]], lang)
            d2 = date.fromisoformat(newest["date"])
            if bn:
                said.append(f"পুরো {lab} শেষবার রিভাইস হয়েছে {_day(d2, today, lang)}-এর মধ্যে: {parts}।")
            else:
                said.append(f"All of {lab} was last revised by {_day(d2, today, lang)}: {parts}.")
    return " ".join(said)


def _week_said(s: dict, lang: str, days: int = 7) -> str:
    bn = lang == "bn"
    today = s["today"]
    since = today - timedelta(days=days - 1)
    logs = [e for e in s["logs"] if e["date"] >= since.isoformat()]
    window = (f"গত {quran.bn_digits(days)} দিনে" if bn else f"In the last {days} days")
    if not logs:
        return f"{window} কোনো রিভিশন লগ হয়নি।" if bn else f"{window}, nothing was logged."
    pages = quran.pages([tuple(r) for e in logs for r in e["ranges"]])
    juz = pages / 20.13
    labels = quran.join(_compress([e for e in logs], lang), lang)
    on_days = len({e["date"] for e in logs})
    weak = sum(1 for e in logs if e["strength"] == "weak")
    if bn:
        said = (f"{window}: {labels}, মোট প্রায় {quran.bn_digits(_round(juz))} পারা ({quran.bn_digits(round(pages))} পৃষ্ঠা), "
                f"{quran.bn_digits(days)} দিনের {quran.bn_digits(on_days)} দিনে।")
        if weak:
            said += f" {quran.bn_digits(weak)}টি কাঁচা হিসেবে লগ।"
        return said
    said = f"{window}: {labels}, about {_round(juz)} juz in all ({round(pages)} pages), on {on_days} of {days} days."
    if weak:
        said += f" {weak} {'was' if weak == 1 else 'were'} marked weak."
    return said


def _round(x: float) -> str:
    return f"{x:.0f}" if x >= 10 or abs(x - round(x)) < 0.05 else f"{x:.1f}"


def _compress(logs: list[dict], lang: str) -> list[str]:
    """"juz 5, 6 and 7" rather than "juz 5, juz 6 and juz 7"; surahs by name; each once."""
    juz, manzil, other = [], [], []
    for e in logs:
        p = e["portion"]
        if p["kind"] == "juz" and p["n"] not in juz:
            juz.append(p["n"])
        elif p["kind"] == "manzil" and p["n"] not in manzil:
            manzil.append(p["n"])
        elif p["kind"] not in ("juz", "manzil"):
            lab = _label(e, lang, True)
            if lab not in other:
                other.append(lab)
    out = []
    if juz:
        out.append(_units_text("juz", sorted(juz), lang))
    if manzil:
        out.append(_units_text("manzil", sorted(manzil), lang))
    return out + other


def _status_said(s: dict, lang: str) -> str:
    bn = lang == "bn"
    if not s["started"]:
        return ("হিফজ ট্র্যাকার প্রস্তুত, এখনো কিছু লগ হয়নি। রিভাইস করে বলো, যেমন “আজ ৫ পারা রিভাইস করেছি”।" if bn else
                "Your hifz tracker is ready, with nothing logged yet. Tell me when you revise, like “I revised juz 5”.")
    kind, count, cycle = s["kind"], s["count"], s["cycle"]
    today = (_units_text(kind, [u["n"] for u in s["today_units"]], lang) if s["today_units"] else
             ("শেষ" if bn else "done"))
    if bn:
        parts = [f"আজ: {today}", f"{quran.bn_digits(cycle)} দিনের মধ্যে রিভাইস: {quran.bn_digits(s['fresh'])}/{quran.bn_digits(count)}"]
        if s["behind"]:
            parts.append(f"পিছিয়ে {quran.bn_digits(len(s['behind']))}টি")
        if s["weak"]:
            parts.append(f"কাঁচা {quran.bn_digits(len(s['weak']))}টি")
    else:
        parts = [f"Today: {today}", f"{s['fresh']} of {count} {kind if kind == 'juz' else 'manzils'} revised in the "
                                    f"last {cycle} days"]
        if s["behind"]:
            parts.append(f"{len(s['behind'])} behind")
        if s["weak"]:
            parts.append(f"{len(s['weak'])} weak")
    said = ("; ".join(parts) + ("।" if bn else "."))
    extra = [x for x in (_streak_text(s, lang), _ramadan_text(lang)) if x]
    if extra:
        said += " " + quran.cap(quran.join(extra, lang)) + ("।" if bn else ".")
    return said + (" পুরো ছবি হিফজ প্যানেলে।" if bn else " The full map is in the Hifz view.")


def _runs(ns: list[int], lang: str) -> str:
    """1, 2, 3, 5, 7, 8 as "1–3, 5, 7–8"."""
    out, i = [], 0
    ns = sorted(ns)
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
            j += 1
        a, b = ns[i], ns[j]
        out.append((quran.bn_digits(a) if lang == "bn" else str(a)) if a == b else
                   f"{quran.bn_digits(a) if lang == 'bn' else a}–{quran.bn_digits(b) if lang == 'bn' else b}")
        i = j + 1
    return ", ".join(out)


def _status_screen(s: dict, lang: str) -> str:
    """The juz (or manzils) by how fresh they are, as short ranges; the Hifz view draws them."""
    if not s["started"]:
        return ""
    bn = lang == "bn"
    cycle = s["cycle"]
    groups = {"fresh": [], "steady": [], "soon": [], "over": [], "never": []}
    for u in s["units"]:
        d = u["days"]
        groups["never" if d is None else "fresh" if d <= cycle / 3 else "steady" if d <= 2 * cycle / 3 else
               "soon" if d <= cycle else "over"].append(u["n"])
    names = ({"fresh": "তাজা", "steady": "ঠিক আছে", "soon": "শিগগির দরকার", "over": "চক্র পেরিয়েছে", "never": "এখনো না"}
             if bn else {"fresh": "Fresh", "steady": "Steady", "soon": "Due soon", "over": "Past the cycle",
                         "never": "Not yet"})
    rows = [f"{'হিফজ' if bn else 'Hifz'} · {_unit_word(s['kind'], lang)} · {quran.bn_digits(cycle) if bn else cycle}"
            f"{' দিনের চক্র' if bn else '-day cycle'}:"]
    rows += [f"- {names[k]}: {_runs(v, lang)}" for k, v in groups.items() if v]
    weak = [u["n"] for u in s["units"] if u["weak"]]
    if weak:
        rows.append(f"- {'কাঁচা' if bn else 'Weak'}: {_runs(weak, lang)}")
    if s["today_units"]:
        rows.append(f"- {'আজ' if bn else 'Today'}: {_runs([u['n'] for u in s['today_units']], lang)}")
    extra = [x for x in (_streak_text(s, lang), _ramadan_text(lang, True)) if x]
    if extra:
        rows.append(" · ".join(quran.cap(x) for x in extra))
    return "\n".join(rows)


def _delete_answer(req: dict, lang: str):
    import javid
    s = _state()
    bn = lang == "bn"
    entries = [e for e in s["entries"] if e["kind"] == "log"]
    if req.get("last") and entries:
        last = entries[-1]
        chosen = [e for e in entries if e.get("batch") == last.get("batch")]
    else:
        day = None if req.get("day") is None else s["today"] + timedelta(days=req["day"])
        chosen = []
        for p in req["portions"]:
            lo, hi = p["ranges"][0]
            match = [e for e in entries if any(a <= hi and lo <= b for a, b in e["ranges"]) and
                     (day is None or e["date"] == day.isoformat())]
            if match:
                newest = max(match, key=lambda e: e["at"])
                chosen += [e for e in match if e["date"] == newest["date"]]
        if not req["portions"] and day is not None:
            chosen = [e for e in entries if e["date"] == day.isoformat()]
    chosen = list({e["id"]: e for e in chosen}.values())
    if not chosen:
        return ("মুছে ফেলার মতো কোনো লগ পাইনি।" if bn else "I couldn't find a log like that to delete."), ""
    labels = [f"{_label(e)} ({_day(date.fromisoformat(e['date']), s['today'], 'en')})" for e in chosen]
    javid.gate_write("hifz_delete", {"ids": [e["id"] for e in chosen], "labels": labels})
    return _approve_said(lang), ""


def _reset_answer(lang: str):
    import javid
    entries = [e for e in _read() if e["kind"] == "log"]
    if not entries:
        return ("মুছে ফেলার মতো কোনো রিভিশন লগ নেই।" if lang == "bn" else "There's no revision history to delete."), ""
    first = min(e["date"] for e in entries)
    javid.gate_write("hifz_reset", {"count": len(entries), "since": first})
    return _approve_said(lang), ""


def _approve_said(lang: str) -> str:
    import javid
    web = javid._write_gate is javid._web_gate
    if lang == "bn":
        return ("এটা তোমার অনুমোদনের অপেক্ষায়: নিচের কার্ডে Approve চাপো, অথবা পরের মেসেজে “হ্যাঁ” বলো।" if web else
                "এটা তোমার অনুমোদনের অপেক্ষায়: নিচে Approve চাপো।")
    return ("That waits for your Approve: tap it on the card below, or say yes as your next message." if web else
            "That waits for your Approve: tap it below.")


# ---------- for the model: a mixed message ("I revised juz 5, and set a timer") ----------
TOOLS = [{"type": "function", "function": {
    "name": "hifz",
    "description": "Mahadi's Quran revision (hifz) tracker, all in code: log what he revised, or answer what's due "
                   "today, what's behind, when he last revised a part, this week's total or his status. Pass the "
                   "revision part of his message in his own words, e.g. 'I revised juz 5, weak' or 'what's due "
                   "today?'. Never work out juz, surahs or ayah numbers yourself, and never quote the Quran.",
    "parameters": {"type": "object", "properties": {
        "request": {"type": "string", "description": "His words about his revision."}}, "required": ["request"]}}}]


def tool(request_text: str) -> str:
    req = request(request_text)
    if not req:
        return ("Not logged: I couldn't tell which part of the Quran he means (a juz, para, manzil or surah). "
                "Ask him.")
    if req["action"] in ("delete", "reset", "undo"):
        return ("Not done: deleting or undoing revision logs is done when he says it as a message of its own "
                "(\"undo that\", \"delete the juz 5 log\"); tell him so.")
    gen = answer(req, "en", via="model")
    while True:
        try:
            next(gen)
        except StopIteration as done:
            spoken, screen = done.value
            return spoken + (f"\n{screen}" if screen else "")


TOOL_MAP = {"hifz": lambda args: tool(str(args.get("request") or ""))}


# ---------- logging from a reminder's buttons (the page and Telegram) ----------
def log_from_alert(alert_id: str, strength: str, via: str) -> str:
    """The reminder's buttons: log exactly the portions it named, for its day, once."""
    st = strength if strength in STRENGTHS else "okay"
    with _lock:
        state = _load_state()
        if alert_id in state.get("logged", {}):
            return "Already logged from this reminder."
        item = timers.get(alert_id)
        info = (item or {}).get("extra", {}).get("hifz")
        if not info:
            return "That reminder has gone; say what you revised instead."
        found = [quran.unit_portion(p["kind"], p["n"]) if p["kind"] in ("juz", "manzil") else
                 {**p, "ranges": [tuple(r) for r in p["ranges"]]} for p in info["portions"]]
        before = _state()
        entries = log([p for p in found if p], st, date.fromisoformat(info["date"]), via)
        state.setdefault("logged", {})[alert_id] = _time()
        _save_state(state)
    return _logged_said(before, entries, "en")


# ---------- reminders (the watches thread) ----------
def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
        if isinstance(s, dict):
            s.setdefault("raised", {})
            s.setdefault("logged", {})
            return s
    except (OSError, ValueError):
        pass
    return {"raised": {}, "logged": {}}


def _save_state(s: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_FILE)


def _at(d: date, when: dict) -> datetime:
    zone = prayers._zone()
    if "at" in when:
        hh, mm = map(int, when["at"].split(":"))
        return datetime(d.year, d.month, d.day, hh, mm, tzinfo=zone)
    dd = prayers.times_for(d, wait=False)
    if dd:
        base = prayers._at(dd, when["after"])
    else:
        hh, mm = map(int, FALLBACK[when["after"]].split(":"))
        base = datetime(d.year, d.month, d.day, hh, mm, tzinfo=zone)
    return base + timedelta(minutes=when["minutes"])


def planned(d: date, cfg: dict | None = None) -> list[dict]:
    """Day d's reminders: the morning one and the nudges, with when they're due and until when they may ring."""
    cfg = cfg or config()
    rem = cfg["reminders"]
    if not (rem["page"] or rem["telegram"]):
        return []
    out = []
    if rem["morning"]:
        at = _at(d, rem["morning"])
        out.append({"key": f"{d.isoformat()}:morning", "kind": "morning", "due": at,
                    "until": at + timedelta(seconds=MORNING_LATE_SEC)})
    for i, w in enumerate(rem["nudges"]):
        at = _at(d, w)
        out.append({"key": f"{d.isoformat()}:nudge:{i}", "kind": "last" if i == len(rem["nudges"]) - 1 and i else "nudge",
                    "due": at, "until": at + timedelta(seconds=NUDGE_SEC)})
    return sorted(out, key=lambda x: x["due"])


def _reminder_texts(item: dict, s: dict) -> tuple[str, dict, list[dict]]:
    """(label, {en, bn}, the portions it names) of a reminder as it's raised."""
    kind = s["kind"]
    units = [u["n"] for u in s["today_units"]]
    found = [{"kind": kind, "n": n} for n in units]
    for w in s["weak_due"]:
        if not any(u["lo"] <= w["lo"] and w["hi"] <= u["hi"] for u in s["today_units"]):
            e = w["entry"]
            found.append({**e["portion"], "ranges": e["ranges"]} if w["whole"] else
                         {"kind": "surah", "n": quran.verse(w["lo"])[0], "ranges": [[w["lo"], w["hi"]]]})
    texts = {}
    for lang in ("en", "bn"):
        bn = lang == "bn"
        what = "; ".join(f"{_unit_word(kind, lang, n)}, {_span_of(s, n, lang)}" for n in units)
        weak = quran.join([w["label_bn" if bn else "label"] for w in s["weak_due"] if not any(u["lo"] <= w["lo"] and w["hi"] <= u["hi"]
                                                                        for u in s["today_units"])][:2], lang)
        st, rd = _streak_text(s, lang), _ramadan_text(lang)
        if item["kind"] == "morning":
            if bn:
                t = f"মাহাদি, আজকের রিভিশন: {what}।" + (f" কাঁচা অংশও আছে: {weak}।" if weak else "")
                t += (f" {st}, চালিয়ে যাও।" if st else " আজই নতুন করে শুরু করো।") + (f" {rd}।" if rd else "")
            else:
                t = f"Mahadi, today's revision: {what}." + (f" {quran.cap(weak)} comes back today too." if weak else "")
                t += (f" It's {st}, keep it going." if st else " Start a fresh streak today.") + (f" {quran.cap(rd)}." if rd else "")
        elif item["kind"] == "nudge":
            short = _units_text(kind, units, lang)
            if bn:
                t = f"মাহাদি, আজকের {short} এখনো বাকি। এখন কুড়ি মিনিট দিলেই " + (f"{st}ের ধারা টিকে থাকবে।" if st else "দিনটা ধরা থাকবে।")
            else:
                t = f"Mahadi, {short} is still waiting today. Twenty quiet minutes now " + \
                    (f"keeps your streak alive: {st}." if st else "keeps today in the chain.")
        else:
            short = _units_text(kind, units, lang)
            if bn:
                t = f"আজকের শেষ ডাক: {short}। অল্প একটু হলেও এখন পড়ে নাও, শিকলটা যেন না ভাঙে।"
            else:
                t = f"Last call for today: {short}. Even part of it now keeps the chain unbroken."
        texts[lang] = t
    label = {"morning": "Today's revision", "nudge": "Revision still waiting", "last": "Last call for revision"}[item["kind"]]
    return f"{label}: {_units_text(kind, units, 'en')}", texts, found


def tick(now: float | None = None) -> list[dict]:
    """Raise what has come due (the watches thread): each reminder once (data/hifz_state.json), LEAD_SEC early so the
    page rings it on its minute, never after its `until`; a nudge only while today's portion isn't logged."""
    ts = _time() if now is None else now
    cfg = config()
    rem = cfg["reminders"]
    channels = None if rem["page"] and rem["telegram"] else ["telegram"] if rem["telegram"] else ["page", "cli"]
    d = datetime.fromtimestamp(ts, prayers._zone()).date()
    raised = []
    with _lock:
        state = _load_state()
        changed = False
        for item in planned(d, cfg):
            due, until = item["due"].timestamp(), item["until"].timestamp()
            if item["key"] in state["raised"] or ts < due - LEAD_SEC or ts >= until:
                continue
            s = _state(d, cfg=cfg)
            state["raised"][item["key"]] = ts
            changed = True
            if not s["today_units"] and not s["weak_due"]:     # already done today: no reminder, no nudge
                print(f"[hifz] {item['key']}: today's revision is done, nothing to remind")
                continue
            try:
                label, texts, found = _reminder_texts(item, s)
            except Exception as e:      # a broken text mustn't stop the tick
                print(f"[hifz] {item['key']}: {type(e).__name__}: {str(e)[:80]}")
                continue
            raised.append(timers.add_alert(label, texts, kind="hifz", due=max(due, ts), until=until, channels=channels,
                                           extra={"hifz": {"date": d.isoformat(), "portions": found,
                                                           "kind": item["kind"]}}))
            print(f"[hifz] raised {item['key']} for {datetime.fromtimestamp(max(due, ts)):%H:%M}")
        if changed:
            keep = ts - KEEP_DAYS * 86400
            state["raised"] = {k: v for k, v in state["raised"].items() if v >= keep}
            state["logged"] = {k: v for k, v in state.get("logged", {}).items() if v >= keep}
            _save_state(state)
    return raised


# ---------- the morning brief's line, the page's panel ----------
def brief_line(d: date, lang: str) -> str:
    """📖 Revision today: juz 13 (Yusuf 53 to Ibrahim 52) · weak: juz 7 · day 12 in a row · 79 days to Ramadan."""
    if not config()["brief"]:
        return ""
    s = _state(d)
    bn = lang == "bn"
    if not s["started"]:
        return ("📖 হিফজ রিভিশন: আজ শুরু করো, রিভাইস করে বলো “আজ ১ পারা রিভাইস করেছি”।" if bn else
                "📖 Hifz revision: start today, and tell Javid “I revised juz 1” when you're done.")
    if s["today_units"]:
        what = "; ".join(f"{_unit_word(s['kind'], lang, u['n'])} ({_span_of(s, u['n'], lang, False)})" for u in s["today_units"])
    else:
        what = "শেষ ✓" if bn else "done ✓"
    parts = [f"{'আজকের রিভিশন' if bn else 'Revision today'}: {what}"]
    if s["weak_due"]:
        parts.append(f"{'কাঁচা' if bn else 'weak'}: {quran.join([w['label_bn' if bn else 'label'] for w in s['weak_due'][:2]], lang)}")
    if s["behind"]:
        parts.append(f"{quran.bn_digits(len(s['behind']))}টি পিছিয়ে" if bn else f"{len(s['behind'])} behind")
    parts += [x for x in (_streak_text(s, lang), _ramadan_text(lang, True)) if x]
    return "📖 " + " · ".join(parts)


def status_json() -> dict:
    """The Hifz panel and view (GET /hifz)."""
    s = _state()
    cfg = s["cfg"]
    kind, cycle = s["kind"], s["cycle"]

    def state_of(u):
        if u["days"] is None:
            return "never"
        if u["days"] <= cycle / 3:
            return "fresh"
        if u["days"] <= 2 * cycle / 3:
            return "steady"
        return "soon" if u["days"] <= cycle else "over"
    def weak_back(u):       # the day a weak part of it comes back
        dues = [w["due"] for w in s["weak"] if w["lo"] <= u["hi"] and u["lo"] <= w["hi"]]
        return min(dues).isoformat() if dues else None
    today_ns = {u["n"] for u in s["today_units"]}
    units = []
    for u in s["units"]:
        s1 = quran.verse(u["lo"])[0]
        units.append({"n": u["n"], "state": state_of(u), "days": u["days"], "last": u["last"] and u["last"].isoformat(),
                      "due": u["due"].isoformat(), "behind": u["behind"] if not u["today_done"] else 0,
                      "weak": u["weak"], "today": u["n"] in today_ns, "done_today": u["today_done"],
                      "weak_back": weak_back(u),
                      "partial": u["partial"] and u["partial"].isoformat(), "ar": quran.surah(s1)["ar"],
                      "start": quran.span_text(u["lo"], u["lo"]).split(" to ")[0],
                      "span": quran.span_text(u["lo"], u["hi"]), "span_bn": quran.span_text(u["lo"], u["hi"], "bn"),
                      "pages": round(quran.pages([(u["lo"], u["hi"])]), 1)})
    r = ramadan()
    week = [e for e in s["logs"] if e["date"] >= (s["today"] - timedelta(days=6)).isoformat()]
    recent = list(reversed([e for e in s["entries"] if e["kind"] == "log"]))[:8]     # newest first, as written
    return {
        "unit": kind, "count": s["count"], "daily": s["daily"], "cycle": cycle, "weak_days": cfg["weak_days"],
        "reciter": cfg["reciter"], "started": s["started"], "today_date": s["today"].isoformat(),
        "today": [{"kind": kind, "n": u["n"], "label": _unit_word(kind, "en", u["n"]),
                   "label_bn": _unit_word(kind, "bn", u["n"]), "span": _span_of(s, u["n"], "en", False),
                   "span_bn": _span_of(s, u["n"], "bn", False), "behind": u["behind"],
                   "pages": round(quran.pages([(u["lo"], u["hi"])]))} for u in s["today_units"]],
        "done_today": [u["n"] for u in s["done_today"]], "tomorrow": [u["n"] for u in s["tomorrow"]],
        "weak_due": [{"label": w["label"], "label_bn": w["label_bn"], "since": w["date"].isoformat(), "due": w["due"].isoformat()}
                     for w in s["weak_due"]],
        "behind": [{"n": u["n"], "days": u["behind"]} for u in s["behind"]],
        "units": units, "fresh": s["fresh"], "streak": s["streak"], "best": s["best"],
        "week": {"pages": round(quran.pages([tuple(x) for e in week for x in e["ranges"]])),
                 "days": len({e["date"] for e in week}), "logs": len(week)},
        "ramadan": r and {"date": r["date"].isoformat(), "days": r["days"],
                          "rounds": round(r["days"] * s["daily"] / s["count"], 1)},
        "recent": [{"id": e["id"], "date": e["date"], "label": _label(e), "label_bn": _label(e, "bn"),
                    "strength": e["strength"], "at": e["at"], "via": e.get("via", "")} for e in recent],
        "line": line_of_the_day("en", s["today"]), "line_bn": line_of_the_day("bn", s["today"]),
        "problems": problems(),
    }
