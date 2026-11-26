"""
Mahadi's Dua & Zikr (2026-11-23): the list he approved, data/duas.yaml (built from his screenshots and checked against
Quran.com, Hisnul Muslim and the six hadith collections; dua_check.py), on the Hifz page's Dua & Zikr tab with a tap
counter for each that starts again at local midnight, and a card a few minutes after each prayer with that prayer's
adhkar, on the page and on Telegram. All of it is code: no model request, and no dua's words from a model.
  - The Arabic is the list's, copied from its source (arabic_from). A Quranic item keeps only its reference: its text
    is Quran.com's (verses/by_key), kept QURAN_KEEP_SEC (six days: Quran Foundation's terms allow a week), with
    Tanzil's text on this Mac (quran_text.py) when Quran.com can't be reached.
  - Nothing is said in Arabic, nor a dua in transliteration: what Javid speaks names the list ("your after-Fajr
    adhkar"), and synthesize strips Arabic script anyway. A Quranic item plays only from a reciter (the page's player).
  - The model's replies are checked (reply_problem): a dua it writes out is rewritten once, then its sentences go.
  - A picture sent with "add this to my evening adhkar" is read on this Mac (Tesseract) and checked as the
    screenshots were (dua_check.check). What would be added is the source's text, never the picture's reading, and it
    waits for Mahadi: an Approve card, and the tab's "Waiting for you" list.
Local and out of git: the counters (data/dua_progress.json), the settings (data/dua_settings.json), what was raised
(data/dua_state.json), the additions waiting (data/dua_pending.json), Quran.com's copies (data/dua_quran/); all 0600.
"""
import json
import os
import re
import secrets
import ssl
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta

import yaml

import prayers
import quran
import quran_text
import timers
import tracing

LIST_FILE = os.path.join("data", "duas.yaml")
SETTINGS_FILE = os.path.join("data", "dua_settings.json")
PROGRESS_FILE = os.path.join("data", "dua_progress.json")
STATE_FILE = os.path.join("data", "dua_state.json")
PENDING_FILE = os.path.join("data", "dua_pending.json")
QURAN_DIR = os.path.join("data", "dua_quran")
INBOX_DIR = os.path.join("data", "dua_inbox")       # Mahadi's screenshots; a picture he adds is kept here too
QURAN_URL = ("https://api.quran.com/api/v4/verses/by_key/{key}?words=true&word_fields=text_uthmani"
             "&fields=text_uthmani")
USER_AGENT = "Javid/1.0 (personal voice assistant)"
QURAN_KEEP_SEC = 6 * 86400      # Quran.com's text is kept six days, then fetched again (its terms: a week at most)
QURAN_RETRY_SEC = 600           # after Quran.com failed, Tanzil's text until this long has passed
LEAD_SEC = 20                   # a card goes into timers.json this early, so the page rings it on its minute
LATE_SEC = 3600                 # and rings up to an hour late (a Mac asleep), never once that prayer's time is over
KEEP_DAYS = 3                   # what was raised, remembered this long
PENDING_DAYS = 14               # an addition nobody approved is dropped after this
RELOAD_SEC = 2                  # the list's file is looked at for changes at most this often
MAX_OFFSET = 120
BACKGROUND = True               # Quran.com is asked in a thread (the suite asks in line, to see every request)

STATUSES = ("verified", "check", "unsourced", "unreadable")
SHOWN = ("verified", "check")                       # unsourced and unreadable ones only when he turns them on
KINDS = ("dua", "quran", "practice")
AFTER = {p: f"after_{p.lower()}" for p in prayers.PRAYERS}
WHEN = ("after_salah", *AFTER.values(), "morning", "evening", "before_sleep", "anytime", "other")
GROUPS = ("after_salah", "morning", "evening", "before_sleep", "anytime", "other")
SLOTS = tuple(p.lower() for p in prayers.PRAYERS) + ("morning", "evening", "before_sleep", "anytime")
NAMES = {"after_salah": ("After salah", "নামাজের পর"), "morning": ("Morning", "সকাল"),
         "evening": ("Evening", "সন্ধ্যা"), "before_sleep": ("Before sleep", "ঘুমানোর আগে"),
         "anytime": ("Anytime", "যেকোনো সময়"), "other": ("Occasions", "বিশেষ সময়ে")}
# The words a card's title and Javid's voice use: "your after-Fajr adhkar", "your morning adhkar".
SAID = {"after_salah": ("after-salah adhkar", "নামাজের পরের জিকির"), "morning": ("morning adhkar", "সকালের জিকির"),
        "evening": ("evening adhkar", "সন্ধ্যার জিকির"), "before_sleep": ("before-sleep adhkar", "ঘুমানোর আগের জিকির"),
        "anytime": ("duas for any time", "যেকোনো সময়ের দোয়া"), "other": ("duas for set occasions", "বিশেষ সময়ের দোয়া")}
DEFAULTS = {"offset_min": 10, "prayers": {p: True for p in prayers.PRAYERS}, "spoken": False,
            "show_unsourced": False, "page": True, "telegram": True, "meaning": "auto"}

_lock = threading.RLock()
_list = {"items": [], "by_id": {}, "problems": [], "stamp": None, "checked": 0.0}
_qfail = {"at": 0.0}
_fetching = threading.Event()
_time = time.time               # the clock (regression.py moves it, with prayers._time and timers._now)


# ---------- files ----------
def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", mode=0o700, exist_ok=True)
    tmp = f"{path}.{secrets.token_hex(3)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


# ---------- the list (data/duas.yaml) ----------
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_KEY = re.compile(r"(\d{1,3}):(\d{1,3})")


def _pos_int(v, hi: int = 100000) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= hi


def _valid_key(k: str) -> bool:
    m = _KEY.fullmatch(k)
    return bool(m) and 1 <= int(m[1]) <= 114 and 1 <= int(m[2]) <= quran.SURAHS[int(m[1]) - 1]["ayahs"]


def _texts(v) -> dict:
    v = v if isinstance(v, dict) else {}
    return {lang: " ".join(str(v[lang]).split()) for lang in ("bn", "en") if v.get(lang)}


def _strs(v) -> list[str]:
    return [" ".join(str(x).split()) for x in (v if isinstance(v, list) else [v] if v else []) if str(x).strip()]


def _item(it) -> tuple[dict | None, str]:
    """One entry, checked and tidied (the Arabic is never changed: only the space around it goes)."""
    if not isinstance(it, dict):
        return None, "not a mapping"
    iid = str(it.get("id") or "")
    if not _ID.fullmatch(iid):
        return None, "its id must be a-z, 0-9 and -"
    title = " ".join(str(it.get("title") or "").split())
    if not title:
        return None, "no title"
    status, kind = it.get("status"), it.get("kind") or "dua"
    if status not in STATUSES:
        return None, f"status must be one of {', '.join(STATUSES)}"
    if kind not in KINDS:
        return None, f"kind must be one of {', '.join(KINDS)}"
    when = it.get("when")
    when = [when] if isinstance(when, str) else when
    if not isinstance(when, list) or not when or any(w not in WHEN for w in when):
        return None, f"when must list some of {', '.join(WHEN)}"
    count = it.get("count")
    if count is not None and not _pos_int(count):
        return None, "count must be a whole number from 1"
    after = it.get("count_after") or {}
    if not isinstance(after, dict) or any(p not in prayers.PRAYERS or not _pos_int(n) for p, n in after.items()):
        return None, "count_after must give prayers (Fajr, Dhuhr, Asr, Maghrib, Isha) a whole number each"
    phases = it.get("phases") or []
    if not isinstance(phases, list) or any(not isinstance(p, dict) or not _pos_int(p.get("count"))
                                           or not str(p.get("label") or "").strip() for p in phases):
        return None, "phases must each have a count and a label"
    arabic = str(it["arabic"]).strip() if it.get("arabic") else None
    if kind == "dua" and status in SHOWN and not arabic:
        return None, "a dua needs its Arabic"
    ref = None
    if kind == "quran":
        q = it.get("quran")
        if not isinstance(q, dict) or not q.get("ayahs"):
            return None, "a Quranic item needs its ayahs"
        ayahs = [str(k) for k in q["ayahs"]]
        if bad := [k for k in ayahs if not _valid_key(k)]:
            return None, f"{bad[0]} isn't an ayah"
        words = {}
        for k, span in (q.get("words") or {}).items():
            k = str(k)
            if k not in ayahs or not isinstance(span, list) or len(span) != 2 or not all(_pos_int(x, 200) for x in span) \
                    or span[0] > span[1]:
                return None, f"the words of {k} must be [first, last] of that ayah"
            words[k] = [int(span[0]), int(span[1])]
        s, a = ayahs[0].split(":")
        ref = {"ayahs": ayahs, "words": words or None, "link": str(q.get("link") or f"https://quran.com/{s}/{a}")}
    try:
        order = int(it["order"]) if it.get("order") is not None else None
    except (TypeError, ValueError):
        order = None
    return {"id": iid, "title": title, "status": status, "kind": kind, "when": list(dict.fromkeys(when)),
            "count": count, "count_after": {p: int(n) for p, n in after.items()},
            "phases": [{"count": int(p["count"]), "label": " ".join(str(p["label"]).split())} for p in phases],
            "arabic": arabic, "arabic_from": str(it.get("arabic_from") or "") or None, "quran": ref,
            "transliteration": _texts(it.get("transliteration")), "meaning": _texts(it.get("meaning")),
            "text_from": {str(k): str(v) for k, v in (it.get("text_from") or {}).items()} if isinstance(it.get("text_from"), dict) else {},
            "reference": _strs(it.get("reference")),
            "links": [x for x in _strs(it.get("links")) if re.match(r"https://[a-z0-9.-]+/", x)],
            "notes": _strs(it.get("notes")), "occasion": " ".join(str(it.get("occasion") or "").split()) or None,
            "claims": [{"says": str(c.get("says") or ""), "verdict": str(c.get("verdict") or "")}
                       for c in (it.get("claims") or []) if isinstance(c, dict)],
            "origin": str(it.get("origin") or "") or None, "order": order,
            "screenshots": [x for x in _strs(it.get("screenshots")) if _PICTURE.fullmatch(x)]}, ""


_PICTURE = re.compile(r"[A-Za-z0-9_.-]{1,120}\.(?:jpe?g|png)")


def _validate(raw) -> tuple[list, list]:
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        return [], ["data/duas.yaml has no list of items"]
    items, problems, seen = [], [], set()
    for n, it in enumerate(raw["items"], 1):
        item, why = _item(it)
        name = it.get("id") if isinstance(it, dict) else None
        if why:
            problems.append(f"item {n}{f' ({name})' if name else ''}: {why}")
        elif item["id"] in seen:
            problems.append(f"item {n}: the id {item['id']} is used twice")
        else:
            seen.add(item["id"])
            items.append(item)
    return items, problems


def _load() -> dict:
    """The list as it is in the file (read again when the file changes); a file that can't be read keeps the last
    good list, with the reason in problems()."""
    with _lock:
        if _list["stamp"] is not None and time.time() - _list["checked"] < RELOAD_SEC:
            return _list
        _list["checked"] = time.time()
        try:
            st = os.stat(LIST_FILE)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = "missing"
        if stamp == _list["stamp"]:
            return _list
        _list["stamp"] = stamp
        if stamp == "missing":
            _list.update(items=[], by_id={}, problems=["There's no data/duas.yaml yet."])
            return _list
        try:
            with open(LIST_FILE, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            _list["problems"] = [f"data/duas.yaml can't be read ({type(e).__name__}); the last good list is used."]
            return _list
        items, problems = _validate(raw)
        _list.update(items=items, by_id={i["id"]: i for i in items}, problems=problems,
                     checked_on=str(raw.get("checked") or "") if isinstance(raw, dict) else "")
        return _list


def reload() -> None:
    with _lock:
        _list.update(stamp=None, checked=0.0)


def items() -> list[dict]:
    return list(_load()["items"])


def get(iid: str) -> dict | None:
    return _load()["by_id"].get(iid)


def problems() -> list[str]:
    return list(_load()["problems"])


def shown(item: dict, s: dict | None = None) -> bool:
    return item["status"] in SHOWN or (s or settings())["show_unsourced"]


# ---------- settings (data/dua_settings.json) ----------
def _check_settings(raw) -> tuple[dict, list[str]]:
    s, bad = json.loads(json.dumps(DEFAULTS)), []
    raw = raw if isinstance(raw, dict) else {}
    if "offset_min" in raw:
        v = raw["offset_min"]
        if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_OFFSET:
            s["offset_min"] = v
        else:
            bad.append(f"offset_min must be 0 to {MAX_OFFSET} minutes")
    if "prayers" in raw:
        v = raw["prayers"]
        if isinstance(v, dict) and all(p in prayers.PRAYERS and isinstance(on, bool) for p, on in v.items()):
            s["prayers"].update(v)
        else:
            bad.append("prayers must say true or false for Fajr, Dhuhr, Asr, Maghrib and Isha")
    for k in ("spoken", "show_unsourced", "page", "telegram"):
        if k in raw:
            if isinstance(raw[k], bool):
                s[k] = raw[k]
            else:
                bad.append(f"{k} must be true or false")
    if "meaning" in raw:
        if raw["meaning"] in ("auto", "bn", "en"):
            s["meaning"] = raw["meaning"]
        else:
            bad.append("meaning must be auto, bn or en")
    return s, bad


def settings() -> dict:
    return _check_settings(_read_json(SETTINGS_FILE, {}))[0]


def save_settings(change: dict) -> tuple[dict, list[str]]:
    """Merge a change into the settings: (the settings now, the problems; nothing is saved when there are any)."""
    if not isinstance(change, dict) or set(change) - set(DEFAULTS):
        return settings(), [f"only {', '.join(DEFAULTS)} can be set"]
    with _lock:
        now = settings()
        merged = {**now, **change, "prayers": {**now["prayers"], **(change.get("prayers") or {})}
                  if isinstance(change.get("prayers"), dict) else change.get("prayers", now["prayers"])}
        s, bad = _check_settings(merged)
        if not bad:
            _write_json(SETTINGS_FILE, s)
        return (now, bad) if bad else (s, [])


# ---------- the counters (data/dua_progress.json): today's only, in the city's time ----------
def _today() -> date:
    return datetime.fromtimestamp(_time(), prayers._zone()).date()


def _progress() -> dict:
    p = _read_json(PROGRESS_FILE, {})
    if p.get("date") != _today().isoformat() or not isinstance(p.get("counts"), dict):
        return {"date": _today().isoformat(), "counts": {}}       # a new day: every counter starts at 0
    return p


def slot_of(group: str, prayer: str | None = None) -> str:
    """Which counter an item uses in a group: after a prayer its own (the after-salah adhkar are said five times a
    day), the morning's, the evening's, before sleep, or the one for any time."""
    if group == "after_salah" or group in AFTER.values():
        return (prayer or next(p for p, g in AFTER.items() if g == group)).lower()
    return group if group in ("morning", "evening", "before_sleep") else "anytime"


def target(item: dict, slot: str) -> int:
    """How many times a counter is complete at: a prayer's own count (the Quls three times after Fajr and Maghrib),
    the phases added up (33 + 33 + 33 + 1), the count, else once."""
    prayer = slot.capitalize() if slot in SLOTS[:5] else None
    if prayer and item["count_after"].get(prayer):
        return item["count_after"][prayer]
    if item["phases"]:
        return sum(p["count"] for p in item["phases"])
    return item["count"] or 1


def counts() -> dict:
    return _progress()["counts"]


def count(slot: str, iid: str, delta: int | None = None, value: int | None = None) -> dict:
    """Tap (delta +1), undo a tap (-1) or set (value, 0 = start again) one counter, kept within 0..its target.
    {slot, id, n, target, done}."""
    item = get(iid)
    if slot not in SLOTS or not item:
        raise KeyError("no such dua or counter")
    goal = target(item, slot)
    with _lock:
        p = _progress()
        n = p["counts"].get(slot, {}).get(iid, 0)
        n = value if value is not None else n + (delta or 0)
        n = max(0, min(goal, int(n)))
        p["counts"].setdefault(slot, {})[iid] = n
        if not n:
            p["counts"][slot].pop(iid, None)
        _write_json(PROGRESS_FILE, p)
    return {"slot": slot, "id": iid, "n": n, "target": goal, "done": n >= goal}


def mark_done(entries: list[dict]) -> int:
    """Every entry ({id, slot}) at its target: Telegram's "All done". How many changed."""
    changed = 0
    with _lock:
        p = _progress()
        for e in entries:
            item = get(e.get("id", ""))
            if not item or e.get("slot") not in SLOTS:
                continue
            goal = target(item, e["slot"])
            if p["counts"].get(e["slot"], {}).get(item["id"], 0) < goal:
                p["counts"].setdefault(e["slot"], {})[item["id"]] = goal
                changed += 1
        if changed:
            _write_json(PROGRESS_FILE, p)
    return changed


# ---------- groups and each prayer's card ----------
def _ordered(found: list[dict]) -> list[dict]:
    return sorted(found, key=lambda i: (i["order"] is None, i["order"] or 0))       # Hisnul Muslim's order first


def group_items(group: str, prayer: str | None = None, s: dict | None = None) -> list[dict]:
    """The items of a group as the tab lists them (after salah: the common ones, then that prayer's own)."""
    s = s or settings()
    every = [i for i in items() if shown(i, s)]
    if group == "after_salah":
        common = _ordered([i for i in every if "after_salah" in i["when"]])
        own = [i for i in every if prayer and AFTER[prayer] in i["when"] and i not in common]
        return common + own
    return _ordered([i for i in every if group in i["when"]])


def card(prayer: str, s: dict | None = None) -> list[dict]:
    """A prayer's card: the after-salah adhkar and that prayer's own (each on that prayer's counter); after Fajr the
    morning adhkar too, after Asr the evening's (on theirs). A dua in both lists (Ayat al-Kursi) is on it once, where
    it comes first. [{id, slot, target, group}]."""
    s = s or settings()
    common = group_items("after_salah", None, s)
    out = [{"item": i, "slot": prayer.lower(), "group": "after_salah" if i in common else AFTER[prayer]}
           for i in group_items("after_salah", prayer, s)]
    extra = {"Fajr": "morning", "Asr": "evening"}.get(prayer)
    if extra:
        out += [{"item": i, "slot": extra, "group": extra} for i in group_items(extra, None, s)
                if all(i is not e["item"] for e in out)]
    return [{"id": e["item"]["id"], "slot": e["slot"], "target": target(e["item"], e["slot"]), "group": e["group"]}
            for e in out]


def _done(entries: list[dict], c: dict | None = None) -> int:
    c = counts() if c is None else c
    return sum(1 for e in entries if c.get(e["slot"], {}).get(e["id"], 0) >= e["target"])


def current_prayer(now: datetime | None = None) -> str | None:
    """The prayer whose time it is (the last one begun today), for the tab's after-salah list; before Fajr, Isha."""
    now = now or datetime.fromtimestamp(_time(), prayers._zone())
    dd = prayers.times_for(now.date(), wait=False)
    if not dd:
        return None
    begun = [p for p in prayers.PRAYERS if prayers._at(dd, p) <= now]
    return begun[-1] if begun else "Isha"


# ---------- Quranic text: Quran.com, kept six days; Tanzil's on this Mac without it ----------
def _get_json(url: str) -> dict:
    """One request (the suite replaces it: nothing in the tests reaches Quran.com)."""
    import certifi
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15, context=ssl.create_default_context(cafile=certifi.where())) as r:
        return json.loads(r.read().decode())


def _qc_path(key: str) -> str:
    return os.path.join(QURAN_DIR, key.replace(":", "_") + ".json")


def _qc_saved(key: str) -> dict | None:
    """Quran.com's copy of an ayah if it's younger than QURAN_KEEP_SEC; an older one is deleted."""
    path = _qc_path(key)
    c = _read_json(path, {})
    if not c:
        return None
    if c.get("key") != key or not c.get("text") or not isinstance(c.get("words"), list) or \
            not 0 <= _time() - float(c.get("fetched") or 0) <= QURAN_KEEP_SEC:
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    return c


def _qc_fetch(key: str) -> dict | None:
    if _time() - _qfail["at"] < QURAN_RETRY_SEC:
        return None
    try:
        v = _get_json(QURAN_URL.format(key=key))["verse"]
        if v.get("verse_key") != key:
            raise ValueError("another ayah came back")
        words = [w["text_uthmani"] for w in v.get("words") or [] if w.get("char_type_name") == "word"]
        text = str(v.get("text_uthmani") or "").strip()
        if not words or not text:
            raise ValueError("no text")
    except Exception as e:
        _qfail["at"] = _time()
        print(f"[duas] Quran.com {key}: {type(e).__name__}; Tanzil's text until it answers")
        return None
    c = {"key": key, "fetched": _time(), "text": text, "words": words}
    _write_json(_qc_path(key), c)
    return c


def quran_keys() -> list[str]:
    return list(dict.fromkeys(k for i in items() if i["quran"] for k in i["quran"]["ayahs"]))


def prefetch(background: bool | None = None) -> None:
    """Fetch the Quranic items' ayahs Quran.com hasn't given in the last six days (server start; a request that found
    one missing). One at a time, once at a time."""
    background = BACKGROUND if background is None else background
    def run():
        try:
            for k in quran_keys():
                if not _qc_saved(k) and not _qc_fetch(k):
                    break               # Quran.com is down: Tanzil's text until QURAN_RETRY_SEC has passed
        finally:
            _fetching.clear()
    if _fetching.is_set():
        return
    _fetching.set()
    if background:
        threading.Thread(target=run, name="dua-quran", daemon=True).start()
    else:
        run()


def _segment(key: str, words: list | None) -> dict | None:
    """One ayah (or the words of it the dua is) as the tab shows it, and where its text came from."""
    s, a = map(int, key.split(":"))
    c = _qc_saved(key)
    if c and (not words or words[1] <= len(c["words"])):
        text = " ".join(c["words"][words[0] - 1:words[1]]) if words else c["text"]
        return {"key": key, "s": s, "a": a, "text": text, "whole": not words, "source": "Quran.com"}
    if not quran_text.ready():
        return None
    if words:
        toks = quran_text.words(s, a)
        first = next((i for i, (_, p) in enumerate(toks) if p == words[0]), None)
        last = next((i for i, (_, p) in enumerate(toks) if p == words[1]), None)
        if first is None or last is None:
            return None
        while last + 1 < len(toks) and not toks[last + 1][1]:       # a pause mark after the last word
            last += 1
        text = " ".join(t for t, _ in toks[first:last + 1])
    else:
        text = quran_text._split(s, a)[0]
    return {"key": key, "s": s, "a": a, "text": text, "whole": not words, "source": "Tanzil"}


def quran_segments(item: dict) -> list[dict]:
    """A Quranic item's text, ayah by ayah: Quran.com's when it's fresh, else Tanzil's; [] when neither is here."""
    if not item.get("quran"):
        return []
    q = item["quran"]
    segs = [_segment(k, (q["words"] or {}).get(k)) for k in q["ayahs"]]
    if any(s and s["source"] != "Quran.com" for s in segs) or None in segs:
        prefetch()
    return [s for s in segs if s]


# ---------- the tab (GET /duas) ----------
def _public(item: dict) -> dict:
    out = {k: item[k] for k in ("id", "title", "status", "kind", "when", "count", "count_after", "phases", "arabic",
                                "arabic_from", "transliteration", "meaning", "text_from", "reference", "links",
                                "notes", "occasion", "claims", "origin")}
    if item["quran"]:
        out["quran"] = {**item["quran"], "segments": quran_segments(item)}
    out["picture"] = item["screenshots"][0] if item["origin"] == "picture" and item["screenshots"] else None
    return out


def tab_json() -> dict:
    s = settings()
    every = items()
    now = datetime.fromtimestamp(_time(), prayers._zone())
    dd = prayers.times_for(now.date(), wait=False)
    times = {}
    if dd:
        cfg = prayers.config()
        for p in prayers.PRAYERS:
            name, at = prayers._prayer_at(dd, p, cfg)
            times[p] = {"name": name, "at": prayers._clock(at), "at_bn": prayers._short(at, "bn")}
    groups = []
    for g in GROUPS:
        if g == "after_salah":
            common = group_items(g, None, s)
            groups.append({"key": g, "en": NAMES[g][0], "bn": NAMES[g][1], "common": [i["id"] for i in common],
                           "own": {p: [i["id"] for i in group_items(g, p, s) if i not in common] for p in prayers.PRAYERS}})
        else:
            groups.append({"key": g, "en": NAMES[g][0], "bn": NAMES[g][1],
                           "items": [i["id"] for i in group_items(g, None, s)]})
    return {"items": {i["id"]: _public(i) for i in every}, "groups": groups,
            "cards": {p: card(p, s) for p in prayers.PRAYERS}, "times": times,
            "date": _today().isoformat(), "counts": counts(), "settings": s, "current": current_prayer(now),
            "hidden": sum(1 for i in every if i["status"] not in SHOWN), "pending": pending_json(),
            "problems": problems(), "checked": _load().get("checked_on") or None}


# ---------- reminders (the watches thread) ----------
def planned(d: date, s: dict | None = None) -> list[dict]:
    """Day d's cards: offset_min after each prayer begins (Jumu'ah on a Friday), until an hour later or the end of
    that prayer's time, whichever is first."""
    s = s or settings()
    if not (s["page"] or s["telegram"]):
        return []
    dd = prayers.times_for(d, wait=False)
    if not dd:
        return []
    cfg, out = prayers.config(), []
    for p in prayers.PRAYERS:
        if not s["prayers"].get(p):
            continue
        name, at = prayers._prayer_at(dd, p, cfg)
        due = at + timedelta(minutes=s["offset_min"])
        until = due + timedelta(seconds=LATE_SEC)
        if p in prayers.WINDOW_END:
            end = prayers._at(dd, prayers.WINDOW_END[p])
            if end - due >= timedelta(minutes=15):
                until = min(until, end)
        out.append({"key": f"{d.isoformat()}:{p}", "prayer": p, "name": name, "at": at, "due": due, "until": until})
    return out


def group_name(group: str, lang: str = "en") -> str:
    """A list's heading: "Evening", "After Fajr only"."""
    if group in NAMES:
        return NAMES[group][1 if lang == "bn" else 0]
    p = group[6:].capitalize()
    return f"শুধু {prayers.OF_BN[p]} পরে" if lang == "bn" else f"After {p} only"


def _card_texts(item: dict) -> tuple[str, dict]:
    p, name, at = item["prayer"], item["name"], item["at"]
    en = f"Time for your after-{prayers._name(name, 'en')} adhkar."
    bn = f"{prayers.OF_BN.get(name, prayers.OF_BN[p])} পরের জিকিরের সময় হয়েছে।"
    return f"After-{prayers._name(name, 'en')} adhkar", {"en": en, "bn": bn}


def tick(now: float | None = None) -> list[dict]:
    """Raise each card that has come due, once (data/dua_state.json), LEAD_SEC early so the page rings it on its
    minute, never after its `until`; none when every dua on it is already done."""
    ts = _time() if now is None else now
    s = settings()
    channels = [c for c, on in (("page", s["page"]), ("telegram", s["telegram"])) if on]
    d = datetime.fromtimestamp(ts, prayers._zone()).date()
    raised = []
    with _lock:
        state = _read_json(STATE_FILE, {})
        state.setdefault("raised", {})
        changed = False
        for day in (d - timedelta(days=1), d):
            for item in planned(day, s):
                due, until = item["due"].timestamp(), item["until"].timestamp()
                if item["key"] in state["raised"] or ts < due - LEAD_SEC or ts >= until:
                    continue
                state["raised"][item["key"]] = ts
                changed = True
                entries = card(item["prayer"], s)
                if not entries or _done(entries) == len(entries):
                    print(f"[duas] {item['key']}: nothing waiting on the card")
                    continue
                label, texts = _card_texts(item)
                raised.append(timers.add_alert(label, texts, kind="dua", due=max(due, ts), until=until,
                                               channels=channels, extra={"dua": {
                                                   "date": day.isoformat(), "prayer": item["prayer"],
                                                   "name": item["name"], "at": prayers._clock(item["at"]),
                                                   "at_bn": prayers._short(item["at"], "bn"),
                                                   "entries": entries, "spoken": s["spoken"]}}))
                print(f"[duas] raised {item['key']} for {datetime.fromtimestamp(max(due, ts)):%H:%M}")
        if changed:
            keep = ts - KEEP_DAYS * 86400
            state["raised"] = {k: v for k, v in state["raised"].items() if v >= keep}
            _write_json(STATE_FILE, state)
    return raised


# ---------- Telegram ----------
def _count_text(item: dict, slot: str | None, lang: str) -> str:
    """How many times, as the source gives it ("" when it gives none: the counter then just marks it said)."""
    if item["phases"]:
        return " + ".join(f"{p['label']} {_num(p['count'], lang)}" for p in item["phases"])
    prayer = slot.capitalize() if slot in SLOTS[:5] else None
    n = item["count_after"].get(prayer) if prayer else None
    n = n or item["count"]
    if not n:
        return ""
    if n == 1:
        return "একবার" if lang == "bn" else "once"
    return f"{_num(n, lang)} বার" if lang == "bn" else f"{n} times"


def telegram_card(extra: dict, lang: str) -> str:
    """The card as Telegram shows it: which prayer's, how many duas, and the button that writes them out (no names
    in transliteration: the Arabic and its meaning come with "Show the text")."""
    d = extra.get("dua") or {}
    name = d.get("name") or d.get("prayer") or ""
    found = [e for e in d.get("entries") or [] if get(e["id"])]
    done = _done(found)
    if lang == "bn":
        head = f"{prayers.OF_BN.get(name, prayers.OF_BN.get(d.get('prayer'), ''))} পরের জিকির ({d.get('at_bn', '')})"
        size = f"কার্ডে {_num(len(found), 'bn')}টি" + (f", {_num(done, 'bn')}টি হয়ে গেছে" if done else "") + "।"
        tap = "আরবি আর অর্থ পড়তে “📖 Show the text” চাপো।"
    else:
        head = f"After-{prayers._name(name, 'en')} adhkar ({prayers._name(name, 'en')} began {d.get('at', '')})"
        size = f"{len(found)} on the card" + (f", {done} already done" if done else "") + "."
        tap = "Tap “📖 Show the text” to read them in Arabic with the meaning."
    return "\n".join((head, size, tap))


def text_lines(item: dict, slot: str | None, lang: str) -> list[str]:
    """An item written out for Telegram or a chat bubble's screen part: its name, its Arabic (the source's, or
    Quran.com's / Tanzil's for an ayah), its count and its reference. Never spoken."""
    times = _count_text(item, slot, lang) if slot else ""
    out = [item["title"] + (f" · {times}" if times else "")]
    if item["quran"]:
        segs = quran_segments(item)
        if segs:
            out.append(" ".join(s["text"] + ("" if not s["whole"] else f" ({s['a']})") for s in segs))
            out.append(f"Quran {', '.join(item['quran']['ayahs'])} · " + ("Quran.com" if all(s["source"] == "Quran.com" for s in segs)
                                                                      else "Tanzil"))
        else:
            out.append(item["quran"]["link"])
    elif item["arabic"]:
        out.append(item["arabic"])
    if item["reference"] and not item["quran"]:
        out.append(" · ".join(item["reference"][:2]))
    return out


def _tg_count(item: dict, slot: str | None, lang: str) -> str:
    """A count for the written-out card, in numbers only ("33 + 33 + 33 + 1", "3 times"): no transliterated words."""
    if item["phases"]:
        return " + ".join(_num(p["count"], lang) for p in item["phases"])
    return _count_text(item, slot, lang)


def _meaning(item: dict, lang: str) -> str:
    """The meaning his list gives, in the reply language (else the other); for a Quranic dua it lacks, Tanzil's
    translation of its whole ayahs on this Mac (Saheeh International / Muhiuddin Khan), named."""
    m = item["meaning"]
    if m.get(lang) or m.get("bn" if lang == "en" else "en"):
        return m.get(lang) or m.get("bn" if lang == "en" else "en")
    q = item["quran"]
    if not q or q["words"]:
        return ""
    quran_text.ensure(fetch=False)          # what's saved on this Mac, never a download from here
    for tl in (lang, "bn" if lang == "en" else "en"):
        if quran_text.ready(tl):
            got = [quran_text._text[tl].get(tuple(map(int, k.split(":")))) for k in q["ayahs"]]
            if all(got):
                name = quran_text.SOURCES[tl]["name_bn" if lang == "bn" else "name"]
                return " ".join(f"{t} ({k.split(':')[1]})" for t, k in zip(got, q["ayahs"])) + f" — {name}, Tanzil"
    return ""


def _ayahs_short(keys: list[str]) -> str:
    """"112, 113, 114" for whole surahs, "2:255", "2:285-286" otherwise."""
    by: dict[int, list[int]] = {}
    for k in keys:
        su, a = map(int, k.split(":"))
        by.setdefault(su, []).append(a)
    out = []
    for su, ayahs in by.items():
        whole = sorted(ayahs) == list(range(1, quran.SURAHS[su - 1]["ayahs"] + 1))
        out.append(str(su) if whole and len(ayahs) > 1 else
                   f"{su}:{min(ayahs)}" + (f"-{max(ayahs)}" if len(ayahs) > 1 else ""))
    return ", ".join(out)


def telegram_block(item: dict, slot: str | None, lang: str) -> list[str]:
    """One dua for "Show the text": its Arabic (the source's, or Quran.com's / Tanzil's for an ayah), its
    meaning and its reference. No transliterated title. Never spoken."""
    out = []
    if item["quran"]:
        segs = quran_segments(item)
        out.append(" ".join(s["text"] + ("" if not s["whole"] else f" ({s['a']})") for s in segs) if segs
                   else item["quran"]["link"])
    elif item["arabic"]:
        out.append(item["arabic"])
    meaning = _meaning(item, lang)
    if meaning:
        out.append(meaning)
    if item["quran"]:
        segs = quran_segments(item)
        out.append(f"Quran {_ayahs_short(item['quran']['ayahs'])}" + (" · Quran.com" if segs and all(
            s["source"] == "Quran.com" for s in segs) else " · Tanzil" if segs else ""))
    elif item["reference"]:
        out.append(" · ".join(item["reference"][:2]))
    return out


def telegram_texts(extra: dict, lang: str, limit: int = 3800) -> list[str]:
    """The card's duas written out (Arabic and meaning), as few Telegram messages as fit."""
    d = extra.get("dua") or {}
    parts, chunk, group = [], "", None
    for n, e in enumerate(d.get("entries") or [], 1):
        item = get(e["id"])
        if not item:
            continue
        head = f"— {group_name(e['group'], lang)} —\n" if e["group"] != group and e["group"] != "after_salah" else ""
        group = e["group"]
        times = _tg_count(item, e["slot"], lang)
        first = f"{_num(n, lang)}." + (f" {times}" if times else "")
        block = head + "\n".join([first, *telegram_block(item, e["slot"], lang)]) + "\n\n"
        if len(chunk) + len(block) > limit and chunk:
            parts.append(chunk.strip())
            chunk = ""
        chunk += block[:limit]
    if chunk.strip():
        parts.append(chunk.strip())
    return parts


def from_alert(alert_id: str) -> dict | None:
    t = timers.get(alert_id)
    return (t or {}).get("extra") if (t or {}).get("kind") == "dua" else None


def done_from_alert(alert_id: str) -> str:
    extra = from_alert(alert_id)
    if not extra:
        return "That card has gone; tick them on the Dua & Zikr tab instead."
    d = extra["dua"]
    if d.get("date") != _today().isoformat():
        return "That card was for another day; today's counters are on the Dua & Zikr tab."
    changed = mark_done(d.get("entries") or [])
    return f"Marked done: {changed} of {len(d.get('entries') or [])}." if changed else "They were already done."


# ---------- questions and requests, answered in code ----------
_W = prayers._alt
DUA_ASK = re.compile(_W(r"du'?aa?'?s?|doa|dowa|adhkar|azkar|athkar|adhkaar|azkaar|zikirs?|zikrs?|dhikrs?|thikr|"
                        r"tasbih|tasbeeh|tasbihs|wazifas?|supplications?|invocations?|dua (?:and|&) zikr",
                        r"দোয়া|দোআ|দুআ|দু'আ|দুয়া|জিকির|যিকির|জিকর|যিকর|আজকার|আযকার|তাসবিহ|তাসবীহ|ওজিফা|অজিফা"), re.I)
_TAB = re.compile(r"(?<![a-z])dua\s*(?:&|and)\s*zikr(?![a-z])", re.I)
_GROUP_WORDS = {
    "morning": _W(r"morning|mornings|sokal|shokal|sakal|subah|dawn", r"সকাল|সকালের|সকালে|ভোরের|ভোরে"),
    "evening": _W(r"evening|evenings|shondha|sondha|shondhar|sondhar|bikel|bikal|afternoon",
                  r"সন্ধ্যা|সন্ধ্যার|সন্ধ্যায়|বিকেল|বিকালের|বিকেলের"),
    "before_sleep": _W(r"(?:before|when going to|for) (?:sleep|sleeping|bed|bedtime|going to (?:bed|sleep))|bedtime|"
                       r"bed time|sleeping|sleep|ghumanor|ghumer", r"ঘুমানোর|ঘুমের|শোয়ার|ঘুমাতে"),
    "after_salah": _W(r"after (?:the |every |each |my |all |a |our )?(?:prayers?|salah|salaah|salat|namaz|namaj|fard|"
                      r"farz|obligatory prayers?|salam)|after-salah|post-?prayer|after prayer",
                      r"নামাজের পর|নামাজ শেষে|সালাম ফিরিয়ে|ফরজ নামাজের পর"),
    "anytime": _W(r"any ?time|all (?:of )?(?:my|the)|everything|whole list|my list|list of", r"সব|যেকোনো"),
}
_GROUPS_RE = {g: re.compile(v, re.I) for g, v in _GROUP_WORDS.items()}
_AFTER_WORD = re.compile(_W(r"after|post", r"পর|পরের|পরে|শেষে"), re.I)
_ADD = re.compile(r"(?<![a-z])(?:add|put|save|include|keep|store)\s+(?:this|it|that|these|this one|that one|"
                  r"the (?:picture|photo|screenshot|image|pic|one)|this (?:picture|photo|screenshot|image|pic|dua))"
                  r"(?![a-z])|যোগ কর|যুক্ত কর|অ্যাড কর|এড কর|রেখে দাও|রাখো", re.I)
_ADD_REF = re.compile(r"(?<![a-z])(?:add|put|include)\s+(?:hisn|quran|qur'?an|sura|surah|ayah|ayat|\d)|"
                      r"(?:হিসনুল|কুরআন|সূরা).{0,40}(?:যোগ কর|যুক্ত কর|অ্যাড কর)", re.I)
_HISN_REF = re.compile(r"(?<![a-z])hisn(?:ul)?[ -]?(?:al[ -])?muslim\s*(?:no\.?|number|#)?\s*([0-9০-৯]{1,3})|"
                       r"হিসনুল মুসলিম\s*(?:#|নং|নম্বর)?\s*([0-9০-৯]{1,3})", re.I)
_PROGRESS = re.compile(r"(?<![a-z])(?:did i|have i|i have|i've)\s+(?:finish|finished|complete|completed|do|done|did|"
                       r"say|said|read|recite|recited)|how many .{0,40}(?:done|left|did i|have i|remaining)|"
                       r"(?:what's|what is|whats) left|still (?:left|waiting)|progress|kotogula|kotogulo|baki|"
                       r"বাকি|শেষ করেছি|হয়েছে কি|কয়টা হয়েছে|কতগুলো", re.I)
_RECITE = re.compile(r"(?<![a-z])(?:recite|say|read)\s+(?:it|them|the|my|this|that)?\s*(?:\w+\s+){0,3}"
                     r"(?:aloud|out loud|to me|for me)|(?<![a-z])recite(?![a-z])|পড়ে শোনাও|শোনাও|তেলাওয়াত", re.I)
_SPECIFIC = re.compile(r"(?<![a-z])(?:du'?aa?|doa|dowa|zikr|dhikr|adhkar|azkar|supplication)s?\s+"
                       r"(?:for|to|of|when|against|before|after|at|on|in|during|while|from)(?![a-z])|"
                       r"(?<![a-z])(?:which|what) (?:du'?aa?|doa|zikr|dhikr|supplication)(?![a-z])|"
                       r"(?<![a-z])(?:the|a) (?:\w+\s+){1,3}(?:du'?aa?|doa)(?![a-z])|"
                       r"দোয়া কী|কোন দোয়া|দোয়াটা|দোয়া কোনটা|এর দোয়া|সময়ের দোয়া|জন্য দোয়া", re.I)
_GENERAL = re.compile(r"(?<![a-z])(?:what is|what's|whats|meaning of|importance of|benefits? of|virtues? of|how (?:to|do i|should i|can i)|"
                      r"why|when (?:is|are|should)|is it|can i|should i|history of|define|explain|tell me about)\s+"
                      r"(?:a |the )?(?:du'?aa?|doa|dhikr|zikr|adhkar|azkar)(?:\s*\?|$)|"
                      r"(?<![a-z])(?:how to|how do i|how should i) (?:make|do|say|read)(?![a-z])|"
                      r"(?<![a-z])(?:importance|benefits?|virtues?|merits?|etiquette|adab|conditions|why|how come)(?![a-z])|"
                      r"(?<![a-z])(?:when|what time|which time) (?:should|do|can|must|is it best|is the best|to)(?![a-z])|"
                      r"(?<![a-z])best time(?![a-z])", re.I)
# Words a dua request is made of besides the group, the prayer and the dua word: anything else left over means the
# message asks for more (or for a dua by its use: then the list is searched for it).
_FILLER = set("""a an the is are was what what's whats which where how many much show open list display see view
bring pull up read give tell me let know please javid hey ok okay my our your of for in at on to from do does did
i i'm im have has had all every each today today's todays now this that these those and also again some any one
ones it them there there's here can could would will you should shall need want like find get look say recite
done finished finish complete completed left remaining still waiting after before when time times prayer prayers
salah salat salaah namaz namaj go to page tab hifz app list section part screen from with make write pray praying
unsourced hidden verified checked sourced ones""".split())
_FILLER_BN = set("""আমার আমাকে আমাদের দেখাও দেখান খোলো খুলে দাও বলো বলুন কী কি কোন কোনটা কোনগুলো পড়ব পড়বো পড়ার
পড়তে এর আর টা টি গুলো গুলি সব আজ আজকের এখন জন্য সময় সময়ের নামাজের নামাজ পর পরে পরের আগে আগের কয়টা
হয়েছে বাকি শেষ করেছি আছে করো দিন লিস্ট তালিকা""".split())
_STOP = _FILLER | {"make", "write", "dua", "duas", "du'a", "du'as", "doa", "adhkar", "azkar", "athkar", "zikr", "zikir", "dhikr",
                   "tasbih", "tasbeeh", "supplication", "supplications", "zikrs", "morning", "evening", "sleep",
                   "sleeping", "bed", "bedtime", "night"}
# What a dua's use is called in his words, mapped onto the list's occasions ("for worry and grief")
_SYNONYMS = {"anxiety": "worry", "anxious": "worry", "stress": "worry", "stressed": "worry", "sad": "grief",
             "sadness": "grief", "depressed": "grief", "depression": "grief", "worried": "worry", "sick": "ill",
             "sickness": "ill", "illness": "ill", "fever": "ill", "loan": "debt", "loans": "debt", "debts": "debt",
             "hard": "difficult", "hardship": "difficult", "tough": "difficult", "problem": "difficult",
             "problems": "difficult", "enemy": "adversary", "enemies": "adversary", "oppressor": "adversary",
             "evil": "protection", "eye": "protection", "kids": "children", "child": "children", "son": "children",
             "daughter": "children", "prostration": "sujud", "sajda": "sujud", "sajdah": "sujud", "sijdah": "sujud",
             "rukoo": "ruku'", "ruku": "ruku'", "bowing": "ruku'", "qadr": "laylat", "friday": "friday",
             "jumuah": "friday", "jumma": "friday", "hurt": "pain", "ache": "pain", "aches": "pain", "painful": "pain",
             "wake": "waking", "woke": "waking", "decision": "istikharah", "choose": "istikharah",
             "choice": "istikharah"}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z']*|[ঀ-৿]+", text.lower().replace("’", "'"))


def _search_keys(item: dict) -> set[str]:
    words = set(_tokens(f"{item['title']} {item['occasion'] or ''} {item['id'].replace('-', ' ')}"))
    return {w.strip("'") for w in words if len(w) > 2} - _STOP


def find(topic_words: list[str], s: dict | None = None) -> list[dict]:
    """The shown items whose title or occasion has every one of the words (each through its synonym)."""
    want = [_SYNONYMS.get(w, w).strip("'") for w in topic_words]
    if not want:
        return []
    out = []
    for i in items():
        if not shown(i, s):
            continue
        keys = _search_keys(i)
        if all(any(k == w or (len(w) >= 4 and k.startswith(w[:max(4, len(w) - 2)])) for k in keys) for w in want):
            out.append(i)
    return out


def request(text: str, docs: list | None = None) -> dict | None:
    """{"action", ...} for a message code answers whole: show a group (after a prayer, morning, evening, before sleep,
    any time), a dua by its use or name, today's progress, or "add this to my evening adhkar" with a picture; and a
    request for a dua that isn't in the list ("missing": never left to the model). None leaves it to the model:
    another request in the message, or a question about duas in general ("what's the virtue of the morning adhkar?",
    whose reply is checked: reply_problem)."""
    t = " ".join((text or "").lower().replace("’", "'").split()).strip(" ?!.।,")
    if not t or len(t.split()) > 18 or not (DUA_ASK.search(t) or _TAB.search(t)):
        return None
    prayer = None
    if (m := prayers._PRAYER.search(t)) and m.lastgroup in prayers.PRAYERS + ("Jumuah",):
        prayer = "Dhuhr" if m.lastgroup == "Jumuah" else m.lastgroup
    groups = [g for g, rx in _GROUPS_RE.items() if rx.search(t)]
    if prayer and not (_AFTER_WORD.search(t) or not groups or groups == ["after_salah"]):
        prayer = None                   # "the evening adhkar before Maghrib": a time, not the after-Maghrib list
    if _ADD.search(t) or _ADD_REF.search(t):
        if prayer:
            when = AFTER[prayer]
        else:
            when = next((g for g in groups if g in ("after_salah", "morning", "evening", "before_sleep")), "anytime")
        return {"action": "add", "when": when, "docs": [d["id"] for d in (docs or [])], "ref": _reference(t)}
    if prayers._ELSE.search(t) or _GENERAL.search(t):
        return None
    rest = t
    for rx in (*_GROUPS_RE.values(), prayers._PRAYER, DUA_ASK, _TAB, _AFTER_WORD):
        rest = rx.sub(" ", rest)
    rest = [w for w in _tokens(rest) if w not in _FILLER and w not in _FILLER_BN
            and not (w[0] >= "\u0980" and len(w) <= 2)]           # a case ending left behind: মাগরিব|ের
    recite = bool(_RECITE.search(t))
    if not rest:
        views = [("after_salah", prayer)] if prayer else []
        covered = {"Fajr": "morning", "Asr": "evening"}.get(prayer)
        views += [(g, None) for g in groups if g not in ("after_salah", "anytime") and g != covered] + \
                 ([("after_salah", None)] if "after_salah" in groups and not prayer else [])
        return {"action": "progress" if _PROGRESS.search(t) else "show", "views": views, "recite": recite}
    found = find(rest)
    if found:
        return {"action": "find", "words": rest, "ids": [i["id"] for i in found], "recite": recite}
    if _SPECIFIC.search(t) or groups or len(rest) <= 3:
        return {"action": "missing", "words": rest, "recite": recite}
    return None


def _reference(t: str) -> dict | None:
    """The source Mahadi named for a dua: {"hisn": n} or {"quran": [surah, first ayah, last ayah]}."""
    if (m := _HISN_REF.search(t)):
        return {"hisn": int((m[1] or m[2]).translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))}
    found, _ = quran.portions(t)
    one = [p for p in found if p.get("kind") == "surah"]
    if len(one) == 1:
        return {"quran": [one[0]["n"], one[0].get("a1"), one[0].get("a2")]}
    return None


def answer(req: dict, lang: str, via: str = "page"):
    """A dua request answered here: a generator (the duas event opens the tab) returning (spoken, screen)."""
    yield ("tool", "duas")
    span = tracing.tool_start("duas")
    try:
        if req["action"] == "add":
            return _add_answer(req, lang, via)      # no events of its own: its Approve cards come through the gate
        return (yield from _answer(req, lang, via))
    finally:
        tracing.tool_end(span)


def _num(n: int, lang: str) -> str:
    return prayers._bn(n) if lang == "bn" else str(n)


def _screen(title: str, entries: list[tuple[dict, str | None]], lang: str, full: bool) -> str:
    """A chat bubble's screen part (never spoken): the duas by name, with their text on Telegram (full)."""
    lines = [f"{title}:"]
    for n, (item, slot) in enumerate(entries, 1):
        if full:
            lines.append(f"{n}. " + "\n".join(text_lines(item, slot, lang)))
        else:
            badge = {"check": " (check)", "unsourced": " (no reliable source)", "unreadable": " (unreadable)"}.get(item["status"], "")
            ref = item["reference"][0] if item["reference"] else ""
            times = _count_text(item, slot, lang) if slot else ""
            lines.append(f"{n}. {item['title']}{badge}" + (f" · {times}" if times else "") + (f" · {ref}" if ref else ""))
    lines.append("" if full else ("আরবি, অর্থ আর কাউন্টার: হিফজ পেজের “দোয়া ও জিকির”।" if lang == "bn"
                                  else "The Arabic, meanings and counters are on the Hifz page's Dua & Zikr tab."))
    return "\n".join(x for x in lines if x)


def _where(when: str, lang: str = "en") -> str:
    """How Javid names a group: "after-Maghrib adhkar", "evening adhkar"."""
    bn = lang == "bn"
    if when.startswith("after_") and when != "after_salah":
        p = when[6:].capitalize()
        return f"{prayers.OF_BN[p]} পরের জিকির" if bn else f"after-{p} adhkar"
    return SAID[when][1 if bn else 0]


def _view(group: str, prayer: str | None, s: dict) -> tuple[str, str, list[dict]]:
    """(how it's said, its screen heading, its entries) of one list: a prayer's card, or a group."""
    bn = s.get("_lang") == "bn"
    if group == "after_salah":
        prayer = prayer or current_prayer() or "Fajr"
        name = "Jumuah" if prayer == "Dhuhr" and _today().weekday() == 4 else prayer
        said = f"{prayers.OF_BN[name]} পরের জিকির" if bn else f"after-{prayers._name(name, 'en')} adhkar"
        head = said if bn else f"After {prayers._name(name, 'en')}"
        return said, head, card(prayer, s)
    entries = [{"id": i["id"], "slot": slot_of(group), "target": target(i, slot_of(group)), "group": group}
               for i in group_items(group, None, s)]
    return SAID[group][1 if bn else 0], NAMES[group][1 if bn else 0], entries


def _answer(req: dict, lang: str, via: str):
    s = {**settings(), "_lang": lang}
    bn = lang == "bn"
    full = via == "telegram"        # Telegram has no tab: the text comes in the screen part
    first = ("আমি দোয়া পড়ে শোনাই না। " if bn else "I don't recite duas aloud. ") if req.get("recite") else ""
    if req["action"] == "missing":
        topic = " ".join(req["words"])
        return first + ((f"“{topic}”-এর দোয়া তোমার দোয়া ও জিকিরের তালিকায় নেই, আর আমি মনে থেকে কোনো দোয়া বলি না। "
                         "দোয়াটার একটা ছবি পাঠিয়ে বলো “এটা আমার সন্ধ্যার জিকিরে যোগ করো”, আমি উৎসের সাথে মিলিয়ে দেখব।") if bn else
                        (f"A dua for “{topic}” isn't in your Dua & Zikr list, and I don't give duas from memory. Send me "
                         "a picture of it with “add this to my evening adhkar” and I'll check it against its source.")), ""
    if req["action"] == "find":
        found = [i for i in (get(x) for x in req["ids"]) if i]
        yield ("duas", {"ids": [i["id"] for i in found]})
        n = _num(len(found), lang)
        return first + (f"তোমার তালিকা থেকে {n}টি দোয়া স্ক্রিনে আছে।" if bn else
                        f"From your list: {n} {'dua' if len(found) == 1 else 'duas'}, on screen."), \
            _screen("তোমার তালিকা থেকে" if bn else "From your list", [(i, None) for i in found], lang, full)
    c = counts()
    if not req["views"]:
        yield ("duas", {"group": "after_salah", "prayer": current_prayer()})
        if req["action"] == "progress":
            done = sum(1 for slot, ids in c.items() for iid, n in ids.items() if get(iid) and n >= target(get(iid), slot))
            return first + (f"আজ তোমার কাউন্টারে {_num(done, lang)}টি দোয়া পূর্ণ হয়েছে।" if bn else
                            f"Today {done} {'dua is' if done == 1 else 'duas are'} complete on your counters."), ""
        n = _num(sum(1 for i in items() if shown(i, s)), lang)
        return first + (f"তোমার দোয়া ও জিকির হিফজ পেজে খোলা হলো: {n}টি।" if bn else
                        f"Your Dua & Zikr list is open on the Hifz page: {n} of them."), ""
    said, screens = [], []
    for k, (group, prayer) in enumerate(req["views"]):
        name, head, entries = _view(group, prayer, s)
        if k == 0:
            yield ("duas", {"group": group, "prayer": (prayer or current_prayer() or "Fajr") if group == "after_salah" else None})
        done, total = _done(entries, c), len(entries)
        if not total:
            said.append(f"{name}: এখনো কিছু নেই" if bn else f"no {name} in your list yet")
            continue
        if req["action"] == "progress":
            left = total - done
            said.append((f"{name}: {_num(total, lang)}টির মধ্যে {_num(done, lang)}টি হয়েছে" +
                         (f", {_num(left, lang)}টি বাকি" if left else ", সব শেষ, মাশাআল্লাহ")) if bn else
                        (f"your {name}: {done} of {total} done today" + (f", {left} left" if left else ", all of them, MashaAllah")))
        else:
            said.append((f"{name}, {_num(total, lang)}টি" + (f" (আজ {_num(done, lang)}টি হয়েছে)" if done else "")) if bn else
                        (f"your {name}, {total}" + (f" ({done} done today)" if done else "")))
        screens.append(_screen(head, [(get(e["id"]), e["slot"]) for e in entries if get(e["id"])], lang, full))
    if req["action"] == "progress":
        spoken = "; ".join(said) + ("।" if bn else ".")
    elif len(said) == 1 and "," in said[0]:
        name, _, rest = said[0].rpartition(", ")
        up = name[:1].upper() + name[1:]
        spoken = (f"তোমার {name} স্ক্রিনে আছে: {rest}।" if bn else f"{up} are on screen: {rest} of them.")
        if not bn and "(" in rest:
            n, _, done = rest.partition(" (")
            spoken = f"{up} are on screen: {n} of them, {done.rstrip(')')}."
    else:
        spoken = ("স্ক্রিনে আছে: " + "; ".join(said) + "।") if bn else ("On screen: " + "; ".join(said) + ".")
    return first + spoken[:1].upper() + spoken[1:], "\n\n".join(screens)


# ---------- adding a dua from a picture: checked like the screenshots, then waiting for Mahadi ----------
def _pending() -> list[dict]:
    keep = _time() - PENDING_DAYS * 86400
    return [p for p in _read_json(PENDING_FILE, []) if isinstance(p, dict) and p.get("created", 0) >= keep]


def _save_pending(found: list[dict]) -> None:
    _write_json(PENDING_FILE, found)


def pending_item(pid: str) -> dict | None:
    return next((p for p in _pending() if p.get("id") == pid), None)


def pending_json() -> list[dict]:
    out = []
    for p in _pending():
        item = p.get("item") or {}
        segs = []
        if item.get("quran"):
            segs = quran_segments({"quran": {"ayahs": item["quran"]["ayahs"], "words": item["quran"].get("words"),
                                             "link": item["quran"].get("link")}})
        out.append({"id": p["id"], "created": p["created"], "when": p["when"], "text": describe_pending(p),
                    "status": item.get("status"), "title": item.get("title"), "arabic": item.get("arabic"),
                    "arabic_from": item.get("arabic_from"), "segments": segs, "reference": item.get("reference") or [],
                    "links": item.get("links") or [], "notes": item.get("notes") or [], "existing": p.get("existing"),
                    "picture": p.get("picture")})
    return out


def describe_pending(p: dict) -> str:
    if p.get("existing"):
        return f"Add “{p['existing']['title']}” to your {_where(p['when'])} too"
    item = p["item"]
    status = {"verified": "matches its source", "check": "close to its source: check it",
              "unsourced": "no reliable source: it stays hidden", "unreadable": "the Arabic couldn't be read"}[item["status"]]
    return f"Add to your {_where(p['when'])}: {item['title']} ({status})"


def _slug(base: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40] or "dua"
    have = {i["id"] for i in items()} | {(p.get("item") or {}).get("id") for p in _pending()}
    n, out = 1, f"added-{stem}"
    while out in have:
        n += 1
        out = f"added-{stem}-{n}"
    return out


def _same_source(found: dict) -> dict | None:
    """The list's item a finding already is (the same ayahs, the same Hisnul Muslim entry or hadith, or its Arabic)."""
    import dua_check
    for i in items():
        mine = {"kind": i["kind"], "quran": i["quran"], "arabic_from": i["arabic_from"] if i["kind"] == "dua" else None}
        if i["kind"] in ("dua", "quran") and (dua_check.same(found, mine) or
                                              (found.get("arabic") and found["arabic"] == i["arabic"])):
            return i
    return None


def _picture_name(doc: dict) -> str:
    ext = ".png" if (doc.get("ext") or "").lower() == ".png" else ".jpg"
    return f"Picture_{datetime.fromtimestamp(_time()):%Y%m%d_%H%M%S}_{doc['id'][:6]}{ext}"


def _keep_picture(name: str, data: bytes) -> None:
    """The picture an addition came from, beside his screenshots (data/dua_inbox), owner-only."""
    os.makedirs(INBOX_DIR, mode=0o700, exist_ok=True)
    fd = os.open(os.path.join(INBOX_DIR, name), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


MAX_FOUND = 5      # duas from one picture, each its own approval


def _entry(found: dict, when: str, picture: str | None) -> dict | str:
    """An addition waiting for approval, or the title of the list's item already in that place."""
    existing = None if found["status"] in ("unsourced", "unreadable") else _same_source(found)
    if existing and when in existing["when"]:
        return existing["title"]
    entry = {"id": secrets.token_hex(5), "created": _time(), "when": when, "picture": picture,
             "existing": {"id": existing["id"], "title": existing["title"]} if existing else None}
    waiting = next((p for p in _pending() if p["when"] == when and _key(p) == _key(entry, found)), None)
    if waiting:
        return {**waiting, "again": True}       # the same addition is already waiting: not a second card
    if existing:
        return entry
    day = datetime.fromtimestamp(_time()).strftime("%Y-%m-%d")
    if found["status"] in ("unsourced", "unreadable"):
        title = f"Dua from your picture ({day})"
    elif found["kind"] == "quran":
        title = next((r for r in found["reference"] if r.startswith("Quran")), found["reference"][0])
    elif found.get("translit"):           # Hisnul Muslim's transliteration, its first phrase, and the number: it can
        m = re.match(r"\(*([^()]+)", found["translit"].strip())     # open with a narrator's name ("'Aa.ishah")
        words = (m[1] if m else found["translit"]).split()
        title = (" ".join(words[:8]) + ("…" if len(words) > 8 else "")).rstrip(".,;: ") + f" ({found['arabic_from']})"
    else:
        title = f"Dua from {found['arabic_from']}"
    item = {"id": _slug(title), "title": title, "status": found["status"], "kind": found["kind"], "when": [when],
            "origin": "picture" if picture else "reference", "screenshots": [picture] if picture else [],
            "reference": found["reference"], "links": found["links"],
            "notes": found["notes"] + [f"Added {'from a picture' if picture else 'by its reference'} on {day}."]}
    if found.get("repeat") and found["repeat"] > 1:
        item["count"] = found["repeat"]
    if found["kind"] == "quran":
        item.update(arabic_from="Quran.com", quran={"ayahs": found["quran"]["ayahs"], "words": found["quran"]["words"],
                                                    "link": found["links"][0] if found["links"] else None})
    elif found.get("arabic"):
        item.update(arabic=found["arabic"], arabic_from=found["arabic_from"])
    if found.get("translit") or found.get("meaning_en"):
        item["transliteration"] = {"en": found["translit"]} if found.get("translit") else {}
        item["meaning"] = {"en": found["meaning_en"]} if found.get("meaning_en") else {}
        item["text_from"] = {k: f"{found['arabic_from']} (English edition)" for k, v in
                             (("transliteration_en", found.get("translit")), ("meaning_en", found.get("meaning_en"))) if v}
    if found["kind"] == "dua" and found["status"] in SHOWN and not item.get("arabic"):
        item["status"] = "check"
    entry["item"] = item
    return entry


def _key(entry: dict, found: dict | None = None) -> tuple:
    """What an addition adds, to tell the same one asked for twice."""
    if entry.get("existing"):
        return ("item", entry["existing"]["id"])
    f = found or entry.get("item") or {}
    if f.get("status") in ("unsourced", "unreadable"):
        return ("picture", entry.get("picture"))
    return (f.get("kind"), f.get("arabic_from"), tuple((f.get("quran") or {}).get("ayahs") or ()))


def _add_answer(req: dict, lang: str, via: str):
    """"Add this to my evening adhkar" with a picture (and/or the source's reference): each dua found becomes an
    addition waiting for Mahadi's approval, with the source's text; nothing is added here."""
    import documents
    import dua_check
    import javid
    bn = lang == "bn"
    docs = [d for d in (documents.load(i) for i in req.get("docs") or []) if d]
    pictures = [d for d in docs if d.get("kind") == "image"]
    ref = req.get("ref")
    if not pictures and not ref:
        return (("দোয়াটার ছবি বা স্ক্রিনশট এই মেসেজের সাথে পাঠাও, অথবা তার রেফারেন্স বলো (যেমন “হিসনুল মুসলিম ৯১”)।" if bn else
                 "Send the dua's picture or screenshot with that message (the 📎 button), or tell me its reference, "
                 "like “Hisnul Muslim 91” or “Quran 2:201”.") if not docs else
                ("আমি শুধু ছবি বা স্ক্রিনশট থেকে দোয়া মিলিয়ে দেখতে পারি।" if bn else
                 "I can only check a dua from a picture or a screenshot, or by its reference."), "")
    picture, findings, reading, data = None, [], "", b""
    if pictures:
        doc = pictures[0]
        with open(documents._path(doc["id"], doc.get("ext", "")), "rb") as f:
            data = f.read()
        reading = dua_check.ocr(data)
        findings = dua_check.find_all(reading)
        picture = _picture_name(doc)
    if ref:
        named = dua_check.from_reference(ref)
        if not named:
            return (("এই রেফারেন্সটা আমার কাছে নেই: হিসনুল মুসলিম ১ থেকে ২৬৭, কুরআন সূরা:আয়াত।" if bn else
                     "I don't have that reference: Hisnul Muslim runs from 1 to 267, and the Quran's as surah:ayah "
                     "(at most 30 ayahs)."), "")
        if pictures:
            named["notes"].append("Your picture matches it." if any(dua_check.same(named, x) for x in findings) else
                                  "Your picture couldn't be matched to it by its reading: compare the two before approving.")
        findings = [named]
    elif not findings:
        findings = [dua_check.nothing_found(reading)]
    already, entries = [], []
    for found in findings[:MAX_FOUND]:
        e = _entry(found, req["when"], picture)
        (already if isinstance(e, str) else entries).append(e)
    fresh = [e for e in entries if not e.get("again")]
    if fresh:
        if picture and any(not e.get("existing") for e in fresh):
            _keep_picture(picture, data)
        else:                       # one already in the list keeps its own screenshots: nothing new to keep
            for e in fresh:
                e["picture"] = None
        with _lock:
            _save_pending(_pending() + fresh)
    for e in entries:                           # a card for each (again for one already waiting: its own card expired)
        javid.gate_write("dua_add", {"pending": e["id"]})
    where = _where(req["when"], lang)
    if not entries:
        return ((f"এটা তোমার {where}-এ আগেই আছে: {already[0]}।" if bn else f"It's in your {where} already: {already[0]}."), "")
    # What's said names the source, never the dua: its title is a transliteration, and Javid doesn't voice a dua.
    sure = [e for e in entries if (e.get("item") or {}).get("status") in ("verified", "check") or e.get("existing")]
    if len(entries) > 1:
        heard = (f"ছবিতে {_num(len(entries), lang)}টি দোয়া পেয়েছি।" if bn else f"I found {len(entries)} duas in the picture.")
    elif sure:
        e = entries[0]
        if e.get("existing"):
            mine = get(e["existing"]["id"])
            src = (mine["reference"] or [mine["arabic_from"] or ""])[0] if mine else ""
            heard = (f"এটা তোমার তালিকায় আছে ({src}), এবার {where} তালিকাতেও যাবে।" if bn else
                     f"It's in your list already ({src}); it would go in your {where} too.")
        else:
            src = (e["item"]["reference"] or [e["item"].get("arabic_from")])[0]
            close = e["item"]["status"] == "check"
            if ref:
                heard = f"{src} তৈরি, যাবে তোমার {where} তালিকায়।" if bn else f"{src} is ready for your {where}."
            else:
                heard = ((f"এটা পেয়েছি: {src}" + (", কয়েকটা অক্ষর আলাদা, উৎসের লেখাটাই রাখা হবে।" if close else "।")) if bn else
                         (f"I found it: {src}" + (", a few letters apart from the picture; the source's wording is kept."
                                                  if close else ".")))
    else:
        status = entries[0]["item"]["status"]
        heard = (("ছবির আরবি পড়তে পারিনি।" if status == "unreadable" else
                  "কুরআন, হিসনুল মুসলিম বা ছয়টি হাদিস গ্রন্থে এটা পাইনি, তাই যোগ করলে লুকানো থাকবে।") if bn else
                 ("I couldn't read the Arabic in the picture clearly." if status == "unreadable" else
                  "I couldn't match it to the Quran, Hisnul Muslim or the six hadith books, so it would stay hidden.") +
                 (" If you know its reference, say “add it to my evening adhkar, Hisnul Muslim 91”." if not bn else
                  " রেফারেন্স জানলে বলো, যেমন “হিসনুল মুসলিম ৯১”।"))
    wait = ("অনুমোদনের অপেক্ষায় আছে: Approve চাপো, অথবা হিফজ পেজের দোয়া ও জিকিরে।" if bn else
            "Nothing is added until you approve: the Approve card, or the Dua & Zikr tab on the Hifz page.")
    lines = []
    for e in entries:
        lines.append(describe_pending(e) + ".")
        lines += [f"  {n}" for n in ((e.get("item") or {}).get("notes") or [])[:3]]
    if already:
        lines.append(("Already in that place: " if not bn else "আগে থেকেই আছে: ") + "; ".join(already))
    return f"{heard} {wait}", "\n".join(lines)


def approve(pid: str) -> str:
    """Put a waiting addition into data/duas.yaml (a copy of the file before is kept as duas.yaml.bak). The outcome."""
    with _lock:
        entry = pending_item(pid)
        if not entry:
            return "That addition isn't waiting any more, so nothing was done."
        with open(LIST_FILE, encoding="utf-8") as f:
            before = f.read()
        after = _with_when(before, entry["existing"]["id"], entry["when"]) if entry.get("existing") else \
            _with_item(before, entry["item"])
        if after is None:
            return "Your list couldn't be changed safely (data/duas.yaml has changed shape), so nothing was done."
        items_after, bad = _validate(yaml.safe_load(after))
        want = entry["existing"]["id"] if entry.get("existing") else entry["item"]["id"]
        hit = next((i for i in items_after if i["id"] == want), None)
        if not hit or entry["when"] not in hit["when"] or len(items_after) < len(items()):
            return "That addition didn't check out against your list, so nothing was done."
        _write_text(LIST_FILE + ".bak", before)
        _write_text(LIST_FILE, after)
        _save_pending([p for p in _pending() if p["id"] != pid])
        reload()
    title = entry["existing"]["title"] if entry.get("existing") else entry["item"]["title"]
    return f"Added to your {_where(entry['when'])}: {title}."


def reject(pid: str) -> bool:
    with _lock:
        entry = pending_item(pid)
        if not entry:
            return False
        left = [p for p in _pending() if p["id"] != pid]
        _save_pending(left)
        name = entry.get("picture")
        if name and not entry.get("existing") and not any(p.get("picture") == name for p in left) and \
                not any(name in i["screenshots"] for i in items()):
            try:
                os.remove(os.path.join(INBOX_DIR, name))
            except OSError:
                pass
    return True


def _write_text(path: str, text: str) -> None:
    tmp = f"{path}.{secrets.token_hex(3)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _with_item(text: str, item: dict) -> str | None:
    """The file with one more item at the end of items:, written as the others are; comments and all else kept."""
    block = yaml.dump([item], allow_unicode=True, sort_keys=False, width=110, default_flow_style=False)
    lines = text.split("\n")
    at = next((n for n, line in enumerate(lines) if re.match(r"not_duas\s*:", line)), None)
    if at is None:
        at = len(lines)
        while at and not lines[at - 1].strip():
            at -= 1
    if not any(re.match(r"items\s*:", line) for line in lines[:at]):
        return None
    return "\n".join(lines[:at] + block.rstrip("\n").split("\n") + lines[at:]) + ("" if text.endswith("\n") else "")


def _with_when(text: str, iid: str, when: str) -> str | None:
    """The file with `when` added to an item's when: list, in place."""
    lines = text.split("\n")
    start = next((n for n, line in enumerate(lines) if re.fullmatch(rf"- id: ['\"]?{re.escape(iid)}['\"]?\s*", line)), None)
    if start is None:
        return None
    end = next((n for n in range(start + 1, len(lines)) if lines[n].startswith("- ") or re.match(r"\S", lines[n])),
               len(lines))
    at = next((n for n in range(start, end) if re.fullmatch(r"  when:\s*", lines[n])), None)
    if at is None:
        return None
    last = at
    while last + 1 < end and re.match(r"  - \S", lines[last + 1]):
        last += 1
    return "\n".join(lines[:last + 1] + [f"  - {when}"] + lines[last + 1:])


# ---------- the model's replies: no dua written out ----------
_FORMULA = re.compile(r"(?<![a-z])(?:all[aā]h?umm?a|rabban[aā]|subh[aā]n(?:a|al)?[- ]?(?:all[aā]h|rabb)|subhanallah|"
                      r"al-?hamdu ?li-?ll[aā]h|alhamdulillah|astaghfiru-?ll[aā]h|astaghfirullah|l[aā] il[aā]ha ill|"
                      r"l[aā] hawla wa l[aā] quwwata|hasb(?:un|iy|i)-?all[aā]hu?|bismill[aā]h|bismika|"
                      r"a['’`]?[uū]dhu bi|y[aā] hayyu|inn[aā] li-?ll[aā]hi|wa bi-?hamdihi|tawakkaltu|"
                      r"rabbi-? ?(?:ighfir|zidni|yassir|ishrah|hab|inni|awzi'?ni|najjini)|ya muqallib)", re.I)
_EN_DUA = re.compile(r"(?:^|[\"“:]\s*)O Allah[,!]", re.I)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+|\n+", text or "") if s.strip()]


def _bad(sentence: str, asked: bool) -> bool:
    if quran._ARABIC.search(sentence):
        return asked
    hits = len(_FORMULA.findall(sentence))
    return bool(hits and (asked or hits >= 2)) or bool(asked and _EN_DUA.search(sentence))


def reply_problem(text: str, user_text: str = "") -> list[str]:
    """What breaks the dua rule in a model's reply: a dua or dhikr written out (Arabic, in transliteration, or an
    English "O Allah, …" prayer) when the message was about duas, or two formulas in a sentence anywhere."""
    asked = bool(DUA_ASK.search(user_text or ""))
    return ["a dua written out"] if any(_bad(s, asked) for s in _sentences(text)) else []


def correction(problems: list[str]) -> str:
    return ("Rewrite your reply without the words of any dua or dhikr: no Arabic, no transliteration, no translation "
            "of one. Javid never gives a dua from memory. Mahadi's own checked list is on the Hifz page's Dua & Zikr "
            "tab: say he can ask 'show my evening adhkar', or send a picture of a dua with 'add this to my evening "
            "adhkar' to have it checked and added.")


def strip_reply(text: str, user_text: str = "", lang: str = "en") -> str:
    asked = bool(DUA_ASK.search(user_text or ""))
    kept = [s for s in _sentences(text) if not _bad(s, asked)]
    if kept:
        return " ".join(kept)
    return ("আমি মনে থেকে কোনো দোয়া বলি না। তোমার যাচাই করা দোয়া ও জিকির হিফজ পেজে আছে: বলো “সন্ধ্যার জিকির দেখাও”।"
            if lang == "bn" else "I don't give duas from memory. Your checked Dua & Zikr list is on the Hifz page: "
                                 "say “show my evening adhkar”.")


# ---------- python duas.py --check: the list as Javid reads it ----------
def check_report() -> tuple[list[str], list[str]]:
    """What --check says of data/duas.yaml: (lines, problems). Every item needs its timing and status, every dua its
    Arabic, every Quranic one ayahs whose text is here (Quran.com's, fetched now if it's missing, or Tanzil's)."""
    from collections import Counter
    reload()
    quran_text.ensure(fetch=False)
    prefetch(background=False)
    every = items()
    found = list(problems())
    lines = [f"{len(every)} items: " + ", ".join(f"{n} {st}" for st, n in Counter(i["status"] for i in every).items())]
    s = {**settings(), "show_unsourced": False}
    for g in GROUPS:
        lines.append(f"  {NAMES[g][0]}: {len(group_items(g, None, s))} shown")
    for p in prayers.PRAYERS:
        lines.append(f"  {p}'s card: {len(card(p, s))}")
    for i in every:
        if i["kind"] == "quran":
            segs = quran_segments(i)
            if len(segs) != len(i["quran"]["ayahs"]):
                found.append(f"{i['id']}: the text of {', '.join(i['quran']['ayahs'])} isn't here (Quran.com or Tanzil)")
            elif any(x["source"] != "Quran.com" for x in segs):
                lines.append(f"  {i['id']}: Tanzil's text (Quran.com didn't answer)")
        elif i["kind"] == "dua" and i["status"] in SHOWN and not i["arabic"]:
            found.append(f"{i['id']}: a dua with no Arabic")
    return lines, found


if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--check"]:
        sys.exit("usage: python duas.py --check")
    out, bad = check_report()
    print("\n".join(out))
    print("\n".join(f"PROBLEM {b}" for b in bad) or "Every item has its timing and status, every dua its Arabic.")
    sys.exit(1 if bad else 0)
