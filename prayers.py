"""
Prayer times, prayer and zikir reminders, the Hijri date and the morning brief (2026-11-22).

The times come from Aladhan (api.aladhan.com, no key) for the city, calculation method and Asr school in
config/prayers.yaml: a month a request, kept in data/prayer_times.json, so a question ("when's Maghrib?") is answered
in code from the file, with no model request and no wait on the network. The server fetches the month at start; a
question that still finds nothing saved waits ANSWER_SEC at most, for that one day.

The watches thread calls tick() every TICK_SEC. It raises each reminder through timers.add_alert, so the page (chime,
banner, the announcement in Javid's voice), the CLI and Telegram deliver it as they deliver a watch's alert: the
heads-up before each prayer (everywhere); the message when a prayer begins, the morning brief and the zikir reminders
(Telegram only: `channels`). None is delivered once its time is over (`until`), and what was raised is kept in
data/prayers_state.json, so a restart doesn't raise it again.

Privacy: Aladhan gets the city, country, method, school and offsets, nothing else. The brief carries the calendar and
the tasks: it's built here (no model sees it) and goes only to the paired Telegram chat.
"""
import json
import math
import os
import re
import shutil
import ssl
import string
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import certifi   # python.org macOS builds have no CA bundle for urllib (see rates.py)
import yaml

import calendar_tools
import timers
import tracing
import weather

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "prayers.yaml")   # beside the code
CACHE_FILE = os.path.join("data", "prayer_times.json")
STATE_FILE = os.path.join("data", "prayers_state.json")
API = "https://api.aladhan.com/v1"
USER_AGENT = "Javid/1.0 (personal voice assistant)"
FETCH_SEC = 10             # a month (1.1-1.2 s measured 2026-11-22), in the background
ANSWER_SEC = 1.2           # a question with nothing saved waits this long for its one day (0.7-0.8 s measured)
RETRY_SEC = 300            # a month that failed is asked for again this much later
RELOAD_SEC = 5             # the settings file is looked at for changes at most this often
LEAD_SEC = 20              # a reminder goes into timers.json this long before it's due: the page syncs every 10 s
START_LATE_SEC = 15 * 60   # "it's Maghrib time" still goes out this late (the Mac just woke up), not later
REMINDER_LATE_SEC = 30 * 60
KEEP_DAYS = 3              # what was raised is remembered this long
BACKGROUND = True          # months are fetched on a thread (regression.py: in line, so a case sees every request)

PRAYERS = ("Fajr", "Dhuhr", "Asr", "Maghrib", "Isha")
SHOWN = ("Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha")
ANCHORS = SHOWN + ("Imsak", "Sunset", "Midnight", "Firstthird", "Lastthird")
NIGHT = ("Midnight", "Firstthird", "Lastthird")   # the night after the day: one before noon is on the next date
TUNE_ORDER = ("Imsak", "Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Sunset", "Isha", "Midnight")   # Aladhan's tune
METHODS = frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23})
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WINDOW_END = {"Fajr": "Sunrise", "Dhuhr": "Asr", "Asr": "Maghrib", "Maghrib": "Isha"}   # Isha's: the next Fajr

NAME_EN = {"Jumuah": "Jumu'ah", "Lastthird": "the night's last third", "Firstthird": "the night's first third",
           "Midnight": "Islamic midnight", "Sunrise": "sunrise", "Sunset": "sunset"}
NAME_BN = {"Fajr": "ফজর", "Sunrise": "সূর্যোদয়", "Dhuhr": "যোহর", "Asr": "আসর", "Maghrib": "মাগরিব", "Isha": "এশা",
           "Jumuah": "জুমা", "Imsak": "ইমসাক", "Sunset": "সূর্যাস্ত", "Midnight": "মধ্যরাত",
           "Firstthird": "রাতের প্রথম তৃতীয়াংশ", "Lastthird": "রাতের শেষ তৃতীয়াংশ"}
OF_BN = {"Fajr": "ফজরের", "Dhuhr": "যোহরের", "Asr": "আসরের", "Maghrib": "মাগরিবের", "Isha": "এশার", "Jumuah": "জুমার"}
HIJRI_EN = ("Muharram", "Safar", "Rabi' al-Awwal", "Rabi' al-Thani", "Jumada al-Ula", "Jumada al-Akhirah", "Rajab",
            "Sha'ban", "Ramadan", "Shawwal", "Dhu al-Qi'dah", "Dhu al-Hijjah")
HIJRI_BN = ("মহররম", "সফর", "রবিউল আউয়াল", "রবিউস সানি", "জমাদিউল আউয়াল", "জমাদিউস সানি", "রজব", "শাবান",
            "রমজান", "শাওয়াল", "জিলকদ", "জিলহজ")
DAYS_BN = ("সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার")
MONTHS_BN = ("জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর",
             "নভেম্বর", "ডিসেম্বর")
CITY_BN = {"dhaka": "ঢাকা", "chittagong": "চট্টগ্রাম", "chattogram": "চট্টগ্রাম", "sylhet": "সিলেট", "khulna": "খুলনা",
           "rajshahi": "রাজশাহী", "barisal": "বরিশাল", "barishal": "বরিশাল", "rangpur": "রংপুর",
           "mymensingh": "ময়মনসিংহ"}
_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

_time = time.time          # the clock (regression.py moves it, with timers._now)


# ---------- settings ----------
_TEXTS = {   # the shipped file's texts, for a file that leaves one out
    "tasbih": {"en": "After the prayer: SubhanAllah 33, Alhamdulillah 33, Allahu Akbar 34.",
               "bn": "নামাজের পর: সুবহানাল্লাহ ৩৩ বার, আলহামদুলিল্লাহ ৩৩ বার, আল্লাহু আকবার ৩৪ বার।"},
    "morning_adhkar": {"en": "Morning adhkar before sunrise ({sunrise}): Ayat al-Kursi; Al-Ikhlas, Al-Falaq and An-Nas "
                             "three times; SubhanAllahi wa bihamdihi 100 times.",
                       "bn": "সকালের জিকির, সূর্যোদয়ের ({sunrise}) আগে: আয়াতুল কুরসি; সূরা ইখলাস, ফালাক ও নাস তিনবার "
                             "করে; সুবহানাল্লাহি ওয়া বিহামদিহি ১০০ বার।"},
    "friday": {"en": "It's Jumu'ah ({jumuah}): Surah Al-Kahf, ghusl, and more salawat on the Prophet ﷺ.",
               "bn": "আজ জুমা ({jumuah}): সূরা কাহফ, গোসল, আর নবীজির (সা.) ওপর বেশি বেশি দরুদ।"},
}


def _defaults() -> dict:
    return {"city": "Dhaka", "country": "Bangladesh", "method": 1, "school": 1, "tune": {}, "hijri_adjust": 0,
            "heads_up": {"minutes": 10, "prayers": list(PRAYERS)},
            "at_start": {"telegram": True, "tasbih": True, "prayers": list(PRAYERS)},
            "jumuah": "", "brief": {"telegram": True, "after_fajr_minutes": 20}, "keep_awake": True,
            "reminders": [], **{k: dict(v) for k, v in _TEXTS.items()}}


_PLACE = re.compile(r"[^\W\d_][^\W_]*(?:[ .'-]+[^\W_]+)*\.?")      # Dhaka, New York, Cox's Bazar, St. Louis
_HHMM = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
_FIELDS = {n.lower() + when for n in ANCHORS + ("Jumuah",) for when in ("", "_tomorrow")}


def _place(v):
    if not isinstance(v, str) or len(v) > 60 or not _PLACE.fullmatch(v.strip()):
        raise ValueError("a place name, like Dhaka")
    return v.strip()


def _number(lo: int, hi: int):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f"a whole number from {lo} to {hi}")
        return v
    return check


def _flag(v):
    if not isinstance(v, bool):
        raise ValueError("true or false")
    return v


def _one_of(name, allowed: tuple) -> str:
    for a in allowed:
        if str(name).strip().lower() == a.lower():
            return a
    raise ValueError(f"{name!r} isn't one of {', '.join(allowed)}")


def _names(allowed: tuple):
    def check(v):
        if not isinstance(v, list):
            raise ValueError(f"a list, like [{', '.join(allowed[:2])}]")
        return list(dict.fromkeys(_one_of(n, allowed) for n in v))
    return check


def _clock_setting(v, empty_ok: bool = True) -> str:
    if empty_ok and v in ("", None):
        return ""
    m = _HHMM.fullmatch(str(v).strip()) if isinstance(v, str) else None
    if not m:
        raise ValueError('a time like "13:15"' + (', or ""' if empty_ok else ""))
    return f"{int(m[1]):02d}:{m[2]}"


def _method(v) -> int:
    if _number(0, 99)(v) not in METHODS:
        raise ValueError(f"{v} isn't one of Aladhan's methods ({', '.join(map(str, sorted(METHODS)))})")
    return v


def _text(v) -> dict:
    if isinstance(v, str):
        v = {"en": v}
    if not isinstance(v, dict) or set(v) - {"en", "bn"} or not isinstance(v.get("en"), str) or \
            not isinstance(v.get("bn", ""), str) or len(v["en"]) > 500 or len(v.get("bn", "")) > 500:
        raise ValueError("a text in English (en:) and, if you like, Bangla (bn:), each up to 500 letters")
    out = {"en": v["en"].strip(), "bn": (v.get("bn") or v["en"]).strip()}
    for t in out.values():
        try:
            unknown = sorted({f for _, f, _, _ in string.Formatter().parse(t) if f} - _FIELDS)
        except ValueError:
            raise ValueError(f"a stray brace in {t[:40]!r}") from None
        if unknown:
            raise ValueError(f"{', '.join('{' + u + '}' for u in unknown)} isn't a time Javid knows")
    return out


def _section(spec: dict, defaults: dict):
    def check(v):
        if not isinstance(v, dict):
            raise ValueError(f"a section with {', '.join(spec)}")
        if unknown := sorted(set(v) - set(spec)):
            raise ValueError(f"unknown {', '.join(unknown)}")
        return {**defaults, **{k: spec[k](v[k]) for k in v}}
    return check


def _tune(v) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError("minutes per time, like {Fajr: 2}")
    return {_one_of(k, TUNE_ORDER): _number(-60, 60)(m) for k, m in v.items()}


def _reminders(v) -> list:
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > 20:
        raise ValueError("a list of at most 20 reminders")
    out, seen = [], set()
    for i, r in enumerate(v, 1):
        if not isinstance(r, dict) or not isinstance(r.get("name"), str) or not 1 <= len(r["name"].strip()) <= 40:
            raise ValueError(f"reminder {i}: each needs a name of up to 40 letters")
        name = " ".join(r["name"].split())
        try:
            if unknown := sorted(set(r) - {"name", "enabled", "days", "after", "before", "at", "minutes", "text",
                                           "page", "telegram"}):
                raise ValueError(f"unknown {', '.join(unknown)}")
            if name.lower() in seen:
                raise ValueError("two reminders have this name")
            when = [k for k in ("after", "before", "at") if k in r]
            if len(when) != 1:
                raise ValueError("give one of after:, before: or at:")
            item = {"name": name, "enabled": _flag(r.get("enabled", True)), "page": _flag(r.get("page", False)),
                    "telegram": _flag(r.get("telegram", True)), "text": _text(r.get("text"))}
            if not (item["page"] or item["telegram"]):
                raise ValueError("page: or telegram: has to be true")
            if when[0] == "at":
                if "minutes" in r:
                    raise ValueError("minutes: goes with after: or before:")
                item["at"] = _clock_setting(r["at"], empty_ok=False)
            else:
                item[when[0]] = _one_of(r[when[0]], ANCHORS)
                item["minutes"] = _number(0, 240)(r.get("minutes", 0))
            days = r.get("days")
            if days not in (None, []) and not isinstance(days, list):
                raise ValueError("days: is a list, like [Fri]")
            item["days"] = list(WEEKDAYS) if not days else [_one_of(str(d)[:3], WEEKDAYS) for d in days]
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from None
        seen.add(name.lower())
        out.append(item)
    return out


_CHECKS = {
    "city": _place, "country": _place, "school": _number(0, 1), "tune": _tune, "hijri_adjust": _number(-2, 2),
    "method": lambda v: _method(v),
    "heads_up": _section({"minutes": _number(0, 60), "prayers": _names(PRAYERS)}, _defaults()["heads_up"]),
    "at_start": _section({"telegram": _flag, "tasbih": _flag, "prayers": _names(PRAYERS)}, _defaults()["at_start"]),
    "jumuah": _clock_setting, "keep_awake": _flag, "reminders": _reminders,
    "brief": _section({"telegram": _flag, "after_fajr_minutes": _number(0, 120)}, _defaults()["brief"]),
    "tasbih": _text, "morning_adhkar": _text, "friday": _text,
}


def _validate(raw) -> tuple[dict, list[str]]:
    """The settings from the file (the defaults for what it leaves out), and what's wrong with it."""
    cfg, problems = _defaults(), []
    if raw is None:
        return cfg, problems
    if not isinstance(raw, dict):
        return cfg, ["it isn't a list of settings"]
    for key, value in raw.items():
        if key not in _CHECKS:
            problems.append(f"{key}: not a setting")
            continue
        try:
            cfg[key] = _CHECKS[key](value)
        except ValueError as e:
            problems.append(f"{key}: {e}")
    return cfg, problems


_cfg = {"stamp": None, "checked": 0.0, "good": None, "problems": []}
_cfg_lock = threading.Lock()


def config() -> dict:
    """The settings, read again when the file changes. A file that doesn't check out isn't used: the last good
    settings stay (the defaults if there were none), and what's wrong is printed once."""
    with _cfg_lock:
        if _cfg["good"] is not None and time.time() - _cfg["checked"] < RELOAD_SEC:
            return _cfg["good"]
        _cfg["checked"] = time.time()
        try:
            st = os.stat(CONFIG_FILE)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp == _cfg["stamp"] and _cfg["good"] is not None:
            return _cfg["good"]
        _cfg["stamp"] = stamp
        try:
            raw = None
            if stamp:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
            cfg, problems = _validate(raw)
        except (OSError, yaml.YAMLError) as e:
            cfg, problems = None, [f"{type(e).__name__}: {' '.join(str(e).split())[:120]}"]
        if problems:
            print(f"[prayers] {CONFIG_FILE} isn't used ({'; '.join(problems)}); "
                  f"{'keeping the last good settings' if _cfg['good'] else 'using the defaults'}")
            _cfg["problems"] = problems
            _cfg["good"] = _cfg["good"] or _defaults()
        else:
            _cfg["good"], _cfg["problems"] = cfg, []
        return _cfg["good"]


def reload() -> None:
    """Forget the settings, so the next call reads the file (regression.py, after moving CONFIG_FILE)."""
    with _cfg_lock:
        _cfg.update(stamp=None, checked=0.0, good=None, problems=[])


def problems() -> list[str]:
    return list(_cfg["problems"])


# ---------- Aladhan and the saved months ----------
_lock = threading.Lock()
_mem = {"c": None}
_fetch = {"busy": set(), "failed": {}}     # month -> when it last failed (time.time)


def _params(cfg: dict) -> dict:
    p = {"city": cfg["city"], "country": cfg["country"], "method": cfg["method"], "school": cfg["school"]}
    if cfg["tune"]:
        p["tune"] = ",".join(str(cfg["tune"].get(k, 0)) for k in TUNE_ORDER)
    return p


def _key(cfg: dict) -> str:
    return json.dumps(_params(cfg), sort_keys=True, ensure_ascii=False)


def _read_cache(key: str) -> dict:
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            c = json.load(f)
        if isinstance(c, dict) and c.get("key") == key and isinstance(c.get("months"), dict):
            c.setdefault("days", {})
            return c
    except (OSError, ValueError):
        pass
    return {"key": key, "tz": None, "months": {}, "days": {}}


def _write_cache(c: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE) or ".", exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False)
    os.replace(tmp, CACHE_FILE)


def _cache(cfg: dict) -> dict:
    """The saved months for these settings (a city or method change starts afresh). Under _lock."""
    key = _key(cfg)
    if _mem["c"] is None or _mem["c"].get("key") != key:
        _mem["c"] = _read_cache(key)
    return _mem["c"]


def forget() -> None:
    """Drop what's in memory (regression.py: a fresh start for each case, after moving CACHE_FILE)."""
    with _lock:
        _mem["c"] = None
        _fetch["busy"].clear()
        _fetch["failed"].clear()


def _get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode())


def _data(answer, kind):
    if not isinstance(answer, dict) or answer.get("code") != 200 or not isinstance(answer.get("data"), kind) \
            or not answer["data"]:
        raise RuntimeError(f"Aladhan answered {str(answer)[:100]}")
    return answer["data"]


_HM = re.compile(r"(\d{1,2}):(\d{2})")


def _compact(d: dict) -> dict:
    """One of Aladhan's days (a month's "04:58 (+06)" or a day's "04:58") as Javid keeps it."""
    dd, mm, yy = (int(x) for x in d["date"]["gregorian"]["date"].split("-"))
    times = {}
    for name in ANCHORS:
        if (m := _HM.match(str(d["timings"].get(name, "")))):
            times[name] = f"{int(m[1]):02d}:{m[2]}"
    if missing := [n for n in SHOWN if n not in times]:
        raise RuntimeError(f"Aladhan gave no {', '.join(missing)}")
    h, meta = d["date"]["hijri"], d.get("meta") or {}
    return {"date": date(yy, mm, dd).isoformat(), "tz": meta.get("timezone"), "times": times,
            "hijri": {"day": int(h["day"]), "month": int(h["month"]["number"]), "year": int(h["year"])},
            "method": (meta.get("method") or {}).get("name", ""), "school": meta.get("school", "")}


def _fetch_month(cfg: dict, y: int, m: int) -> None:
    month = f"{y:04d}-{m:02d}"
    url = f"{API}/calendarByCity/{y}/{m}?{urllib.parse.urlencode(_params(cfg))}"
    try:
        days = [_compact(d) for d in _data(_get_json(url, FETCH_SEC), list)]
    except Exception as e:
        with _lock:
            _fetch["failed"][month] = time.time()
            _fetch["busy"].discard(month)
        print(f"[prayers] couldn't get {month} from Aladhan: {type(e).__name__}: {str(e)[:80]}")
        return
    with _lock:
        c = _cache(cfg)
        if c["key"] == _key(cfg):
            c["months"][month] = days
            c["tz"] = days[0]["tz"] or c.get("tz")
            c["months"] = {k: c["months"][k] for k in sorted(c["months"])[-4:]}
            c["days"] = {k: v for k, v in c["days"].items() if k[:7] not in c["months"]}
            _write_cache(c)
        _fetch["busy"].discard(month)
        _fetch["failed"].pop(month, None)


def _zone(dd: dict | None = None):
    tz = (dd or {}).get("tz") or (_mem["c"] or {}).get("tz")
    try:
        return ZoneInfo(tz) if tz else datetime.now().astimezone().tzinfo
    except Exception:
        return datetime.now().astimezone().tzinfo


def _now() -> datetime:
    with _lock:
        _cache(config())
    return datetime.fromtimestamp(_time(), _zone())


def prefetch(background: bool | None = None) -> None:
    """Fetch the months around today that aren't saved (the server at start; a lookup that finds one missing)."""
    cfg = config()
    today = datetime.fromtimestamp(_time(), _zone()).date()
    todo = []
    with _lock:
        c = _cache(cfg)
        for d in (today - timedelta(days=2), today, today + timedelta(days=7)):   # Hijri shifts, next Friday
            month = f"{d.year:04d}-{d.month:02d}"
            if month in c["months"] or month in _fetch["busy"] or \
                    time.time() - _fetch["failed"].get(month, -1e12) < RETRY_SEC or (d.year, d.month) in todo:
                continue
            _fetch["busy"].add(month)
            todo.append((d.year, d.month))
    for y, m in todo:
        if BACKGROUND if background is None else background:
            threading.Thread(target=_fetch_month, args=(cfg, y, m), name="prayers-fetch", daemon=True).start()
        else:
            _fetch_month(cfg, y, m)


def _saved(cfg: dict, iso: str) -> dict | None:
    with _lock:
        c = _cache(cfg)
        return next((x for x in c["months"].get(iso[:7], ()) if x["date"] == iso), None) or c["days"].get(iso)


def times_for(d: date, wait: bool = True) -> dict | None:
    """d's times. One that isn't saved: with wait (a question) Aladhan is asked for that day, for ANSWER_SEC at most,
    while its month comes in the background; without (the tick), None. None too when Aladhan can't be reached."""
    cfg = config()
    iso = d.isoformat()
    if (hit := _saved(cfg, iso)):
        return hit
    prefetch()
    if (hit := _saved(cfg, iso)) or not wait:
        return hit
    url = f"{API}/timingsByCity/{d:%d-%m-%Y}?{urllib.parse.urlencode(_params(cfg))}"
    try:
        one = _compact(_data(_get_json(url, ANSWER_SEC), dict))
    except Exception as e:
        print(f"[prayers] couldn't get {iso} from Aladhan: {type(e).__name__}: {str(e)[:80]}")
        return None
    with _lock:
        c = _cache(cfg)
        if c["key"] == _key(cfg):
            c["days"][iso] = one
            c["tz"] = c.get("tz") or one["tz"]
            _write_cache(c)
    return one


def _at(dd: dict, name: str) -> datetime:
    """One of the day's times, in the city's zone; a night time before noon is on the next date."""
    hh, mm = map(int, dd["times"][name].split(":"))
    d = date.fromisoformat(dd["date"])
    at = datetime(d.year, d.month, d.day, hh, mm, tzinfo=_zone(dd))
    return at + timedelta(days=1) if name in NIGHT and hh < 12 else at


def _friday(dd: dict) -> bool:
    return date.fromisoformat(dd["date"]).weekday() == 4


def _jumuah_at(dd: dict, cfg: dict) -> datetime:
    if not cfg["jumuah"]:
        return _at(dd, "Dhuhr")
    hh, mm = map(int, cfg["jumuah"].split(":"))
    d = date.fromisoformat(dd["date"])
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=_zone(dd))


def _prayer_at(dd: dict, p: str, cfg: dict) -> tuple[str, datetime]:
    """(name, time) of a prayer that day: on a Friday, Dhuhr is Jumu'ah (at the mosque's time if it's set)."""
    if p == "Dhuhr" and _friday(dd):
        return "Jumuah", _jumuah_at(dd, cfg)
    return p, _at(dd, p)


def hijri(d: date) -> dict | None:
    """d's Hijri date: Aladhan's, moved by hijri_adjust days for the local moon sighting."""
    dd = times_for(d + timedelta(days=config()["hijri_adjust"]))
    return dd["hijri"] if dd else None


# ---------- wording ----------
def _bn(x) -> str:
    return str(x).translate(_TO_BN)


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def _part_bn(h: int) -> str:
    return ("রাত" if h < 4 or h >= 20 else "ভোর" if h < 6 else "সকাল" if h < 12 else "দুপুর" if h < 15 else
            "বিকেল" if h < 17 else "সন্ধ্যা")


def _clock_bn(dt: datetime, at: bool = False) -> str:
    """ভোর ৪টা ৫৮ মিনিট; at=True: ...মিনিটে, ...৫টায়."""
    words = f"{_part_bn(dt.hour)} {dt.hour % 12 or 12}টা" + (f" {dt.minute} মিনিট" if dt.minute else "")
    return _bn(words + (("ে" if dt.minute else "য়") if at else ""))


def _short(dt: datetime, lang: str) -> str:
    return _bn(f"{_part_bn(dt.hour)} {dt.hour % 12 or 12}:{dt.minute:02d}") if lang == "bn" else _clock(dt)


def _say(dt: datetime, lang: str, at: bool = False) -> str:
    return _clock_bn(dt, at) if lang == "bn" else _clock(dt)


def _span(secs: float, lang: str) -> str:
    mins = max(0, math.ceil(secs / 60))
    h, m = divmod(mins, 60)
    if lang == "bn":
        return " ".join(p for p in (f"{_bn(h)} ঘণ্টা" if h else "", f"{_bn(m)} মিনিট" if m else "") if p) or \
            "এক মিনিটেরও কম"
    return " and ".join(p for p in (f"{h} hour{'s' if h != 1 else ''}" if h else "",
                                    f"{m} minute{'s' if m != 1 else ''}" if m else "") if p) or "under a minute"


def _name(p: str, lang: str) -> str:
    return NAME_BN.get(p, p) if lang == "bn" else NAME_EN.get(p, p)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _date_text(d: date, lang: str) -> str:
    if lang == "bn":
        return f"{DAYS_BN[d.weekday()]}, {_bn(d.day)} {MONTHS_BN[d.month - 1]}"
    return f"{d:%A}, {d.day} {d:%B}"


def _city(cfg: dict, lang: str) -> str:
    return CITY_BN.get(cfg["city"].lower(), cfg["city"]) if lang == "bn" else cfg["city"]


def hijri_text(h: dict, lang: str) -> str:
    if lang == "bn":
        return f"{_bn(h['day'])} {HIJRI_BN[h['month'] - 1]} {_bn(h['year'])} হিজরি"
    return f"{h['day']} {HIJRI_EN[h['month'] - 1]} {h['year']} AH"


def _school(cfg: dict, lang: str) -> str:
    if lang == "bn":
        return "হানাফি আসর" if cfg["school"] == 1 else "শাফেয়ি আসর"
    return "Hanafi Asr" if cfg["school"] == 1 else "Shafi'i Asr"


def _no_times(lang: str) -> str:
    return ("এখন নামাজের সময় পাচ্ছি না: Aladhan সাড়া দিচ্ছে না, আর ওই দিনের সময় আমার কাছে রাখা নেই।" if lang == "bn"
            else "I can't get the prayer times right now: Aladhan isn't answering, and I have nothing saved for "
                 "that day.")


def _when_word(d: date, today: date, lang: str) -> str:
    if d == today:
        return "আজ" if lang == "bn" else "today"
    if d == today + timedelta(days=1):
        return "কাল" if lang == "bn" else "tomorrow"
    return _date_text(d, lang) if lang == "bn" else f"on {_date_text(d, lang)}"


class _Fields(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _fill(text: str, d: date, lang: str, cfg: dict) -> str:
    """A configured text with its {fajr}, {maghrib_tomorrow}... filled in from d's (and the next day's) times."""
    fields = _Fields()
    for when, dd in (("", times_for(d, wait=False)), ("_tomorrow", times_for(d + timedelta(days=1), wait=False))):
        if dd:
            for n in ANCHORS:
                fields[n.lower() + when] = _say(_at(dd, n), lang)
            fields["jumuah" + when] = _say(_jumuah_at(dd, cfg), lang)
    return text.format_map(fields)


# ---------- questions, answered in code ----------
def _alt(latin: str, bengali: str = "") -> str:
    """Words to match: the Latin ones whole ("isha" isn't in "Aisha"), the Bengali ones as they are, since a case
    ending follows them (মাগরিবের)."""
    return rf"(?<![a-z])(?:{latin})(?![a-z])" + (f"|{bengali}" if bengali else "")


_WORDS = {   # each prayer by name, and by the time of day it's called after ("the morning prayer", সকালের নামাজ)
    "Fajr": _alt(r"fajr|fajar|fazr|fozor|fojor|fazar|(?:morning|dawn) prayer", "ফজর|(?:ভোরের|সকালের) নামাজ"),
    "Sunrise": _alt(r"sunrise|shuruq|ishraq", "সূর্যোদয়|সূর্য ওঠ"),
    "Dhuhr": _alt(r"dhuhr|zuhr|zohr|zuhur|zohar|duhr|dhuhur|johor|zohor|(?:noon|midday) prayer",
                  "যোহর|জোহর|যুহর|জুহর|দুপুরের নামাজ"),
    "Asr": _alt(r"asr|asar|asor|afternoon prayer", "আসর|বিকেলের নামাজ"),
    "Maghrib": _alt(r"maghrib|magrib|maghreb|magreb|mogrib|mugrib|maghrob|(?:evening|sunset) prayer",
                    "মাগরিব|মাগরীব|সন্ধ্যার নামাজ"),
    "Isha": _alt(r"isha'?|esha|ishaa|isya", "এশা|ইশা(?!রা)"),
    "Jumuah": _alt(r"jumu'?ah|jummah|jumm?a|jumah|friday prayers?", "জুমা|জুম্মা|জুমআ"),
    "Iftar": _alt(r"iftar|iftaar|ifter", "ইফতার"),
    "Sehri": _alt(r"sehri|sehar|suhoor|suhur|sahur|sahri", "সেহরি|সাহরি"),
    "Tahajjud": _alt(r"tahajjud|tahajud|tahajjod", "তাহাজ্জুদ"),
}
_PRAYER = re.compile("|".join(f"(?P<{k}>{v})" for k, v in _WORDS.items()), re.I)
_ANY_PRAYER = "|".join(_WORDS.values())
_GENERIC = _alt(r"prayers?|namaz|namaj|namaaz|salah|salat|salaah|adhan|azan|athan|waqt", "নামাজ|নামায|আজান|ওয়াক্ত")
_HIJRI = re.compile(_alt(r"hijri|islamic date|arabic date|islamic month|arabic month|lunar date|islamic calendar|"
                         r"hijri calendar", "হিজরি|আরবি তারিখ|আরবি মাস|ইসলামি তারিখ|চাঁদের তারিখ"), re.I)
_WHEN = re.compile(_alt(r"when|what time|time|times|timing|timings|schedule|kokhon|koytay|koyta|kotay|somoy|shomoy|"
                        r"waqt|wakt|starts?|begins?|ends?|until|till|left|next|now|yet|today|tonight|tomorrow|soon",
                        "কখন|কয়টায়|কটায়|কয়টা|সময়|ওয়াক্ত|শুরু|শেষ|কতক্ষণ|আজ|কাল|পরের|এখন"), re.I)
_ALL = re.compile(_alt(r"(?:prayer|namaz|namaj|salah|salat|salaah) ?(?:times?|timings?|schedule|waqt)|"
                       r"(?:times?|timings?|schedule) (?:of|for) (?:the |today'?s |tomorrow'?s )?"
                       r"(?:prayers?|namaz|salah|salat)", "নামাজের (?:সময়|ওয়াক্ত)|ওয়াক্তের সময়"), re.I)
_NEXT = re.compile(_alt(r"next (?:prayer|namaz|salah|salat|one|waqt)|which prayer (?:is )?(?:next|now)|"
                        r"what prayer is (?:it|next|now)|current prayer",
                        "পরের নামাজ|পরের ওয়াক্ত|এখন কোন (?:নামাজ|ওয়াক্ত)"), re.I)
_LEFT = re.compile(r"how (?:long|much time|many (?:minutes|hours))(?: is (?:it|left|there))?\s*"
                   r"(?:until|till|til|before|to|left|for|remain)|time left|minutes left|কতক্ষণ বাকি|আর কতক্ষণ|"
                   r"কত সময় বাকি|কতক্ষণ পর", re.I)
_NOW = re.compile(r"(?<![a-z])(?:has|did) \S+(?: time)? (?:started|begun|come in|come)|is it (?:\S+ )?(?:time )?"
                  r"(?:yet|now)|(?:started|begun) yet|শুরু হয়েছে|হয়ে গেছে|ওয়াক্ত হয়েছে", re.I)
_END = re.compile(r"(?<![a-z])(?:when does|when will|what time does|till when|until when) \S+(?: time)? "
                  r"(?:end|finish|close|be over|last)|(?:end|ending) (?:of|time)|lasts? until|শেষ কখন|কখন শেষ|"
                  r"শেষ হবে|কতক্ষণ থাকে", re.I)
_REMIND = re.compile(r"(?<![a-z])(?:remind|notify|alert|ping|wake|warn) me(?![a-z])|(?:tell|let) me (?:know )?when|"
                     r"(?<![a-z])(?:turn on|switch on|enable)\s+(?:the |my )?(?:prayer |namaz |salah )?"
                     r"(?:reminders?|alerts?|notifications?)(?![a-z])|জানিও|মনে করিয়ে|জাগিয়ে", re.I)
# Another request in the same message goes to the model, which has prayer_times too; so do statements and other
# questions about a prayer ("I missed Fajr", "how many rakats is Maghrib?", "what is Tahajjud?").
_ELSE = re.compile(_alt(r"weather|forecast|rain|raining|temperature|umbrella|tasks?|to-?dos?|timer|alarm|calendar|"
                        r"meetings?|appointments?|events?|news|headlines?|price|bitcoin|crypto|stocks?|dollar|taka|"
                        r"exchange|search|google|look up|email|message|save|delete|add|plan|translate",
                        "আবহাওয়া|বৃষ্টি|খবর|দাম|টাইমার|অ্যালার্ম|ক্যালেন্ডার|মিটিং|টাস্ক"), re.I)
_NOT_TIME = re.compile(r"(?<![a-z])(?:i|we)(?:'ve| have)? (?:prayed|missed|did|finished|offered|performed|made|forgot|"
                       r"skipped|will pray|am praying|want to pray|need to pray)(?![a-z])|how (?:many|to|do i|should i)"
                       rf"(?![a-z])|^what(?:'s| is| are) (?:the )?(?:{'|'.join(_WORDS.values())})"
                       r"(?: prayers?| namaz| salah)?$|"
                       r"what does|(?<![a-z])(?:why|meaning|dua|du'a|rak'?ahs?|rakats?|nafl|witr|fard|farz)(?![a-z])|"
                       r"পড়েছি|পড়লাম|পড়িনি|কত রাকাত|কিভাবে|কীভাবে|কেন|মানে|দোয়া|দুআ", re.I)
_DAY = re.compile(_alt(r"day after tomorrow|today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|"
                       r"sunday", r"আগামীকাল|(?<![\u0980-\u09ff])(?:আজ|কাল|পরশু)"), re.I)   # not the কাল in সকাল
_DAYS_EN = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# The words a prayer-time question is made of besides the prayer and the day: anything else left over means the message
# asks for more than the times ("tell me a fun fact and when is Maghrib"), so it goes to the model.
_FILLER = set("""a an the is are was what what's whats when when's whens which how long much many until till before
after time times timing timings day now yet next prayer prayers namaz namaj salah salat waqt schedule please javid hey
ok okay tell me let know give show say of for in at on to does do did will it its it's has have begun began started
start starts begin begins end ends ending finish last lasts left remaining remain there there's be by my again also
and kokhon koytay koyta kotay somoy shomoy er date month calendar remind notify alert wake ping warn reminders
reminder alerts notifications turn switch enable up i is it's soon come comes""".split())
_FILLER_BN = set("""কখন কয়টায় কটায় কয়টা কত কতটায় কতক্ষণ বাকি আর সময় সময়সূচি সময়সূচী ওয়াক্ত শুরু শেষ হবে হয়েছে হলো
হয় পরের এখন কি কী তারিখ আজকের কালকের বলো বল বলবে জানাও জানিও দাও আছে থাকে থাকবে পর্যন্ত কোন কোনটা নামাজ
নামাজের আগে পরে মনে করিয়ে দিও দিবে আমাকে""".split())


def _leftover(t: str) -> list[str]:
    rest = re.sub(_GENERIC, " ", _DAY.sub(" ", _HIJRI.sub(" ", _PRAYER.sub(" ", t))), flags=re.I)
    words = re.findall(r"[a-z']+|[\u0980-\u09ff]+", rest)
    return [w for w in words if w not in _FILLER and w not in _FILLER_BN
            and not (w[0] >= "\u0980" and len(w) <= 3)]     # a case ending left behind: মাগরিব|ের


ASK = re.compile(rf"(?:{_ANY_PRAYER}).*?(?:{_WHEN.pattern})|(?:{_WHEN.pattern}).*?(?:{_ANY_PRAYER})|{_ALL.pattern}|"
                 rf"{_HIJRI.pattern}|{_NEXT.pattern}", re.I | re.S)    # REQUIRED_TOOLS: then prayer_times is a must
WORDS = re.compile(rf"{_ANY_PRAYER}|{_GENERIC}|{_HIJRI.pattern}", re.I)    # TOOL_GROUPS: these send prayer_times


def request(text: str) -> dict | None:
    """{"action", "prayer", "day"} for a message that is only a prayer-time question; None leaves it to the model."""
    t = " ".join((text or "").lower().replace("’", "'").split()).strip(" ?!.।,")
    if not t or len(t.split()) > 14 or _ELSE.search(t) or _NOT_TIME.search(t) or _leftover(t):
        return None
    day = (m[0] if (m := _DAY.search(t)) else "")
    if _HIJRI.search(t):
        return {"action": "hijri", "prayer": None, "day": day}
    prayer = m.lastgroup if (m := _PRAYER.search(t)) else None
    if _REMIND.search(t):
        return {"action": "reminders", "prayer": prayer, "day": ""} if prayer or re.search(_GENERIC, t) else None
    if _NEXT.search(t):
        return {"action": "next", "prayer": None, "day": ""}
    if not prayer:
        return {"action": "all", "prayer": None, "day": day} if _ALL.search(t) else None
    if not (_WHEN.search(t) or len(t.split()) <= 3):
        return None
    action = "left" if _LEFT.search(t) else "now" if _NOW.search(t) else "end" if _END.search(t) else "when"
    return {"action": action, "prayer": prayer, "day": day}


def _day_of(word: str, today: date) -> date:
    w = (word or "").lower()
    if w in ("", "today", "tonight", "আজ"):
        return today
    if w in ("tomorrow", "কাল", "আগামীকাল"):
        return today + timedelta(days=1)
    if w in ("day after tomorrow", "পরশু"):
        return today + timedelta(days=2)
    if w in _DAYS_EN:
        return today + timedelta(days=(_DAYS_EN.index(w) - today.weekday()) % 7)
    try:
        return date.fromisoformat(w)
    except ValueError:
        return today


def answer(req: dict, lang: str):
    """A prayer question answered here: a generator (the prayer_times tool event, for the page and the trace)
    returning (spoken, screen)."""
    yield ("tool", "prayer_times")
    span = tracing.tool_start("prayer_times")
    try:
        return _answer(req, lang)
    finally:
        tracing.tool_end(span)


def _answer(req: dict, lang: str) -> tuple[str, str]:
    cfg, now = config(), _now()
    today, act = now.date(), req["action"]
    if act == "reminders":
        return _reminders_said(cfg, now, lang), ""
    d = _day_of(req.get("day", ""), today)
    if act == "hijri":
        h = hijri(d)
        if not h:
            return _no_times(lang), ""
        if lang == "bn":
            said = f"{_when_word(d, today, 'bn')} {hijri_text(h, 'bn')}।"
        elif d in (today, today + timedelta(days=1)):
            said = f"{_cap(_when_word(d, today, 'en'))} is {hijri_text(h, 'en')}."
        else:
            said = f"{_date_text(d, 'en')} is {hijri_text(h, 'en')}."
        return said, _hijri_note(cfg, lang)
    dd = times_for(d)
    if not dd:
        return _no_times(lang), ""
    if act == "all":
        return _all_said(dd, d == today, now, cfg, lang), _card(dd, now, cfg, lang)
    if act == "next":
        nxt = _next(now, cfg)
        if not nxt:
            return _no_times(lang), ""
        name, at = nxt
        return _next_said(name, at, now, lang), _card(times_for(at.date()) or dd, now, cfg, lang)
    said = _prayer_said(req["prayer"], act, d, dd, now, cfg, lang)
    return said, _card(dd, now, cfg, lang)


def _next(now: datetime, cfg: dict) -> tuple[str, datetime] | None:
    for d in (now.date(), now.date() + timedelta(days=1)):
        dd = times_for(d)
        if not dd:
            return None
        for p in PRAYERS:
            name, at = _prayer_at(dd, p, cfg)
            if at > now:
                return name, at
    return None


def _current(now: datetime) -> tuple[str, datetime, datetime] | None:
    """The prayer whose time it is now (name, began, ends), or None (after sunrise, before Dhuhr)."""
    yday, today, tmrw = (times_for(now.date() + timedelta(days=k)) for k in (-1, 0, 1))
    spans = [("Isha", _at(yday, "Isha"), _at(today, "Fajr"))] if yday and today else []
    if today:
        spans += [(p, _at(today, p), _at(today, WINDOW_END[p])) for p in WINDOW_END]
        if tmrw:
            spans.append(("Isha", _at(today, "Isha"), _at(tmrw, "Fajr")))
    return next((s for s in spans if s[1] <= now < s[2]), None)


def _day_times(dd: dict, cfg: dict) -> list[tuple[str, datetime]]:
    """The six times shown, Jumu'ah in Dhuhr's place on a Friday."""
    return [_prayer_at(dd, p, cfg) if p in PRAYERS else (p, _at(dd, p)) for p in SHOWN]


def _all_said(dd: dict, is_today: bool, now: datetime, cfg: dict, lang: str) -> str:
    d = date.fromisoformat(dd["date"])
    items = [_prayer_at(dd, p, cfg) for p in PRAYERS]
    if lang == "bn":
        listed = ", ".join(f"{_name(n, 'bn')} {_clock_bn(at)}" for n, at in items[:-1])
        whose = {now.date(): "আজকের", now.date() + timedelta(days=1): "কালকের"}.get(d, f"{_date_text(d, 'bn')}-এর")
        said = (f"{whose} নামাজের সময় ({_city(cfg, 'bn')}): {listed}, আর "
                f"{_name(items[-1][0], 'bn')} {_clock_bn(items[-1][1])}।")
    else:
        listed = ", ".join(f"{_name(n, 'en')} at {_clock(at)}" for n, at in items[:-1])
        said = (f"{_cap(_when_word(d, now.date(), 'en'))} in {cfg['city']}: {listed} and "
                f"{_name(items[-1][0], 'en')} at {_clock(items[-1][1])}.")
    if is_today and (nxt := _next(now, cfg)):
        left = _span((nxt[1] - now).total_seconds(), lang)
        said += (f" পরের নামাজ {_name(nxt[0], 'bn')}, আর {left} পর।" if lang == "bn" else
                 f" Next is {_name(nxt[0], 'en')}, in {left}.")
    return said


def _next_said(name: str, at: datetime, now: datetime, lang: str) -> str:
    left = _span((at - now).total_seconds(), lang)
    if lang == "bn":
        later = "কাল " if at.date() != now.date() else ""
        return f"পরের নামাজ {later}{_name(name, 'bn')}, {_clock_bn(at, True)}, আর {left} পর।"
    return f"Next is {_name(name, 'en')} at {_clock(at)}{' tomorrow' if at.date() != now.date() else ''}, in {left}."


def _prayer_said(p: str, act: str, d: date, dd: dict, now: datetime, cfg: dict, lang: str) -> str:
    today = now.date()
    if p == "Iftar":
        return _special("Iftar", "Maghrib", d, dd, now, cfg, lang)
    if p == "Sehri":
        return _sehri(now, lang)
    if p == "Tahajjud":
        return _tahajjud(now, lang)
    if p == "Jumuah":
        return _jumuah(d, now, cfg, lang)
    name, at = _prayer_at(dd, p, cfg) if p in PRAYERS else (p, _at(dd, p))
    if d != today:
        when = _when_word(d, today, lang)
        return (f"{when} {_name(name, 'bn')} {_clock_bn(at, True)}।" if lang == "bn" else
                f"{_cap(_name(name, 'en'))} {when}{',' if when.startswith('on ') else ''} is at {_clock(at)}.")
    cur = _current(now)
    if act == "end" and p in PRAYERS:
        return _end_said(p, now, cfg, lang)
    if act == "now" and p in PRAYERS:
        if cur and cur[0] == p:
            nxt = _next(now, cfg)
            return (f"হ্যাঁ, {_name(p, 'bn')} শুরু হয়েছে {_clock_bn(cur[1], True)}" +
                    (f"; {_name(nxt[0], 'bn')} {_clock_bn(nxt[1], True)}।" if nxt else "।") if lang == "bn" else
                    f"Yes: {_name(p, 'en')} began at {_clock(cur[1])}" +
                    (f", and {_name(nxt[0], 'en')} is at {_clock(nxt[1])}." if nxt else "."))
        if at > now:
            left = _span((at - now).total_seconds(), lang)
            return (f"এখনো না: {_name(name, 'bn')} {_clock_bn(at, True)}, আর {left} পর।" if lang == "bn" else
                    f"Not yet: {_name(name, 'en')} is at {_clock(at)}, in {left}.")
    if at > now:
        left = _span((at - now).total_seconds(), lang)
        if act == "left":
            return (f"{_name(name, 'bn')} আর {left} পর, {_clock_bn(at, True)}।" if lang == "bn" else
                    f"{_cap(_name(name, 'en'))} is in {left}, at {_clock(at)}.")
        return (f"আজ {_name(name, 'bn')} {_clock_bn(at, True)}, আর {left} পর।" if lang == "bn" else
                f"{_cap(_name(name, 'en'))} is at {_clock(at)} today, in {left}.")
    if cur and cur[0] == p:
        end_name = WINDOW_END.get(p, "Fajr")
        return (f"{_name(p, 'bn')} শুরু হয়েছে {_clock_bn(cur[1], True)}, থাকবে {_name(end_name, 'bn')} "
                f"({_clock_bn(cur[2])}) পর্যন্ত।" if lang == "bn" else
                f"{_name(p, 'en')} began at {_clock(cur[1])} and lasts until {_name(end_name, 'en')}, at "
                f"{_clock(cur[2])}.")
    tmrw = times_for(today + timedelta(days=1))
    later = (_prayer_at(tmrw, p, cfg)[1] if p in PRAYERS else _at(tmrw, p)) if tmrw else None
    if lang == "bn":
        return f"আজ {_name(name, 'bn')} ছিল {_clock_bn(at, True)}" + \
            (f"; কাল {_clock_bn(later, True)}।" if later else "।")
    return f"{_cap(_name(name, 'en'))} was at {_clock(at)} today" + \
        (f"; tomorrow it's at {_clock(later)}." if later else ".")


def _end_said(p: str, now: datetime, cfg: dict, lang: str) -> str:
    for k in (0, 1):
        dd = times_for(now.date() + timedelta(days=k))
        nxt = times_for(now.date() + timedelta(days=k + 1))
        if not dd:
            break
        end_name = WINDOW_END.get(p, "Fajr")
        end = _at(dd, end_name) if p != "Isha" else (_at(nxt, "Fajr") if nxt else None)
        if end and end > now:
            tail = " tomorrow" if end.date() != now.date() else ""
            if lang == "bn":
                return f"{OF_BN.get(p, _name(p, 'bn'))} সময় থাকে {_name(end_name, 'bn')} পর্যন্ত, {_clock_bn(end)}।"
            return f"{_name(p, 'en')} lasts until {_name(end_name, 'en')}, at {_clock(end)}{tail}."
    return _no_times(lang)


def _special(label: str, p: str, d: date, dd: dict, now: datetime, cfg: dict, lang: str) -> str:
    """Iftar: Maghrib's time, said as iftar."""
    at = _at(dd, p)
    if d == now.date() and at > now:
        left = _span((at - now).total_seconds(), lang)
        return (f"আজ ইফতার মাগরিবে, {_clock_bn(at, True)}, আর {left} পর।" if lang == "bn" else
                f"Iftar is at Maghrib, {_clock(at)} today, in {left}.")
    if d != now.date():
        when = _when_word(d, now.date(), lang)
        return (f"{when} ইফতার {_clock_bn(at, True)}।" if lang == "bn" else f"Iftar {when} is at {_clock(at)}.")
    tmrw = times_for(d + timedelta(days=1))
    later = f"; কাল {_clock_bn(_at(tmrw, p), True)}" if tmrw and lang == "bn" else \
        f"; tomorrow it's at {_clock(_at(tmrw, p))}" if tmrw else ""
    return (f"আজ ইফতার ছিল {_clock_bn(at, True)}{later}।" if lang == "bn" else
            f"Iftar was at {_clock(at)} today{later}.")


def _sehri(now: datetime, lang: str) -> str:
    for k in (0, 1):
        dd = times_for(now.date() + timedelta(days=k))
        if dd and (at := _at(dd, "Fajr")) > now:
            when = "আজ" if k == 0 else "কাল"
            left = _span((at - now).total_seconds(), lang)
            return (f"সেহরি শেষ করতে হবে ফজরের আগে, {when} {_clock_bn(at, True)}, আর {left} পর।" if lang == "bn" else
                    f"Sehri ends by Fajr, at {_clock(at)} {'today' if k == 0 else 'tomorrow'}, in {left}.")
    return _no_times(lang)


def _tahajjud(now: datetime, lang: str) -> str:
    for k in (-1, 0):
        dd = times_for(now.date() + timedelta(days=k))
        fajr_day = times_for(now.date() + timedelta(days=k + 1))
        if not dd or not fajr_day:
            continue
        start, fajr = _at(dd, "Lastthird"), _at(fajr_day, "Fajr")
        if now < start:
            left = _span((start - now).total_seconds(), lang)
            return (f"রাতের শেষ তৃতীয়াংশ, তাহাজ্জুদের ভালো সময়, শুরু হবে {_clock_bn(start, True)}, আর {left} পর; ফজর "
                    f"{_clock_bn(fajr, True)}।" if lang == "bn" else
                    f"The night's last third, a good time for Tahajjud, begins at {_clock(start)}, in {left}; Fajr is "
                    f"at {_clock(fajr)}.")
        if now < fajr:
            if lang == "bn":
                return f"রাতের শেষ তৃতীয়াংশ শুরু হয়েছে {_clock_bn(start, True)}; ফজর {_clock_bn(fajr, True)}।"
            return f"The night's last third began at {_clock(start)}; Fajr is at {_clock(fajr)}."
    return _no_times(lang)


def _jumuah(d: date, now: datetime, cfg: dict, lang: str) -> str:
    friday = d + timedelta(days=(4 - d.weekday()) % 7)
    if friday == now.date() and (dd := times_for(friday)) and _jumuah_at(dd, cfg) < now:
        friday += timedelta(days=7)
    dd = times_for(friday)
    if not dd:
        return _no_times(lang)
    at = _jumuah_at(dd, cfg)
    dhuhr = "" if cfg["jumuah"] else (" যোহরের সময়ে" if lang == "bn" else " at Dhuhr's time,")
    if friday == now.date():
        left = _span((at - now).total_seconds(), lang)
        return (f"আজ জুমা{dhuhr} {_clock_bn(at, True)}, আর {left} পর।" if lang == "bn" else
                f"Jumu'ah is{dhuhr} {_clock(at)} today, in {left}.")
    return (f"জুমা {_date_text(friday, 'bn')},{dhuhr} {_clock_bn(at, True)}।" if lang == "bn" else
            f"Jumu'ah is on {_date_text(friday, 'en')},{dhuhr} {_clock(at)}.")


def _card(dd: dict, now: datetime, cfg: dict, lang: str) -> str:
    """The day's times for the screen, the next one marked, with the Hijri date and where the times come from."""
    d = date.fromisoformat(dd["date"])
    nxt = _next(now, cfg)
    up = ("Dhuhr" if nxt[0] == "Jumuah" else nxt[0]) if nxt and nxt[1].date() == d else None
    head = (f"নামাজের সময় · {_city(cfg, 'bn')}, {_date_text(d, 'bn')}:" if lang == "bn" else
            f"Prayer times · {cfg['city']}, {_date_text(d, 'en')}:")
    lines = [head]
    for p in SHOWN:
        at = _at(dd, p)
        mark = (" · পরের নামাজ" if lang == "bn" else " · next") if p == up else ""
        friday = "" if p != "Dhuhr" or not _friday(dd) or cfg["jumuah"] else " (জুমা)" if lang == "bn" else " (Jumu'ah)"
        lines.append(f"- {_name(p, lang) if lang == 'bn' else p}{friday}  {_short(at, lang)}{mark}")
        if p == "Dhuhr" and _friday(dd) and cfg["jumuah"]:
            lines.append(f"- {_name('Jumuah', lang)}  {_short(_jumuah_at(dd, cfg), lang)}")
    h = hijri(d)
    lines.append(" · ".join(filter(None, [hijri_text(h, lang) if h else "",
                                          f"Aladhan: {dd.get('method') or 'method ' + str(cfg['method'])}; "
                                          f"{_school(cfg, lang)}"])))
    return "\n".join(lines)


def _hijri_note(cfg: dict, lang: str) -> str:
    if not cfg["hijri_adjust"]:
        return ""
    n = cfg["hijri_adjust"]
    if lang == "bn":
        moved = "পিছিয়ে" if n < 0 else "এগিয়ে"
        return (f"হিজরি তারিখ: Aladhan-এর ক্যালেন্ডার, স্থানীয় চাঁদ দেখার জন্য {_bn(abs(n))} দিন {moved} "
                "(config/prayers.yaml)")
    return (f"Hijri date: Aladhan's calendar moved {abs(n)} day{'s' if abs(n) != 1 else ''} "
            f"{'back' if n < 0 else 'forward'} for the local moon sighting (config/prayers.yaml)")


def _reminders_said(cfg: dict, now: datetime, lang: str) -> str:
    hu, st = cfg["heads_up"], cfg["at_start"]
    if not (hu["minutes"] and hu["prayers"]):
        return ("নামাজের রিমাইন্ডার বন্ধ আছে; config/prayers.yaml-এ চালু করা যায়।" if lang == "bn" else
                "Prayer reminders are off. They're set in config/prayers.yaml.")
    which = ("each prayer" if len(hu["prayers"]) == 5 else " and ".join(hu["prayers"])) if lang != "bn" else \
        ("প্রতি ওয়াক্তের" if len(hu["prayers"]) == 5 else ", ".join(OF_BN[p] for p in hu["prayers"]))
    nxt = next((i for d in (now.date(), now.date() + timedelta(days=1)) for i in _planned(d, cfg)
                if i["kind"] == "prayer" and i["due"] > now), None)
    if lang == "bn":
        said = f"নামাজের রিমাইন্ডার চালু আছে: {which} {_bn(hu['minutes'])} মিনিট আগে পেজে আর টেলিগ্রামে জানাই"
        said += ", আর ওয়াক্ত শুরু হলে টেলিগ্রামে বার্তা যায়।" if st["telegram"] and st["prayers"] else "।"
        if nxt:
            said += f" পরেরটা {_clock_bn(nxt['due'], True)}, {OF_BN.get(nxt['name'], nxt['name'])} জন্য।"
        return said
    said = f"Prayer reminders are on: I ring {hu['minutes']} minutes before {which} on the page and on Telegram"
    said += ", and Telegram gets a message when each one begins." if st["telegram"] and st["prayers"] else "."
    if nxt:
        said += f" The next is at {_clock(nxt['due'])}, for {_name(nxt['name'], 'en')}."
    return said


# ---------- the model's tool (a prayer question with something else in it) ----------
def prayer_times(day: str = "today") -> str:
    cfg, now = config(), _now()
    d = _day_of(str(day or "today").strip(), now.date())
    dd = times_for(d)
    if not dd:
        return _no_times("en")
    h = hijri(d)
    shown = ", ".join(f"{_cap(_name(n, 'en'))} {_clock(at)}" + (" (iftar)" if n == "Maghrib" else "")
                      for n, at in _day_times(dd, cfg))
    hijri_part = f" ({hijri_text(h, 'en')})" if h else ""
    out = (f"{_date_text(d, 'en')}{hijri_part} in {cfg['city']}, {cfg['country']}: {shown}. Source: Aladhan, "
           f"{dd.get('method') or 'method ' + str(cfg['method'])}, {_school(cfg, 'en')}.")
    if d == now.date() and (nxt := _next(now, cfg)):
        out += (f" It's {_clock(now)} now: next is {_name(nxt[0], 'en')} at {_clock(nxt[1])}, in "
                f"{_span((nxt[1] - now).total_seconds(), 'en')}.")
    return out


TOOLS = [{
    "type": "function",
    "function": {
        "name": "prayer_times",
        "description": "Prayer times (Fajr, Sunrise, Dhuhr, Asr, Maghrib, Isha) for Mahadi's city on a day, the next "
                       "prayer and the Hijri date, from Aladhan. Call it for any prayer-time, iftar or sehri question "
                       "and use only its times. Reminders before each prayer are already on; don't set a timer for "
                       "them unless he asks for an extra one.",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string", "description": "'today', 'tomorrow', a weekday or YYYY-MM-DD; default today."}}},
    },
}]
TOOL_MAP = {"prayer_times": lambda args: prayer_times(**args)}


# ---------- what the tick raises ----------
def _planned(d: date, cfg: dict) -> list[dict]:
    """What day d's times raise, in order: {"key", "kind", "name", "at", "due", "until", "channels"} (+ "reminder").
    The tick raises them; /reminders lists them."""
    dd = times_for(d, wait=False)
    if not dd:
        return []
    out, hu, st = [], cfg["heads_up"], cfg["at_start"]
    for p in hu["prayers"] if hu["minutes"] else ():
        name, at = _prayer_at(dd, p, cfg)
        out.append({"key": f"{dd['date']}:heads:{p}", "kind": "prayer", "name": name, "at": at,
                    "due": at - timedelta(minutes=hu["minutes"]), "until": at, "channels": None})
    for p in st["prayers"] if st["telegram"] else ():
        name, at = _prayer_at(dd, p, cfg)
        out.append({"key": f"{dd['date']}:start:{p}", "kind": "prayer_start", "name": name, "at": at, "due": at,
                    "until": at + timedelta(seconds=START_LATE_SEC), "channels": ["telegram"]})
    if cfg["brief"]["telegram"]:
        due = _at(dd, "Fajr") + timedelta(minutes=cfg["brief"]["after_fajr_minutes"])
        out.append({"key": f"{dd['date']}:brief", "kind": "brief", "name": "Morning brief", "at": due, "due": due,
                    "until": max(_at(dd, "Dhuhr"), due + timedelta(minutes=30)), "channels": ["telegram"]})
    for r in cfg["reminders"]:
        if not r["enabled"]:
            continue
        if "at" in r:
            hh, mm = map(int, r["at"].split(":"))
            due = datetime(d.year, d.month, d.day, hh, mm, tzinfo=_zone(dd))
        else:
            base = _at(dd, r.get("after") or r["before"])
            due = base + timedelta(minutes=r["minutes"]) * (1 if "after" in r else -1)
        if WEEKDAYS[due.weekday()] not in r["days"]:
            continue
        channels = None if r["page"] and r["telegram"] else ["telegram"] if r["telegram"] else ["page", "cli"]
        out.append({"key": f"{dd['date']}:rem:{r['name']}", "kind": "devotion", "name": r["name"], "at": due,
                    "due": due, "until": due + timedelta(seconds=REMINDER_LATE_SEC), "channels": channels,
                    "reminder": r})
    return sorted(out, key=lambda i: i["due"])


def _raised_texts(item: dict, cfg: dict, ts: float) -> tuple[str, dict]:
    """(label, {"en", "bn"}) of an item at the moment it's raised."""
    kind, name, at = item["kind"], item["name"], item["at"]
    if kind == "prayer":
        mins = max(1, math.ceil((at.timestamp() - max(ts, item["due"].timestamp())) / 60))
        return (f"{_name(name, 'en')} in {mins} minute{'s' if mins != 1 else ''}",
                {"en": f"Mahadi, {_name(name, 'en')} is in {mins} minute{'s' if mins != 1 else ''}, at {_clock(at)}.",
                 "bn": f"মাহাদি, {_bn(mins)} মিনিট পর {_name(name, 'bn')}, {_clock_bn(at, True)}।"})
    if kind == "prayer_start":
        tasbih = cfg["tasbih"] if cfg["at_start"]["tasbih"] and name != "Jumuah" else {"en": "", "bn": ""}
        return (f"{_name(name, 'en')} time",
                {"en": f"It's {_name(name, 'en')} time ({_clock(at)}). {tasbih['en']}".strip(),
                 "bn": f"{OF_BN.get(name, _name(name, 'bn'))} ওয়াক্ত শুরু হয়েছে ({_clock_bn(at)})। "
                       f"{tasbih['bn']}".strip()})
    if kind == "brief":
        return "Morning brief", brief(date.fromisoformat(item["key"][:10]))
    r, on = item["reminder"], item["due"].date()
    return r["name"], {lang: _fill(r["text"][lang], on, lang, cfg) for lang in ("en", "bn")}


_state_lock = threading.Lock()


def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
        if isinstance(s, dict) and isinstance(s.get("raised"), dict):
            return s
    except (OSError, ValueError):
        pass
    return {"raised": {}}


def _save_state(s: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_FILE)


def tick(now: float | None = None) -> list[dict]:
    """Raise what has come due (the watches thread, every TICK_SEC): each item once (data/prayers_state.json), up
    to LEAD_SEC early so the page rings it on its minute, and never once its time is over. Returns the alerts."""
    ts = _time() if now is None else now
    cfg = config()
    today = datetime.fromtimestamp(ts, _zone()).date()
    raised = []
    with _state_lock:
        state = _load_state()
        for d in (today - timedelta(days=1), today, today + timedelta(days=1)):
            for item in _planned(d, cfg):
                due, until = item["due"].timestamp(), item["until"].timestamp()
                if item["key"] in state["raised"] or ts < due - LEAD_SEC or ts >= until:
                    continue
                try:
                    label, texts = _raised_texts(item, cfg, ts)
                except Exception as e:      # one broken text mustn't stop the rest
                    print(f"[prayers] {item['key']}: {type(e).__name__}: {str(e)[:80]}")
                    continue
                raised.append(timers.add_alert(label, texts, kind=item["kind"], due=max(due, ts), until=until,
                                               channels=item["channels"]))
                state["raised"][item["key"]] = ts
                print(f"[prayers] raised {item['key']} for {datetime.fromtimestamp(max(due, ts)):%H:%M}")
        if raised:
            state["raised"] = {k: v for k, v in state["raised"].items() if v >= ts - KEEP_DAYS * 86400}
            _save_state(state)
    return raised


# ---------- the morning brief ----------
def brief(d: date) -> dict:
    """Day d's morning brief in both languages: the date and the Hijri date, the prayer times, the weather, the
    calendar, the open tasks and the morning adhkar (and Friday's line). Built here: no model sees it."""
    cfg = config()
    dd, h = times_for(d), hijri(d)
    weather_line, cal, tasks = _weather_lines(cfg), _calendar_lines(d), _task_lines()
    out = {}
    for lang in ("en", "bn"):
        head = (f"শুভ সকাল, মাহাদি। {_date_text(d, 'bn')}" if lang == "bn" else
                f"Good morning, Mahadi. {_date_text(d, 'en')}") + (f" · {hijri_text(h, lang)}" if h else "")
        if dd:
            times = " · ".join(f"{_cap(_name(n, lang))} {_short(at, lang)}" for n, at in _day_times(dd, cfg))
            times = f"{_city(cfg, lang)}: {times}"
        else:
            times = "নামাজের সময় এখন পাওয়া যাচ্ছে না।" if lang == "bn" else "Prayer times: unavailable right now."
        lines = [head, f"🕌 {times}", f"🌤 {weather_line[lang]}", f"📅 {cal[lang]}", f"✅ {tasks[lang]}"]
        if dd and cfg["morning_adhkar"][lang]:
            lines.append(f"🤲 {_fill(cfg['morning_adhkar'][lang], d, lang, cfg)}")
        if dd and _friday(dd) and cfg["friday"][lang]:
            lines.append(f"📖 {_fill(cfg['friday'][lang], d, lang, cfg)}")
        out[lang] = "\n".join(lines)
    return out


def _deg_bn(x) -> str:
    return "অজানা" if x is None else _bn(f"{round(x)}°")


def _weather_lines(cfg: dict) -> dict:
    try:
        loc = weather.locate(f"{cfg['city']}, {cfg['country']}")
        fc = weather.forecast(loc, 6)
    except Exception:
        return {"en": "Weather: unavailable right now.", "bn": "আবহাওয়া: এখন পাওয়া যাচ্ছে না।"}
    local_now = datetime.fromtimestamp(_time(), ZoneInfo(loc["tz"])).replace(tzinfo=None)
    today = next((x for x in fc["days"] if x["date"] == local_now.date()), None)
    now = fc["now"]
    en = f"Weather: {weather._deg(now['temp'])} now, {now['desc'] or 'no sky report'}"
    bn = f"আবহাওয়া: এখন {_deg_bn(now['temp'])}"
    if today:
        en += (f"; high {weather._deg(today['high'])}, low {weather._deg(today['low'])}; "
               f"{weather._rain(today, fc['hours'], local_now)}")
        bn += f", আজ সর্বোচ্চ {_deg_bn(today['high'])}, সর্বনিম্ন {_deg_bn(today['low'])}"
        if today.get("rain_chance") is not None:
            bn += f", বৃষ্টির সম্ভাবনা {_bn(round(today['rain_chance']))}%"
    return {"en": f"{en} ({fc['source']}).", "bn": f"{bn} ({fc['source']})।"}


def _calendar_lines(d: date) -> dict:
    try:
        if calendar_tools.status() != "connected":
            if os.path.exists(calendar_tools.TOKEN_FILE):
                return {"en": "Calendar: I couldn't read it, Google's sign-in has expired. Reconnect it from the "
                              "Dashboard.",
                        "bn": "ক্যালেন্ডার: পড়তে পারিনি, Google-এর সাইন-ইনের মেয়াদ শেষ। ড্যাশবোর্ড থেকে আবার যুক্ত "
                              "করো।"}
            return {"en": "Calendar: not connected (connect it from the Dashboard).",
                    "bn": "ক্যালেন্ডার: যুক্ত করা নেই (ড্যাশবোর্ড থেকে যুক্ত করো)।"}
        events = calendar_tools._fetch(*calendar_tools._day_bounds(d))
    except Exception:
        return {"en": "Calendar: unavailable right now.", "bn": "ক্যালেন্ডার: এখন পাওয়া যাচ্ছে না।"}
    if not events:
        return {"en": "Calendar: nothing scheduled today.", "bn": "ক্যালেন্ডার: আজ কিছু নেই।"}
    en, bn = [], []
    for ev in events[:6]:
        start, _, all_day = calendar_tools._event_times(ev)
        title = " ".join(str(ev.get("summary") or "(no title)").split())[:60]
        en.append(f"{'all day' if all_day else calendar_tools._fmt_time(start)} {title}")
        bn.append(f"{'সারাদিন' if all_day else _short(start, 'bn')} {title}")
    more = len(events) - len(en)
    return {"en": f"Calendar: {'; '.join(en)}" + (f"; and {more} more" if more > 0 else "") + ".",
            "bn": f"ক্যালেন্ডার: {'; '.join(bn)}" + (f"; আরও {_bn(more)}টি" if more > 0 else "") + "।"}


def _task_lines() -> dict:
    import javid    # here, not at the top: javid imports this module
    pending = [re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*", "", t) for t in javid._read_lines()
               if not t.startswith("✓")]
    if not pending:
        return {"en": "Tasks: none open.", "bn": "কাজ: কিছু বাকি নেই।"}
    shown = "; ".join(pending[:5])
    more = len(pending) - min(len(pending), 5)
    return {"en": f"Tasks: {len(pending)} open: {shown}" + (f"; and {more} more" if more else "") + ".",
            "bn": f"কাজ: {_bn(len(pending))}টি বাকি: {shown}" + (f"; আরও {_bn(more)}টি" if more else "") + "।"}


# ---------- Telegram's /prayers and /reminders ----------
def telegram_times(lang: str) -> str:
    now = _now()
    dd = times_for(now.date())
    if not dd:
        return _no_times(lang)
    nxt = _next(now, config())
    head = _next_said(*nxt, now, lang) if nxt else ""
    return "\n\n".join(filter(None, [head, _card(dd, now, config(), lang)]))


def telegram_schedule(lang: str) -> str:
    """What goes out in the next 24 hours, and where."""
    cfg, now = config(), _now()
    items = [i for d in (now.date(), now.date() + timedelta(days=1)) for i in _planned(d, cfg)
             if now < i["due"] < now + timedelta(hours=24)]
    bn = lang == "bn"
    if not items:
        return ("পরের ২৪ ঘণ্টায় কোনো রিমাইন্ডার নেই; সেগুলো config/prayers.yaml-এ।" if bn else
                "No reminders in the next 24 hours. They're set in config/prayers.yaml.")
    lines = ["পরের ২৪ ঘণ্টার রিমাইন্ডার:" if bn else "Reminders in the next 24 hours:"]
    for i in items:
        name, mins = _name(i["name"], lang), cfg["heads_up"]["minutes"]
        if i["kind"] == "prayer":
            what = f"{_bn(mins)} মিনিট পর {name}" if bn else f"{name} in {mins} minutes"
        elif i["kind"] == "prayer_start":
            what = f"{name} শুরু" if bn else f"{name} begins"
        elif i["kind"] == "brief":
            what = "সকালের ব্রিফ" if bn else "Morning brief"
        else:
            what = i["name"]
        everywhere = i["channels"] is None
        where = ("পেজ আর টেলিগ্রাম" if everywhere else "টেলিগ্রাম") if bn else \
            ("page and Telegram" if everywhere else "Telegram")
        later = (" (কাল)" if bn else " (tomorrow)") if i["due"].date() != now.date() else ""
        lines.append(f"- {_short(i['due'], lang)}{later}  {what} · {where}")
    lines.append("config/prayers.yaml-এ বদলানো যায়।" if bn else "Change them in config/prayers.yaml.")
    return "\n".join(lines)


# ---------- keeping the Mac awake ----------
_spawn = lambda args: subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,   # noqa: E731
                                       stderr=subprocess.DEVNULL)


def keep_awake(pid: int | None = None) -> bool:
    """While the server runs and the Mac is on power, it doesn't idle to sleep (macOS `caffeinate -s -w <pid>`:
    released when the server stops; on battery, nothing), so Fajr's messages go out. Closing the lid still sleeps
    it. True if it was asked for."""
    if not config()["keep_awake"] or sys.platform != "darwin" or not shutil.which("caffeinate"):
        return False
    try:
        _spawn(["caffeinate", "-s", "-w", str(pid or os.getpid())])
    except OSError as e:
        print(f"[prayers] couldn't keep the Mac awake: {e}")
        return False
    return True
