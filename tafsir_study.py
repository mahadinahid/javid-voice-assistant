"""
The tafsir study plan and the tadabbur notebook (2026-11-24), for understanding what Mahadi has memorised, starting
with what he recites in salah. All of it code: no model request.

The path: Juz Amma first, from An-Nas back to An-Naba, then Al-Fatiha, then Al-Baqarah onward (PATH). A day's
passage is daily.min to daily.max ayahs (5 to 10, config/tafsir.yaml): it ends where the boundary tafsir's own passage
ends when that is known on this Mac (Ibn Kathir abridged: Al-Mulk 1-5), else at one of Tanzil's rukus, else evenly;
it stays within its surah, except that short surahs join the ones after them up to daily.max (Al-Ikhlas and
Al-Masad). Today's passages (`target` a day) are fixed once worked out, so the morning nudge, the Today card and
"today's tafsir" agree all day. Progress is the ayahs studied, per surah and per juz; "continue where I left off" is
the first one not studied, in the path's order. The streak counts the days he studied; a day missed only pauses it,
and nothing ever says so in a reproachful way.

The nudge rides the watches thread like hifz.py's: `nudge` minutes after Fajr (35 by default, after hifz's at 25),
once a day, while today's passage waits, on the page (its own banner and a soft chime; spoken only if `spoken`) and
on Telegram (with "Show the tafsir" and "Studied" buttons). When a juz is due for hifz revision, the Today card and
the hifz reminder offer its tafsir too (hifz_offer).

The notebook (data/tadabbur.json, 0600, gitignored, never sent to a model): for a passage, a reflection, a question,
an action, tags and a mark (studied or revisit); searchable, and exported as Markdown. It holds his own words and the
passage's reference, never a tafsir's text.
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

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "tafsir.yaml")
STUDY_FILE = os.path.join("data", "tafsir_study.json")      # what he studied, today's passages
NOTES_FILE = os.path.join("data", "tadabbur.json")           # his notebook
STATE_FILE = os.path.join("data", "tafsir_reminders.json")   # the nudges raised
RELOAD_SEC = 5
LEAD_SEC = 20
LATE_SEC = 5 * 3600          # the morning nudge rings until about Dhuhr
KEEP_DAYS = 3
MAX_TEXT = 4000              # a note's field
MAX_TAGS = 12
TAG_SUGGESTIONS = ("tawakkul", "akhirah", "dua", "stories of prophets", "sabr", "shukr", "tawbah")
PATH = list(range(114, 77, -1)) + [1] + list(range(2, 78))
_time = time.time
_lock = threading.RLock()


# ---------- settings: config/tafsir.yaml ----------
def _defaults() -> dict:
    return {"daily": {"min": 5, "max": 10}, "target": 1, "nudge": {"after": "Fajr", "minutes": 35}, "page": True,
            "telegram": True, "spoken": False, "boundary": 169, "prefetch_days": 3, "hifz_link": True}


def _validate(raw) -> tuple[dict, list[str]]:
    import hifz
    import tafsir
    cfg, problems = _defaults(), []
    if raw is None:
        return cfg, problems
    if not isinstance(raw, dict):
        return cfg, ["it isn't a list of settings"]
    for k, v in raw.items():
        try:
            if k == "daily":
                if not isinstance(v, dict) or set(v) - {"min", "max"}:
                    raise ValueError("min: and max: ayahs")
                d = {**cfg["daily"], **v}
                if not all(isinstance(d[x], int) and not isinstance(d[x], bool) for x in ("min", "max")) or \
                        not 1 <= d["min"] <= d["max"] <= 40:
                    raise ValueError("min and max are whole numbers, 1 ≤ min ≤ max ≤ 40")
                v = d
            elif k == "target":
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 5:
                    raise ValueError("1 to 5 passages a day")
            elif k == "nudge":
                v = hifz._when(v, "nudge")
            elif k in ("page", "telegram", "spoken", "hifz_link"):
                if not isinstance(v, bool):
                    raise ValueError("true or false")
            elif k == "boundary":
                if v not in tafsir.TAFSIRS:
                    raise ValueError(f"one of the tafsirs' ids: {', '.join(map(str, tafsir.TAFSIRS))}")
            elif k == "prefetch_days":
                if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 6:
                    raise ValueError("0 to 6 (Quran.com's text is kept six days)")
            else:
                raise ValueError("not a setting")
            cfg[k] = v
        except ValueError as e:
            problems.append(f"{k}: {e}")
    return cfg, problems


_cfg = {"cfg": None, "mtime": None, "checked": 0.0, "problems": []}


def config() -> dict:
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
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError:
            raw = None
        except yaml.YAMLError as e:
            _cfg["problems"] = [f"config/tafsir.yaml can't be read: {str(e).splitlines()[0]}"]
            return _cfg["cfg"] or _defaults()
        cfg, problems = _validate(raw)
        if problems and problems != _cfg["problems"]:
            print(f"[tafsir] config/tafsir.yaml: {'; '.join(problems)} (those settings aren't used)")
        _cfg.update(cfg=cfg, mtime=mtime, problems=problems)
        return cfg


def reload() -> None:
    with _lock:
        _cfg.update(cfg=None, mtime=None, checked=0.0)


def problems() -> list[str]:
    config()
    return list(_cfg["problems"])


# ---------- files ----------
def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{threading.get_ident()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, type(default)) else default
    except (OSError, ValueError):
        return default


def _study() -> dict:
    s = _read_json(STUDY_FILE, {})
    s.setdefault("studied", [])
    s.setdefault("today", {})
    return s


def _today() -> date:
    return datetime.fromtimestamp(_time(), prayers._zone()).date()


# ---------- passages ----------
def _key(i: int) -> str:
    return quran.key(i)


def _idx(k: str) -> int:
    return quran.index(*map(int, k.split(":")))


def studied_set(entries: list | None = None) -> set[int]:
    out = set()
    for e in entries if entries is not None else _study()["studied"]:
        out.update(range(e["lo"], e["hi"] + 1))
    return out


def _boundary_ends(s: int, cfg: dict) -> set[int]:
    """Where the boundary tafsir's passages end in surah s, from what's on this Mac (no request)."""
    import tafsir
    try:
        data = tafsir._surah_cache(cfg["boundary"], s)
    except Exception:
        return set()
    return {_idx(g["keys"][-1]) for g in data["groups"] if g.get("keys")}


def _ruku_ends(lo: int, hi: int) -> set[int]:
    return {r - 1 for r in quran.RUKU_START if lo < r <= hi + 1}


def _passage(ranges: list[tuple[int, int]], done: set[int] | None = None) -> dict:
    lo, hi = min(r[0] for r in ranges), max(r[1] for r in ranges)
    ayahs = sum(b - a + 1 for a, b in ranges)
    if len(ranges) > 1:         # short surahs joined: named in the path's order ("Al-Ikhlas and Al-Masad")
        ns = [quran.verse(a)[0] for a, _b in ranges]
        label, label_bn = quran.join([quran.name(n) for n in ns], "en"), quran.join([quran.name(n, "bn") for n in ns], "bn")
    else:
        label, label_bn = quran.span_text(lo, hi), quran.span_text(lo, hi, "bn")
    out = {"first": _key(lo), "last": _key(hi), "ranges": [list(r) for r in ranges], "ayahs": ayahs,
           "label": label, "label_bn": label_bn,
           "view": {"kind": "range", "first": _key(lo), "last": _key(hi)}}
    if done is not None:
        out["done"] = all(i in done for a, b in ranges for i in range(a, b + 1))
    return out


def _from(start: int, taken: set[int], cfg: dict) -> list[tuple[int, int]]:
    """The passage that starts at ayah `start` (an index): to a boundary within daily.min..max, never into an ayah
    already studied (or taken), never past its surah but for short surahs joined in the path's order."""
    mn, mx = cfg["daily"]["min"], cfg["daily"]["max"]
    s = quran.verse(start)[0]
    lo, hi = quran.surah_range(s)
    stop = next((i for i in range(start, hi + 1) if i in taken), hi + 1) - 1      # before an ayah already studied
    rem = stop - start + 1
    if start == lo and stop == hi and rem < mn:          # a short surah: whole, with the next ones in the path
        ranges, total, joined = [(lo, hi)], rem, [s]
        for n in PATH[PATH.index(s) + 1:]:
            nlo, nhi = quran.surah_range(n)
            if n not in (min(joined) - 1, max(joined) + 1) or total + (nhi - nlo + 1) > mx or \
                    any(i in taken for i in range(nlo, nhi + 1)):
                break
            ranges.append((nlo, nhi))
            joined.append(n)
            total += nhi - nlo + 1
        return ranges
    if rem <= mx:
        return [(start, stop)]
    target = (mn + mx) / 2
    rukus = _ruku_ends(start, stop)
    seg = min([e for e in rukus | {stop} if e >= start + mx - 1] or [stop])     # balanced within its ruku
    seg_len = seg - start + 1
    parts = max(1, round(seg_len / target))
    if math.ceil(seg_len / parts) > mx:
        parts = math.ceil(seg_len / mx)
    size = min(mx, max(mn, math.ceil(seg_len / parts)))
    window = [e for e in range(start + mn - 1, start + mx) if stop - e == 0 or stop - e >= mn]
    cands = [e for e in window if e in _boundary_ends(s, cfg)]
    if cands:                   # the boundary tafsir's own passage (or two short ones together): the first that fits
        return [(start, min(cands))]
    cands = [e for e in window if e in rukus]
    if cands:
        return [(start, min(cands, key=lambda x: (abs((x - start + 1) - size), x)))]
    e = start + size - 1
    if 0 < stop - e < mn:
        e = start + math.ceil(rem / 2) - 1 if rem < mn + mx else stop - mn
    return [(start, e)]


def next_passage(taken: set[int] | None = None, cfg: dict | None = None) -> dict | None:
    """The first passage not studied, in the path's order (None: the whole Quran is studied)."""
    cfg = cfg or config()
    taken = studied_set() if taken is None else taken
    for s in PATH:
        lo, hi = quran.surah_range(s)
        start = next((i for i in range(lo, hi + 1) if i not in taken), None)
        if start is not None:
            return _passage(_from(start, taken, cfg))
    return None


def passage_of(p: dict) -> dict:
    """{first, last} as a passage (its label, whether it's studied)."""
    return _passage([(_idx(p["first"]), _idx(p["last"]))], studied_set())


def passage_at(key: str, cfg: dict | None = None) -> dict:
    """The study passage holding ayah `key`: today's if it's in one, else its surah's passage as the plan cuts the
    surah from its start (the same passages whatever was studied before)."""
    cfg = cfg or config()
    i = _idx(key)
    done = studied_set()
    for p in today():
        if any(a <= i <= b for a, b in p["ranges"]):
            return p
    lo, hi = quran.surah_range(quran.verse(i)[0])
    start = lo
    while start <= hi:
        r = _from(start, set(), cfg)
        if any(a <= i <= b for a, b in r):
            return _passage(r, done)
        start = r[0][1] + 1
    return _passage([(i, i)], done)


def today(d: date | None = None) -> list[dict]:
    """Today's passages (`target`), worked out once a day and kept, each with done: studied or not."""
    d = d or _today()
    with _lock:
        s = _study()
        done = studied_set(s["studied"])
        t = s["today"]
        if t.get("date") == d.isoformat() and t.get("passages"):
            return [_passage([tuple(r) for r in p["ranges"]], done) for p in t["passages"]]
        cfg = config()
        taken, out = set(done), []
        for _ in range(cfg["target"]):
            p = next_passage(taken, cfg)
            if not p:
                break
            out.append(p)
            for a, b in p["ranges"]:
                taken.update(range(a, b + 1))
        s["today"] = {"date": d.isoformat(), "passages": [{"ranges": p["ranges"]} for p in out]}
        _write_json(STUDY_FILE, s)
        return [_passage([tuple(r) for r in p["ranges"]], done) for p in out]


def mark(first: str, last: str, studied: bool = True, via: str = "page", d: date | None = None) -> dict:
    """Studied (or not after all): ayahs first..last on day d (today)."""
    lo, hi = sorted((_idx(first), _idx(last)))
    if hi - lo > 300:
        raise ValueError("at most 300 ayahs at a time")
    with _lock:
        s = _study()
        if studied:
            s["studied"].append({"lo": lo, "hi": hi, "date": (d or _today()).isoformat(), "at": _time(), "via": via})
        else:
            keep = []
            for e in s["studied"]:
                if e["hi"] < lo or e["lo"] > hi:
                    keep.append(e)
                    continue
                if e["lo"] < lo:
                    keep.append({**e, "hi": lo - 1})
                if e["hi"] > hi:
                    keep.append({**e, "lo": hi + 1})
            s["studied"] = keep
        _write_json(STUDY_FILE, s)
    return plan_json()


def streak(entries: list | None = None, d: date | None = None) -> dict:
    """{days: how many days he studied in all, status: today | ready (yesterday was a study day) | paused | new}. A
    missed day pauses it; nothing resets it."""
    entries = _study()["studied"] if entries is None else entries
    d = d or _today()
    days = sorted({e["date"] for e in entries})
    status = "new" if not days else "today" if d.isoformat() in days else \
        "ready" if (d - timedelta(days=1)).isoformat() in days else "paused"
    return {"days": len(days), "status": status, "last": days[-1] if days else None}


def progress(done: set[int] | None = None) -> dict:
    done = studied_set() if done is None else done
    surahs = []
    for n in range(1, 115):
        lo, hi = quran.surah_range(n)
        k = sum(1 for i in range(lo, hi + 1) if i in done)
        surahs.append({"n": n, "studied": k, "total": hi - lo + 1})
    juz = []
    for n in range(1, 31):
        lo, hi = quran.unit_range("juz", n)
        juz.append({"n": n, "studied": sum(1 for i in range(lo, hi + 1) if i in done), "total": hi - lo + 1})
    amma = juz[29]
    path_done = 0
    for s in PATH:
        if surahs[s - 1]["studied"] == surahs[s - 1]["total"]:
            path_done += 1
        else:
            break
    return {"surahs": surahs, "juz": juz, "juz_amma": {"studied": amma["studied"], "total": amma["total"]},
            "overall": {"studied": len(done), "total": quran.TOTAL}, "surahs_done_in_order": path_done}


def hifz_offer(d: date | None = None) -> list[dict]:
    """When hifz.py has a juz (or manzil) due today: the first passage of it not studied, to study its tafsir too."""
    if not config()["hifz_link"]:
        return []
    import hifz
    try:
        found = hifz.today_portions()
    except Exception:
        return []
    taken, out = studied_set(), []
    for p in found:
        lo, hi = p["ranges"][0][0], p["ranges"][-1][1]
        start = next((i for i in range(lo, hi + 1) if i not in taken), None)
        if start is None:
            continue
        ranges = [(a, min(b, hi)) for a, b in _from(start, taken, config())]
        label = quran.portion_label(p) if p.get("kind") else quran.span_text(lo, hi)
        out.append({"portion": label, "portion_bn": quran.portion_label(p, "bn") if p.get("kind") else "",
                    "passage": _passage(ranges, taken)})
    return out


# ---------- the page ----------
def plan_json() -> dict:
    cfg = config()
    s = _study()
    done = studied_set(s["studied"])
    t = today()
    nxt = next_passage({*done, *(i for p in t for a, b in p["ranges"] for i in range(a, b + 1))}, cfg)
    pr = progress(done)
    return {"today": t, "next": nxt, "continue": next_passage(done, cfg), "streak": streak(s["studied"]),
            "progress": {"juz_amma": pr["juz_amma"], "overall": pr["overall"],
                         "surahs": [x for x in pr["surahs"] if x["studied"]],
                         "juz": [x for x in pr["juz"] if x["studied"]]},
            "hifz": hifz_offer(), "revisit": [n for n in notes() if n["mark"] == "revisit"][:12],
            "daily": cfg["daily"], "target": cfg["target"], "nudge": cfg["nudge"], "problems": problems(),
            "recent": sorted(s["studied"], key=lambda e: -e["at"])[:6]}


def surah_progress(n: int) -> dict:
    lo, hi = quran.surah_range(n)
    done = studied_set()
    return {"n": n, "studied": [_key(i) for i in range(lo, hi + 1) if i in done]}


# ---------- the morning nudge (the watches thread) ----------
def planned(d: date, cfg: dict | None = None) -> list[dict]:
    import hifz
    cfg = cfg or config()
    if not cfg["nudge"] or not (cfg["page"] or cfg["telegram"]):
        return []
    at = hifz._at(d, cfg["nudge"])
    return [{"key": f"{d.isoformat()}:morning", "due": at, "until": at + timedelta(seconds=LATE_SEC)}]


def nudge_texts(passages: list[dict], st: dict) -> tuple[str, dict]:
    what_en = quran.join([p["label"] for p in passages], "en")
    what_bn = quran.join([p["label_bn"] for p in passages], "bn")
    en = f"Today's tafsir: {what_en}."
    bn = f"আজকের তাফসীর: {what_bn}।"
    if st["days"] and st["status"] != "paused":
        en += f" Day {st['days'] + 1} of your study."
        bn += f" তোমার পড়ার {quran.bn_digits(st['days'] + 1)}তম দিন।"
    return f"Today's tafsir: {what_en}", {"en": en, "bn": bn}


def tick(now: float | None = None) -> list[dict]:
    """Raise the morning nudge once a day (data/tafsir_reminders.json), LEAD_SEC early, never after its `until`,
    and not when today's passages are already studied."""
    ts = _time() if now is None else now
    cfg = config()
    channels = [c for c, on in (("page", cfg["page"]), ("cli", cfg["page"]), ("telegram", cfg["telegram"])) if on]
    d = datetime.fromtimestamp(ts, prayers._zone()).date()
    raised = []
    with _lock:
        state = _read_json(STATE_FILE, {})
        state.setdefault("raised", {})
        changed = False
        for item in planned(d, cfg):
            due, until = item["due"].timestamp(), item["until"].timestamp()
            if item["key"] in state["raised"] or ts < due - LEAD_SEC or ts >= until:
                continue
            state["raised"][item["key"]] = ts
            changed = True
            ps = today(d)
            waiting = [p for p in ps if not p["done"]]
            if not waiting:
                print(f"[tafsir] {item['key']}: today's tafsir is studied, no nudge")
                continue
            label, texts = nudge_texts(waiting, streak(d=d))
            raised.append(timers.add_alert(label, texts, kind="tafsir", due=max(due, ts), until=until, channels=channels,
                                           extra={"tafsir": {"date": d.isoformat(), "spoken": cfg["spoken"],
                                                             "passages": [{k: p[k] for k in ("first", "last", "label",
                                                                                              "label_bn", "ranges")}
                                                                          for p in waiting]}}))
            print(f"[tafsir] raised {item['key']} for {datetime.fromtimestamp(max(due, ts)):%H:%M}")
        if changed:
            keep = ts - KEEP_DAYS * 86400
            state["raised"] = {k: v for k, v in state["raised"].items() if v >= keep}
            _write_json(STATE_FILE, state)
    return raised


def from_alert(alert_id: str) -> dict | None:
    t = timers.get(alert_id)
    return ((t or {}).get("extra") or {}).get("tafsir") if (t or {}).get("kind") == "tafsir" else None


def studied_from_alert(alert_id: str, via: str = "telegram") -> str:
    info = from_alert(alert_id)
    if not info:
        return "That reminder has gone; mark it on the Hifz page instead."
    with _lock:
        state = _read_json(STATE_FILE, {})
        state.setdefault("done", {})
        if alert_id in state["done"]:
            return "Already marked studied."
        for p in info["passages"]:
            mark(p["first"], p["last"], True, via, date.fromisoformat(info["date"]))
        state["done"][alert_id] = _time()
        _write_json(STATE_FILE, state)
    return f"Marked studied: {quran.join([p['label'] for p in info['passages']], 'en')}."


def telegram_texts(alert_id: str, lang: str, limit: int = 3800) -> list[str]:
    """The nudge's "Show the tafsir": the first chosen tafsir's text for each passage, with its name, author and link,
    in as few messages as fit (from this Mac when it's kept)."""
    import tafsir
    info = from_alert(alert_id)
    if not info:
        return []
    out = []
    tid = tafsir.chosen(lang)[0]
    for p in info["passages"]:
        lo, hi = _idx(p["first"]), _idx(p["last"])
        i, seen = lo, set()
        while i <= hi:
            k = _key(i)
            try:
                got = tafsir.passage(tid, k, lang)
            except (LookupError, tafsir.Unavailable, ValueError) as e:
                out.append(f"{tafsir.TAFSIRS[tid]['name']}: {k}: {e}")
                i += 1
                continue
            if tuple(got["keys"]) not in seen:
                seen.add(tuple(got["keys"]))
                text = tafsir.plain([b for sec in got["sections"] for b in sec["blocks"]])
                head = f"📖 {got['tafsir']['name']} · {got['tafsir']['author']} · Quran.com\n{got['label']}\n\n"
                body = text if len(head) + len(text) <= limit else text[:limit - len(head) - 60].rsplit(" ", 1)[0] + " …"
                out.append(f"{head}{body}\n\n{got['link']}")
            i = max(i + 1, _idx(got["last"]) + 1)
    return out


# ---------- prefetch: the next days' passages, so today's opens offline ----------
_prefetching = threading.Event()


def prefetch(days: int | None = None, langs: tuple = ("en", "bn")) -> int:
    """Fetch the next `days` days' passages (the boundary tafsir first, so the passages end where it ends), in the
    tafsirs he reads and the compare pair, with their words and the surah's introduction. One request at a time;
    stops when Quran.com doesn't answer. Returns how many passages it fetched."""
    import tafsir
    if _prefetching.is_set():
        return 0
    _prefetching.set()
    try:
        cfg = config()
        days = cfg["prefetch_days"] if days is None else days
        taken, n = studied_set(), 0
        ids = list(dict.fromkeys([cfg["boundary"]] + [x for lang in langs for x in tafsir.chosen(lang)] +
                                 [x for lang in langs for x in tafsir.COMPARE[lang]]))
        for _ in range(days * cfg["target"]):
            nxt = next_passage(taken, cfg)
            if not nxt:
                break
            start = _idx(nxt["first"])
            window = range(start, min(start + cfg["daily"]["max"], quran.surah_range(quran.verse(start)[0])[1] + 1))
            try:
                _cover(cfg["boundary"], window)
                p = next_passage(taken, cfg)        # worked out again, now with the boundary tafsir's passages
                keys = [_key(i) for a, b in p["ranges"] for i in range(a, b + 1)]
                for tid in ids:
                    _cover(tid, [_idx(k) for k in keys])
                for a, b in p["ranges"]:
                    s, a1 = quran.verse(a)
                    tafsir.words(s, a1, quran.verse(b)[1])
                    tafsir.surah_info(s)
            except tafsir.Unavailable:
                break
            for a, b in p["ranges"]:
                taken.update(range(a, b + 1))
            n += 1
        return n
    finally:
        _prefetching.clear()


def _cover(tid: int, idxs) -> None:
    import tafsir
    covered = set()
    for i in idxs:
        if i in covered:
            continue
        try:
            g = tafsir.group(tid, _key(i))
        except KeyError:
            return
        if g:
            covered.update(_idx(k) for k in g["keys"])


def prefetch_soon() -> None:
    threading.Thread(target=prefetch, name="tafsir-prefetch", daemon=True).start()


# ---------- the notebook ----------
_TAG = re.compile(r"[\w][\w '\-’]{0,31}", re.U)


def _notes() -> list[dict]:
    return _read_json(NOTES_FILE, {}).get("notes", []) if isinstance(_read_json(NOTES_FILE, {}), dict) else []


def _is_name(n: dict) -> bool:
    """A note on one of Allah's names (asma.py's tab) rather than on a passage."""
    return "name" in n


def _name_of(n: dict) -> dict | None:
    import asma
    return asma.get(n["name"])


def notes(q: str = "", tag: str = "") -> list[dict]:
    """His notes: the passages' in the Quran's order, then the names' in the list's order; q: words in any of them
    (or a surah's name, or a name's meaning or transliteration), tag: one tag."""
    passages = sorted((n for n in _notes() if not _is_name(n)), key=lambda n: (_idx(n["first"]), _idx(n["last"])))
    on_names = sorted((n for n in _notes() if _is_name(n) and _name_of(n)), key=lambda n: _name_of(n)["n"])
    out = passages + on_names
    if tag:
        t = tag.strip().lower()
        out = [n for n in out if t in n["tags"]]
    if q.strip():
        words = [w for w in quran.tokens(q) if w]
        def text(n):
            mine = [n["reflection"], n["question"], n["action"], " ".join(n["tags"])]
            if _is_name(n):
                import asma
                e = _name_of(n)
                return " ".join(mine + [e["arabic"], *e["transliteration"].values(), asma._lat(e["transliteration"].get("en", "")),
                                        *e["meaning"].values(), str(e["n"]), "name names asma"]).lower()
            s = int(n["first"].split(":")[0])
            return " ".join(mine + [quran.name(s), quran.name(s, "bn"), quran.surah(s)["ar"]] +
                            list(quran.surah(s).get("aka", []))).lower()
        out = [n for n in out if all(w in text(n) or any(w in tg for tg in n["tags"]) for w in words)]
    return [_note_public(n) for n in out]


def _note_public(n: dict) -> dict:
    if _is_name(n):
        import asma
        e = _name_of(n)
        return {**n, "label": asma.label(e), "label_bn": asma.label(e, "bn"), "n": e["n"], "arabic": e["arabic"]}
    lo, hi = _idx(n["first"]), _idx(n["last"])
    return {**n, "label": quran.span_text(lo, hi), "label_bn": quran.span_text(lo, hi, "bn")}


def _clean_tags(v) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > MAX_TAGS:
        raise ValueError(f"tags: a list of up to {MAX_TAGS}")
    out = []
    for t in v:
        t = " ".join(str(t).split()).strip().lower()
        if not t:
            continue
        if not _TAG.fullmatch(t):
            raise ValueError(f"a tag is a word or a few, up to 32 letters: {t[:20]!r}")
        if t not in out:
            out.append(t)
    return out


def _fields(body: dict) -> dict:
    fields = {}
    for k in ("reflection", "question", "action"):
        v = body.get(k, "")
        if not isinstance(v, str) or len(v) > MAX_TEXT:
            raise ValueError(f"{k}: text up to {MAX_TEXT} characters")
        fields[k] = v.strip()
    fields["tags"] = _clean_tags(body.get("tags"))
    return fields


def _write_notes(all_notes: list[dict]) -> None:
    data = _read_json(NOTES_FILE, {})
    data = data if isinstance(data, dict) else {}
    _write_json(NOTES_FILE, {**data, "notes": all_notes})       # whatever else the file holds stays


def save_note(body: dict) -> dict:
    """Add or change the note on a passage (first..last): reflection, question, action, tags, mark; or, with `name`
    (one of Allah's names, asma.py), the note on that name (no mark: a name's learning marks are asma.py's)."""
    if not isinstance(body, dict):
        raise ValueError("a note is a JSON object")
    if "name" in body:
        return save_name_note(body)
    try:
        first, last = str(body["first"]), str(body["last"])
        lo, hi = _idx(first), _idx(last)
    except (KeyError, ValueError, TypeError, IndexError):
        raise ValueError("first and last: ayahs like 67:1") from None
    if lo > hi or hi - lo > 300 or not (0 <= lo < quran.TOTAL and hi < quran.TOTAL):
        raise ValueError("first and last in order, at most 300 ayahs")
    if quran.key(lo) != first or quran.key(hi) != last:
        raise ValueError("those aren't ayahs")
    fields = _fields(body)
    m = body.get("mark", "") or ""
    if m not in ("", "studied", "revisit"):
        raise ValueError("mark: studied, revisit or nothing")
    fields["mark"] = m
    with _lock:
        all_notes = _notes()
        n = next((x for x in all_notes if not _is_name(x) and x["first"] == first and x["last"] == last), None)
        was = n["mark"] if n else ""
        if n is None:
            n = {"id": secrets.token_hex(5), "first": first, "last": last, "created": _time()}
            all_notes.append(n)
        n.update(fields, updated=_time())
        if not any(n[k] for k in ("reflection", "question", "action", "tags", "mark")):
            all_notes = [x for x in all_notes if x is not n]
        _write_notes(all_notes)
    if m == "studied" and was != "studied":
        mark(first, last, True, "note")
    return _note_public(n)


def save_name_note(body: dict) -> dict:
    """The note on one of Allah's names: his reflection, a question, an action, tags. An empty one is removed."""
    import asma
    e = asma.get(str(body.get("name") or ""))
    if not e:
        raise ValueError("name: one of the names' ids, like al-wadud")
    if body.get("mark"):
        raise ValueError("a name's note has no mark: learned and reviewing are on its card")
    fields = _fields(body)
    with _lock:
        all_notes = _notes()
        n = next((x for x in all_notes if _is_name(x) and x["name"] == e["id"]), None)
        if n is None:
            n = {"id": secrets.token_hex(5), "name": e["id"], "created": _time()}
            all_notes.append(n)
        n.update(fields, mark="", updated=_time())
        if not any(n[k] for k in ("reflection", "question", "action", "tags")):
            all_notes = [x for x in all_notes if x is not n]
        _write_notes(all_notes)
    return _note_public(n)


def delete_note(nid: str) -> bool:
    with _lock:
        all_notes = _notes()
        keep = [n for n in all_notes if n["id"] != nid]
        if len(keep) == len(all_notes):
            return False
        _write_notes(keep)
        return True


def note_for(first: str, last: str) -> dict | None:
    return next((_note_public(n) for n in _notes() if not _is_name(n) and n["first"] == first and n["last"] == last), None)


def note_for_name(nid: str) -> dict | None:
    return next((_note_public(n) for n in _notes() if _is_name(n) and n["name"] == nid and _name_of(n)), None)


def export_md(found: list[dict] | None = None, q: str = "", tag: str = "") -> str:
    """The notebook as Markdown: by surah, each passage with its link on Quran.com, its tags and mark, his words."""
    found = notes(q, tag) if found is None else found
    lines = ["# Tadabbur notes", "", f"Exported from Javid on {_today().isoformat()}" +
             (f" · search: {q}" if q else "") + (f" · tag: {tag}" if tag else "") + f" · {len(found)} notes", ""]
    surah = None
    for n in found:
        if _is_name(n):
            continue
        s = int(n["first"].split(":")[0])
        if s != surah:
            surah = s
            lines += [f"## {quran.surah(s)['n']}. {quran.name(s)} ({quran.surah(s)['ar']})", ""]
        a1, a2 = n["first"].split(":")[1], n["last"].split(":")[1]
        lines.append(f"### {n['label']}")
        meta = [f"[Quran.com](https://quran.com/{s}/{a1}" + (f"-{a2}" if a2 != a1 or n["first"].split(':')[0] != n["last"].split(':')[0] else "") + ")"]
        if n["tags"]:
            meta.append("tags: " + ", ".join(n["tags"]))
        if n["mark"]:
            meta.append({"studied": "✓ studied", "revisit": "↻ revisit"}[n["mark"]])
        meta.append("updated " + datetime.fromtimestamp(n["updated"], prayers._zone()).strftime("%Y-%m-%d"))
        lines += [" · ".join(meta), ""]
        lines += _note_body(n)
    named = [n for n in found if _is_name(n)]
    if named:
        lines += ["## The names of Allah (Asma ul-Husna)", ""]
        for n in named:
            lines.append(f"### {n['n']}. {n['arabic']} · {n['label'].split(': ', 1)[1]}")
            meta = (["tags: " + ", ".join(n["tags"])] if n["tags"] else []) + \
                ["updated " + datetime.fromtimestamp(n["updated"], prayers._zone()).strftime("%Y-%m-%d")]
            lines += [" · ".join(meta), ""] + _note_body(n)
    return "\n".join(lines).rstrip() + "\n"


def _note_body(n: dict) -> list[str]:
    out = []
    for k, word in (("reflection", "Reflection"), ("question", "Question"), ("action", "Action")):
        if n[k]:
            body = n[k].replace("\n", "\n> ")
            out += [f"**{word}**", "", f"> {body}", ""]
    return out


# ---------- answers (tafsir.answer hands these over) ----------
NOTE_FILLER = quran._nfc({"show", "me", "my", "the", "notes", "note", "on", "about", "tagged", "tag", "with", "all",
                          "tadabbur", "reflection", "reflections", "study", "find", "search", "list", "open", "any",
                          "do", "i", "have", "what", "are", "for", "in", "of", "please", "javid", "আমার", "নোট",
                          "নোটগুলো", "দেখাও", "সম্পর্কে", "নিয়ে", "নিয়ে", "তাদাব্বুর", "কী", "কি", "আছে", "সব", "খোলো"})


def notes_answer(topic: str, lang: str):
    """"Show my notes on tawakkul": by tag first, else by words. Personal: answered here, never by a model."""
    bn = lang == "bn"
    found = notes(tag=topic) if topic and notes(tag=topic) else notes(q=topic)
    yield ("tool", "tafsir_notes")
    yield ("tafsir", {"notes": {"q": topic}})
    if not found:
        said = (f"“{topic}” নিয়ে তোমার কোনো নোট নেই।" if topic else "তোমার নোটবুক এখনো খালি।") if bn else \
               (f"You have no notes on “{topic}” yet." if topic else "Your notebook is empty so far.")
        return said, ""
    where = quran.join([n["label_bn" if bn else "label"] for n in found[:3]], lang)
    more = len(found) - 3
    if bn:
        said = f"“{topic}” নিয়ে তোমার {quran.bn_digits(len(found))}টি নোট: {where}" if topic else \
            f"তোমার {quran.bn_digits(len(found))}টি নোট: {where}"
        said += (f", আরও {quran.bn_digits(more)}টি" if more > 0 else "") + "। পর্দায় আছে।"
    else:
        said = (f"You have {len(found)} note{'s' if len(found) != 1 else ''} on “{topic}”: {where}" if topic else
                f"You have {len(found)} note{'s' if len(found) != 1 else ''}: {where}") + \
               (f", and {more} more" if more > 0 else "") + ". They're on your screen."
    lines = []
    for n in found[:12]:
        lines.append(f"{n['label_bn' if bn else 'label']}:" + (f" [{', '.join(n['tags'])}]" if n["tags"] else "") +
                     (" ↻" if n["mark"] == "revisit" else " ✓" if n["mark"] == "studied" else ""))
        for k, w in (("reflection", ("Reflection", "ভাবনা")), ("question", ("Question", "প্রশ্ন")),
                     ("action", ("Action", "করণীয়"))):
            if n[k]:
                lines.append(f"• {w[1] if bn else w[0]}: {n[k][:300]}")
    return said, "\n".join(lines)


def today_answer(lang: str, via: str):
    bn = lang == "bn"
    ps = today()
    st = streak()
    if not ps:
        return ("পুরো কুরআনের তাফসীর পড়া হয়ে গেছে, আলহামদুলিল্লাহ।" if bn else
                "You've studied the tafsir of the whole Quran, Alhamdulillah."), ""
    waiting = [p for p in ps if not p["done"]]
    show = waiting[0] if waiting else ps[0]
    yield ("tafsir", {"open": {"key": show["first"], "passage": {"first": show["first"], "last": show["last"]},
                               "study": True}})
    what = quran.join([p["label_bn" if bn else "label"] for p in (waiting or ps)], lang)
    if waiting:
        said = (f"আজকের তাফসীর: {what}।" + (" স্টাডি প্যানেলে খোলা হলো।" if via == "page" else "")) if bn else \
               (f"Today's tafsir: {what}." + (" It's open in the study panel." if via == "page" else ""))
    else:
        nxt = plan_json()["next"]
        said = (f"আজকের তাফসীর পড়া শেষ, মাশাআল্লাহ। এরপর: {nxt['label_bn']}।" if nxt else "আজকের তাফসীর পড়া শেষ।") if bn else \
               (f"Today's tafsir is done, MashaAllah. Next: {nxt['label']}." if nxt else "Today's tafsir is done.")
    if st["days"]:
        said += (f" এ পর্যন্ত {quran.bn_digits(st['days'])} দিন পড়েছ।" if bn else
                 f" {st['days']} day{'s' if st['days'] != 1 else ''} of study so far.")
    return said, ""


def studied_answer(key: str | None, lang: str, passage: dict | None = None):
    bn = lang == "bn"
    p = passage or (passage_at(key) if key else None)
    if not p:
        return ("কোন অংশ? রিডারে সেটা খোলো।" if bn else "Which passage? Open it in the reader first."), ""
    mark(p["first"], p["last"], True, "voice")
    yield ("tafsir", {"refresh": True})
    st = streak()
    return ((f"{p['label_bn']} পড়া হয়েছে বলে রাখলাম।" + (f" {quran.bn_digits(st['days'])} দিন হলো।" if st["days"] else ""))
            if bn else (f"Marked {p['label']} as studied." + (f" That's {st['days']} day{'s' if st['days'] != 1 else ''} of study."
                                                              if st["days"] else ""))), ""
