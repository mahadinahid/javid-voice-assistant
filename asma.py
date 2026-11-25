"""
Asma ul-Husna (2026-11-25): Allah's names on the Hifz page, to learn, understand and live by them ("whoever counts
them enters Paradise": Bukhari 2736, Muslim 2677), from sound sources only. All of it code: no model request.

The list is Mahadi's approved data/asma_ul_husna.yaml (gitignored, 0600; built and checked 2026-11-25, see its
`sources`): Tirmidhi 3507's order as the main list (Allah first, then 98 names, as that hadith's text has them),
each name marked with Ibn Uthaymeen's list (al-Qawa'id al-Muthla) or as only in one of them, the Arabic verbatim
from the hadith, the transliteration and meaning in English and Bangla from its two editions, the Quran's
occurrences (references only), where Sa'di explains it (a place only: his words are Quran.com's, shown live, kept
six days at most), and how to use it: a dua or dhikr with it from a hadith graded sahih or hasan (its reference and
grading), a Quranic dua by reference, an entry of his own approved Dua & Zikr list, and always "call upon Allah by
it in your own dua" (Al-A'raf 7:180).
  - Nothing is written from a model, and nothing beyond the yaml: _entry checks every name (its Arabic, both
    meanings, a source for each), every use (a hadith's grading sahih or hasan, a count or a promised reward only
    inside a sound hadith's own use), and drops a use whose text came out damaged (the hadith data's U+FFFD).
  - Javid's voices (Guy, Pradeep) never say a name: what's spoken gives its number and its meaning, "its card is on
    your screen", never the transliteration or the Arabic (speakable). The model's replies are checked too
    (reply_problem): a meaning, count or benefit it gives for a name is rewritten once, then dropped.
  - Learning: the name of the day (the list in order from the first day, one a day, the day turning at Fajr: it
    rides the hifz reminder after Fajr), flashcards both ways, and "learned / reviewing" marks spaced as hifz.py
    spaces its juz: a learned name comes back after a cycle (99 / `daily`), stalest first, `daily` a day; one still
    reviewing comes back `weak_days` after it was marked, until it's marked learned.
  - Reflections per name go in his Tadabbur notebook (tafsir_study: data/tadabbur.json), searchable and exported
    with his passage notes. His marks are data/asma_progress.json, his settings data/asma_settings.json; all 0600.
No audio: neither Quran.com nor another verified source publishes a recitation of the names (checked 2026-11-25),
so none is played; a Quranic dua's ayah plays from the Hifz player's reciters like any ayah.
"""
import json
import math
import os
import re
import secrets
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta

import yaml

import prayers
import quran

LIST_FILE = os.path.join("data", "asma_ul_husna.yaml")
PROGRESS_FILE = os.path.join("data", "asma_progress.json")
SETTINGS_FILE = os.path.join("data", "asma_settings.json")
SAADI = 91                      # Tafsir as-Sa'di on Quran.com, where the yaml's explanations are
RELOAD_SEC = 2
COUNT = 99
NEW_IN_DECK = 3                 # names not marked yet that today's flashcards add after what's due
MAX_EXPLAIN_WORDS = 30          # Sa'di's explanation is cut at his first stop, at most this many words
FALLBACK_FAJR = "05:00"         # the day turns here when the prayer times can't be had
DEFAULTS = {"daily": 5, "weak_days": 2, "morning": True, "meaning": "auto"}
MARKS = ("learned", "reviewing", "new")
USE_KINDS = ("your own dua", "Quranic dua", "from your Dua & Zikr list", "dua (You are …)", "dua (calling: yā …)",
             "dhikr")
HADITH_KINDS = ("dua (You are …)", "dua (calling: yā …)", "dhikr")
_time = time.time               # the clock (regression.py moves it with prayers._time and timers._now)
_lock = threading.RLock()
_list = {"names": [], "by_id": {}, "by_n": {}, "problems": [], "dropped": [], "note": {}, "uthaymeen_only": [],
         "status": "", "stamp": None, "checked": 0.0}


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


# ---------- the list: data/asma_ul_husna.yaml, checked ----------
_AR = re.compile("[\u0600-\u06FF]")
_DAMAGED = "\ufffd"
_SOUND = re.compile(r"^\s*(?:sahih|hasan)\b", re.I)
_UNSOUND = re.compile(r"da'?[iī]f|\bdaif\b|\bweak\b|mawdu|munkar|fabricat|shadh", re.I)
_SOUND_REF = re.compile(r"Sahih al-Bukhari|Sahih Muslim|\((?:Sahih|Hasan)\b[^)]*\)", re.I)
# A count of recitations ("×3", "100x", "33 times", "১০০ বার"), and a promised result: only inside a sound hadith's use.
_COUNT = re.compile(r"\b\d+\s*(?:x|×|times?)(?![a-z])|[×]\s*\d+|\b(?:once|twice|thrice)\b|[0-9০-৯]+\s*বার\b", re.I)
_BENEFIT = re.compile(r"\b(?:benefits?|rewards?|virtues?|whoever (?:says|recites|reads)|will (?:be granted|receive|get|"
                      r"enter|be forgiven|be protected)|cures?|protection from|guaranteed)\b|ফজিলত|ফযীলত|উপকার", re.I)
_KEY = re.compile(r"(\d{1,3}):(\d{1,3})")


def _valid_key(k) -> bool:
    m = _KEY.fullmatch(str(k))
    return bool(m) and 1 <= int(m[1]) <= 114 and 1 <= int(m[2]) <= quran.SURAHS[int(m[1]) - 1]["ayahs"]


def _text(v) -> str:
    return " ".join(str(v).split()) if v else ""


def _texts(v) -> dict:
    v = v if isinstance(v, dict) else {}
    return {lang: _text(v[lang]) for lang in ("en", "bn") if v.get(lang)}


def _refs(v) -> list[str]:
    return [_text(x) for x in (v if isinstance(v, list) else [v] if v else []) if _text(x)]


def _links(v) -> list[str]:
    return [x for x in _refs(v) if re.fullmatch(r"https://[a-z0-9.-]+/\S*", x)]


def sound_grade(grade: str) -> bool:
    """A hadith's grading as the yaml records it: "Sahih (in Sahih al-Bukhari)", "Sahih (Al-Albani)", "Hasan Sahih
    (Al-Albani)", "Hasan (Al-Albani)" are sound; a Da'if or a missing grading isn't."""
    return bool(grade) and bool(_SOUND.match(grade)) and not _UNSOUND.search(grade)


def _use(u, n: int) -> tuple[dict | None, str]:
    """One "how to use this name", checked: its kind, its reference, a hadith's sound grading, a count or a promised
    reward only where a sound hadith gives it (the own-dua and Quranic lines never; a hadith's and a Dua & Zikr entry's
    are sound), no damaged text. (use, why it was left out)."""
    if not isinstance(u, dict):
        return None, "not a mapping"
    kind = u.get("kind")
    if kind not in USE_KINDS:
        return None, f"unknown kind {kind!r}"
    refs = _refs(u.get("reference"))
    if not refs:
        return None, f"{kind} without a reference"
    texts = [_text(u.get(k)) for k in ("arabic", "title", "note", "en", "bn")]
    if any(_DAMAGED in t for t in texts + [_text(u.get("hadith_en")), _text(u.get("hadith_bn"))]):
        return None, f"{refs[0]}: its text came out damaged in the source data (U+FFFD)"
    out = {"kind": kind, "reference": refs, "link": _links(u.get("link"))[:1] and _links(u.get("link"))[0] or None}
    counted = any(_COUNT.search(t) for t in texts[1:])
    promised = any(_BENEFIT.search(t) for t in texts[1:])
    if kind == "your own dua":
        if u.get("key") != "7:180" or not _texts({"en": u.get("en"), "bn": u.get("bn")}).keys() >= {"en", "bn"}:
            return None, "the own-dua line is 7:180, in English and Bangla"
        if counted or promised:
            return None, "the own-dua line gives a count or a benefit"
        out.update(key="7:180", en=_text(u["en"]), bn=_text(u["bn"]))
    elif kind == "Quranic dua":
        if not _valid_key(u.get("key")):
            return None, f"{u.get('key')!r} isn't an ayah"
        if counted or promised:
            return None, f"{refs[0]}: a Quranic dua's line gives a count or a benefit"
        out.update(key=str(u["key"]))
    elif kind == "from your Dua & Zikr list":
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", str(u.get("id") or "")):
            return None, "no id of a Dua & Zikr item"
        if not any(_SOUND_REF.search(r) for r in refs):
            return None, f"{u.get('id')}: none of its references is in Bukhari or Muslim or graded sahih or hasan"
        if not _text(u.get("arabic")):
            return None, f"{u.get('id')}: no Arabic"
        out.update(id=str(u["id"]), title=_text(u.get("title")), arabic=_text(u["arabic"]),
                   arabic_from=_text(u.get("arabic_from")), sound=True)
    else:
        grade = _text(u.get("grade"))
        if not sound_grade(grade):
            return None, f"{refs[0]}: graded {grade or 'nothing'}, not sahih or hasan"
        if not _text(u.get("arabic")) or not _text(u.get("arabic_from")):
            return None, f"{refs[0]}: no Arabic, or no source for it"
        out.update(grade=grade, arabic=_text(u["arabic"]), arabic_from=_text(u["arabic_from"]),
                   hadith=_texts({"en": u.get("hadith_en"), "bn": u.get("hadith_bn")}),
                   text_from={str(k): _text(v) for k, v in (u.get("text_from") or {}).items()}
                   if isinstance(u.get("text_from"), dict) else {}, sound=True)
    out["note"] = _text(u.get("note")) or None
    out["counted"] = counted or promised        # a sound hadith's own number or reward, said in its own words
    return out, ""


def _entry(x) -> tuple[dict | None, list[str], list[str]]:
    """One name, checked: (the name, why it's left out, the uses left out). A name needs its number, id, Arabic,
    both meanings and a source for each, and at least the own-dua use."""
    if not isinstance(x, dict):
        return None, ["not a mapping"], []
    n = x.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= COUNT:
        return None, [f"number {n!r} isn't 1 to {COUNT}"], []
    nid = str(x.get("id") or "")
    bad = []
    if not re.fullmatch(r"[a-z0-9][a-z0-9'-]{0,63}", nid):
        bad.append("its id must be a-z, 0-9, ' and -")
    arabic = _text(x.get("arabic"))
    if not arabic or not _AR.search(arabic) or _DAMAGED in arabic:
        bad.append("no Arabic")
    if not _text(x.get("arabic_from")):
        bad.append("no source for its Arabic")
    tl, mean = _texts(x.get("transliteration")), _texts(x.get("meaning"))
    if set(mean) != {"en", "bn"}:
        bad.append("needs its meaning in English and Bangla")
    if "en" not in tl:
        bad.append("no transliteration")
    src = x.get("text_from") if isinstance(x.get("text_from"), dict) else {}
    need = [f"meaning_{lang}" for lang in mean] + [f"transliteration_{lang}" for lang in tl]
    if missing := [k for k in need if not _text(src.get(k))]:
        bad.append(f"no source for {', '.join(missing)}")
    if any(_COUNT.search(t) for t in mean.values()):        # ("the one who benefits" is An-Nafi's meaning)
        bad.append("a count in its meaning")
    if bad:
        return None, bad, []
    uses, dropped = [], []
    for u in x.get("use") or []:
        got, why = _use(u, n)
        if got:
            uses.append(got)
        else:
            dropped.append(why)
    if not any(u["kind"] == "your own dua" for u in uses):
        return None, ["no own-dua line (7:180)"], dropped
    lists = x.get("lists") if isinstance(x.get("lists"), dict) else {}
    uth = lists.get("uthaymeen") if isinstance(lists.get("uthaymeen"), dict) else None
    q = x.get("quran") if isinstance(x.get("quran"), dict) else {}
    definite = [str(k) for k in q.get("definite") or [] if _valid_key(k)]
    indefinite = [str(k) for k in q.get("indefinite") or [] if _valid_key(k)]
    if len(definite) != len(q.get("definite") or []) or len(indefinite) != len(q.get("indefinite") or []):
        dropped.append("a Quran reference that isn't an ayah")
    e = x.get("explanation") if isinstance(x.get("explanation"), dict) else None
    explanation = None
    if e and _valid_key(e.get("key")) and _text(e.get("quoted")):
        explanation = {"source": _text(e.get("source")), "key": str(e["key"]), "quoted": str(e["quoted"]).strip(),
                       "link": _links(e.get("link"))[0] if _links(e.get("link")) else None,
                       "with": [str(w) for w in e.get("explains_with") or []]}
    return {
        "n": n, "id": nid, "arabic": arabic, "arabic_from": _text(x["arabic_from"]), "transliteration": tl,
        "meaning": mean, "text_from": {str(k): _text(v) for k, v in src.items()},
        "lists": {"tirmidhi_3507": lists.get("tirmidhi_3507"),
                  "uthaymeen": uth and {"list": uth.get("list"), "as_written": _text(uth.get("as_written")),
                                        "his_source": _text(uth.get("his_source")) or None},
                  "only_in": lists.get("only_in")},
        "quran": {"definite": definite, "indefinite": indefinite,
                  "count": q.get("count") if isinstance(q.get("count"), int) else len(definite) + len(indefinite),
                  "count_words": q.get("count_words"), "count_ayahs": q.get("count_ayahs"),
                  "key": str(q["key"]) if _valid_key(q.get("key")) else None, "note": _text(q.get("note")) or None},
        "explanation": explanation, "use": uses, "links": _links(x.get("links")),
        "status": x.get("status") if x.get("status") in ("verified", "check") else "check",
        "checks": _refs(x.get("checks")), "notes": _refs(x.get("notes")),
    }, [], dropped


def _validate(raw) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("names"), list):
        return {"names": [], "problems": ["data/asma_ul_husna.yaml has no list of names"], "dropped": []}
    names, problems, dropped, seen_n, seen_id, seen_ar = [], [], [], set(), set(), set()
    for i, x in enumerate(raw["names"], 1):
        e, why, gone = _entry(x)
        label = f"name {x.get('n', i) if isinstance(x, dict) else i}"
        dropped += [f"{label}: {g}" for g in gone]
        if why:
            problems.append(f"{label}: {'; '.join(why)}")
            continue
        skel = _ar_full(e["arabic"])
        if e["n"] in seen_n or e["id"] in seen_id or skel in seen_ar:
            problems.append(f"{label}: its number, id or Arabic is there twice")
            continue
        seen_n.add(e["n"])
        seen_id.add(e["id"])
        seen_ar.add(skel)
        names.append(e)
    names.sort(key=lambda e: e["n"])
    if len(names) != COUNT:
        problems.append(f"{len(names)} names, not {COUNT}")
    note = raw.get("note") if isinstance(raw.get("note"), dict) else {}
    uth = [{"arabic": _text(u.get("arabic_as_written")), "list": u.get("list"), "his_no": u.get("his_no"),
            "his_source": _text(u.get("his_source")) or None}
           for u in raw.get("uthaymeen_only") or [] if isinstance(u, dict) and _text(u.get("arabic_as_written"))]
    return {"names": names, "problems": problems, "dropped": dropped, "note": _note(note), "uthaymeen_only": uth,
            "status": _text(raw.get("status")), "sources": raw.get("sources") if isinstance(raw.get("sources"), dict) else {}}


def _note(note: dict) -> dict:
    """The page's note on the list: the sahih hadith (each one's own words, from its number on: the Bengali edition
    of Bukhari 2736 carries the chapter's report before it), the weak grading of Tirmidhi 3507, Ibn Uthaymeen."""
    sahih = []
    for h in note.get("sahih") or []:
        if not isinstance(h, dict) or not _text(h.get("reference")):
            continue
        bn = _text(h.get("bn"))
        num = re.search(r"(\d+)$", _text(h["reference"]))
        if num and (at := bn.find(quran.bn_digits(num[1]) + ".")) > 0:
            bn = bn[at:]
        sahih.append({"reference": _text(h["reference"]), "link": _links(h.get("link"))[0] if _links(h.get("link")) else None,
                      "grade": _text(h.get("grade")), "en": _text(h.get("en")), "bn": bn})
    lst = note.get("list") if isinstance(note.get("list"), dict) else {}
    uth = note.get("uthaymeen") if isinstance(note.get("uthaymeen"), dict) else {}
    return {"sahih": sahih,
            "list": {"reference": _text(lst.get("reference")), "link": _links(lst.get("link"))[0] if _links(lst.get("link")) else None,
                     "grades": _refs(lst.get("grades")), "tirmidhi_said": _text(lst.get("tirmidhi_said")),
                     "en": _text(lst.get("en")), "bn": _text(lst.get("bn"))},
            "uthaymeen": {k: _text(uth.get(k)) for k in ("not_fixed", "ihsa", "gathered", "hesitation", "idafa")
                          if uth.get(k)} | {"source": uth.get("source") if isinstance(uth.get("source"), dict) else {}}}


def _load() -> dict:
    """The list as the file has it (read again when the file changes); a file that can't be read keeps the last good
    list, with the reason in problems()."""
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
            _list.update(names=[], by_id={}, by_n={}, problems=["There's no data/asma_ul_husna.yaml."], dropped=[],
                         note={}, uthaymeen_only=[], status="")
            return _list
        try:
            with open(LIST_FILE, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as e:
            _list["problems"] = [f"data/asma_ul_husna.yaml can't be read ({type(e).__name__}); the last good list is used."]
            return _list
        v = _validate(raw)
        _list.update(v, by_id={e["id"]: e for e in v["names"]}, by_n={e["n"]: e for e in v["names"]})
        _index()
        return _list


def reload() -> None:
    with _lock:
        _list.update(stamp=None, checked=0.0)


def names() -> list[dict]:
    return list(_load()["names"])


def get(nid) -> dict | None:
    """A name by its id ("al-wadud") or its number."""
    L = _load()
    if isinstance(nid, int) or (isinstance(nid, str) and nid.isdigit()):
        return L["by_n"].get(int(nid))
    return L["by_id"].get(str(nid))


def problems() -> list[str]:
    return list(_load()["problems"])


def approved() -> bool:
    return _load()["status"].lower().startswith("approved")


# ---------- matching a name in any script ----------
ARTICLES = {"al", "ar", "as", "ash", "at", "ad", "adh", "az", "an", "ath", "ul", "el", "ya", "yaa"}
_JOINERS = {"al", "ul", "il", "wal", "wa", "l"}
_BN_ARTICLES = ("আল", "আর", "আস", "আশ", "আত", "আদ", "আয", "আয়", "আন", "আয্", "ইয়া", "ইয়া")
_BN_SWAP = str.maketrans({"ী": "ি", "ূ": "ু", "ঈ": "ই", "ঊ": "উ", "ৃ": "ৃ", "\u200c": None, "\u200d": None, "-": None,
                          " ": None, "্": None, "়": None, "ঃ": None})


def _lat(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _phon(s: str) -> str:
    """A transliteration's sound, spelling-proof: Wadūd = wadood, Ḥafīẓ = hafidh = hafiz, Qawiyy = kawi."""
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("dh", "z"), ("q", "k"), ("ph", "f"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("ou", "u"),
                 ("uu", "u"), ("aa", "a"), ("o", "u"), ("e", "i")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return re.sub(r"(?<=u)w$|(?<=i)y$", "", s)


def lat_keys(text: str) -> set[str]:
    """The keys a Latin-script name gives: with its article and without ("arrahman", "rahman"), joiners dropped
    ("Dhul-Jalali wal-Ikram")."""
    toks = [t for t in re.split(r"[^a-z]+", _lat(text).replace("'", "").replace("ʿ", "").replace("ʾ", "")
                                .replace("’", "").replace("`", "")) if t]
    if not toks:
        return set()
    out = {_phon("".join(toks))}
    rest = toks[1:] if toks[0] in ARTICLES and len(toks) > 1 else toks
    out.add(_phon("".join(rest)))
    inner = [t for i, t in enumerate(rest) if i == 0 or t not in _JOINERS]
    out.add(_phon("".join(inner)))
    return {k for k in out if len(k) >= 2}


def _bn_key(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "")
    s = re.sub(r"[^\u0980-\u09FF\- ]", "", s).strip()
    for a in sorted(_BN_ARTICLES, key=len, reverse=True):
        if re.match(f"{a}[- ]", s):
            s = s[len(a):]
            break
    s = s.translate(_BN_SWAP)
    return re.sub("[ুি]$", "", s)


def _ar_key(text: str) -> str:
    import dua_check
    s = dua_check.skel(text)
    return s[1:] if s.startswith("ل") and len(s) > 2 else s       # ال without its alef is ل


def _ar_full(text: str) -> str:
    """The Arabic's letters with its alefs (al-Wālī isn't al-Waliyy), without harakat or al-."""
    import dua_check
    s = dua_check.norm(text).replace(" ", "")
    return s[2:] if s.startswith("ال") and len(s) > 3 else s


_idx = {"lat": {}, "bn": {}, "ar": {}, "ar_full": {}}


def _index() -> None:
    """Each key to the names it may be: two names can sound alike (al-Majīd and al-Mājid are both "majid")."""
    lat, bn, ar, full = {}, {}, {}, {}
    for e in _list["names"]:
        for k in lat_keys(e["transliteration"].get("en", "")) | (lat_keys("Allah") if e["n"] == 1 else set()):
            lat.setdefault(k, []).append(e["n"])
        for k in {_bn_key(e["transliteration"].get("bn", ""))} | ({_bn_key("আল্লাহ")} if e["n"] == 1 else set()):
            if k:
                bn.setdefault(k, []).append(e["n"])
        ar.setdefault(_ar_key(e["arabic"]), []).append(e["n"])
        full.setdefault(_ar_full(e["arabic"]), []).append(e["n"])
    _idx.update(lat=lat, bn=bn, ar=ar, ar_full=full)


def match_names(text: str) -> list[int]:
    """The names `text` may be (a transliteration in any spelling, its Arabic, its Bangla transliteration): one, two
    when they sound alike, or none."""
    _load()
    t = (text or "").strip()
    if not t:
        return []
    if _AR.search(t):
        return _idx["ar_full"].get(_ar_full(t)) or _idx["ar"].get(_ar_key(t)) or []
    if re.search("[\u0980-\u09FF]", t):
        return _idx["bn"].get(_bn_key(t)) or []
    for k in sorted(lat_keys(t), key=len, reverse=True):
        if k in _idx["lat"]:
            return _idx["lat"][k]
    return []


def match_name(text: str) -> int | None:
    found = match_names(text)
    return found[0] if found else None


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹০১২৩৪৫৬৭৮৯", "0123456789" * 3)


def search(q: str) -> list[int]:
    """The names a search box's words find, best first: a number in any digits (48, ٤٨, ৪৮), the Arabic (with or
    without its harakat and al-), the transliteration in any spelling or a beginning of it, the English or Bangla
    meaning, the Bangla transliteration."""
    L = _load()
    q = " ".join((q or "").split())
    if not q:
        return [e["n"] for e in L["names"]]
    digits = q.translate(_DIGITS)
    if re.fullmatch(r"\d{1,2}", digits):
        return [int(digits)] if int(digits) in L["by_n"] else []
    scored = []
    if _AR.search(q):
        k, kf = _ar_key(q), _ar_full(q)
        for e in L["names"]:
            if _ar_full(e["arabic"]) == kf:
                scored.append((0, e["n"]))
            elif _ar_key(e["arabic"]) == k:
                scored.append((1, e["n"]))
            elif k and k in _ar_key(e["arabic"]):
                scored.append((2, e["n"]))
    elif re.search("[\u0980-\u09FF]", q):
        k, raw = _bn_key(q), unicodedata.normalize("NFC", q)
        for e in L["names"]:
            tl = _bn_key(e["transliteration"].get("bn", ""))
            mean = unicodedata.normalize("NFC", e["meaning"].get("bn", ""))
            if k and tl == k:
                scored.append((0, e["n"]))
            elif k and tl.startswith(k):
                scored.append((1, e["n"]))
            elif raw in mean:
                scored.append((2, e["n"]))
    else:
        keys = lat_keys(q)
        words = [re.sub(r"(?:ing|ers|er|ness|s)$", "", w) if len(w) > 5 else w
                 for w in re.findall(r"[a-z]+", _lat(q)) if w not in ("the", "a", "an", "of", "one", "who")]
        for e in L["names"]:
            nk = lat_keys(e["transliteration"].get("en", ""))
            mean = _lat(e["meaning"].get("en", ""))
            if keys & nk:
                scored.append((0, e["n"]))
            elif any(len(k) >= 3 and any(x.startswith(k) for x in nk) for k in keys):
                scored.append((1, e["n"]))
            elif words and all(re.search(rf"\b{re.escape(w)}", mean) for w in words):
                scored.append((2, e["n"]))
    best = {}
    for rank, n in scored:
        best[n] = min(rank, best.get(n, 9))
    return sorted(best, key=lambda n: (best[n], n))


# ---------- what may be spoken ----------
def speakable(text: str) -> bool:
    """Javid's voices never say a name: no Arabic, and no word that is one of the names in transliteration (Bangla
    included). A name's meaning ("the Loving") is fine."""
    if _AR.search(text or ""):
        return False
    _load()
    toks = re.findall(r"[A-Za-zÀ-ž'ʿʾ’-]+", text or "")
    for i in range(len(toks)):
        for w in (3, 2, 1):
            chunk = " ".join(toks[i:i + w])
            if len(toks[i:i + w]) == w and _named(chunk):
                return False
    for tok in re.findall(r"[\u0980-\u09FF]+(?:[- ][\u0980-\u09FF]+)?", text or ""):
        if tok.split("-")[0].split(" ")[0] in _BN_ARTICLES and _idx["bn"].get(_bn_key(tok)):
            return False
    return True


def _named(chunk: str) -> bool:
    """A Latin chunk names one of the 99 with its article or "ya" ("Al-Wadud", "ar rahman", "Ya Latif")."""
    toks = [t for t in re.split(r"[^a-z]+", _lat(chunk).replace("'", "").replace("’", "")) if t]
    return len(toks) > 1 and toks[0] in ARTICLES and match_name(" ".join(toks)) is not None


# ---------- the day, the name of the day ----------
def _zone():
    return prayers._zone()


def _fajr(d: date) -> datetime:
    dd = prayers.times_for(d, wait=False)
    if dd:
        return prayers._at(dd, "Fajr")
    hh, mm = map(int, FALLBACK_FAJR.split(":"))
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=_zone())


def learning_day(ts: float | None = None) -> date:
    """The day the name of the day and the marks count in: it turns at Fajr (before it, it's still yesterday)."""
    now = datetime.fromtimestamp(_time() if ts is None else ts, _zone())
    return now.date() if now >= _fajr(now.date()) else now.date() - timedelta(days=1)


def _progress() -> dict:
    p = _read_json(PROGRESS_FILE, {})
    p.setdefault("log", [])
    return p


def _start(p: dict | None = None) -> date:
    """The rotation's first day: the day the names were first opened (kept, so it stays the same)."""
    with _lock:
        p = p or _progress()
        try:
            return date.fromisoformat(p["start"])
        except (KeyError, TypeError, ValueError):
            p["start"] = learning_day().isoformat()
            _write_json(PROGRESS_FILE, p)
            return date.fromisoformat(p["start"])


def name_of_day(d: date | None = None) -> dict | None:
    """The list in its order, one a day from the first day, round again after the 99th."""
    L = _load()["names"]
    if not L:
        return None
    d = d or learning_day()
    return L[(d - _start()).days % len(L)]


# ---------- settings ----------
def _check_settings(raw) -> tuple[dict, list[str]]:
    s, bad = dict(DEFAULTS), []
    if not isinstance(raw, dict):
        return s, ["settings are a JSON object"]
    for k, v in raw.items():
        if k == "daily":
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 33:
                bad.append("daily: 1 to 33 names a day")
                continue
        elif k == "weak_days":
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 14:
                bad.append("weak_days: 1 to 14 days")
                continue
        elif k == "morning":
            if not isinstance(v, bool):
                bad.append("morning: true or false")
                continue
        elif k == "meaning":
            if v not in ("auto", "en", "bn"):
                bad.append("meaning: auto, en or bn")
                continue
        else:
            bad.append(f"{k}: not a setting")
            continue
        s[k] = v
    return s, bad


def settings() -> dict:
    return _check_settings(_read_json(SETTINGS_FILE, {}))[0]


def save_settings(change: dict) -> tuple[dict, list[str]]:
    """Merge `change` into the settings; nothing is saved when any value is wrong."""
    with _lock:
        now = settings()
        if not isinstance(change, dict):
            return now, ["settings are a JSON object"]
        merged, bad = _check_settings({**now, **change})
        if bad:
            return now, bad
        _write_json(SETTINGS_FILE, merged)
        return merged, []


# ---------- learned / reviewing, spaced as hifz.py spaces its juz ----------
def mark(nid, what: str, via: str = "page") -> dict:
    """Mark a name learned, still reviewing, or new again (unmarked). KeyError: no such name; ValueError: no such mark."""
    e = get(nid)
    if not e:
        raise KeyError(nid)
    if what not in MARKS:
        raise ValueError(f"a mark is one of {', '.join(MARKS)}")
    with _lock:
        p = _progress()
        _start(p)
        p = _progress()
        entry = {"id": e["id"], "n": e["n"], "mark": what, "date": learning_day().isoformat(), "at": _time(), "via": via}
        p["log"].append(entry)
        _write_json(PROGRESS_FILE, p)
    return entry


def state(d: date | None = None, log: list | None = None, cfg: dict | None = None) -> dict:
    """Every name's mark and when it comes back, today's reviews and the names still reviewing that are due."""
    cfg = cfg or settings()
    d = d or learning_day()
    L = _load()["names"]
    log = _progress()["log"] if log is None else log
    entries = sorted((x for x in log if isinstance(x, dict) and x.get("date", "") <= d.isoformat()
                      and x.get("mark") in MARKS), key=lambda x: (x["date"], x.get("at", 0)))
    latest, before_today = {}, {}
    for x in entries:
        if x["date"] < d.isoformat():
            before_today[x["id"]] = x
        latest[x["id"]] = x
    cycle = math.ceil(max(1, len(L)) / cfg["daily"])
    per = {}
    for e in L:
        x = latest.get(e["id"])
        m = x["mark"] if x and x["mark"] != "new" else None
        last = date.fromisoformat(x["date"]) if m else None
        due = last + timedelta(days=cfg["weak_days"] if m == "reviewing" else cycle) if m else None
        per[e["id"]] = {"n": e["n"], "mark": m, "last": last, "due": due, "today": bool(x and x["date"] == d.isoformat()),
                        "marks": sum(1 for y in entries if y["id"] == e["id"] and y["mark"] != "new")}
    reviewed = sum(1 for nid, s in per.items() if s["today"] and s["mark"] and
                   (before_today.get(nid) or {}).get("mark") == "learned")       # "not yet" isn't a review
    learned_due = sorted((s for s in per.values() if s["mark"] == "learned" and s["due"] <= d and not s["today"]),
                         key=lambda s: (s["last"], s["n"]))
    review_today = learned_due[:max(0, cfg["daily"] - reviewed)]
    reviewing_due = sorted((s for s in per.values() if s["mark"] == "reviewing" and s["due"] <= d and not s["today"]),
                           key=lambda s: (s["last"], s["n"]))
    days = sorted({x["date"] for x in entries if x["mark"] != "new"})
    streak, run, prev = 0, 0, None
    for iso in days:
        o = date.fromisoformat(iso).toordinal()
        run = run + 1 if prev is not None and o == prev + 1 else 1
        prev = o
    if prev is not None and prev >= d.toordinal() - 1:
        streak = run
    return {"day": d, "cycle": cycle, "daily": cfg["daily"], "weak_days": cfg["weak_days"], "per": per,
            "review_today": [s["n"] for s in review_today], "reviewing_due": [s["n"] for s in reviewing_due],
            "behind": [s["n"] for s in learned_due if s["due"] < d],
            "learned": sum(1 for s in per.values() if s["mark"] == "learned"),
            "reviewing": sum(1 for s in per.values() if s["mark"] == "reviewing"),
            "new": sum(1 for s in per.values() if not s["mark"]), "reviewed_today": reviewed, "streak": streak}


def deck(which: str = "due", d: date | None = None) -> list[dict]:
    """Flashcards: "due" = the names still reviewing that are due, today's reviews, the name of the day if it isn't
    marked, then a few not marked yet (NEW_IN_DECK); or "all", "learned", "reviewing" in the list's order."""
    st = state(d)
    by_n = {s["n"]: s for s in st["per"].values()}
    if which == "all":
        return [{"n": n, "why": "all"} for n in sorted(by_n)]
    if which in ("learned", "reviewing"):
        return [{"n": n, "why": which} for n in sorted(by_n) if by_n[n]["mark"] == which]
    out, seen = [], set()
    def add(n, why):
        if n not in seen:
            seen.add(n)
            out.append({"n": n, "why": why})
    for n in st["reviewing_due"]:
        add(n, "reviewing")
    for n in st["review_today"]:
        add(n, "review")
    today = name_of_day(st["day"])
    if today and not by_n[today["n"]]["mark"] and not by_n[today["n"]]["today"]:
        add(today["n"], "today")
    fresh = [n for n in sorted(by_n) if not by_n[n]["mark"] and not by_n[n]["today"] and n not in seen]
    for n in fresh[:NEW_IN_DECK]:
        add(n, "new")
    return out


# ---------- Sa'di's explanation, live from Quran.com (six days at most), never kept here ----------
def _quotes_others(e: dict, quoted: str) -> bool:
    """Sa'di quotes the name with another, or with "Allah" (as at Adh-Dhariyat 58): his explanation then runs to his
    full stop, not his first comma, as it was cut for the review (2026-11-25)."""
    import dua_check
    own = _ar_full(e["arabic"])
    for t in quoted.replace("{", " ").replace("}", " ").split():
        if _ar_full(t) == own:
            continue
        if dua_check.skel(t).lstrip("وفب") in ("لله", "له") or any(n != e["n"] for n in match_names(t)):
            return True
    return False



def explanation(nid, fetch: bool = True) -> dict | None:
    """Sa'di's own words on the name where the yaml says he explains it: the text after the name he quotes, to his
    first stop (MAX_EXPLAIN_WORDS, ten more when he explains it with the names beside it), as Quran.com gives it.
    None when the yaml has no place; {"missing": True} when the text there doesn't quote it any more. Raises
    tafsir.Unavailable when Quran.com can't be reached and it isn't kept."""
    import tafsir
    e = get(nid)
    if not e or not e["explanation"]:
        return None
    x = e["explanation"]
    p = tafsir.passage(SAADI, x["key"], "ar", fetch=fetch)
    meta = tafsir.meta(SAADI)
    base = {"key": x["key"], "label": p["label_en"], "link": x["link"] or p["link"], "tafsir": meta["name_en"],
            "author": meta["author"], "source": "Quran.com", "fetched": p["fetched"], "with": x["with"]}
    for sec in p["sections"]:
        for b in sec["blocks"]:
            runs = b.get("runs") or []
            for i, (txt, kind) in enumerate(runs):
                if kind != "q" or txt.strip() != x["quoted"]:
                    continue
                after = ""
                for t2, k2 in runs[i + 1:]:
                    if k2 == "q":
                        break
                    after += t2
                after = after.replace("\u200f", "").strip()
                lead = ""
                m = re.match(r"^(أي\s*:\s*)", after)
                if m:
                    lead, after = m.group(1), after[m.end():]
                wide = _quotes_others(e, x["quoted"])
                stop = r"(?<=[.؛])" if wide else r"(?<=[.،؛])"
                cut = re.split(stop, after, maxsplit=1)[0].strip()
                words = cut.split()
                limit = MAX_EXPLAIN_WORDS + (10 if wide else 0)
                if len(words) > limit:
                    cut = " ".join(words[:limit]) + " …"
                if len(cut.split()) >= 2:
                    return {**base, "quoted": x["quoted"], "text": (lead + cut).strip()}
    return {**base, "quoted": x["quoted"], "missing": True}


# ---------- a Quranic dua's ayah: Quran.com's text (six days, duas.py's copies), Tanzil's on this Mac without it ----------
def ayah(key: str) -> dict | None:
    import duas
    seg = duas._segment(key, None)
    if not seg or seg["source"] != "Quran.com":
        prefetch([key])
        if not BACKGROUND:              # fetched in line (the suite): read it again
            seg = duas._segment(key, None) or seg
    return seg


_fetching = threading.Event()
BACKGROUND = True


def quran_keys() -> list[str]:
    return list(dict.fromkeys(u["key"] for e in names() for u in e["use"] if u["kind"] == "Quranic dua"))


def prefetch(keys: list[str] | None = None, background: bool | None = None) -> None:
    """Fetch the Quranic duas' ayahs Quran.com hasn't given in six days (server start, or one found missing)."""
    import duas
    background = BACKGROUND if background is None else background
    keys = keys if keys is not None else quran_keys()

    def run():
        try:
            for k in keys:
                if not duas._qc_saved(k) and not duas._qc_fetch(k):
                    break
        finally:
            _fetching.clear()
    if _fetching.is_set():
        return
    _fetching.set()
    if background:
        threading.Thread(target=run, name="asma-quran", daemon=True).start()
    else:
        run()


# ---------- the page (GET /asma, GET /asma/name/{id}) ----------
def _brief(e: dict, st: dict) -> dict:
    s = st["per"][e["id"]]
    sound = [u for u in e["use"] if u.get("sound") or u["kind"] == "Quranic dua"]
    return {"n": e["n"], "id": e["id"], "arabic": e["arabic"], "transliteration": e["transliteration"],
            "meaning": e["meaning"], "status": e["status"], "uthaymeen": (e["lists"]["uthaymeen"] or {}).get("list"),
            "only_in": e["lists"]["only_in"], "quran": e["quran"]["count"] if e["n"] != 1 else e["quran"]["count_ayahs"],
            "duas": len(sound), "explained": bool(e["explanation"]), "mark": s["mark"],
            "due": s["due"] and s["due"].isoformat(), "last": s["last"] and s["last"].isoformat(), "today": s["today"]}


def label(e: dict, lang: str = "en") -> str:
    """"Name 48: the Loving" (for the notebook and the voice: a number and a meaning, never the name)."""
    if lang == "bn":
        return f"নাম {quran.bn_digits(e['n'])}: {e['meaning']['bn']}"
    return f"Name {e['n']}: {e['meaning']['en']}"


def tab_json() -> dict:
    L = _load()
    st = state()
    today = name_of_day(st["day"])
    return {"names": [_brief(e, st) for e in L["names"]], "today": today and today["n"], "day": st["day"].isoformat(),
            "progress": {k: st[k] for k in ("learned", "reviewing", "new", "cycle", "daily", "weak_days", "streak",
                                             "reviewed_today")} | {"due": len(deck("due", st["day"])),
                                                                    "review_today": st["review_today"],
                                                                    "reviewing_due": st["reviewing_due"]},
            "settings": settings(), "note": L["note"], "uthaymeen_only": L["uthaymeen_only"],
            "approved": approved(), "problems": L["problems"], "dropped": L["dropped"]}


def name_json(nid) -> dict | None:
    """One name's card: everything the yaml has, the Quranic duas' text, its Dua & Zikr entries as his list has them,
    his mark and his note in the notebook. Sa'di's words come separately (GET /asma/explain/{id}: Quran.com)."""
    import duas
    import tafsir_study
    e = get(nid)
    if not e:
        return None
    st = state()
    uses = []
    for u in e["use"]:
        u = dict(u)
        if u["kind"] == "Quranic dua":
            seg = ayah(u["key"])
            u["text"] = seg and {"text": seg["text"], "source": seg["source"], "a": seg["a"]}
        elif u["kind"] == "from your Dua & Zikr list":
            item = duas.get(u["id"])
            u["item"] = item and {"title": item["title"], "transliteration": item["transliteration"],
                                  "meaning": item["meaning"], "status": item["status"], "count": item["count"]}
        uses.append(u)
    s = st["per"][e["id"]]
    return {**e, "use": uses, "mark": s["mark"], "due": s["due"] and s["due"].isoformat(),
            "last": s["last"] and s["last"].isoformat(), "marked_today": s["today"],
            "note": tafsir_study.note_for_name(e["id"]), "label": label(e), "label_bn": label(e, "bn")}


def morning_line(lang: str, d: date | None = None) -> str:
    """The name of the day for the reminder after Fajr (spoken: its number and meaning only, never the name)."""
    if not settings()["morning"]:
        return ""
    e = name_of_day(d)
    if not e:
        return ""
    if lang == "bn":
        return f" আজকের আল্লাহর নাম: {quran.bn_digits(e['n'])} নম্বর, “{e['meaning']['bn']}”, হিফজ পেজে।"
    return f" Today's name of Allah: number {e['n']}, “{e['meaning']['en']}”, on the Hifz page."


# ---------- asked by voice, answered in code ----------
_TODAY = re.compile(r"(?<![a-z])(?:today'?s|todays|this day'?s)\s+(?:name|asma)(?:\s+of\s+allah)?(?![a-z])|"
                    r"(?<![a-z])name\s+of\s+the\s+day(?![a-z])|(?<![a-z])(?:the\s+)?daily\s+name(?![a-z])|"
                    r"আজকের\s+(?:আল্লাহর\s+)?নাম|আজকের\s+আসমা", re.I)
_MEANING = re.compile(r"(?<![a-z])(?:mean|means|meaning|meanings|explain|explanation|translate|translation|"
                      r"what\s+is|what's|whats|who\s+is|tell\s+me\s+about|about)(?![a-z])|মানে|অর্থ|ব্যাখ্যা|কী|কি", re.I)
_DUA = re.compile(r"(?<![a-z])(?:du'?aa?'?s?|duas|doa|dowa|supplications?|dhikr|zikr|zikir|adhkar|azkar|invoke|"
                  r"call\s+(?:up)?on)(?![a-z])|দোয়া|দোআ|দুআ|জিকির|যিকির", re.I)
_OPEN = re.compile(r"(?<![a-z])(?:asma'?\s*-?\s*ul\s*-?\s*husna|asmaul\s*husna|asma\s+al\s*-?\s*husna|99\s+names|"
                   r"ninety[- ]nine\s+names|names\s+of\s+allah|allah'?s\s+names)(?![a-z])|আসমাউল\s*হুসনা|"
                   r"আল্লাহর\s+(?:৯৯|নিরানব্বই)\s*(?:টি\s*)?নাম|الأسماء الحسنى", re.I)
_CARDS = re.compile(r"(?<![a-z])(?:flash\s*cards?|test\s+me|quiz\s+me|practi[cs]e)(?![a-z])|ফ্ল্যাশকার্ড|পরীক্ষা\s+নাও", re.I)
_NUMBER = re.compile(r"(?<![a-z])(?:name\s+(?:no\.?\s*|number\s+|#)?([0-9০-৯]{1,2})|(?:the\s+)?([0-9]{1,2})(?:st|nd|rd|th)\s+name|"
                     r"([0-9০-৯]{1,2})\s*(?:নম্বর|নং)\s*নাম)(?![a-z0-9])", re.I)
_RECITE = re.compile(r"(?<![a-z])(?:recite|say|read|pronounce|chant)\s+(?:it|them|the\s+names?|allah'?s\s+names?|"
                     r"the\s+99|al|ar|as|ya)|পড়ে\s+শোনাও|উচ্চারণ", re.I)
_ELSE = re.compile(r"(?<![a-z])(?:timer|remind|reminder|weather|news|task|calendar|price|bitcoin|and\s+(?:also|then)|"
                   r"also\s+tell|set\s+a)(?![a-z])", re.I)
_FILLER = set("""a an the is are was what what's whats which how does do did can could would will you please javid tell me
show open give find read explain me my of to in on for with by about name names allah allah's mean means meaning meanings
explanation translate translation who this that it its one today today's todays day daily dua duas du'a du'as doa dowa
supplication supplications dhikr zikr zikir adhkar azkar there any some has have an authentic sound sahih containing
using uses use invoke call upon on hey o oh and or i number no #  card flashcards flashcard test quiz practise practice
page let's lets see""".split())
_FILLER_BN = set("""আমাকে আমার দেখাও বলো কী কি মানে অর্থ ব্যাখ্যা নাম আল্লাহর আল্লাহ আজকের দোয়া দোআ দুআ জিকির যিকির
দিয়ে দিয়ে দিয়ে সহ এর এই কোন কোনটা আছে নম্বর নং টি টা খোলো""".split())
_FILLER_AR = set("ما معنى معني اسم الله شرح دعاء ادعية بـ".split())


def _words(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text or "").replace("’", "'")
    return re.findall(r"[A-Za-zÀ-ž'ʿʾ`]+(?:-[A-Za-zÀ-ž'ʿʾ`]+)*|[0-9০-৯]+|[\u0980-\u09FF]+(?:-[\u0980-\u09FF]+)*|"
                      r"[\u0600-\u06FF]+|#", t)


_ASKS_MEANING = re.compile(r"(?<![a-z])(?:mean|means|meaning)(?![a-z])|মানে|অর্থ", re.I)


def find_names(text: str) -> tuple[list[int], set[int]]:
    """The names said in `text` (with its article, "ya", or in Arabic or Bangla script; a bare one when the message
    is about the names), and the word positions they took."""
    ws = _words(text)
    ctx = bool(re.search(r"(?<![a-z])(?:name|names|asma|husna|allah)(?![a-z])|নাম|اسم", text or "", re.I))
    found, used, i = [], set(), 0
    while i < len(ws):
        hit = None
        for w in (5, 4, 3, 2, 1):
            chunk = ws[i:i + w]
            if len(chunk) < w:
                continue
            s = " ".join(chunk)
            if s.lower() in _FILLER and w == 1 and s.lower() != "allah":
                continue
            ns = match_names(s)
            if not ns:
                continue
            n = ns[0]
            first = _lat(chunk[0]).replace("'", "")
            article = first in ARTICLES or bool(re.match(r"(?:al|ar|as|ash|at|ad|az|an|ya)-", first)) or \
                bool(_AR.search(s)) or bool(re.search("[\u0980-\u09FF]", s))
            if n == 1 and s.lower() == "allah" and not re.search(r"(?<![a-z])(?:mean|meaning|name\s+allah)", text or "", re.I):
                continue
            bare_ok = bool(_ASKS_MEANING.search(text or "")) and min(map(len, lat_keys(s) or {""})) >= 5
            if article or ctx or w > 1 or bare_ok:
                hit = (ns, w)
                break
        if hit:
            found.extend(hit[0])
            used.update(range(i, i + hit[1]))
            i += hit[1]
        else:
            i += 1
    return list(dict.fromkeys(found)), used


def _said_names(text: str) -> list[str]:
    """The words of `text` that name something (find_names' spans), to tell one word with two names apart from two."""
    ws, (_, used) = _words(text), find_names(text)
    spans, prev = [], None
    for i in sorted(used):
        if prev is None or i != prev + 1:
            spans.append([])
        spans[-1].append(ws[i])
        prev = i
    return [" ".join(x) for x in spans]


def _number_in(text: str) -> int | None:
    m = _NUMBER.search(text or "")
    if not m:
        return None
    n = int(next(g for g in m.groups() if g).translate(_DIGITS))
    return n if 1 <= n <= COUNT else None


def _leftover(text: str, used: set[int]) -> list[str]:
    out = []
    for i, w in enumerate(_words(text)):
        if i in used:
            continue
        lw = _lat(w).replace("'", "'")
        if lw in _FILLER or w in _FILLER_BN or w in _FILLER_AR or re.fullmatch(r"[0-9০-৯]+(?:st|nd|rd|th)?", lw) or lw in ARTICLES:
            continue
        if re.fullmatch(r"(?:st|nd|rd|th)", lw):
            continue
        out.append(w)
    return out


def request(text: str) -> dict | None:
    """A message that is all about the names, answered here: "what does Al-Wadud mean?", "today's name", "dua with
    Al-'Afuww", "name 48", "open Asma ul-Husna", "flashcards of the names". Anything else (a question about the names
    in general, another request in the same message) goes to the model, under reply_problem's rule."""
    t = " ".join((text or "").split())
    if not t or len(t) > 200 or _ELSE.search(t) or not _load()["names"]:
        return None
    if _RECITE.search(t) and (find_names(t)[0] or _OPEN.search(t)):
        return {"kind": "recite"}
    if _TODAY.search(t):
        rest = _leftover(_TODAY.sub(" ", t), set())
        if not rest:
            return {"kind": "today", "dua": bool(_DUA.search(t))}
        return None
    found, used = find_names(t)
    num = _number_in(t)
    if num and not found:
        found = [num]
        used = set()
        t_rest = _NUMBER.sub(" ", t)
    else:
        t_rest = t
    if found:
        if _leftover(t_rest, used if t_rest is t else set()):
            return None
        alike = len(found) > 1 and len(_said_names(t)) == 1      # one word, two names that sound alike
        if len(found) > 1 and not alike:
            return None
        if _DUA.search(t):
            return {"kind": "dua", "n": found[0], "alike": found if alike else None}
        if _MEANING.search(t) or num or len(_words(t)) <= 3:
            return {"kind": "meaning", "n": found[0], "alike": found if alike else None}
        return None
    if _OPEN.search(t):
        rest = _leftover(_OPEN.sub(" ", t), set())
        if not rest:
            return {"kind": "cards" if _CARDS.search(t) else "open"}
        if _CARDS.search(t) and not _leftover(_CARDS.sub(" ", _OPEN.sub(" ", t)), set()):
            return {"kind": "cards"}
    return None


_LIST_REF = re.compile(r"^(.*?)\s*\(((?:Hasan\s+)?Sahih|Hasan)\s*\(([^)]+)\)\)$", re.I)


def _grade_said(u: dict, lang: str) -> str:
    """"Sunan Abi Dawud 1495, graded sahih by Al-Albani"; a hadith in Bukhari or Muslim is its reference alone."""
    ref, g = u["reference"][0], u.get("grade") or ""
    if not g and (m := _LIST_REF.match(ref)):
        ref, g = m[1], f"{m[2]} ({m[3]})"
    if not g or "in Sahih" in g:
        return ref
    word, who = g.split("(")[0].strip(), g.split("(")[-1].rstrip(")") if "(" in g else ""
    if lang == "bn":
        return f"{ref}, {'হাসান' if word.lower().startswith('hasan') else 'সহীহ'}" + (f" ({who})" if who else "")
    return f"{ref}, graded {word.lower()}" + (f" by {who}" if who else "")


def _card_lines(e: dict, lang: str, full: bool) -> list[str]:
    bn = lang == "bn"
    lines = [f"{e['n']} · {e['arabic']} · {e['transliteration'].get('en', '')}"
             + (f" · {e['transliteration']['bn']}" if e["transliteration"].get("bn") else "")]
    lines.append(("অর্থ" if bn else "Meaning") + f": {e['meaning']['en']} · {e['meaning']['bn']}")
    lst = e["lists"]
    uth = lst["uthaymeen"]
    where = (f"Tirmidhi 3507 no. {e['n']}" + (f" · Ibn Uthaymeen ({'Quran' if uth['list'] == 'quran' else 'Sunnah' if uth['list'] == 'sunnah' else 'compound name'})"
                                            if uth else " · only in Tirmidhi 3507's list"))
    lines.append(("তালিকা" if bn else "Lists") + f": {where}")
    if e["n"] == 1:
        lines.append(("কুরআনে" if bn else "In the Quran") + f": {e['quran']['count_words']} times in {e['quran']['count_ayahs']} ayahs")
    elif e["quran"]["count"]:
        refs = (e["quran"]["definite"] + e["quran"]["indefinite"])[:6]
        lines.append(("কুরআনে" if bn else "In the Quran") + f": {e['quran']['count']} · " + ", ".join(refs)
                     + (" …" if e["quran"]["count"] > 6 else ""))
    else:
        lines.append("কুরআনে এই রূপে নেই" if bn else "Not in the Quran in this form")
    if full:
        sound = [u for u in e["use"] if u.get("sound") or u["kind"] == "Quranic dua"]
        if sound:
            lines.append(("দোয়ায়" if bn else "In duas") + ": " + "; ".join(
                u["reference"][0] + (f" ({u['grade']})" if u.get("grade") else "") for u in sound[:4]))
        lines.append(next(u[lang] for u in e["use"] if u["kind"] == "your own dua"))
        if e["status"] == "check":
            lines.append(("যাচাই বাকি" if bn else "Still to check") + ": " + "; ".join(e["checks"][:2]))
    return lines


def answer(req: dict, lang: str, via: str = "page"):
    """(spoken, screen). Spoken: a number and a meaning, never a name; the screen has the Arabic and the sources."""
    bn = lang == "bn"
    yield ("tool", "asma")
    k = req["kind"]
    page = via == "page"
    if k == "recite":
        return (("আল্লাহর নামগুলো আমি পড়ে শোনাই না: হিফজ পেজের আসমাউল হুসনা ট্যাবে আরবি আর অর্থ আছে।" if bn else
                 "I don't recite Allah's names aloud: their Arabic and meanings are on the Hifz page's Asma ul-Husna tab."), "")
    if k in ("open", "cards"):
        yield ("asma", {"open": True, "cards": k == "cards"})
        st = state()
        if bn:
            said = "আসমাউল হুসনা খোলা হলো" + ("।" if k == "open" else ", ফ্ল্যাশকার্ডে।") + \
                   f" {quran.bn_digits(st['learned'])}টি শেখা, {quran.bn_digits(st['reviewing'])}টি রিভিউতে।"
        else:
            said = ("Asma ul-Husna is open" + (", on the flashcards." if k == "cards" else ".") +
                    f" {st['learned']} learned, {st['reviewing']} still reviewing.")
        return said, ""
    if k == "today":
        e = name_of_day()
        if not e:
            return ("নামের তালিকা পাওয়া যাচ্ছে না।" if bn else "The list of names isn't here."), ""
        req = {"kind": "dua" if req.get("dua") else "meaning", "n": e["n"], "today": True}
    if req.get("alike"):                # al-Majīd and al-Mājid: both, said apart by number and meaning
        es = [get(n) for n in req["alike"]]
        yield ("asma", {"open": es[0]["id"], "alike": [x["id"] for x in es]})
        parts = [(f"{quran.bn_digits(x['n'])} নম্বর, “{x['meaning']['bn']}”" if bn else f"number {x['n']}, “{x['meaning']['en']}”")
                 for x in es]
        said = (f"এই উচ্চারণে {quran.bn_digits(len(es))}টি নাম আছে: {quran.join(parts, lang)}। দুটিই পর্দায়।" if bn else
                f"{len(es)} names sound like that: {quran.join(parts, lang)}. Both cards are on your screen.")
        return _spoken(said, es[0], lang), "\n\n".join("\n".join(_card_lines(x, lang, req["kind"] == "meaning")) for x in es)
    e = get(req["n"])
    if not e:
        return ("এই নম্বরে কোনো নাম নেই।" if bn else "There's no name with that number in the list."), ""
    yield ("asma", {"open": e["id"]})
    where = (" কার্ডটি পর্দায় খোলা।" if bn else " Its card is open on your screen.") if page else ""
    if req["kind"] == "meaning":
        if req.get("today"):
            said = (f"আজকের নাম {quran.bn_digits(e['n'])} নম্বর: “{e['meaning']['bn']}”।" if bn else
                    f"Today's name is number {e['n']}: “{e['meaning']['en']}”.")
        else:
            said = (f"তালিকার {quran.bn_digits(e['n'])} নম্বর নামের অর্থ “{e['meaning']['bn']}”।" if bn else
                    f"Name {e['n']} in your list means “{e['meaning']['en']}”.")
        return _spoken(said + where, e, lang), "\n".join(_card_lines(e, lang, True))
    sound = [u for u in e["use"] if u.get("sound")]
    quranic = [u for u in e["use"] if u["kind"] == "Quranic dua"]
    if not sound and not quranic:
        said = (f"{quran.bn_digits(e['n'])} নম্বর নাম দিয়ে নির্দিষ্ট কোনো দোয়া যাচাই করা উৎসে পাইনি। নিজের দোয়ায় এই নামে আল্লাহকে ডাকুন, আল-আ'রাফ ৭:১৮০।"
                if bn else f"No dua in the sources checked calls on Allah by name {e['n']}. Call upon Him by it in your own dua, "
                           f"as Al-A'raf 7:180 says.")
        screen = _card_lines(e, lang, False) + [next(u[lang] for u in e["use"] if u["kind"] == "your own dua")]
        return _spoken(said + where, e, lang), "\n".join(screen)
    first = sound[0] if sound else quranic[0]
    total = len(sound) + len(quranic)
    if bn:
        said = f"{quran.bn_digits(e['n'])} নম্বর নাম দিয়ে {quran.bn_digits(total)}টি দোয়া আছে; প্রথমটি {_grade_said(first, 'bn')}।"
    else:
        said = (f"There {'is' if total == 1 else 'are'} {total} authentic dua{'' if total == 1 else 's'} with name {e['n']}"
                f"{' in your list and the sources checked' if sound else ' in the Quran'}; the first is "
                f"{_grade_said(first, 'en') if first.get('sound') else first['reference'][0]}.")
    lines = _card_lines(e, lang, False)
    for u in sound[:6]:
        lines.append(f"{u['reference'][0]}" + (f" · {u['grade']}" if u.get("grade") else "")
                     + (f" · {u['title']}" if u.get("title") else ""))
        lines.append(u["arabic"])
        if u.get("link"):
            lines.append(u["link"])
        if u.get("counted"):
            lines.append("The hadith gives a number of times: it's in its own words." if not bn else
                         "হাদিসে সংখ্যা দেওয়া আছে: হাদিসের নিজের ভাষায়।")
    for u in quranic[:4]:
        lines.append(f"{u['reference'][0]} (" + ("আয়াতটি কার্ডে, Quran.com থেকে" if bn else "the ayah is on the card, from Quran.com") + ")")
    return _spoken(said + where, e, lang), "\n".join(lines)


def _spoken(said: str, e: dict, lang: str) -> str:
    """What's spoken, checked: should a meaning ever hold a name's word, it goes rather than be said."""
    if speakable(said):
        return said
    return (f"{quran.bn_digits(e['n'])} নম্বর নামের কার্ড পর্দায় আছে।" if lang == "bn" else
            f"Name {e['n']}'s card is on your screen.")


# ---------- the model's replies: no meaning, count or benefit of a name from memory ----------
_SAYS = re.compile(r"(?<![a-z])(?:means?|meaning|refers?\s+to|signif\w*|translat\w*|i\.e\.|denot\w*|"
                   r"the\s+one\s+who|is\s+the\s+(?:most|all|one)|describes?)(?![a-z])|মানে|অর্থ", re.I)
_RECITAL = re.compile(r"(?<![a-z])(?:recit\w*|repeat\w*|say(?:ing)?|chant\w*|read(?:ing)?)(?![a-z]).{0,60}"
                      r"(?:\b\d+\s*(?:times|x)\b|daily|every\s+(?:day|morning|night))|\b\d+\s*(?:times|x)\b|"
                      r"[0-9০-৯]+\s*বার", re.I)


def _mentions(sentence: str) -> bool:
    ws = _words(sentence)
    for i in range(len(ws)):
        for w in (4, 3, 2, 1):
            chunk = ws[i:i + w]
            if len(chunk) == w and (_named(" ".join(chunk)) or (w == 1 and _AR.search(chunk[0]) and match_name(chunk[0]))):
                return True
    return False


def _bad(sentence: str) -> bool:
    return _mentions(sentence) and bool(_SAYS.search(sentence) or _RECITAL.search(sentence) or _BENEFIT.search(sentence))


def reply_problem(text: str) -> list[str]:
    """A model reply that gives a name of Allah's meaning, a count to recite it or a benefit of it: not from the
    approved list, so it's rewritten once, then those sentences go (javid.think_stream)."""
    if not text or not _load()["names"]:
        return []
    return ["asma"] if any(_bad(s) for s in quran._sentences(text)) else []


def correction(problems: list[str]) -> str:
    return ("Rewrite your reply without giving the meaning of any of Allah's names, a number of times to recite one, "
            "or a benefit of it: Javid gives those only from Mahadi's approved Asma ul-Husna list. If he wants one, say "
            "he can ask “what does Al-Wadud mean?” or “a dua with Al-'Afuww” and Javid will show its card.")


def strip_reply(text: str, lang: str = "en") -> str:
    kept = [s for s in quran._sentences(text) if not _bad(s)]
    if kept:
        return " ".join(kept)
    return ("আল্লাহর নামের অর্থ বা আমল আমি নিজে থেকে বলি না। জিজ্ঞেস করো যেমন “আল-ওয়াদুদ মানে কী?”, আমি তোমার অনুমোদিত "
            "তালিকা থেকে কার্ডটি দেখাব।" if lang == "bn" else
            "I don't give the meaning or practice of Allah's names from memory. Ask, for example, “what does Al-Wadud "
            "mean?” and I'll show its card from your approved list.")


# ---------- python asma.py --check ----------
def check_report() -> tuple[list[str], list[str]]:
    """What the list holds and what was left out of it: (lines, problems)."""
    reload()
    L = _load()
    ns = L["names"]
    lines = [f"{len(ns)} names: {sum(e['status'] == 'verified' for e in ns)} verified, "
             f"{sum(e['status'] == 'check' for e in ns)} to check" + (" (approved)" if approved() else " (not approved yet)")]
    both = sum(1 for e in ns if e["lists"]["uthaymeen"])
    lines.append(f"Lists: {both} in Ibn Uthaymeen's too, {len(ns) - both} only in Tirmidhi 3507's, "
                 f"{len(L['uthaymeen_only'])} only in Ibn Uthaymeen's")
    lines.append(f"In the Quran: {sum(1 for e in ns if e['quran']['count'] or e['n'] == 1)}; Sa'di explains "
                 f"{sum(1 for e in ns if e['explanation'])}")
    kinds = {}
    for e in ns:
        for u in e["use"]:
            kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    lines.append("Uses: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
    with_dua = sum(1 for e in ns if any(u.get("sound") or u["kind"] == "Quranic dua" for u in e["use"]))
    lines.append(f"Names with a sound dua, dhikr or Quranic dua: {with_dua}")
    for d in L["dropped"]:
        lines.append(f"left out: {d}")
    return lines, list(L["problems"])


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        out, bad = check_report()
        print("\n".join(out))
        for p in bad:
            print(f"PROBLEM: {p}")
        sys.exit(1 if bad else 0)
    print("python asma.py --check   reads data/asma_ul_husna.yaml and says what it holds and what was left out")
