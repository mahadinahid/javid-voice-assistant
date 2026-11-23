"""
The Quran's text and two translations, kept on this Mac for the reader (2026-11-23).

Source: Tanzil (tanzil.net), as Mahadi chose on 2026-11-23: its Uthmani text 1.1 (Creative Commons Attribution 3.0:
copied verbatim, never changed, credited with a link to tanzil.net), and its translations by Saheeh International
(en.sahih) and Muhiuddin Khan (bn.bengali), which Tanzil offers for non-commercial use. Quran.com's own text can't be
kept offline without an approved account: Quran Foundation's Developer Terms (2026-11-20) allow a lasting offline copy
only through its Content Sync API, and anything from its regular API for a week at most.

The three files are downloaded once into data/quran/ (gitignored: none of it goes into the repository), written 0600,
and checked at every load: each surah's ayah count against quran_meta.json (6,236 in all) and each file's SHA-256
against the one recorded when it was downloaded. A file that fails either is downloaded again. Nothing here is ever
spoken (javid.synthesize takes Arabic script out) or given to a model: the page shows it, with its credits.

Tanzil writes the Bismillah at the start of the first ayah of every surah but Al-Fatiha (where it is the first ayah)
and At-Tawbah (which has none). The text stays as it is; the reader shows that Bismillah above the surah instead of
inside the ayah (`_split`).
"""
import hashlib
import json
import os
import ssl
import threading
import time
import unicodedata
import urllib.request

import quran

DIR = os.path.join("data", "quran")            # beside the other data: the folder the server runs in
MANIFEST = "manifest.json"
USER_AGENT = "Javid/1.0 (personal voice assistant)"
SOURCES = {
    "ar": {"file": "quran-uthmani.txt",
           "url": "https://tanzil.net/pub/download/index.php?marks=true&sajdah=true&rub=true&tatweel=true"
                  "&quranType=uthmani&outType=txt-2&agree=true",
           "name": "Tanzil Quran Text (Uthmani 1.1)", "name_bn": "তানজিলের উসমানি পাঠ (১.১)",
           "link": "https://tanzil.net", "terms": "CC BY 3.0, copied verbatim"},
    "en": {"file": "en.sahih.txt", "url": "https://tanzil.net/trans/?transID=en.sahih&type=txt-2",
           "name": "Saheeh International", "name_bn": "সহিহ ইন্টারন্যাশনাল",
           "link": "https://tanzil.net/trans/", "terms": "via Tanzil, for non-commercial use"},
    "bn": {"file": "bn.bengali.txt", "url": "https://tanzil.net/trans/?transID=bn.bengali&type=txt-2",
           "name": "Muhiuddin Khan", "name_bn": "মুহিউদ্দীন খান",
           "link": "https://tanzil.net/trans/", "terms": "via Tanzil, for non-commercial use"},
}
LANGS = tuple(SOURCES)
MAX_AYAHS = 700         # one reader request: a juz (at most 564 ayahs, juz 30) or a surah (Al-Baqarah, 286)
RETRY_SEC = 600         # after a failed download, try again no sooner than this

_lock = threading.RLock()
_text: dict[str, dict] = {}         # lang -> {(s, a): text}, once checked
_state: dict[str, str] = {}         # lang -> "ready" | "missing" | "bad" | "fetching" | "failed"
_problems: dict[str, list] = {}
_failed_at: dict[str, float] = {}
_fetching = threading.Event()
_words: dict[tuple, list] = {}      # (s, a) -> [[token, word position or 0], ...]
_time = time.time


def _get(url: str) -> bytes:
    """One download (the suite replaces it with a fake: nothing in the tests reaches Tanzil)."""
    import certifi
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as r:
        return r.read()


def parse(raw: str) -> dict:
    """Tanzil's txt-2 format: one "surah|ayah|text" line per ayah; lines starting with "#" are its notes (the
    licence), kept in the file and skipped here."""
    table = {}
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"a line isn't surah|ayah|text: {line[:40]!r}")
        k = (int(parts[0]), int(parts[1]))
        if k in table:
            raise ValueError(f"{k[0]}:{k[1]} is there twice")
        table[k] = parts[2].strip()
    return table


def problems_of(table: dict) -> list[str]:
    """What's wrong with a parsed file: each surah's ayahs against quran_meta.json (6,236 in all), none empty."""
    out = []
    for s in quran.SURAHS:
        have = sum(1 for a in range(1, s["ayahs"] + 1) if table.get((s["n"], a)))
        if have != s["ayahs"]:
            out.append(f"surah {s['n']}: {have} of {s['ayahs']} ayahs")
    extra = [k for k in table if not (1 <= k[0] <= 114 and 1 <= k[1] <= quran.SURAHS[k[0] - 1]["ayahs"])]
    if extra:
        out.append(f"{len(extra)} lines that aren't ayahs (first: {extra[0][0]}:{extra[0][1]})")
    if not out and len(table) != quran.TOTAL:
        out.append(f"{len(table)} ayahs, not {quran.TOTAL}")
    return out


def _path(lang: str) -> str:
    return os.path.join(DIR, SOURCES[lang]["file"])


def _manifest() -> dict:
    try:
        with open(os.path.join(DIR, MANIFEST), encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: str, data: bytes) -> None:
    os.makedirs(DIR, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _load(lang: str) -> tuple[dict | None, str, list]:
    """The saved file, if it's whole: (table, state, problems)."""
    path = _path(lang)
    if not os.path.exists(path):
        return None, "missing", []
    with open(path, "rb") as f:
        data = f.read()
    want = _manifest().get(lang, {}).get("sha256")
    if want and hashlib.sha256(data).hexdigest() != want:
        return None, "bad", ["the file changed since it was downloaded (its checksum differs)"]
    try:
        table = parse(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, "bad", [str(e)]
    bad = problems_of(table)
    return (None, "bad", bad) if bad else (table, "ready", [])


def _download(lang: str) -> tuple[dict | None, list]:
    data = _get(SOURCES[lang]["url"])
    table = parse(data.decode("utf-8"))
    bad = problems_of(table)
    if bad:
        return None, bad
    _write(_path(lang), data)
    m = _manifest()
    m[lang] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "ayahs": len(table),
               "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": SOURCES[lang]["url"]}
    _write(os.path.join(DIR, MANIFEST), json.dumps(m, indent=1).encode())
    return table, []


def ensure(fetch: bool = True) -> dict:
    """Load what's saved and whole; download what isn't (fetch), each file once per RETRY_SEC after a failure.
    Run at server start in a thread, and by a reader request that finds the text missing. Returns status()."""
    with _lock:
        todo = []
        for lang in LANGS:
            if lang in _text:
                continue
            table, st, bad = _load(lang)
            if table:
                _text[lang], _state[lang], _problems[lang] = table, "ready", []
                continue
            _state[lang], _problems[lang] = st, bad
            if bad:
                print(f"[quran] {SOURCES[lang]['file']}: {bad[0]}; downloading it again")
            todo.append(lang)
        if fetch:
            todo = [lang for lang in todo if _time() - _failed_at.get(lang, 0) >= RETRY_SEC]
        if not fetch or not todo or _fetching.is_set():
            return status()
        _fetching.set()
        for lang in todo:
            _state[lang] = "fetching"
    try:
        for lang in todo:
            try:
                table, bad = _download(lang)
            except Exception as e:      # offline, Tanzil down: the reader says so, and it's tried again later
                table, bad = None, [f"couldn't download it ({type(e).__name__})"]
            with _lock:
                if table:
                    _text[lang], _state[lang], _problems[lang] = table, "ready", []
                    _failed_at.pop(lang, None)
                    print(f"[quran] {SOURCES[lang]['name']}: downloaded and checked ({len(table):,} ayahs)")
                else:
                    _state[lang], _problems[lang], _failed_at[lang] = "failed", bad, _time()
                    print(f"[quran] {SOURCES[lang]['file']}: {bad[0]}")
    finally:
        _fetching.clear()
    return status()


def ensure_soon() -> None:
    """A download in the background, unless the text is there or one is running."""
    if not ready() and not _fetching.is_set():
        threading.Thread(target=ensure, daemon=True).start()


def ready(lang: str = "ar") -> bool:
    return lang in _text


def status() -> dict:
    with _lock:
        return {"ready": "ar" in _text, "langs": {lang: _state.get(lang, "missing") for lang in LANGS},
                "problems": {lang: p for lang, p in _problems.items() if p}, "fetching": _fetching.is_set(),
                "credits": credits()}


def credits() -> dict:
    return {lang: {k: SOURCES[lang][k] for k in ("name", "name_bn", "link", "terms")} for lang in LANGS}


def forget() -> None:
    """Drop what's loaded (the suite, between cases)."""
    with _lock:
        _text.clear()
        _state.clear()
        _problems.clear()
        _failed_at.clear()
        _words.clear()


# ---------- the reader's ayahs ----------
def _is_word(token: str) -> bool:
    """A word has a letter; a pause mark (small high letters), the sajdah sign or the rub el hizb sign alone isn't
    one. Quran.com's word timings count words the same way (checked on Al-Mulk: every ayah's count matched)."""
    return any(unicodedata.category(c) == "Lo" for c in token)


def bismillah() -> str:
    return _text.get("ar", {}).get((1, 1), "")


def _split(s: int, a: int) -> tuple[str, bool]:
    """(the ayah's own text, whether a Bismillah was in front of it)."""
    t = _text["ar"][(s, a)]
    b = bismillah()
    if a == 1 and s not in (1, 9) and b and t.startswith(b + " "):
        return t[len(b) + 1:], True
    return t, False


def words(s: int, a: int) -> list[list]:
    """The ayah's tokens for the page: [token, its word position from 1], a mark alone with position 0."""
    k = (s, a)
    if k not in _words:
        out, n = [], 0
        for tok in _split(s, a)[0].split():
            if _is_word(tok):
                n += 1
                out.append([tok, n])
            else:
                out.append([tok, 0])
        _words[k] = out
    return _words[k]


def word_count(s: int, a: int) -> int | None:
    if not ready():
        return None
    return sum(1 for _, p in words(s, a) if p)


def reader(ranges: list, langs: list[str] | tuple = ("en",)) -> dict:
    """What the reader shows for ayah index ranges (a surah, a juz, a stretch): the parts by surah, each with its
    names, whether to show the Bismillah above it, and its ayahs (tokens, and the translations asked for)."""
    if not ready():
        raise LookupError("the Quran's text isn't downloaded yet")
    langs = [x for x in langs if x in ("en", "bn") and ready(x)]
    parts, count = [], 0
    for lo, hi in quran.merge(ranges):
        i = lo
        while i <= hi and count < MAX_AYAHS:
            s, a1 = quran.verse(i)
            last = min(hi, quran.surah_range(s)[1])
            a2 = min(quran.verse(last)[1], a1 + MAX_AYAHS - count - 1)
            meta = quran.SURAHS[s - 1]
            verses = []
            for a in range(a1, a2 + 1):
                v = {"k": f"{s}:{a}", "a": a, "w": words(s, a)}
                if langs:
                    v["tr"] = {x: _text[x][(s, a)] for x in langs}
                verses.append(v)
            parts.append({"s": s, "en": meta["en"], "ar": meta["ar"], "bn": meta["bn"], "ayahs": meta["ayahs"],
                          "type": meta["type"], "bismillah": a1 == 1 and _split(s, 1)[1], "verses": verses})
            count += len(verses)
            i = quran.index(s, a2) + 1
    return {"parts": parts, "count": count, "bismillah_text": bismillah(), "langs": langs, "credits": credits()}
