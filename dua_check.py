"""
The check a dua goes through before it joins Mahadi's Dua & Zikr list (duas.py): the same one Stage 1 ran over his
196 screenshots on 2026-11-23. A picture he sends with "add this to my evening adhkar" is read on this Mac (Tesseract,
Arabic) and its words are looked for, as letter skeletons, in:
  - the Quran, in Tanzil's text already on this Mac (quran_text.py): a match keeps only the reference (surah:ayah and
    the word range), and the text Javid shows comes from Quran.com (duas.py), never from the picture;
  - Hisnul Muslim (hisnmuslim.com's Arabic and English editions);
  - the six hadith collections (fawazahmed0/hadith-api: Bukhari, Muslim with Abdul Baqi's numbers, Abu Dawud,
    Tirmidhi, Nasa'i, Ibn Majah, with Al-Albani's grades).
What's kept is always the source's own wording, cut out of the source; the picture only finds it. The OCR's Arabic
is never shown as a dua: with no source found the item is unsourced and stays hidden. Hisnul Muslim and the hadith
data live in data/dua_sources/ (copied there 2026-11-23; ensure_sources() downloads what's missing); sunnah.com refuses
automated requests, so its links are given for Mahadi to open, never fetched.

Matching: Arabic is reduced to its letters (no harakat, no spaces, no alef: السموات and السماوات are one), a match is
the reduced text found whole in a source, or a window of the source scored as a whole so gaps count (a phrase from
two separate sayings isn't "found"). MATCH and above is the same dua; CLOSE and above differs in a few letters.
"""
import array
import difflib
import io
import json
import os
import re
import ssl
import threading
import urllib.request

import quran
import quran_text

DIR = os.path.join("data", "dua_sources")
HISN_FILE = "hisnul_muslim.json"
BOOKS = ("bukhari", "muslim", "abudawud", "tirmidhi", "nasai", "ibnmajah")
BOOK_NAME = {"bukhari": "Sahih al-Bukhari", "muslim": "Sahih Muslim", "abudawud": "Sunan Abi Dawud",
             "tirmidhi": "Jami' at-Tirmidhi", "nasai": "Sunan an-Nasa'i", "ibnmajah": "Sunan Ibn Majah"}
HADITH_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-{book}.min.json"
HISN_URL = "https://www.hisnmuslim.com/api/{lang}/{name}.json"
USER_AGENT = "Javid/1.0 (personal voice assistant)"
MATCH, CLOSE = 0.97, 0.90
MIN_LETTERS = 10            # fewer letters than this can't be told apart from a hundred other texts
OCR_LANGS = "ara"
OCR = None                  # the suite's fake reader: a function of the picture's bytes giving its text

_lock = threading.RLock()
_cache: dict = {}

# ---------- reducing Arabic to its letters ----------
DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
SWAP = str.maketrans({"ٱ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ی": "ي", "ک": "ك", "ہ": "ه", "ۃ": "ة",
                      "ة": "ه", "ؤ": "و", "ئ": "ي", "ھ": "ه", "ے": "ي"})
ALEFISH = "اأإآٱءٰ"
MARKS = "".join(chr(c) for c in list(range(0x0610, 0x061B)) + list(range(0x064B, 0x0660)) + [0x0670, 0x0640] +
                list(range(0x06D6, 0x06EE)))


def norm(s: str) -> str:
    s = DIAC.sub("", (s or "").translate(SWAP)).replace("ء", "")
    s = re.sub("[^ء-ي ]", " ", s)
    return " ".join(s.split())


def skel(s: str) -> str:
    return norm(s).replace(" ", "").replace("ا", "")


def skel_map(s: str) -> tuple[str, list]:
    """(the letters, the index in s of each one): to cut a source's own words back out of it."""
    out, pos = [], []
    for i, ch in enumerate(s or ""):
        c = ch.translate(SWAP)
        if DIAC.match(c) or c in " ءا" or not ("ء" <= c <= "ي"):
            continue
        out.append(c)
        pos.append(i)
    return "".join(out), pos


def cut(text: str, start: int, end: int, whole_words: bool = False) -> str:
    """The source's own words for letters start..end-1, with the alefs and marks at their edges (and, whole_words,
    out to the spaces around them: a close match's window can begin inside a word)."""
    sk, pos = skel_map(text)
    if not pos or start >= len(pos):
        return ""
    a, b = pos[start], pos[min(end, len(pos)) - 1] + 1
    while a > 0:
        j = a - 1
        while j >= 0 and text[j] in MARKS:
            j -= 1
        if j >= 0 and text[j] in ALEFISH:
            a = j
            continue
        break
    while b < len(text) and (text[b] in ALEFISH or text[b] in MARKS):
        b += 1
    if whole_words:
        while a > 0 and not text[a - 1].isspace():
            a -= 1
        while b < len(text) and not text[b].isspace():
            b += 1
    return text[a:b].strip(" \u200f\u200e.\u060c()﴿﴾")    # the source's own words, without its quote marks


def tight(q: str, t: str) -> tuple[float, tuple | None]:
    """1.0 when q is in t; else the window of t around their longest common run, scored as a whole (gaps cost)."""
    if not q or not t:
        return 0.0, None
    i = t.find(q)
    if i >= 0:
        return 1.0, (i, i + len(q))
    m = difflib.SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    if m.size < 3:
        return 0.0, None
    best = (0.0, None)
    for pad in (0, 4, 10):
        a = max(0, m.a - m.b - pad)
        b = min(len(t), m.a + (len(q) - m.b) + pad)
        r = difflib.SequenceMatcher(None, q, t[a:b], autojunk=False).ratio()
        if r > best[0]:
            best = (r, (a, b))
    return best


# ---------- reading a picture ----------
def ocr(data: bytes) -> str:
    """The picture's text, read on this Mac by Tesseract's Arabic model (never a language model: its Arabic is only
    used to find the dua in a source, never shown)."""
    if OCR:
        return OCR(data)
    import pytesseract
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img).convert("L")
    if max(img.size) < 1600:                    # small text reads better a little larger
        img = img.resize((img.width * 2, img.height * 2))
    return pytesseract.image_to_string(img, lang=OCR_LANGS)


# ---------- the sources ----------
def _get(url: str) -> bytes:
    """One download (the suite replaces it: nothing in the tests reaches the network)."""
    import certifi
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context(cafile=certifi.where())) as r:
        return r.read()


def _write(path: str, data: bytes) -> None:
    os.makedirs(DIR, mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def status() -> dict:
    """Which sources are on this Mac."""
    have = {b: os.path.exists(os.path.join(DIR, f"ara-{b}.json")) for b in BOOKS}
    if not quran_text.ready():
        quran_text.ensure(fetch=False)
    return {"hisn": os.path.exists(os.path.join(DIR, HISN_FILE)), "hadith": have, "quran": quran_text.ready()}


def ensure_sources() -> list[str]:
    """Download what's missing (a hadith book from jsDelivr, Hisnul Muslim from hisnmuslim.com). The problems, if any."""
    problems = []
    for b in BOOKS:
        path = os.path.join(DIR, f"ara-{b}.json")
        if not os.path.exists(path):
            try:
                data = _get(HADITH_URL.format(book=b))
                json.loads(data)
                _write(path, data)
            except Exception as e:
                problems.append(f"{BOOK_NAME[b]}: {type(e).__name__}")
    path = os.path.join(DIR, HISN_FILE)
    if not os.path.exists(path):
        try:
            book = []
            idx_ar = json.loads(_get(HISN_URL.format(lang="ar", name="husn_ar")))
            idx_en = json.loads(_get(HISN_URL.format(lang="en", name="husn_en")))
            en_titles = {c["ID"]: c.get("TITLE") for c in next(iter(idx_en.values()))}
            for ch in next(iter(idx_ar.values())):
                cid = ch["ID"]
                ar = next(iter(json.loads(_get(HISN_URL.format(lang="ar", name=cid)).decode("utf-8-sig"), strict=False).values()))
                try:
                    en = next(iter(json.loads(_get(HISN_URL.format(lang="en", name=cid)).decode("utf-8-sig"), strict=False).values()))
                except Exception:
                    en = []
                book.append({"chapter": cid, "title_ar": ch.get("TITLE"), "title_en": en_titles.get(cid),
                             "items_ar": ar, "items_en": en})
            _write(path, json.dumps(book, ensure_ascii=False).encode())
        except Exception as e:
            problems.append(f"Hisnul Muslim: {type(e).__name__}")
    with _lock:
        _cache.clear()
    return problems


def _hisn() -> dict:
    with _lock:
        if "hisn" not in _cache:
            out = {}
            try:
                with open(os.path.join(DIR, HISN_FILE), encoding="utf-8") as f:
                    book = json.load(f)
            except (OSError, ValueError):
                book = []
            for ch in book:
                en = {e.get("ID"): e for e in (ch.get("items_en") or []) if isinstance(e, dict)}
                for a in ch.get("items_ar") or []:
                    ar = a["ARABIC_TEXT"]
                    quoted = re.findall(r"\(\((.*?)\)\)", ar, flags=re.S) + re.findall(r"﴿(.*?)﴾", ar, flags=re.S)
                    core = " ".join(quoted) if quoted else ar
                    e = en.get(a["ID"]) or {}
                    out[a["ID"]] = {"id": a["ID"], "chapter": ch["chapter"], "chapter_en": (ch.get("title_en") or "").strip(),
                                    "ar": ar.strip(), "core": core.strip(), "repeat": a.get("REPEAT"),
                                    "translit": (e.get("LANGUAGE_ARABIC_TRANSLATED_TEXT") or "").strip(),
                                    "en": (e.get("TRANSLATED_TEXT") or "").strip(), "sk": skel(ar)}
            _cache["hisn"] = out
        return _cache["hisn"]


def _hadith() -> dict:
    with _lock:
        if "hadith" not in _cache:
            out = {}
            for b in BOOKS:
                rows = []
                try:
                    with open(os.path.join(DIR, f"ara-{b}.json"), encoding="utf-8") as f:
                        data = json.load(f)["hadiths"]
                except (OSError, ValueError, KeyError):
                    data = []
                for h in data:
                    try:
                        an = float(h.get("arabicnumber"))
                    except (TypeError, ValueError):
                        an = None
                    rows.append({"book": b, "n": h["hadithnumber"], "an": an, "text": h["text"],
                                 "sk": skel_map(h["text"])[0], "grades": h.get("grades") or []})
                if b == "muslim":       # a narration without its Abdul Baqi number: between 597 and 599 it's 598
                    for i, h in enumerate(rows):
                        if h["an"] is None:
                            p = next((rows[j]["an"] for j in range(i - 1, -1, -1) if rows[j]["an"] is not None), None)
                            n = next((rows[j]["an"] for j in range(i + 1, len(rows)) if rows[j]["an"] is not None), None)
                            if p is not None and n is not None and int(n) - int(p) in (0, 2):
                                h["an"] = int(p) + (1 if int(n) - int(p) == 2 else 0)
                out[b] = rows
            _cache["hadith"] = out
        return _cache["hadith"]


def _quran() -> dict | None:
    """Tanzil's text as one string of letters, with each letter's (surah, ayah, word position)."""
    with _lock:
        if "quran" not in _cache:
            if not quran_text.ready():
                quran_text.ensure(fetch=False)      # the text saved on this Mac (the server loads it at start)
            if not quran_text.ready():
                return None
            letters, where_s, where_a, where_p = [], array.array("H"), array.array("H"), array.array("H")
            for s in range(1, 115):
                for a in range(1, quran.SURAHS[s - 1]["ayahs"] + 1):
                    for tok, p in quran_text.words(s, a):
                        if not p:
                            continue
                        k = skel(tok)
                        letters.append(k)
                        where_s.extend([s] * len(k))
                        where_a.extend([a] * len(k))
                        where_p.extend([p] * len(k))
            _cache["quran"] = {"sk": "".join(letters), "s": where_s, "a": where_a, "p": where_p}
        return _cache["quran"]


def forget() -> None:
    with _lock:
        _cache.clear()


# ---------- grades, references ----------
def _grade(h: dict) -> tuple[str, bool]:
    if h["book"] in ("bukhari", "muslim"):
        return f"Sahih (in {BOOK_NAME[h['book']]})", True
    g = {x["name"]: x["grade"] for x in h["grades"]}
    txt = g.get("Al-Albani") or next(iter(g.values()), "")
    who = "Al-Albani" if "Al-Albani" in g else (next(iter(g), "") or "")
    sound = bool(re.search(r"sahih|hasan", txt, re.I)) and not re.search(r"da.?if|mawdu|munkar|shadh|maqtu|batil", txt, re.I)
    return (f"{txt} ({who})" if txt else "no grade in the data"), sound


def _ref(h: dict) -> tuple[str, str]:
    num = int(h["n"]) if h["book"] != "muslim" else (int(h["an"]) if h["an"] else None)
    if num is None:
        return f"{BOOK_NAME[h['book']]} (no. {h['n']} in the data)", ""
    return f"{BOOK_NAME[h['book']]} {num}", f"https://sunnah.com/{h['book']}:{num}"


# ---------- finding a text ----------
def _find_quran(q: str) -> dict | None:
    """The ayahs (and word range) whose letters match q best: exact, else a window found by voting on 8-letter slices."""
    Q = _quran()
    if not Q or len(q) < MIN_LETTERS:
        return None
    big = Q["sk"]
    i = big.find(q)
    if i >= 0:
        score, span = 1.0, (i, i + len(q))
    else:
        votes: dict = {}
        for off in range(0, max(1, len(q) - 8), 4):
            piece, start, hits = q[off:off + 8], 0, 0
            while hits < 60:
                j = big.find(piece, start)
                if j < 0:
                    break
                votes[j - off] = votes.get(j - off, 0) + 1
                start, hits = j + 1, hits + 1
        if not votes:
            return None
        base = max(votes, key=votes.get)
        if votes[base] < 2:
            return None
        lo, hi = max(0, base - 12), min(len(big), base + len(q) + 12)
        score, win = tight(q, big[lo:hi])
        if not win:
            return None
        span = (lo + win[0], lo + win[1])
    if score < CLOSE:
        return None
    a, b = span[0], span[1] - 1
    first, last = (Q["s"][a], Q["a"][a], Q["p"][a]), (Q["s"][b], Q["a"][b], Q["p"][b])
    keys, words = [], {}
    s, ay = first[0], first[1]
    while True:
        k = f"{s}:{ay}"
        keys.append(k)
        n = quran_text.word_count(s, ay) or 0
        w1 = first[2] if (s, ay) == first[:2] else 1
        w2 = last[2] if (s, ay) == last[:2] else n
        if (w1, w2) != (1, n):
            words[k] = [w1, w2]
        if (s, ay) == last[:2] or len(keys) > 40:
            break
        ay += 1
        if ay > quran.SURAHS[s - 1]["ayahs"]:
            s, ay = s + 1, 1
    return {"score": round(score, 3), "ayahs": keys, "words": words or None}


def _find_hisn(q: str) -> list[dict]:
    out = []
    for h in _hisn().values():
        if q in h["sk"]:
            j = h["sk"].find(q)
            out.append({"id": h["id"], "score": 1.0, "span": (j, j + len(q))})
    if out:
        return out
    best = None
    for h in _hisn().values():
        sc, span = tight(q, h["sk"])
        if span and (best is None or sc > best["score"]):
            best = {"id": h["id"], "score": round(sc, 3), "span": span}
    return [best] if best and best["score"] >= CLOSE else []


def _find_hadith(q: str, limit: int = 6) -> list[dict]:
    H = _hadith()
    hits = []
    for b in BOOKS:
        for h in H.get(b, []):
            j = h["sk"].find(q)
            if j >= 0:
                hits.append((1.0, h, (j, j + len(q))))
    if not hits and len(q) >= 14:
        k = len(q) // 3
        parts = [q[:k], q[k:2 * k], q[2 * k:]]
        for b in BOOKS:
            for h in H.get(b, []):
                if sum(p in h["sk"] for p in parts) >= 2:
                    sc, span = tight(q, h["sk"])
                    if sc >= CLOSE:
                        hits.append((sc, h, span))
    hits.sort(key=lambda x: (-x[0], BOOKS.index(x[1]["book"])))
    out = []
    for sc, h, span in hits[:limit]:
        grade, sound = _grade(h)
        label, url = _ref(h)
        out.append({"ref": label, "url": url, "score": round(sc, 3), "grade": grade, "sound": sound,
                    "text": cut(h["text"], span[0], span[1], whole_words=sc < 1)})
    return out


HARAKAT = re.compile("[\u064B-\u0652\u0670]")
_LETTER = re.compile("[\u0621-\u064A\u0671]")
_JUNK = re.compile(r"[0-9A-Za-z\[\]{}|<>@#$%^&*_=+~`\"“”!?\\/©®]")
_COUNT = re.compile(r"\(?\s*(?:x\s*\d{1,5}|\d{1,5}\s*(?:x|times|বার)|[٠-٩]{1,5})\s*\)?", re.I)
MIN_LINE = 15               # a line's letters to be looked for on its own (a shorter one joins the next)
MIN_CLOSE = 20              # a reading this long may match a few letters apart; shorter, only exactly
MIN_WINDOW = 25             # inside a long hadith a shorter reading finds a lookalike stretch by chance
COMMON = 3                  # a short exact reading found in more places than this is a common phrase: not one dua
HADITH_CLOSE = 0.95         # a long hadith has many stretches to resemble: a close match there must be closer
AMBIGUOUS = 0.015           # two sources' wordings scoring this close: the picture can't tell which


def arabic_lines(text: str) -> list[str]:
    """The lines of a reading that are a dua's Arabic. Tesseract's Arabic model also turns a screenshot's English and
    Bangla captions into Arabic letters: those come with digits, Latin letters and symbols and almost no harakat,
    while a dua as posted is vowelled. A line passes with few such marks, and harakat unless it is long and clean."""
    out = []
    for raw in (text or "").splitlines():
        line = _COUNT.sub(" ", raw).strip(" \u200e\u200f*•·-–—")
        toks = line.split()
        arabic = [t for t in toks if _LETTER.search(t) and not _JUNK.search(t)]
        if toks and len(arabic) >= 2 and len(arabic) >= 0.6 * len(toks):
            line = " ".join(arabic)             # "(سبحان الله) 17وج": the caption's digits stuck to the end go
        letters = len(_LETTER.findall(line))
        if letters < 4:
            continue
        junk = len(_JUNK.findall(line))
        marks = len(HARAKAT.findall(line))
        if junk > 0.08 * (letters + junk):
            continue
        if marks < 0.12 * letters and (junk or letters < 25):
            continue
        out.append(" ".join(line.split()))
    return out


def _places(q: str, limit: int) -> int:
    """In how many places (ayahs, Hisnul Muslim entries, hadiths) a stretch of letters is found exactly, up to limit."""
    n = 0
    Q = _quran()
    if Q:
        start = 0
        while n < limit and (j := Q["sk"].find(q, start)) >= 0:
            n += 1
            start = j + 1
    for h in _hisn().values():
        if n >= limit:
            break
        n += q in h["sk"]
    for b in BOOKS:
        for h in _hadith().get(b, []):
            if n >= limit:
                return n
            n += q in h["sk"]
    return n


def _best(q: str) -> dict | None:
    """The source a stretch of letters is from: the Quran when it's as good as any (Hisnul Muslim quotes ayahs too),
    then Hisnul Muslim, then the hadith books (Bukhari and Muslim first, then the sound ones). None when the reading
    is too short to tell, is a phrase found in many places, or only resembles a source."""
    if len(q) < MIN_LINE:
        return None
    cands = []
    if (f := _find_quran(q)):
        cands.append((f["score"], 2, {"what": "quran", **f}))
    for h in _find_hisn(q):
        cands.append((h["score"], 1, {"what": "hisn", **h}))
    for h in _find_hadith(q, limit=3):
        if h["score"] >= 1 or len(q) >= MIN_WINDOW:
            cands.append((h["score"] - (0 if h["sound"] else 0.002), 0, {"what": "hadith", **h}))
    cands = [c for c in cands if c[0] >= 0.999 or (c[0] >= CLOSE and len(q) >= MIN_CLOSE
                                                   and (c[2]["what"] != "hadith" or c[0] >= HADITH_CLOSE))]
    if not cands:
        return None
    cands.sort(key=lambda c: (round(c[0], 3), c[1]), reverse=True)
    best = cands[0]
    if best[0] >= 0.999 and len(q) < MIN_WINDOW and _places(q, COMMON + 1) > COMMON:
        return None
    rival = next((c for c in cands[1:] if c[2]["what"] != "quran" and _source_key(c[2]) != _source_key(best[2])), None)
    if best[0] < 0.999 and best[2]["what"] != "quran" and rival and best[0] - rival[0] < AMBIGUOUS:
        return None                 # two sources' wordings this close: which one it is, the picture can't tell
    if best[2]["what"] == "hisn" and (quoted := _quoted_ayahs(best[2])):
        return quoted               # Hisnul Muslim quoting the Quran (Ayat al-Kursi in the morning adhkar): the ayahs
    return best[2]


def _quoted_ayahs(f: dict) -> dict | None:
    """The ayahs a Hisnul Muslim match is, when the stretch matched lies in one of its ﴿…﴾ quotations."""
    h = _hisn()[f["id"]]
    part = h["sk"][f["span"][0]:f["span"][1]]
    for quote in re.findall(r"﴿(.*?)﴾", h["ar"], flags=re.S):
        if part and part[:12] in skel(quote) and (found := _find_quran(part)):
            return {"what": "quran", **found, "score": f["score"]}
    return None


def _source_key(f: dict) -> str:
    if f["what"] == "quran":
        return "q:" + f["ayahs"][0].split(":")[0]       # one surah: neighbouring ayahs are one recitation
    return f"h:{f['id']}" if f["what"] == "hisn" else f"d:{f['ref']}"


def _finding(lines: list[str], f: dict) -> dict:
    """One dua found in the picture: its source's own text (never the reading's), reference and verdict."""
    q = skel(" ".join(lines))
    again = _best(q)
    if again and _source_key(again) == _source_key(f):
        f = again
    score = f["score"]
    out = {"status": "verified" if score >= MATCH else "check", "kind": "dua", "score": round(score, 3),
           "arabic": None, "arabic_from": None, "quran": None, "reference": [], "links": [], "grade": None,
           "notes": [], "lines": len(lines)}
    if f["what"] == "quran":
        part = " (part of the ayah)" if f["words"] and len(f["ayahs"]) == 1 else ""
        a, b = f["ayahs"][0], f["ayahs"][-1]
        out.update(kind="quran", quran={"ayahs": f["ayahs"], "words": f["words"]}, arabic_from="Quran.com",
                   reference=[f"Quran {a}" + (f"–{b.split(':')[1]}" if b != a else "") + part],
                   links=[f"https://quran.com/{a.replace(':', '/')}"])
        if score < MATCH:
            out["notes"].append("The picture's wording differs from the ayah in a few letters; Javid shows the Quran's.")
        return out
    if f["what"] == "hisn":
        h = _hisn()[f["id"]]
        text_cut = cut(h["ar"], f["span"][0], f["span"][1], whole_words=score < 1)
        out.update(arabic=text_cut, arabic_from=f"Hisnul Muslim #{h['id']}", reference=[f"Hisnul Muslim #{h['id']}"],
                   when_hint=h["chapter_en"], translit=h["translit"] or None, meaning_en=h["en"] or None,
                   repeat=h["repeat"] if isinstance(h["repeat"], int) else None)
        for m in (_find_hadith(skel(text_cut), limit=2) if len(skel(text_cut)) >= MIN_WINDOW else []):
            if m["sound"] and m["score"] >= MATCH:
                out["reference"].append(m["ref"] if m["ref"].startswith(("Sahih al-Bukhari", "Sahih Muslim"))
                                        else f"{m['ref']} ({m['grade']})")
                out["links"].append(m["url"])
    else:
        ref = f["ref"] if f["ref"].startswith(("Sahih al-Bukhari", "Sahih Muslim")) else f"{f['ref']} ({f['grade']})"
        out.update(arabic=f["text"], arabic_from=f["ref"], grade=f["grade"], reference=[ref],
                   links=[f["url"]] if f["url"] else [])
        if not f["sound"]:
            out.update(status="unsourced", notes=[f"Found in {f['ref']}, graded {f['grade']}: not a sound source."])
            return out
    if out["status"] == "check":
        out["notes"].append("The picture's wording differs from the source in a few letters; Javid would keep the source's.")
    return out


def find_all(text: str) -> list[dict]:
    """Every dua in a reading, in the picture's order: each line (a short one with the next) is looked for in the
    sources, and neighbouring lines from the same source are one dua. [] when nothing was found."""
    lines = arabic_lines(text)
    runs, i = [], 0
    while i < len(lines):
        j, q = i, skel(lines[i])
        while len(q) < MIN_LINE and j + 1 < len(lines):
            j += 1
            q = skel(" ".join(lines[i:j + 1]))
        f = _best(q)
        if f:
            if runs and _source_key(runs[-1][1]) == _source_key(f):
                runs[-1][0].extend(lines[i:j + 1])
                if f["score"] > runs[-1][1]["score"]:
                    runs[-1][1].update(score=f["score"])
            else:
                runs.append((list(lines[i:j + 1]), f))
        i = j + 1
    found, seen = [], set()
    for ls, f in runs:
        one = _finding(ls, dict(f))
        key = (one["arabic_from"], tuple((one["quran"] or {}).get("ayahs") or ()), one["arabic"])
        if key not in seen:
            seen.add(key)
            found.append(one)
    return found


def nothing_found(text: str) -> dict:
    """The verdict on a reading where no dua was found: Arabic that isn't in the sources, or none read at all."""
    read = arabic_lines(text)
    return {"status": "unsourced" if read else "unreadable", "kind": "dua", "score": 0.0, "arabic": None,
            "arabic_from": None, "quran": None, "reference": [], "links": [], "grade": None, "lines": len(read),
            "notes": ["Not found in the Quran, Hisnul Muslim or the six hadith collections." if read else
                      "No Arabic could be read from the picture."]}


def check(text: str) -> dict:
    """The picture's dua when there is one (the first found), else why not. find_all gives each of several."""
    found = find_all(text)
    return found[0] if found else nothing_found(text)


MAX_REF_AYAHS = 30


def from_reference(ref: dict) -> dict | None:
    """A dua by the reference Mahadi gave ({"hisn": n} or {"quran": [s, a1, a2]}), from the source itself: Hisnul
    Muslim's own text (its first ((…)) part; an entry that is the Quran becomes its ayahs), or the ayahs, whose text
    comes from Quran.com. None when there is no such entry."""
    if "quran" in ref:
        s, a1, a2 = ref["quran"]
        if not 1 <= s <= 114:
            return None
        last = quran.SURAHS[s - 1]["ayahs"]
        a1, a2 = a1 or 1, a2 or (a1 if a1 else last)
        if not 1 <= a1 <= a2 <= last or a2 - a1 + 1 > MAX_REF_AYAHS:
            return None
        keys = [f"{s}:{a}" for a in range(a1, a2 + 1)]
        return {"status": "verified", "kind": "quran", "score": 1.0, "arabic": None, "arabic_from": "Quran.com",
                "quran": {"ayahs": keys, "words": None}, "grade": None, "lines": 0,
                "reference": [f"Quran {s}:{a1}" + (f"–{a2}" if a2 != a1 else "")],
                "links": [f"https://quran.com/{s}/{a1}"], "notes": []}
    h = _hisn().get(int(ref.get("hisn") or 0))
    if not h:
        return None
    parts = re.findall(r"\(\((.*?)\)\)", h["ar"], flags=re.S)
    quotes = re.findall(r"﴿(.*?)﴾", h["ar"], flags=re.S)
    notes = []
    if not parts and quotes:                     # an entry that is ayahs (Ayat al-Kursi, the three Quls)
        keys = []
        for q in quotes:
            if (f := _find_quran(skel(q))) and f["score"] >= MATCH:
                keys += [k for k in f["ayahs"] if k not in keys]
        if not keys:
            return None
        return {"status": "verified", "kind": "quran", "score": 1.0, "arabic": None, "arabic_from": "Quran.com",
                "quran": {"ayahs": keys, "words": None}, "grade": None, "lines": 0,
                "reference": [f"Hisnul Muslim #{h['id']}", f"Quran {keys[0]}" + (f"–{keys[-1]}" if len(keys) > 1 else "")],
                "links": [f"https://quran.com/{keys[0].replace(':', '/')}"], "notes": notes}
    if len(parts) > 1:
        notes.append(f"Hisnul Muslim #{h['id']} has {len(parts)} parts; the first is kept.")
    arabic = " ".join((parts[0] if parts else h["ar"]).split())
    out = {"status": "verified", "kind": "dua", "score": 1.0, "arabic": arabic, "arabic_from": f"Hisnul Muslim #{h['id']}",
           "quran": None, "grade": None, "lines": 0, "reference": [f"Hisnul Muslim #{h['id']}"], "links": [],
           "notes": notes, "translit": h["translit"] or None, "meaning_en": h["en"] or None,
           "repeat": h["repeat"] if isinstance(h["repeat"], int) else None, "when_hint": h["chapter_en"]}
    if len(skel(arabic)) >= MIN_WINDOW:     # a short phrase is in many hadiths: only Hisnul Muslim's number then
        for m in _find_hadith(skel(arabic), limit=2):
            if m["sound"] and m["score"] >= MATCH:
                out["reference"].append(m["ref"] if m["ref"].startswith(("Sahih al-Bukhari", "Sahih Muslim"))
                                        else f"{m['ref']} ({m['grade']})")
                out["links"].append(m["url"])
    return out


def same(a: dict, b: dict) -> bool:
    """Whether two findings are one dua: the same ayahs, or the same source entry."""
    if a["kind"] == "quran" or b["kind"] == "quran":
        return a["kind"] == b["kind"] == "quran" and bool(set(a["quran"]["ayahs"]) & set(b["quran"]["ayahs"]))
    return bool(a.get("arabic_from")) and a.get("arabic_from") == b.get("arabic_from")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--sources"]:       # download what's missing of Hisnul Muslim and the six hadith books
        bad = ensure_sources()
        print("\n".join(bad) or "The sources are here.")
        sys.exit(1 if bad else 0)
    if sys.argv[1:] == ["--status"]:
        print(json.dumps(status(), indent=1))
        sys.exit(0)
    sys.exit("usage: python dua_check.py --status | --sources")
