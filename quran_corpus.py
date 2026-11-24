"""
The Quran's words: each word's root, lemma and grammar from the Quranic Arabic Corpus (2026-11-24), for the Hifz page's
word cards and "where else is this root?".

Source: the Quranic Arabic Corpus, morphology version 0.4 (corpus.quran.com, (C) 2011 Kais Dukes, GNU General Public
License with its terms of use: verbatim copies only, never changed; the source named and linked wherever it's used).
Mahadi downloaded the file himself (corpus.quran.com asks for an email address) into data/quran_corpus/; it stays
there as it came (gitignored, like all of data/), and Javid only reads it: nothing is rewritten, merged or cleaned in
the file. Its forms are in the Corpus's extended Buckwalter transliteration; they're turned into Arabic letters when
shown (`arabic`), never saved back.

A word is located as surah:ayah:word, counted the way the reader counts words (quran_text.words: a pause mark alone
isn't one). The Corpus builds on Tanzil's Uthmani text, so the counts agree for 6,230 of the 6,236 ayahs; where they
don't (2:181, 8:6, 13:37, 37:130: a word Tanzil 1.1 writes in two pieces), `ayah` gives nothing for that ayah, and the
page shows no word cards there rather than the wrong word's grammar.

The descriptions of the tags ("verb · perfect · form IV · 3rd person masculine singular") are Javid's own wording of the
Corpus's tag set, in English and Bangla.
"""
import collections
import hashlib
import json
import os
import re
import threading

DIR = os.path.join("data", "quran_corpus")
FILE_NAME = "quranic-corpus-morphology-0.4.txt"
MANIFEST = "manifest.json"
HEADER = ("Quranic Arabic Corpus (morphology, version 0.4)", "CHANGING IT IS NOT ALLOWED")
LINK = "https://corpus.quran.com"
CREDIT = "Quranic Arabic Corpus 0.4 (corpus.quran.com), GNU GPL"
TOTAL_AYAHS = 6236

_lock = threading.Lock()
_db: dict = {}          # "words": {(s, a): [word, ...]}, "roots": {root: [(s, a, w), ...]}, "problems": [...]

# The Corpus's extended Buckwalter transliteration (its documentation's table)
_BW = {"'": "ء", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ", "A": "ا", "b": "ب",
       "p": "ة", "t": "ت", "v": "ث", "j": "ج", "H": "ح", "x": "خ", "d": "د",
       "*": "ذ", "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص", "D": "ض",
       "T": "ط", "Z": "ظ", "E": "ع", "g": "غ", "_": "ـ", "f": "ف", "q": "ق",
       "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و", "Y": "ى",
       "y": "ي", "F": "ً", "N": "ٌ", "K": "ٍ", "a": "َ", "u": "ُ", "i": "ِ",
       "~": "ّ", "o": "ْ", "^": "ٓ", "#": "ٔ", "`": "ٰ", "{": "ٱ", ":": "ۜ",
       "@": "۟", '"': "۠", "[": "ۢ", ";": "ۣ", ",": "ۥ", ".": "ۦ", "!": "ۨ",
       "-": "۪", "+": "۫", "%": "۬", "]": "ۭ"}


def arabic(bw: str) -> str:
    """Buckwalter → Arabic script, for showing (the file keeps its own form)."""
    return "".join(_BW.get(c, "") for c in bw or "")


def root_letters(root: str) -> str:
    """A root as its letters apart: ktb → ك ت ب."""
    return " ".join(arabic(c) for c in root)


# ---------- the tags, in words ----------
POS = {
    "N": ("noun", "বিশেষ্য"), "PN": ("proper noun", "নামবাচক বিশেষ্য"), "V": ("verb", "ক্রিয়া"),
    "ADJ": ("adjective", "বিশেষণ"), "PRON": ("pronoun", "সর্বনাম"), "DEM": ("demonstrative", "নির্দেশক সর্বনাম"),
    "REL": ("relative pronoun", "সম্বন্ধবাচক সর্বনাম"), "P": ("preposition", "অব্যয় (হরফে জার)"),
    "CONJ": ("conjunction", "সংযোজক"), "NEG": ("negative particle", "না-বাচক অব্যয়"),
    "ACC": ("accusative particle", "নসবদানকারী অব্যয়"), "DET": ("the", "নির্দিষ্টবাচক আল"),
    "T": ("time adverb", "কালবাচক ক্রিয়াবিশেষণ"), "LOC": ("place adverb", "স্থানবাচক ক্রিয়াবিশেষণ"),
    "REM": ("resumption particle", "পুনরারম্ভক অব্যয়"), "SUB": ("subordinating conjunction", "অধীন সংযোজক"),
    "COND": ("conditional particle", "শর্তবাচক অব্যয়"), "INTG": ("interrogative particle", "প্রশ্নবাচক অব্যয়"),
    "EMPH": ("emphatic particle", "জোরদায়ক অব্যয়"), "VOC": ("vocative particle", "সম্বোধক অব্যয়"),
    "RES": ("restriction particle", "সীমাবদ্ধক অব্যয়"), "CERT": ("particle of certainty", "নিশ্চয়তাবাচক অব্যয়"),
    "RSLT": ("result particle", "ফলবাচক অব্যয়"), "PRO": ("prohibition particle", "নিষেধবাচক অব্যয়"),
    "PRP": ("purpose particle", "উদ্দেশ্যবাচক অব্যয়"), "CIRC": ("circumstantial particle", "অবস্থাবাচক অব্যয়"),
    "SUP": ("supplemental particle", "সম্পূরক অব্যয়"), "PREV": ("preventive particle", "প্রতিরোধক অব্যয়"),
    "FUT": ("future particle", "ভবিষ্যৎবাচক অব্যয়"), "RET": ("retraction particle", "প্রত্যাহারবাচক অব্যয়"),
    "EXP": ("exceptive particle", "ব্যতিক্রমবাচক অব্যয়"), "INC": ("inceptive particle", "সূচনাবাচক অব্যয়"),
    "CAUS": ("particle of cause", "কারণবাচক অব্যয়"), "IMPV": ("imperative particle", "আদেশবাচক অব্যয়"),
    "EXL": ("explanation particle", "ব্যাখ্যাবাচক অব্যয়"), "AMD": ("amendment particle", "সংশোধক অব্যয়"),
    "INT": ("particle of interpretation", "ব্যাখ্যাবাচক অব্যয়"), "ANS": ("answer particle", "উত্তরবাচক অব্যয়"),
    "EXH": ("exhortation particle", "উৎসাহবাচক অব্যয়"), "SUR": ("surprise particle", "বিস্ময়বাচক অব্যয়"),
    "AVR": ("aversion particle", "অনীহাবাচক অব্যয়"), "INL": ("Quranic initials", "হুরুফে মুকাত্তাআত"),
    "EQ": ("equalization particle", "সমতাবাচক অব্যয়"), "COM": ("comitative particle", "সহগামী অব্যয়"),
    "IMPN": ("imperative verbal noun", "আদেশবাচক ক্রিয়াবাচক বিশেষ্য"),
}
_ASPECT = {"PERF": ("perfect (past)", "অতীতকাল"), "IMPF": ("imperfect (present or future)", "বর্তমান/ভবিষ্যৎ"),
           "IMPV": ("imperative", "আদেশসূচক")}
_MOOD = {"IND": ("indicative", "সাধারণ"), "SUBJ": ("subjunctive", "নসবযুক্ত"), "JUS": ("jussive", "জযমযুক্ত")}
_CASE = {"NOM": ("nominative", "পেশযুক্ত (রফা)"), "ACC": ("accusative", "যবরযুক্ত (নসব)"),
         "GEN": ("genitive", "যেরযুক্ত (জর)")}
_PERSON = {"1": ("1st person", "উত্তম পুরুষ"), "2": ("2nd person", "মধ্যম পুরুষ"), "3": ("3rd person", "নাম পুরুষ")}
_GENDER = {"M": ("masculine", "পুংলিঙ্গ"), "F": ("feminine", "স্ত্রীলিঙ্গ")}
_NUMBER = {"S": ("singular", "একবচন"), "D": ("dual", "দ্বিবচন"), "P": ("plural", "বহুবচন")}
_PREFIX = {"Al+": ("al- (the)", "আল (নির্দিষ্টবাচক)"), "w:CONJ+": ("wa- (and)", "ওয়া (এবং)"),
           "w:REM+": ("wa- (and, resuming)", "ওয়া (পুনরারম্ভ)"), "w:CIRC+": ("wa- (while)", "ওয়া (অবস্থা)"),
           "w:SUP+": ("wa-", "ওয়া"), "w:P+": ("wa- (by, an oath)", "ওয়া (শপথ)"), "w:COM+": ("wa- (with)", "ওয়া (সঙ্গে)"),
           "f:CONJ+": ("fa- (so)", "ফা (অতঃপর)"), "f:REM+": ("fa- (then)", "ফা"), "f:RSLT+": ("fa- (then, as a result)", "ফা (ফলে)"),
           "f:SUP+": ("fa-", "ফা"), "f:CAUS+": ("fa- (because)", "ফা (কারণ)"), "bi+": ("bi- (with, by)", "বি (দ্বারা)"),
           "ka+": ("ka- (like)", "কা (মতো)"), "l:P+": ("li- (for, to)", "লি (জন্য)"), "l:PRP+": ("li- (so that)", "লি (যাতে)"),
           "l:EMPH+": ("la- (indeed)", "লা (নিশ্চয়ই)"), "l:IMPV+": ("li- (let)", "লি (আদেশ)"), "sa+": ("sa- (will)", "সা (ভবিষ্যৎ)"),
           "ta+": ("ta- (by, an oath)", "তা (শপথ)"), "ya+": ("yā (O)", "ইয়া (হে)"), "ha+": ("hā (attention)", "হা (মনোযোগ)"),
           "A:INTG+": ("a- (is it?)", "আ (প্রশ্ন)"), "A:EQ+": ("a- (whether)", "আ (সমতা)")}


def _pgn(code: str, lang: int) -> list[str]:
    """3MS → 3rd person, masculine, singular; MP → masculine, plural."""
    out = []
    for ch in code:
        for table in (_PERSON, _GENDER, _NUMBER):
            if ch in table:
                out.append(table[ch][lang])
                break
    return out


def _segment(tag: str, feats: list[str], lang: int) -> str:
    """One segment of a word in words: "verb, perfect (past), form IV, 3rd person, masculine, singular"."""
    if feats and feats[0] == "PREFIX":
        key = feats[1] if len(feats) > 1 else ""
        return _PREFIX.get(key, (f"prefix {key.rstrip('+')}", f"উপসর্গ {key.rstrip('+')}"))[lang]
    if feats and feats[0] == "SUFFIX":
        pron = next((f.split(":", 1)[1] for f in feats if f.startswith("PRON:")), "")
        if pron:
            return (("suffix pronoun: " if lang == 0 else "সর্বনাম প্রত্যয়: ") + ", ".join(_pgn(pron, lang))).strip()
        return POS.get(tag, (tag, tag))[lang]
    words = [POS.get(tag, (tag, tag))[lang]]
    for f in feats:
        if f in _ASPECT:
            words.append(_ASPECT[f][lang])
        elif re.fullmatch(r"\((I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\)", f):
            words.append(("form " if lang == 0 else "বাব ") + f.strip("()"))
        elif f == "PASS":
            words.append("passive" if lang == 0 else "কর্মবাচ্য")
        elif f == "PCPL":
            words.append("participle" if lang == 0 else "কৃদন্ত")
        elif f == "ACT":
            words.append("active" if lang == 0 else "কর্তৃবাচ্য")
        elif f == "VN":
            words.append("verbal noun" if lang == 0 else "ক্রিয়াবাচক বিশেষ্য (মাসদার)")
        elif f == "INDEF":
            words.append("indefinite" if lang == 0 else "অনির্দিষ্ট")
        elif f.startswith("MOOD:") and f[5:] in _MOOD:
            words.append(_MOOD[f[5:]][lang])
        elif f in _CASE:
            words.append(_CASE[f][lang])
        elif f and re.fullmatch(r"[123]?[MF]?[SDP]?", f):      # 3MS, MP, F, 1P…
            words.extend(_pgn(f, lang))
    return ", ".join(words)


# ---------- the file ----------
def path() -> str:
    return os.path.join(DIR, FILE_NAME)


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse(lines) -> tuple[dict, dict, list[str]]:
    words: dict = collections.defaultdict(dict)       # (s, a) → {w: [(form, tag, feats), ...]}
    header, problems = [], []
    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("#"):
            header.append(line)
            continue
        if not line.startswith("("):
            continue
        try:
            loc, form, tag, features = line.split("\t")
            s, a, w, _g = map(int, loc[1:-1].split(":"))
        except ValueError:
            problems.append(f"a line it can't read: {line[:40]!r}")
            continue
        words[(s, a)].setdefault(w, []).append((form, tag, features.split("|")))
    head = "\n".join(header)
    for must in HEADER:
        if must not in head:
            problems.append(f"its header doesn't say “{must}”: is it the Corpus's own file, unchanged?")
    out, roots = {}, collections.defaultdict(list)
    for (s, a), ws in words.items():
        if sorted(ws) != list(range(1, len(ws) + 1)):
            problems.append(f"{s}:{a}: its words aren't numbered 1 to {len(ws)}")
            continue
        out[(s, a)] = []
        for w in range(1, len(ws) + 1):
            segs = ws[w]
            stem = next((x for x in segs if x[2] and x[2][0] == "STEM"), segs[0])
            feats = stem[2]
            root = next((f.split(":", 1)[1] for f in feats if f.startswith("ROOT:")), "")
            lemma = next((f.split(":", 1)[1] for f in feats if f.startswith("LEM:")), "")
            pos = next((f.split(":", 1)[1] for f in feats if f.startswith("POS:")), stem[1])
            out[(s, a)].append({"segs": segs, "root": root, "lemma": lemma, "pos": pos})
            if root:
                roots[root].append((s, a, w))
    if len(out) != TOTAL_AYAHS:
        problems.append(f"it has {len(out)} ayahs, not {TOTAL_AYAHS}")
    return out, dict(roots), problems


def load(force: bool = False) -> dict:
    """Read the file once (in about a second), check it's the Corpus's own and whole, and index the roots. Its SHA-256
    is noted beside it the first time (data/quran_corpus/manifest.json): a file that changes later is said so."""
    with _lock:
        if _db and not force:
            return _db
        _db.clear()
        p = path()
        if not os.path.exists(p):
            _db.update(words={}, roots={}, problems=[f"the Corpus's file isn't there yet ({p})"], ready=False)
            return _db
        with open(p, encoding="utf-8") as f:
            words, roots, problems = _parse(f)
        digest = _sha(p)
        mpath = os.path.join(DIR, MANIFEST)
        try:
            with open(mpath, encoding="utf-8") as f:
                known = json.load(f).get("sha256")
        except (OSError, ValueError):
            known = None
            try:
                fd = os.open(mpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"file": FILE_NAME, "sha256": digest}, f)
            except OSError:
                pass
        if known and known != digest:
            problems.append("the file isn't the one first read here (its SHA-256 changed): if you downloaded a new "
                            "version, delete data/quran_corpus/manifest.json")
        _db.update(words=words, roots=roots, problems=problems, ready=not problems, sha256=digest)
        return _db


def forget() -> None:
    with _lock:
        _db.clear()


def status() -> dict:
    db = load()
    return {"ready": db["ready"], "problems": db["problems"], "credit": CREDIT, "link": LINK}


def ayah(s: int, a: int, count: int | None) -> list[dict] | None:
    """Each word of an ayah, as the word card shows it ({root, root_ar, lemma_ar, pos, en, bn}); None when the
    Corpus isn't there or its words don't match the reader's `count`."""
    db = load()
    if not db["ready"]:
        return None
    ws = db["words"].get((s, a))
    if not ws or count is None or len(ws) != count:
        return None
    return [describe(x) for x in ws]


def describe(x: dict) -> dict:
    segs = x["segs"]
    return {"root": x["root"], "root_ar": root_letters(x["root"]) if x["root"] else "",
            "lemma_ar": arabic(x["lemma"]), "pos": x["pos"],
            "pos_en": POS.get(x["pos"], (x["pos"], x["pos"]))[0], "pos_bn": POS.get(x["pos"], (x["pos"], x["pos"]))[1],
            "form_ar": arabic("".join(sg[0] for sg in segs)),
            "en": [_segment(tag, feats, 0) for _, tag, feats in segs],
            "bn": [_segment(tag, feats, 1) for _, tag, feats in segs],
            "count": len(load()["roots"].get(x["root"], [])) if x["root"] else 0}


def root_locations(root: str) -> list[tuple[int, int, int]]:
    return list(load()["roots"].get(root, []))


def word_at(s: int, a: int, w: int) -> dict | None:
    ws = load()["words"].get((s, a)) or []
    return describe(ws[w - 1]) if 1 <= w <= len(ws) else None
