"""
The Quran's structure, the recitation player's audio, and the rule that the Quran's words never come from a model
(2026-11-22, with the hifz tracker in hifz.py).

Structure: quran_meta.json (in the repo) has the 114 surahs (the Arabic name, the English and Bangla names people say,
the number of ayahs) and where each juz, manzil, hizb quarter and page of the Madani mushaf begins. It was built from
Tanzil's Quran metadata 1.0 (tanzil.net, Creative Commons Attribution 3.0) and is checked against Quran.com with
`python quran.py --check`, never copied from it: Quran Foundation's Developer Terms count its raw data in a public
source repository as redistribution, and this repository is public. Every ayah has an index 0..6235 (`index`,
`verse`); a portion is a list of (first, last) index ranges.

Audio: Quran.com lists each reciter's ayah files on public hosts (checked 2026-11-22: no key, no sign-up, open to a
web page): Husary (its recitation 6) on its QuranicAudio mirror of everyayah.com, Alafasy (7) on verses.quran.com.
`plan()` gives the page the files in order, the reciter's Bismillah (his Al-Fatiha 1:1) before a surah that opens with
one, and the same recording on everyayah.com as each file's backup. The page plays them in its own player: recitation
never goes through edge-tts, and Javid keeps no copy (the terms allow no rehosting and at most a week's cache). The
reciter and Quran Foundation are credited on the player.

Quran text: no model says the Quran's words, a translation or a verse number. A request for them is answered here (it
says so, links to Quran.com and offers to play it); a question about the structure (how many ayahs, which juz) is
answered from quran_meta.json; and every model reply is checked (reply_problem: Arabic script, a verse reference or an
ayah number, a quoted verse), rewritten once, then cut (strip_reply). javid.synthesize never gets Arabic script.
"""
import bisect
import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.request
from datetime import date

META_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quran_meta.json")
API = "https://api.quran.com/api/v4"     # only `python quran.py --check` calls it
USER_AGENT = "Javid/1.0 (personal voice assistant)"
MAX_ITEMS = 1200        # a play request covers at most this many files (about five juz)

# Unicode ranges, spelled out: Bengali, and the Arabic blocks (letters, presentation forms).
BN = f"{chr(0x0980)}-{chr(0x09FF)}"
AR = "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
                                                 (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)))
_BN_MARKS = "".join(map(chr, (0x09BC, 0x09CD, 0x0981, 0x0983)))      # nukta, hasanta, chandrabindu, visarga


def _nfc(x):
    """Bengali written in this file and Bengali Whisper writes are compared in one form (NFC)."""
    if isinstance(x, str):
        return unicodedata.normalize("NFC", x)
    if isinstance(x, dict):
        return {_nfc(k): v for k, v in x.items()}
    return type(x)(_nfc(v) for v in x)


def _re(pattern: str, flags=re.I) -> re.Pattern:
    return re.compile(_nfc(pattern), flags)


with open(META_FILE, encoding="utf-8") as _f:
    META = json.load(_f)
SURAHS = META["surahs"]
TOTAL = META["ayahs"]
_START = [0]            # the index of each surah's first ayah (and TOTAL at the end)
for _s in SURAHS:
    _START.append(_START[-1] + _s["ayahs"])
assert _START[-1] == TOTAL == 6236 and len(SURAHS) == 114


def index(s: int, a: int) -> int:
    """Surah s, ayah a (both from 1) as an index 0..6235."""
    if not (1 <= s <= 114 and 1 <= a <= SURAHS[s - 1]["ayahs"]):
        raise ValueError(f"{s}:{a} isn't an ayah")
    return _START[s - 1] + a - 1


def verse(i: int) -> tuple[int, int]:
    s = bisect.bisect_right(_START, i)
    return s, i - _START[s - 1] + 1


def key(i: int) -> str:
    s, a = verse(i)
    return f"{s}:{a}"


def surah(n: int) -> dict:
    return SURAHS[n - 1]


def surah_range(n: int) -> tuple[int, int]:
    return _START[n - 1], _START[n] - 1


def _ranges(starts: list) -> list[tuple[int, int]]:
    idx = [index(s, a) for s, a in starts] + [TOTAL]
    return [(idx[k], idx[k + 1] - 1) for k in range(len(starts))]


UNITS = {"juz": _ranges(META["juz"]), "manzil": _ranges(META["manzil"]), "quarter": _ranges(META["quarters"])}
COUNT = {"juz": 30, "manzil": 7}
_UNIT_START = {k: [r[0] for r in v] for k, v in UNITS.items()}
_PAGE_START = [index(s, a) for s, a in META["pages"]]


def unit_range(kind: str, n: int) -> tuple[int, int]:
    return UNITS[kind][n - 1]


def unit_of(kind: str, i: int) -> int:
    return bisect.bisect_right(_UNIT_START[kind], i)


def merge(ranges: list) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def units_in(kind: str, ranges: list) -> list[int]:
    """The juz (or manzils) the ranges cover whole."""
    whole = merge(ranges)
    return [n for n, (a, b) in enumerate(UNITS[kind], 1) if any(lo <= a and b <= hi for lo, hi in whole)]


def _page_pos(i: int) -> float:
    """Where index i begins, in pages of the Madani mushaf (0 = the first page's top, 604 = the end)."""
    if i >= TOTAL:
        return float(len(_PAGE_START))
    p = bisect.bisect_right(_PAGE_START, i)
    first, nxt = _PAGE_START[p - 1], (_PAGE_START[p] if p < len(_PAGE_START) else TOTAL)
    return p - 1 + (i - first) / max(1, nxt - first)


def page_of(i: int) -> int:
    return bisect.bisect_right(_PAGE_START, i)


def pages(ranges: list) -> float:
    """How many mushaf pages the ranges are (a juz is about 20), each ayah counted where its page puts it."""
    return sum(_page_pos(hi + 1) - _page_pos(lo) for lo, hi in ranges)


# ---------- names ----------
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn_digits(x) -> str:
    return str(x).translate(_TO_BN)


def name(n: int, lang: str = "en", spoken: bool = False) -> str:
    s = SURAHS[n - 1]
    if lang == "bn":
        return s["bn"]
    return s["en"].replace("'", "") if spoken else s["en"]


def surah_title(n: int, lang: str = "en", spoken: bool = False) -> str:
    return f"সূরা {name(n, 'bn')}" if lang == "bn" else f"Surah {name(n, 'en', spoken)}"


def span_text(lo: int, hi: int, lang: str = "en", spoken: bool = False) -> str:
    """Where a range starts and ends: "Yusuf 53 to Ibrahim 52", "Al-Baqarah 142 to 252", "all of Al-Mulk"."""
    (s1, a1), (s2, a2) = verse(lo), verse(hi)
    n1, n2 = name(s1, lang, spoken), name(s2, lang, spoken)
    whole = s1 == s2 and a1 == 1 and a2 == SURAHS[s1 - 1]["ayahs"]
    if lang == "bn":
        if whole:
            return f"পুরো সূরা {n1}"
        if s1 == s2:
            return f"{n1} {bn_digits(a1)} থেকে {bn_digits(a2)}"
        return f"{n1} {bn_digits(a1)} থেকে {n2} {bn_digits(a2)}"
    if whole:
        return f"all of {n1}"
    return f"{n1} {a1} to {a2}" if s1 == s2 else f"{n1} {a1} to {n2} {a2}"


def _key_en(s: str) -> str:
    """A spelling-proof key for an English name: Yaseen, Ya-Sin and yasin are one; so are Taubah and At-Tawbah."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"^(?:a[lnrstzd]|ash|adh|ath)[\s\-'`’‘]+", "", s)
    s = re.sub(r"[^a-z]", "", s)
    for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ou", "u"), ("kh", "k"),
                 ("gh", "g"), ("dh", "z"), ("th", "s"), ("ph", "f"), ("q", "k"), ("w", "u"), ("y", "i"), ("o", "u")):
        s = s.replace(a, b)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:-1] if len(s) > 3 and s.endswith("h") else s


_BN_MAP = str.maketrans(_nfc({"ঈ": "ই", "ী": "ি", "ঊ": "উ", "ূ": "ু", "য": "জ", "শ": "স", "ষ": "স", "ছ": "স",
                              "ণ": "ন", "ো": "ু", "ও": "উ", "ৌ": "ু", "ঐ": "ই", "ৈ": "ি", "ং": "ঙ"}))
_BN_ARTICLE = _re(r"^(?:আলে|আল|আন|আত|আর|আস|আশ|আয|আদ|আজ)[\s\-]+", 0)


def _key_bn(s: str) -> str:
    """The same for a Bangla name: ইয়াসিন and ইয়াসীন are one, so are কাহফ and আল-কাহফ, মুযযাম্মিল and মুজাম্মিল."""
    s = re.sub(f"[{_BN_MARKS}]", "", unicodedata.normalize("NFD", s))
    s = _BN_ARTICLE.sub("", unicodedata.normalize("NFC", s))
    s = re.sub(f"[^{BN}]", "", s).translate(_BN_MAP)
    s = s.replace("কব", "ক").replace("তব", "ত")          # ক্ব, ত্ব (ق, ط) once the hasanta is gone
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:-1] if len(s) > 2 and s.endswith("হ") else s


# Names that are also prayers, people or everyday words count only after "surah": "Surah Al-Asr", not a bare "Asr".
_NEEDS_SURAH_EN = {_key_en(w) for w in (
    "fajr", "asr", "jumuah", "juma", "sad", "qaf", "tin", "fil", "nas", "ala", "nun", "rum", "tur", "nur", "saba",
    "saff", "duha", "zuha", "layl", "insan", "naba", "amma", "mumin", "rad", "hajj", "jinn", "talaq", "fath", "qadr",
    "qadar", "nasr", "muhammad", "mohammed", "hud", "nuh", "maryam", "yusuf", "ibrahim", "yunus", "luqman", "fatah",
    "dahr", "lahab", "rahman", "rehman", "qalam", "kalam", "room", "noor", "hood", "teen", "feel")}
_NEEDS_SURAH_BN = {_key_bn(w) for w in (
    "ফজর", "ফাজর", "আসর", "জুমা", "জুমুআ", "সাদ", "কাফ", "ক্বাফ", "তিন", "তীন", "ফিল", "ফীল", "নাস", "আলা", "নূন",
    "রুম", "রূম", "তুর", "তূর", "নূর", "নুর", "সাবা", "সফ", "সাফ", "দুহা", "দোহা", "লাইল", "ইনসান", "নাবা", "মুমিন",
    "রাদ", "হজ", "জিন", "তালাক", "কদর", "নাসর", "মুহাম্মদ", "হুদ", "নূহ", "মারইয়াম", "ইউসুফ", "ইবরাহিম", "ইউনুস",
    "লুকমান", "রহমান", "কলম", "কালাম", "দাহর", "লাহাব")}


def _build_index() -> dict:
    idx: dict[str, int] = {}
    for s in SURAHS:
        keys = [("en:", _key_en(w)) for w in [s["en"], *s["aka"]]] + [("bn:", _key_bn(w)) for w in [s["bn"], *s["aka_bn"]]]
        for lang, k in keys:
            if k and idx.setdefault(lang + k, s["n"]) != s["n"]:
                raise ValueError(f"the name key {k!r} is both surah {idx[lang + k]} and {s['n']}")
    return idx


_NAMES = _build_index()
_BN_CASE = _nfc(("ের", "এর", "র", "কে", "টা", "টি", "ে", "য়"))


def find_name(words: list[str]) -> tuple[int, int, bool] | None:
    """(surah, words used, needs "surah" first) for a surah's name at the start of `words`: up to three words ("bani
    israil", "ha mim sajdah", "আলে ইমরান"), a Bengali one with its case ending (কাহফের). None if there's none."""
    for size in (3, 2, 1):
        if len(words) < size or any(w in (":", "-") or w.isdigit() for w in words[:size]):
            continue
        chunk = words[:size]
        bengali = any(re.search(f"[{BN}]", w) for w in chunk)
        tries = [" ".join(chunk)]
        if bengali:
            last = chunk[-1]
            tries += [" ".join(chunk[:-1] + [last[:-len(x)]]) for x in _BN_CASE if last.endswith(x) and len(last) > len(x) + 1]
        for t in tries:
            k = _key_bn(t) if bengali else _key_en(t)
            n = _NAMES.get(("bn:" if bengali else "en:") + k) if k else None
            if n is None and not bengali and re.match(r"^a[lnrstzd][a-z]{3,}$", t):       # "alkahf", written joined
                k = _key_en(t[2:])
                n = _NAMES.get("en:" + k)
            if n is None and bengali and t.startswith(_nfc("আল")) and len(t) > 4:
                k = _key_bn(t[2:])
                n = _NAMES.get("bn:" + k)
            if n is not None:
                return n, size, k in (_NEEDS_SURAH_BN if bengali else _NEEDS_SURAH_EN)
    return None


# ---------- words: text, numbers, portions ----------
def norm(text: str) -> str:
    """Lower case, Bengali digits as ASCII ones, punctuation as spaces (":" and "-" stay: 2:255, juz 5-7)."""
    t = unicodedata.normalize("NFC", text).translate(_BN_DIGITS).lower().replace("’", "'").replace("‘", "'")
    t = re.sub(f"[^\\w\\s:'\\-{BN}]", " ", t)
    t = re.sub(f"(?<=\\d)(?=[a-z{BN}])|(?<=[a-z{BN}])(?=\\d)", " ", t)     # "5ম", "5th", "juz5"
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    """Words, with a hyphenated name kept whole (Ya-Sin, Al-Mulk, ইয়া-সীন); digits, ":" and "-" apart (2:255, 5-7)."""
    return re.findall(f"[a-z]+(?:[-']+[a-z]+)*|[0-9]+|[{BN}]+(?:-[{BN}]+)*|:|-", norm(text))


_EN_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90}
_EN_ORD = {w: i for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_EN_ORD.update({"thirtieth": 30, "twenty-first": 21})
_BN_NUM = _nfc({w: i for i, w in enumerate(
    "শূন্য এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ এগারো বারো তেরো চৌদ্দ পনেরো ষোলো সতেরো আঠারো উনিশ বিশ একুশ বাইশ "
    "তেইশ চব্বিশ পঁচিশ ছাব্বিশ সাতাশ আঠাশ উনত্রিশ ত্রিশ".split())})
_BN_NUM.update(_nfc({"দু": 2, "ছ": 6, "ষোল": 16, "এগার": 11, "বার": 12, "তের": 13, "পনের": 15, "সতের": 17,
                     "আঠার": 18, "চোদ্দ": 14, "আটাশ": 28, "ঊনত্রিশ": 29, "ঊনিশ": 19}))
_BN_ORD = _nfc({"প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুর্থ": 4, "পঞ্চম": 5, "ষষ্ঠ": 6, "সপ্তম": 7, "অষ্টম": 8,
                "নবম": 9, "দশম": 10, "একাদশ": 11, "দ্বাদশ": 12})
_BANGLISH_NUM = {"ek": 1, "dui": 2, "tin": 3, "char": 4, "panch": 5, "pach": 5, "choy": 6, "chhoy": 6, "sat": 7,
                 "shat": 7, "aat": 8, "noy": 9, "dosh": 10, "egaro": 11, "baro": 12, "tero": 13, "choddo": 14,
                 "ponero": 15, "sholo": 16, "solo": 16, "shotero": 17, "sotero": 17, "atharo": 18, "unish": 19,
                 "bish": 20, "ekush": 21, "baish": 22, "teish": 23, "chobbish": 24, "pochish": 25, "chabbish": 26,
                 "satash": 27, "athash": 28, "untrish": 29, "trish": 30, "tirish": 30}
_TIMES_WORDS = _nfc({"once": 1, "twice": 2, "thrice": 3, "একবার": 1, "দুবার": 2, "দু'বার": 2, "তিনবার": 3, "চারবার": 4,
                     "পাঁচবার": 5, "দশবার": 10})
TIMES = _nfc({"times", "time", "বার", "baar", "bar", "x"})
ORD_TAIL = _nfc({"ম", "য়", "র্থ", "ষ্ঠ", "তম", "th", "st", "nd", "rd", "টা", "টি", "নং", "নম্বর", "নাম্বার", "number",
                 "no", "num", "তম"})


def number(words: list[str], i: int, banglish: bool = False) -> tuple[int, int] | None:
    """(value, words used) for the number at words[i]: 5, ৫, five, twenty five, fifth, পাঁচ, পঞ্চম; with banglish
    (next to a juz word only: "panch para") panch, baro too. None if there's none."""
    if i >= len(words):
        return None
    w = words[i]
    if w.isdigit():
        return int(w), 1
    if "-" in w and w.split("-")[0] in _EN_TENS and 0 < _EN_NUM.get(w.split("-")[1], 0) < 10:   # twenty-five
        return _EN_TENS[w.split("-")[0]] + _EN_NUM[w.split("-")[1]], 1
    if w in _EN_TENS:
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if nxt in _EN_NUM and 0 < _EN_NUM[nxt] < 10:
            return _EN_TENS[w] + _EN_NUM[nxt], 2
        if nxt == "-" and i + 2 < len(words) and 0 < _EN_NUM.get(words[i + 2], 0) < 10:
            return _EN_TENS[w] + _EN_NUM[words[i + 2]], 3
        return _EN_TENS[w], 1
    for table in (_EN_NUM, _EN_ORD, _BN_NUM, _BN_ORD):
        if w in table:
            return table[w], 1
    if banglish and w in _BANGLISH_NUM:
        return _BANGLISH_NUM[w], 1
    return None


def _skip_tail(words: list[str], j: int) -> int:
    while j < len(words) and words[j] in ORD_TAIL:
        j += 1
    return j


JUZ_WORDS = _nfc({"juz", "juzz", "jus", "joz", "jooz", "juzu", "juza", "jews", "juice", "para", "paara", "parah",
                  "parra", "sipara", "siparah", "পারা", "পাড়া", "জুজ", "জুয", "জুয়", "সিপারা", "পারাহ", "পারার",
                  "পারাটা", "পারাটি"})
MANZIL_WORDS = _nfc({"manzil", "manzel", "manjil", "mamzil", "manzils", "মঞ্জিল", "মনজিল", "মানজিল", "মঞ্জিলের",
                     "মঞ্জিলটা"})
SURAH_WORDS = _nfc({"surah", "sura", "surat", "soorah", "soora", "suraa", "সূরা", "সুরা", "ছুরা", "সুরাহ", "সূরাহ",
                    "সুরাত", "সূরাটা", "সূরাটি"})
AYAH_WORDS = _nfc({"ayah", "ayat", "aya", "ayaat", "ayahs", "ayats", "verse", "verses", "ayas", "আয়াত", "আয়াহ",
                   "আয়াতে", "আয়াতটি", "আয়াতটা", "আয়াতের", "আয়াতগুলো"})
RANGE = _nfc({"to", "-", "through", "thru", "till", "until", "upto", "থেকে", "হতে", "theke", "porjonto", "পর্যন্ত"})
AND = _nfc({"and", "&", "আর", "ও", "এবং", "ar", "o", "plus"})
_FROM = _nfc({"from", "starting", "start", "থেকে", "theke"})
_ONWARD = _nfc({"onwards", "onward", "on", "শেষ", "end", "পর্যন্ত"})


def _numbers(words: list[str], i: int, banglish: bool) -> tuple[list[int], int]:
    """A number, a range ("5 to 7", "৫ থেকে ৭") or a list ("5 and 6", "5, 6 and 7") at words[i]: (values, words)."""
    first = number(words, i, banglish)
    if not first:
        return [], 0
    vals = [first[0]]
    j = _skip_tail(words, i + first[1])
    if j < len(words) and words[j] in RANGE and (to := number(words, j + 1, banglish)):
        if vals[0] <= to[0] <= vals[0] + 30:
            return list(range(vals[0], to[0] + 1)), _skip_tail(words, j + 1 + to[1]) - i
        return vals, j - i
    while j < len(words) and words[j] in AND and (more := number(words, j + 1, banglish)):
        vals.append(more[0])
        j = _skip_tail(words, j + 1 + more[1])
    return vals, j - i


def _unit_portion(kind: str, n: int) -> dict | None:
    if not 1 <= n <= COUNT[kind]:
        return None
    return {"kind": kind, "n": n, "ranges": [unit_range(kind, n)]}


unit_portion = _unit_portion


def _surah_portion(n: int, a1: int | None = None, a2: int | None = None) -> dict:
    count = SURAHS[n - 1]["ayahs"]
    a1 = 1 if a1 is None else a1
    a2 = count if a2 is None else a2
    if not (1 <= a1 <= a2 <= count):
        return {"kind": "bad_ayah", "n": n, "a1": a1, "a2": a2, "count": count, "ranges": []}
    p = {"kind": "surah", "n": n, "ranges": [(index(n, a1), index(n, a2))]}
    if (a1, a2) != (1, count):
        p.update(a1=a1, a2=a2)
    return p


def _ayah_spec(words: list[str], k: int, count: int) -> tuple[int | None, int | None, int]:
    """The ayahs after a surah's name: "ayah 5", "ayahs 5 to 10", "5-10", "from ayah 10", "10 onwards",
    "১০ নম্বর আয়াত থেকে". (first, last, where it ends); (None, None, k) when there's none."""
    start = k
    from_word = k < len(words) and words[k] in _FROM - {"থেকে"}
    if from_word:
        k += 1
    ayah_word = k < len(words) and words[k] in AYAH_WORDS
    if ayah_word:
        k += 1
    n = number(words, k)
    if not n or not (ayah_word or from_word or words[k].isdigit()):
        return None, None, start
    j = _skip_tail(words, k + n[1])
    if j < len(words) and words[j] in TIMES:        # "surah ikhlas 3 times": a count, not an ayah
        return None, None, start
    if j < len(words) and words[j] in AYAH_WORDS:
        j += 1
    a1 = n[0]
    if j < len(words) and words[j] in RANGE and words[j] != "থেকে" and (to := number(words, j + 1)):
        j = _skip_tail(words, j + 1 + to[1])
        if j < len(words) and words[j] in AYAH_WORDS:
            j += 1
        return a1, to[0], j
    if j < len(words) and words[j] == "থেকে":     # ১০ থেকে ১৫, or ১০ নম্বর আয়াত থেকে (to the end)
        if (to := number(words, j + 1)):
            j = _skip_tail(words, j + 1 + to[1])
            if j < len(words) and words[j] in AYAH_WORDS:
                j += 1
            return a1, to[0], j
        j += 1
        while j < len(words) and words[j] in _ONWARD:
            j += 1
        return a1, count, j
    if j < len(words) and words[j] in _ONWARD:
        return a1, count, j + 1
    return (a1, count, j) if from_word else (a1, a1, j)


def portions(text: str, words: list[str] | None = None, bare_names: bool = True) -> tuple[list[dict], set[int]]:
    """The portions a message names, and the positions of the words that named them: juz/para N (ranges, lists,
    "amma para"), manzil N, Surah X (with "ayah 5", "5 to 10", "from ayah 10", "first 10 ayahs of"), 2:255. A bare
    surah name counts too (bare_names) unless it's also a prayer, a person or an everyday word."""
    words = tokens(text) if words is None else words
    out, used = [], set()
    i = 0
    while i < len(words):
        w = words[i]
        # juz / para / manzil and its number after: "juz 5", "para 12 to 14", "manzil 3"
        if w in JUZ_WORDS or w in MANZIL_WORDS:
            kind = "juz" if w in JUZ_WORDS else "manzil"
            vals, n_used = _numbers(words, i + 1, True)
            got = [p for p in (_unit_portion(kind, v) for v in vals) if p]
            if got:
                out += got
                used.update(range(i, i + 1 + n_used))
                i += 1 + n_used
                continue
        # "amma para", "tabarak para", "last juz", "আমপারা"
        if w in _nfc(("amma", "tabarak", "last", "first", "শেষ", "প্রথম", "তাবারাক", "আম্মা")) and \
                i + 1 < len(words) and words[i + 1] in JUZ_WORDS:
            n = 30 if w in _nfc(("amma", "last", "শেষ", "আম্মা")) else 29 if w in _nfc(("tabarak", "তাবারাক")) else 1
            out.append(_unit_portion("juz", n))
            used.update((i, i + 1))
            i += 2
            continue
        if w in _nfc(("ampara", "আমপারা", "আমপারার", "আমপারাটা")):
            out.append(_unit_portion("juz", 30))
            used.add(i)
            i += 1
            continue
        # a number first: "5 para", "৫ নম্বর পারা", "5th juz", "৫ থেকে ৭ পারা"; or 2:255, 67:1-5
        vals, n_used = _numbers(words, i, True)
        if vals:
            j = i + n_used
            if j < len(words) and (words[j] in JUZ_WORDS or words[j] in MANZIL_WORDS):
                kind = "juz" if words[j] in JUZ_WORDS else "manzil"
                got = [p for p in (_unit_portion(kind, v) for v in vals) if p]
                if got:
                    out += got
                    used.update(range(i, j + 1))
                    i = j + 1
                    continue
            j = i + 1
            if words[i].isdigit() and j < len(words) and words[j] == ":" and (a := number(words, j + 1)) \
                    and words[j + 1].isdigit() and 1 <= int(words[i]) <= 114:
                a2, k = a[0], j + 2
                if k < len(words) and words[k] in RANGE and (b := number(words, k + 1)) and words[k + 1].isdigit():
                    a2, k = b[0], k + 2
                out.append(_surah_portion(int(words[i]), a[0], a2))
                used.update(range(i, k))
                i = k
                continue
        # "first 10 ayahs of Surah Al-Kahf", "last ten ayahs of kahf"
        if w in _nfc(("first", "last", "প্রথম", "শেষ")) and (cnt := number(words, i + 1)) and \
                i + 1 + cnt[1] < len(words) and words[i + 1 + cnt[1]] in AYAH_WORDS:
            j = i + 2 + cnt[1]
            if j < len(words) and words[j] in ("of", "from", "in"):
                j += 1
            has_word = j < len(words) and words[j] in SURAH_WORDS
            j += has_word
            found = find_name(words[j:])
            if found and (has_word or not found[2]):
                n, size = found[0], found[1]
                count = SURAHS[n - 1]["ayahs"]
                k = min(cnt[0], count)
                first = w in _nfc(("first", "প্রথম"))
                out.append(_surah_portion(n, 1 if first else count - k + 1, k if first else count))
                used.update(range(i, j + size))
                i = j + size
                continue
        # Surah X, or a bare name; then its ayahs
        has_word = w in SURAH_WORDS
        j = i + 1 if has_word else i
        found = find_name(words[j:]) if j < len(words) else None
        if not found and has_word and j < len(words) and words[j].isdigit() and 1 <= int(words[j]) <= 114:
            found = (int(words[j]), 1, False)
        if found and (has_word or (bare_names and not found[2])):
            n, size = found[0], found[1]
            k = j + size
            # Bengali puts the ayahs after a case ending: "সূরা কাহফের প্রথম ১০ আয়াত"
            if k < len(words) and words[k] in _nfc(("প্রথম", "শেষ")) and (cnt := number(words, k + 1)) and \
                    k + 1 + cnt[1] < len(words) and words[k + 1 + cnt[1]] in AYAH_WORDS:
                count = SURAHS[n - 1]["ayahs"]
                c = min(cnt[0], count)
                first = words[k] == _nfc("প্রথম")
                out.append(_surah_portion(n, 1 if first else count - c + 1, c if first else count))
                used.update(range(i, k + 2 + cnt[1]))
                i = k + 2 + cnt[1]
                continue
            a1, a2, k = _ayah_spec(words, k, SURAHS[n - 1]["ayahs"])
            out.append(_surah_portion(n, a1, a2))
            used.update(range(i, k))
            i = k
            continue
        i += 1
    return out, used


def portion_label(p: dict, lang: str = "en", spoken: bool = False) -> str:
    if p["kind"] == "juz":
        return f"পারা {bn_digits(p['n'])}" if lang == "bn" else f"juz {p['n']}"
    if p["kind"] == "manzil":
        return f"মঞ্জিল {bn_digits(p['n'])}" if lang == "bn" else f"manzil {p['n']}"
    if p.get("a1"):
        if p["a1"] == p["a2"]:
            return (f"সূরা {name(p['n'], 'bn')}, আয়াত {bn_digits(p['a1'])}" if lang == "bn" else
                    f"{name(p['n'], 'en', spoken)}, ayah {p['a1']}")
        return (f"সূরা {name(p['n'], 'bn')} {bn_digits(p['a1'])}–{bn_digits(p['a2'])}" if lang == "bn" else
                f"{name(p['n'], 'en', spoken)} {p['a1']}{' to ' if spoken else '–'}{p['a2']}")
    return surah_title(p["n"], lang, spoken)


def join(items: list[str], lang: str = "en") -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {'আর' if lang == 'bn' else 'and'} {items[-1]}"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


# ---------- recitation ----------
RECITERS = {
    "husary": {"name": "Mahmoud Khalil Al-Husary", "short": "Husary", "name_bn": "মাহমুদ খলিল আল-হুসারি",
               "qurancom": 6, "url": "https://mirrors.quranicaudio.com/everyayah/Husary_64kbps/{s:03d}{a:03d}.mp3",
               "backup": "https://everyayah.com/data/Husary_128kbps/{s:03d}{a:03d}.mp3",
               "words": _nfc(("husary", "husari", "hussary", "hussari", "husory", "hosary", "husri", "husaree",
                              "khalil", "হুসারি", "হুসারী", "হুসরি", "হোসারি", "হুসারির", "হুসারীর"))},
    "alafasy": {"name": "Mishary Rashid Alafasy", "short": "Alafasy", "name_bn": "মিশারি রাশিদ আল-আফাসি",
                "qurancom": 7, "url": "https://verses.quran.com/Alafasy/mp3/{s:03d}{a:03d}.mp3",
                "backup": "https://everyayah.com/data/Alafasy_128kbps/{s:03d}{a:03d}.mp3",
                "words": _nfc(("alafasy", "afasy", "alafasi", "afasi", "mishary", "mishari", "mishary's", "mishari's",
                               "rashid", "মিশারি", "মিশারী", "আফাসি", "আলাফাসি", "আফাসির", "মিশারির", "মিশারীর"))},
}
DEFAULT_RECITER = "husary"
CREDIT = "Quran.com (Quran Foundation)"
BACKUP_CREDIT = "everyayah.com"


def reciter_in(words: list[str]) -> tuple[str, set[int]] | None:
    for i, w in enumerate(words):
        for rid, r in RECITERS.items():
            if w in r["words"]:
                return rid, {i}
    return None


def _file(rid: str, s: int, a: int) -> dict:
    r = RECITERS[rid]
    return {"key": f"{s}:{a}", "s": s, "a": a, "url": r["url"].format(s=s, a=a), "backup": r["backup"].format(s=s, a=a)}


def plan(ranges: list, rid: str | None = None, title: str = "", lang: str = "en", times: int = 1) -> dict:
    """What the page's player plays, in order: each ayah's file (and its backup), the reciter's Bismillah before a
    surah that opens with one (not Al-Fatiha, where it's the first ayah, nor At-Tawbah); at most MAX_ITEMS files.
    times: the whole of it, that many times over ("Surah Al-Ikhlas three times")."""
    rid = rid if rid in RECITERS else DEFAULT_RECITER
    items = []
    for lo, hi in ranges:
        for i in range(lo, hi + 1):
            if len(items) >= MAX_ITEMS:
                break
            s, a = verse(i)
            if a == 1 and s not in (1, 9):
                items.append({**_file(rid, 1, 1), "key": f"{s}:0", "s": s, "a": 0, "bismillah": True})
            items.append(_file(rid, s, a))
    r = RECITERS[rid]
    first, last = verse(ranges[0][0]), verse(ranges[-1][1])
    touched = sorted({verse(i)[0] for lo, hi in ranges for i in (lo, hi)} |
                     {n for lo, hi in ranges for n in range(verse(lo)[0], verse(hi)[0] + 1)})
    return {"title": title, "reciter": rid, "reciter_name": r["name_bn"] if lang == "bn" else r["name"],
            "credit": CREDIT, "backup_credit": BACKUP_CREDIT, "items": items, "times": max(1, min(int(times), 20)),
            "first": f"{first[0]}:{first[1]}", "last": f"{last[0]}:{last[1]}",
            "surahs": {str(n): {"en": name(n), "bn": name(n, "bn"), "ar": surah(n)["ar"], "ayahs": surah(n)["ayahs"]}
                       for n in touched}}


# ---------- requests answered in code: play, the player's controls, the structure, Quran text ----------
_PLAY_WORDS = _nfc({"play", "recite", "listen", "hear", "শোনাও", "শুনাও", "চালাও", "বাজাও", "প্লে", "শোনান", "শুনান",
                    "শোনাবে", "শুনাবে", "চালাবে", "চালান", "sunao", "shunao", "chalao", "shonao", "shunaw"})
_FILLER = _nfc({
    "the", "a", "an", "please", "javid", "jabid", "javed", "jawed", "hey", "ok", "okay", "now", "it", "this", "that",
    "to", "me", "for", "can", "you", "could", "would", "will", "i", "want", "wanna", "let", "lets", "let's", "on",
    "with", "by", "in", "of", "recitation", "quran", "qur'an", "koran", "reciter", "voice", "audio", "some", "just",
    "up", "out", "loud", "aloud", "full", "whole", "entire", "beautiful", "kindly", "my", "from", "go", "ahead",
    "listen", "hear", "recite", "play", "start", "put", "us", "again", "sheikh", "shaikh", "qari", "imam", "by",
    "reading", "recited", "recitation's", "so", "and", "also", "then", "do", "like", "love", "would", "be", "great",
    "জাভিদ", "জাভেদ", "একটু", "দয়া", "করে", "করো", "কর", "করুন", "দাও", "দিন", "তো", "এটা", "এই", "আমাকে", "আমার", "জন্য",
    "কুরআন", "কোরআন", "তিলাওয়াত", "পুরো", "সম্পূর্ণ", "দিয়ে", "কন্ঠে", "কণ্ঠে", "গলায়", "এখন", "শুনতে", "চাই",
    "শুনব", "শোনাতে", "ক্বারি", "কারী", "শায়খ", "koro", "kor", "pura", "puro", "ekhon", "diye", "dao", "ta", "ti",
    "please", "ektu"})
_TODAY = _re(r"\btoday'?s? (?:revision|portion|juz|para|lesson|manzil|sabaq|amukhta)\b|\bwhat'?s due\b|"
             r"আজকের (?:পারা|রিভিশন|পড়া|অংশ)|ajker (?:para|revision)")
_TODAY_WORDS = _nfc({"today's", "todays", "today", "revision", "portion", "juz", "para", "lesson", "manzil", "due",
                     "what's", "whats", "sabaq", "amukhta", "আজকের", "পারা", "রিভিশন", "পড়া", "অংশ", "ajker"})
_NAMED_PASSAGE = _re(r"\bayat ?(?:ul|al|-al|ul-)? ?kursi\b|\bayatul ?kursi\b|\bkursi\b|\bamana ?(?:r|al) ?rasul\b|"
                     r"আয়াতুল কুরসি|আয়াতুল কুরসী|কুরসি|আমানার রাসূল")
_TEXT_ASK = _re(
    r"\b(?:translate|translation|translated|meaning|meanings|tafsir|tafseer|tafsiir|interpret|interpretation|"
    r"explain|explanation|transliteration)\b.{0,60}\b(?:surah|sura|ayah|ayat|aya|verse|verses|quran|qur'an|koran)\b|"
    r"\b(?:surah|sura|ayah|ayat|verse|verses|quran|qur'an|koran)\b.{0,60}\b(?:translate|translation|meaning|tafsir|"
    r"tafseer|interpretation|transliteration)\b|"
    r"\bwhat (?:does|do|did) (?:the )?(?:quran|qur'an|koran|allah|god|surah [\w'-]+|ayah [\w'-]+|verse [\w'-]+)"
    r"(?: [\w'-]+){0,3} (?:say|says|said|mean|tell|teach)\b|"
    r"\b(?:recite|quote|read|tell|give|show|write|send|share|say|text of)\b.{0,40}\b(?:ayah|ayat|aya|verse|verses|"
    r"ayahs|ayats|ayatul)\b|"
    r"\b(?:a|any|some|one|the best|a few) (?:quranic |qur'anic )?(?:verse|ayah|ayat|verses|ayahs|quote|quotes) "
    r"(?:about|on|for|from|of|that|which)\b|"
    r"\b(?:quran|qur'an|koran)(?:ic)? (?:verse|verses|quote|quotes|text|words)\b|\bthe quran says\b|"
    r"\bwhat(?:'s| is| are| was)\b.{0,25}\b(?:ayah|ayat|aya|verse|verses|ayahs)\b|"
    r"অনুবাদ|তরজমা|তাফসির|তাফসীর|অর্থ কী|অর্থ কি|মানে কী|মানে কি|"
    r"(?:কুরআনে|কোরআনে|কুরআন|কোরআন).{0,20}(?:কী বলা|কি বলা|কী বলে|কি বলে|কী আছে|কি আছে)|"
    r"আয়াত.{0,25}(?:বলো|বলুন|লিখে|লিখো|কী|কি)")
_STRUCTURE_AYAHS = _re(r"\bhow many (?:ayahs?|ayats?|ayaat|verses?|ayas)\b|\b(?:ayahs?|ayats?|verses?) (?:are )?"
                       r"(?:there )?in\b|\bhas how many\b|কয়টি আয়াত|কতটি আয়াত|কত আয়াত|কয়টা আয়াত|কতগুলো আয়াত|"
                       r"koyta ayat|koto ayat")
_STRUCTURE_WHERE = _re(r"\b(?:which|what) (?:juz|para|parah|manzil)\b|\bin which (?:juz|para|manzil)\b|"
                       r"\bwhere (?:does|do|is)\b|\b(?:starts?|begins?|ends?) where\b|"
                       r"কোন পারা|কোন পারায়|কোন পারাতে|কোথা থেকে শুরু|কোথায় শুরু|কোথায় শেষ|kon para")
_STRUCTURE_COUNT = _re(r"\bhow many (?:surahs?|suras?|ayahs?|verses?|juz|paras?|manzils?) (?:are )?(?:there )?"
                       r"(?:in|does) (?:the )?(?:whole )?(?:quran|qur'an|koran)\b|"
                       r"কুরআনে (?:কয়টি|কতটি|কত) (?:সূরা|আয়াত|পারা)")
_QUESTION_WORDS = _nfc({
    "how", "many", "are", "is", "there", "in", "which", "what", "where", "does", "do", "start", "starts", "begin",
    "begins", "end", "ends", "the", "of", "has", "have", "it", "a", "find", "located", "can", "i", "number", "javid",
    "please", "tell", "me", "contain", "contains", "kon", "koyta", "koto", "ayat", "ayahs", "ayah", "ayats", "verses",
    "verse", "পারা", "পারায়", "পারাতে", "কোন", "কয়টি", "কতটি", "কত", "কয়টা", "কতগুলো", "আছে", "কোথা", "থেকে",
    "শুরু", "শেষ", "কোথায়", "হয়", "তে", "এ", "এর", "এই", "ওই", "কি", "কী", "and", "juz", "para", "manzil"}) | SURAH_WORDS | AYAH_WORDS


def _player_active(player) -> bool:
    return isinstance(player, dict) and bool(player.get("active"))


def leftover(words: list[str], used: set[int], allowed: set[str]) -> list[str]:
    return [w for i, w in enumerate(words) if i not in used and w not in allowed]


def request(text: str, player: dict | None = None) -> dict | None:
    """What a message asks of the recitation player or about the Quran, when code can answer all of it; else None.
    player: the page's player as it is ({active, playing, key, surah, ayah, reciter}), for "next ayah", "repeat this
    ayah three times", "pause"."""
    t = norm(text)
    words = tokens(text)
    if not words:
        return None
    if (req := _control(words, t, player)):
        return req
    if (req := _play(words, t)):
        return req
    structure = _STRUCTURE_AYAHS.search(t) or _STRUCTURE_WHERE.search(t)
    if _NAMED_PASSAGE.search(t) or (_TEXT_ASK.search(t) and not structure):
        return {"action": "text", "portions": portions(text, words)[0]}
    if _STRUCTURE_COUNT.search(t):
        return {"action": "count"}
    if structure:
        found, used = portions(text, words)
        if found and not leftover(words, used, _QUESTION_WORDS):
            return {"action": "ayahs" if _STRUCTURE_AYAHS.search(t) else "where", "portions": found}
    return None


def _play(words: list[str], t: str) -> dict | None:
    verbs = {i for i, w in enumerate(words) if w in _PLAY_WORDS}
    if not verbs:
        return None
    rec = reciter_in(words)
    times = times_in(words)
    used = verbs | (rec[1] if rec else set()) | times[1]
    if _TODAY.search(t):
        today = {i for i, w in enumerate(words) if w in _TODAY_WORDS}
        if not leftover(words, used | today, _FILLER):
            return {"action": "play", "today": True, "portions": [], "reciter": rec[0] if rec else None,
                    "times": times[0] or 1}
        return None
    found, named = portions(" ".join(words), words)
    if not found or leftover(words, used | named, _FILLER | AND):
        return None
    bad = next((p for p in found if p["kind"] == "bad_ayah"), None)
    return {"action": "play", "today": False, "portions": [p for p in found if p["kind"] != "bad_ayah"],
            "reciter": rec[0] if rec else None, "bad": bad, "times": times[0] or 1}


def times_in(words: list[str]) -> tuple[int | None, set[int]]:
    """"3 times", "three times", "thrice", "৩ বার", "তিনবার": (count, its words)."""
    for i, w in enumerate(words):
        if w in _TIMES_WORDS:
            return _TIMES_WORDS[w], {i}
        if w in TIMES and w != "x" and i > 0 and (n := number(words, i - 1, True)):
            return n[0], {i - 1, i}
    return None, set()


_C = {k: _nfc(v) for k, v in {
    "pause": {"pause", "hold", "wait", "পজ", "থামো", "থামুন", "thamo", "thamen", "pouse"},
    "stop": {"stop", "enough", "বন্ধ", "থামাও", "bondho", "bandho", "off", "স্টপ"},
    "resume": {"resume", "continue", "unpause", "chalao", "chalu", "চালাও", "চালু", "চালিয়ে", "play", "resumes"},
    "next": {"next", "skip", "forward", "পরের", "পরেরটা", "porer", "porerta"},
    "prev": {"previous", "back", "before", "আগের", "আগেরটা", "ager", "agerta", "prev"},
    "repeat": {"repeat", "again", "রিপিট", "abar", "আবার", "replay"},
}.items()}
_CTL_EXTRA = _nfc({"this", "that", "it", "a", "bit", "moment", "second", "minute", "go", "to", "jump", "skip",
                   "from", "start", "at", "the", "once", "more", "one", "is", "on", "keep", "going", "carry", "turn",
                   "আয়াত", "টা", "এটা", "এই", "একটু", "করো", "থেকে", "koro", "ta", "dao", "দাও", "it's", "its",
                   "ayah", "verse", "ayat", "please", "recitation", "quran", "koran", "qur'an", "the", "now", "for",
                   "switch", "change", "use", "try", "instead", "reciter", "to", "with", "যাও", "শুরু", "করুন"})
_NAMED_PLAYER = _re(r"\brecitation\b|\breciting\b|\bquran\b|\bqur'an\b|\bkoran\b|তিলাওয়াত|কুরআন|কোরআন")


def _control(words: list[str], t: str, player) -> dict | None:
    """The player's commands. Bare ones ("pause", "next", "repeat") only while the page has a recitation open; with
    "the recitation" or "the Quran" in them ("stop the recitation") also when it hasn't, so Javid can say nothing's
    playing."""
    active = _player_active(player)
    if not active and not _NAMED_PLAYER.search(t):
        return None
    found, _ = portions(" ".join(words), words, bare_names=False)
    if found:
        return None         # "play Surah Ya-Sin": a new play request
    times = times_in(words)
    ayah = ayah_number(words, exclude=times[1])
    allowed = _FILLER | AYAH_WORDS | _CTL_EXTRA | set().union(*_C.values())
    taken = times[1] | (ayah[1] if ayah else set())

    def clean(extra: set = frozenset()) -> bool:
        return not leftover(words, taken | set(extra), allowed)
    has = {k for k, ws in _C.items() if any(w in ws for w in words)}
    rec = reciter_in(words)
    if rec and not ayah and clean(rec[1]):
        return {"action": "reciter", "reciter": rec[0], "active": active, "player": player}
    seek = ayah and (any(w in _nfc(("from", "থেকে", "theke")) for w in words) or
                     re.search(r"\b(?:go to|jump to|skip to|start at|take me to)\b", t))
    if seek and clean():
        return {"action": "seek", "ayah": ayah[0], "active": active, "player": player}
    resume_verb = bool(has & {"resume"}) and any(w in _C["resume"] - {"play"} for w in words)
    if (("repeat" in has and not resume_verb) or times[0]) and clean():
        return {"action": "repeat", "ayah": ayah[0] if ayah else None, "times": max(1, min(times[0] or 1, 20)),
                "active": active, "player": player}
    for k in ("next", "prev", "pause", "stop", "resume"):
        if k in has and not ayah and clean():
            return {"action": k, "active": active}
    return None


def ayah_number(words: list[str], exclude: set[int] = frozenset()) -> tuple[int, set[int]] | None:
    """"ayah 5", "verse 5", "৫ নম্বর আয়াত", "the 5th ayah": the number and its words (not the count of "3 times")."""
    for i, w in enumerate(words):
        if w not in AYAH_WORDS:
            continue
        if i + 1 not in exclude and (n := number(words, i + 1)):
            return n[0], set(range(i, i + 1 + n[1]))
        j = i - 1
        while j >= 0 and words[j] in ORD_TAIL:
            j -= 1
        if j >= 0 and j not in exclude and (n := number(words, j)):
            return n[0], set(range(j, i + 1))
    return None


def answer(req: dict, lang: str):
    """A generator: its ("tool", "quran"), ("recite", command) and ("sources", links) events, then (spoken, screen).
    A "recite" command is the page player's: play (with its plan), pause, resume, stop, next, prev, repeat, seek,
    reciter. quiet: the reply is shown, not spoken (Javid doesn't talk over a recitation he has just started)."""
    yield ("tool", "quran")
    act = req["action"]
    if act == "text":
        return (yield from _text_answer(req, lang))
    if act == "count":
        return (f"কুরআনে ১১৪টি সূরা, {bn_digits(f'{TOTAL:,}')}টি আয়াত (হাফসের গণনায়), ৩০টি পারা আর ৭টি মঞ্জিল।"
                if lang == "bn" else
                f"The Quran has 114 surahs and {TOTAL:,} ayahs (in Hafs's count), in 30 juz and 7 manzils."), ""
    if act in ("ayahs", "where"):
        return _structure(req, lang), ""
    if act == "play":
        return (yield from _play_answer(req, lang))
    return (yield from _control_answer(req, lang))


def _structure(req: dict, lang: str) -> str:
    said = []
    for p in req["portions"]:
        if p["kind"] == "bad_ayah":
            said.append(f"সূরা {name(p['n'], 'bn')}-এ {bn_digits(p['count'])}টি আয়াত।" if lang == "bn" else
                        f"{name(p['n'], 'en', True)} has {p['count']} ayahs.")
            continue
        lo, hi = p["ranges"][0]
        juzs = list(range(unit_of("juz", lo), unit_of("juz", hi) + 1))
        jt = join([bn_digits(j) if lang == "bn" else str(j) for j in juzs], lang)
        if p["kind"] == "surah" and not p.get("a1"):
            s = SURAHS[p["n"] - 1]
            if req["action"] == "ayahs":
                said.append(f"সূরা {s['bn']}-এ {bn_digits(s['ayahs'])}টি আয়াত, পারা {jt}-এ।" if lang == "bn" else
                            f"{name(s['n'], 'en', True)} has {s['ayahs']} ayahs. It's in juz {jt}.")
            else:
                said.append(f"সূরা {s['bn']} পারা {jt}-এ।" if lang == "bn" else
                            f"{name(s['n'], 'en', True)} is in juz {jt}" +
                            (f": {span_text(lo, hi, 'en', True)}." if len(juzs) > 1 else "."))
                if len(juzs) > 1 and lang != "bn":
                    said[-1] = (f"{name(s['n'], 'en', True)} runs across juz {jt}: juz {juzs[1]} begins at "
                                f"{span_text(UNITS['juz'][juzs[1] - 1][0], UNITS['juz'][juzs[1] - 1][0], 'en', True).split(' to ')[0]}.")
        elif p["kind"] == "surah":
            said.append(f"{portion_label(p, 'bn')} পারা {jt}-এ।" if lang == "bn" else
                        f"{cap(portion_label(p, 'en', True))} is in juz {jt}.")
        else:
            said.append(f"{portion_label(p, 'bn')}: {span_text(lo, hi, 'bn')}।" if lang == "bn" else
                        f"{cap(portion_label(p))} runs from {span_text(lo, hi, 'en', True)}.")
    return " ".join(said)


def link(p: dict | None) -> dict:
    """The Quran.com page for a portion (a source chip under the reply)."""
    if p and p["kind"] == "surah":
        tail = (f"/{p['a1']}" if p.get("a1") and p["a1"] == p["a2"] else f"/{p['a1']}-{p['a2']}" if p.get("a1") else "")
        url, title = f"https://quran.com/{p['n']}{tail}", f"{portion_label(p)} on Quran.com"
    elif p and p["kind"] == "juz":
        url, title = f"https://quran.com/juz/{p['n']}", f"Juz {p['n']} on Quran.com"
    else:
        url, title = "https://quran.com", "Quran.com"
    return {"n": 1, "url": url, "domain": "quran.com", "name": "Quran.com", "title": title, "official": False,
            "date": None, "date_text": ""}


def _text_answer(req: dict, lang: str):
    p = next((x for x in req["portions"] if x["kind"] in ("surah", "juz")), None)
    yield ("sources", [link(p)])
    target = (surah_title(p["n"], lang, True) if p["kind"] == "surah" else portion_label(p, lang)) if p else \
        surah_title(67, lang, True)
    play = f"“{target} চালাও”" if lang == "bn" else f"“play {target}”"
    if lang == "bn":
        return (f"কুরআনের শব্দ, অনুবাদ বা আয়াতের নম্বর আমি নিজে থেকে বলি না, যাতে একটি শব্দও ভুল না হয়। Quran.com-এ "
                f"পড়তে পারো, লিংক নিচে; আর শুনতে চাইলে বলো {play}।"), ""
    return ("I don't quote, translate or number the Quran myself, so I never get a word of it wrong. You can read it on "
            f"Quran.com (the link is below), and I can play it for you: say {play}."), ""


def _play_answer(req: dict, lang: str):
    import hifz      # the default reciter and today's portion are the tracker's
    rid = req.get("reciter") or hifz.config()["reciter"]
    if req.get("bad") and not req["portions"]:
        b = req["bad"]
        return (f"সূরা {name(b['n'], 'bn')}-এ {bn_digits(b['count'])}টি আয়াত।" if lang == "bn" else
                f"{name(b['n'], 'en', True)} has {b['count']} ayahs."), ""
    found = hifz.today_portions() if req.get("today") else req["portions"]
    if not found:
        return ("আজকের রিভিশন শেষ, মাশাআল্লাহ। কিছু শুনতে চাইলে সূরার নাম বলো।" if lang == "bn" else
                "Today's revision is done, MashaAllah. Name a surah if you'd like to hear one."), ""
    ranges = [r for p in found for r in p["ranges"]]
    title = cap(join([portion_label(p, lang) for p in found], lang))
    cmd = {"action": "play", "quiet": True, **plan(ranges, rid, title, lang, req.get("times") or 1)}
    yield ("recite", cmd)
    times = cmd["times"]
    extra = (f" · {bn_digits(times)} বার" if lang == "bn" else f" · {times} times") if times > 1 else ""
    return f"▶ {title} · {cmd['reciter_name']}{extra}", ""


def _control_answer(req: dict, lang: str):
    act = req["action"]
    if not req.get("active"):
        return ("এখন কোনো তিলাওয়াত চলছে না। শুনতে চাইলে বলো “সূরা মুলক চালাও”।" if lang == "bn" else
                "Nothing is playing. Say “play Surah Al-Mulk” to start one."), ""
    if act == "reciter":
        r = RECITERS[req["reciter"]]
        yield ("recite", {"action": "reciter", "quiet": True, "reciter": req["reciter"],
                          "reciter_name": r["name_bn"] if lang == "bn" else r["name"]})
        return f"↺ {r['name_bn'] if lang == 'bn' else r['name']}", ""
    # an ayah outside what's playing ("repeat ayah 5" while ayahs 10 to 30 play): play it, from where it is
    if act in ("seek", "repeat") and req.get("ayah") and (pl := req.get("player") or {}).get("surah"):
        s = int(pl["surah"])
        count = SURAHS[s - 1]["ayahs"]
        if req["ayah"] > count:
            return (f"সূরা {name(s, 'bn')}-এ {bn_digits(count)}টি আয়াত।" if lang == "bn" else
                    f"{name(s, 'en', True)} has {count} ayahs."), ""
        target = index(s, req["ayah"])
        try:
            lo, hi = (index(*map(int, str(pl[k]).split(":"))) for k in ("first", "last"))
        except (KeyError, ValueError, TypeError):
            lo = hi = None
        if lo is None or not lo <= target <= hi:
            span = (target, target) if act == "repeat" else (target, surah_range(s)[1])
            title = f"{name(s, lang)} {bn_digits(req['ayah']) if lang == 'bn' else req['ayah']}"
            cmd = {"action": "play", "quiet": True, **plan([span], pl.get("reciter"), title, lang,
                                                           req.get("times", 1) if act == "repeat" else 1)}
            yield ("recite", cmd)
            return f"▶ {title} · {cmd['reciter_name']}", ""
    cmd = {"action": act, "quiet": True}
    if act == "repeat":
        cmd.update(ayah=req.get("ayah"), times=req["times"])
        if lang == "bn":
            shown = f"🔁 {'আয়াত ' + bn_digits(req['ayah']) if req.get('ayah') else 'এই আয়াত'}, {bn_digits(req['times'])} বার"
        else:
            shown = f"🔁 {'Ayah ' + str(req['ayah']) if req.get('ayah') else 'This ayah'}, {req['times']}×"
    elif act == "seek":
        cmd.update(ayah=req["ayah"])
        shown = f"⤳ আয়াত {bn_digits(req['ayah'])} থেকে" if lang == "bn" else f"⤳ From ayah {req['ayah']}"
    else:
        labels = {"en": {"pause": "⏸ Paused", "resume": "▶ Resumed", "stop": "⏹ Stopped", "next": "⏭ Next ayah",
                         "prev": "⏮ Previous ayah"},
                  "bn": {"pause": "⏸ থামানো হলো", "resume": "▶ আবার চলছে", "stop": "⏹ বন্ধ", "next": "⏭ পরের আয়াত",
                         "prev": "⏮ আগের আয়াত"}}
        shown = labels["bn" if lang == "bn" else "en"][act]
    yield ("recite", cmd)
    return shown, ""


# ---------- the Quran-text rule on every model reply ----------
_ARABIC = re.compile(f"[{AR}]")
_ARABIC_RUN = re.compile(f"[{AR}][{AR}\\s]*")
_QURAN_CTX = _re(r"\b(?:quran|qur'an|qur’an|koran|surah|sura|surat|ayah|ayat|ayahs|aya|verse|verses|allah|revealed|"
                 r"revelation|scripture)\b|কুরআন|কোরআন|সূরা|সুরা|আয়াত|আল্লাহ")
_REF = re.compile(r"\b(\d{1,3})\s*[:：]\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?\b")
_AYAH_N = _re(r"\b(?:ayah|ayat|aya|verse|verses|ayahs|ayats)\s*(?:no\.?|number|#)?\s*(\d{1,3})\b|"
              r"\b(\d{1,3})(?:st|nd|rd|th)?\s+(?:ayah|ayat|verse)\b|\b(\d{1,3})\s+(?:ayahs|ayats|verses)\b|"
              r"আয়াত\s*(?:নং|নম্বর)?\s*([০-৯0-9]{1,3})|([০-৯0-9]{1,3})\s*(?:নং|নম্বর|টি|টা)?\s*আয়াত")
_QUOTE = _re(r"[\"“”«»]|\b(?:allah|god|the quran|qur'an|the koran|it|he) (?:says|said|states|tells us|declares)\b|"
             r"আল্লাহ (?:বলেন|বলেছেন)|কুরআনে বলা")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?।])\s+|\n+", text) if s.strip()]


_PRAYER_NAMED = {62, 89, 103}     # Al-Jumu'ah, Al-Fajr, Al-'Asr


def _named_refs(s: str) -> set[str]:
    """Numbers right after a surah's name: "Al-Baqarah 255", "Yusuf 53 to Ibrahim 52". The names that are also
    prayers count only after "surah" ("Fajr 4:58" is a time)."""
    words, out = tokens(s), set()
    for i in range(len(words)):
        has_word = words[i] in SURAH_WORDS
        found = find_name(words[i + has_word:])
        if found and (has_word or found[0] not in _PRAYER_NAMED):
            k = i + has_word + found[1]
            if k < len(words) and words[k] in AYAH_WORDS:
                k += 1
            if k < len(words) and words[k].isdigit():
                out.add(words[k])
                if k + 2 < len(words) and words[k + 1] in RANGE and words[k + 2].isdigit():
                    out.add(words[k + 2])
    return out


def _bad_sentence(s: str, allowed: str) -> str | None:
    if _ARABIC.search(s):
        return "arabic"
    if not _QURAN_CTX.search(s) and not _named_refs(s):
        return None
    nums = {g.translate(_BN_DIGITS) for rx in (_REF, _AYAH_N) for m in rx.finditer(s) for g in m.groups() if g}
    nums |= _named_refs(s)
    if nums and not nums <= set(re.findall(r"\d+", allowed.translate(_BN_DIGITS))):
        return "numbers"
    if _QUOTE.search(s):
        return "quote"
    return None


def reply_problem(text: str, allowed: str = "") -> list[str]:
    """What breaks the Quran-text rule in a model's reply: Arabic script; a verse reference or an ayah number the
    turn's own tool results don't give (`allowed`); a quoted verse ("Allah says…"). Empty when it's fine."""
    return sorted({p for s in _sentences(text) if (p := _bad_sentence(s, allowed))})


def correction(problems: list[str]) -> str:
    return ("Rewrite your reply without any Quran text: no Arabic, no translation or quotation of a verse, no verse "
            "references, ayah numbers or counts. Javid never quotes the Quran from memory. If Mahadi asked for a verse, "
            "say he can read it on Quran.com or hear it: 'play Surah ...'.")


def strip_reply(text: str, allowed: str = "", lang: str = "en") -> str:
    """The reply without the sentences that break the rule (the last resort, after one rewrite)."""
    kept = [s for s in _sentences(text) if not _bad_sentence(s, allowed)]
    if kept:
        return " ".join(kept)
    return ("কুরআনের শব্দ বা আয়াতের নম্বর আমি নিজে থেকে বলি না। Quran.com-এ পড়তে পারো, অথবা বলো “সূরা মুলক চালাও”।"
            if lang == "bn" else "I don't quote the Quran or number its verses myself. You can read it on Quran.com, "
                                 "or say “play Surah Al-Mulk” and I'll play it.")


def strip_arabic(text: str) -> str:
    """Speech never carries Arabic script (javid.synthesize): it's taken out, and the rest is said."""
    return re.sub(r"\s{2,}", " ", _ARABIC_RUN.sub(" ", text)).strip() if _ARABIC.search(text) else text


# ---------- checked against Quran.com once: python quran.py --check ----------
def _get(url: str) -> dict:
    import certifi
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20, context=ssl.create_default_context(cafile=certifi.where())) as r:
        return json.loads(r.read().decode())


def _ar(s: str) -> str:
    s = re.sub(f"[{chr(0x064B)}-{chr(0x065F)}{chr(0x0670)}{chr(0x06E1)}]", "", s)
    return s.translate(str.maketrans({chr(0x0623): chr(0x0627), chr(0x0625): chr(0x0627), chr(0x0622): chr(0x0627),
                                      chr(0x0671): chr(0x0627), chr(0x0649): chr(0x064A)}))


def check(get=_get) -> tuple[bool, list[str]]:
    """Compare quran_meta.json with Quran.com's public API: each surah's ayah count and pages, the Arabic names, each
    juz's first and last ayah, each manzil's start and length, and where the two reciters' ayah files are. (ok, report
    lines). Nothing it reads is kept: only the outcome goes into quran_meta.json's "checked"."""
    lines, bad = [], 0
    chapters = get(f"{API}/chapters")["chapters"]
    names_differ = []
    for c in chapters:
        s = SURAHS[c["id"] - 1]
        lo, hi = surah_range(c["id"])
        mine = [page_of(lo), page_of(hi)]
        if c["verses_count"] != s["ayahs"]:
            bad += 1
            lines.append(f"surah {c['id']}: {s['ayahs']} ayahs here, {c['verses_count']} on Quran.com")
        if c.get("pages") and list(c["pages"]) != mine:
            bad += 1
            lines.append(f"surah {c['id']}: pages {mine} here, {c['pages']} on Quran.com")
        if _ar(c["name_arabic"]) != _ar(s["ar"]):
            names_differ.append(c["id"])
    if len(chapters) != 114:
        bad += 1
    lines.append(f"surahs: {len(chapters)} checked (ayah counts and pages); the Arabic spelling differs (hamza, "
                 f"alif forms) for {len(names_differ)}: {names_differ}")
    seen = {}
    for j in get(f"{API}/juzs")["juzs"]:
        seen.setdefault(j["juz_number"], j)
    for n in range(1, 31):
        m = seen[n]["verse_mapping"]
        s1, s2 = min(int(k) for k in m), max(int(k) for k in m)
        a1, a2 = int(m[str(s1)].split("-")[0]), int(m[str(s2)].split("-")[-1])
        if (index(s1, a1), index(s2, a2)) != unit_range("juz", n):
            bad += 1
            lines.append(f"juz {n}: {span_text(*unit_range('juz', n))} here, {s1}:{a1} to {s2}:{a2} on Quran.com")
    lines.append("juz: 30 checked (first and last ayah)")
    for n in range(1, 8):
        first = get(f"{API}/verses/by_manzil/{n}?per_page=1&page=1")
        s, a = map(int, first["verses"][0]["verse_key"].split(":"))
        lo, hi = unit_range("manzil", n)
        if index(s, a) != lo or first["pagination"]["total_records"] != hi - lo + 1:
            bad += 1
            lines.append(f"manzil {n}: {key(lo)} with {hi - lo + 1} ayahs here, {s}:{a} with "
                         f"{first['pagination']['total_records']} on Quran.com")
    lines.append("manzils: 7 checked (start and length)")
    for rid, r in RECITERS.items():
        got = get(f"{API}/recitations/{r['qurancom']}/by_chapter/67?per_page=1")["audio_files"][0]["url"]
        got = "https:" + got if got.startswith("//") else got if got.startswith("http") else "https://verses.quran.com/" + got
        if got != r["url"].format(s=67, a=1):
            bad += 1
            lines.append(f"{rid}: Quran.com lists {got}, Javid plays {r['url'].format(s=67, a=1)}")
    lines.append("reciters: Husary (6) and Alafasy (7): their ayah files are where Javid plays them from")
    return bad == 0, lines


def _write_checked(result: dict) -> None:
    with open(META_FILE, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r'^ "checked": .*,$', lambda m: f' "checked": {json.dumps(result, ensure_ascii=False)},', text,
                  count=1, flags=re.M)
    tmp = META_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, META_FILE)


if __name__ == "__main__":
    if sys.argv[1:] != ["--check"]:
        sys.exit("usage: python quran.py --check   (compares quran_meta.json with Quran.com's public API)")
    ok, report = check()
    print("\n".join(report))
    print("OK: quran_meta.json matches Quran.com" if ok else "MISMATCH: see above")
    if ok:
        _write_checked({"date": date.today().isoformat(), "against": "api.quran.com/api/v4",
                        "what": "the ayah counts and pages of all 114 surahs, the 30 juz, the 7 manzils, and where "
                                "the reciters' ayah files are"})
    sys.exit(0 if ok else 1)
