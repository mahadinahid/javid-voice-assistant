"""
Exchange rates: CNY, BDT, CZK and USD, with when each rate was published, answered in code.

USD->CNY and USD->CZK come from the European Central Bank's reference rates (official; via the Frankfurter
API, which gives the ECB's date: the ECB sets them around 16:00 CET). The ECB doesn't quote taka, so BDT
comes from ExchangeRate-API (a market-data aggregator, updated once a day; it reports the update time).
Cross rates are computed here, not by the model. No API key needed; results are cached for CACHE_SEC.

A rates question (ASK: "exchange rates", the yuan, the koruna) is answered by answer(), spoken and screen
parts in English or Bangla, with no LLM request. It used to be the exchange-rates skill, where the model only
copied these figures into sentences, and got the sources wrong: "1 Czech koruna is 5.6556 taka, based on the
European Central Bank's reference rate", when the taka half is ExchangeRate-API's. request() leaves to the
model what this can't answer whole: another currency (the euro, gold), or another request in the same message;
the model still has the exchange_rates tool for those. A lone "dollar to taka" question stays a web search
(Bangladesh Bank publishes the official rate).
"""
import json
import re
import ssl
import time
import urllib.request
from datetime import date, datetime, timezone

import certifi   # macOS python.org builds have no CA bundle for urllib; groq/httpx already depend on certifi

import tracing

ECB_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY,CZK"
AGG_URL = "https://open.er-api.com/v6/latest/USD"
CACHE_SEC = 15 * 60
TIMEOUT_SEC = 8
CURRENCIES = ("USD", "BDT", "CNY", "CZK")
NAMES = {"USD": "US dollar", "BDT": "taka", "CNY": "yuan", "CZK": "Czech koruna"}
_PLURAL = {**NAMES, "USD": "US dollars"}

_cache: dict = {}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Javid/1.0"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=ctx) as r:
        return json.loads(r.read().decode())


def fetch() -> dict:
    """{"rates": {code: per 1 USD}, "sources": {code: {name, official, when, day}}, "errors": [...]};
    raises if nothing came back."""
    hit = _cache.get("rates")
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    rates, sources, errors = {"USD": 1.0}, {}, []
    try:
        ecb = _get_json(ECB_URL)
        day = date.fromisoformat(ecb["date"])
        for code in ("CNY", "CZK"):
            rates[code] = float(ecb["rates"][code])
            sources[code] = {"name": "European Central Bank", "official": True, "day": day,
                             "when": f"{day:%B} {day.day} reference rate (set about 16:00 CET)"}
    except Exception as e:
        errors.append(f"ECB: {type(e).__name__}")
    try:
        agg = _get_json(AGG_URL)
        if agg.get("result") != "success":
            raise ValueError(agg.get("error-type", "failed"))
        at = datetime.fromtimestamp(agg["time_last_update_unix"], timezone.utc)
        for code in ("BDT", "CNY", "CZK"):
            if code not in rates:    # the ECB's official figure wins when there is one
                rates[code] = float(agg["rates"][code])
                sources[code] = {"name": "ExchangeRate-API", "official": False, "day": at.date(),
                                 "when": f"updated {at:%B} {at.day}, {at:%H:%M} UTC"}
    except Exception as e:
        errors.append(f"ExchangeRate-API: {type(e).__name__}")
    if len(rates) == 1:
        raise RuntimeError("; ".join(errors) or "no rates")
    data = {"rates": rates, "sources": sources, "errors": errors}
    _cache["rates"] = (time.time(), data)
    return data


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if x >= 10 else f"{x:,.4f}".rstrip("0").rstrip(".")


def _board(data: dict) -> list[str]:
    """The screen lines every answer ends with: each rate per US dollar with its source, then the taka rates."""
    rates, sources = data["rates"], data["sources"]
    lines = []
    for code in ("BDT", "CNY", "CZK"):
        if code in rates:
            s = sources[code]
            lines.append(f"1 USD = {_fmt(rates[code])} {code} · {s['name']}{' (official)' if s['official'] else ''}, "
                         f"{s['when']}")
        else:
            lines.append(f"1 USD = {code} unavailable right now")
    if "BDT" in rates:
        lines.append("In taka: " + " · ".join(f"1 {c} = {_fmt(rates['BDT'] / rates[c])} BDT" for c in ("CNY", "CZK")
                                              if c in rates))
    if data["errors"]:
        lines.append("Not available: " + "; ".join(data["errors"]))
    return lines


def exchange_rates() -> str:
    """The tool the model gets when answer() leaves a question to it (another currency or request with it)."""
    try:
        data = fetch()
    except Exception as e:
        return (f"Couldn't get exchange rates right now ({e}). Tell Mahadi the rates are unavailable; "
                "don't give any figures.")
    rates = data["rates"]
    lines = ["Per 1 US dollar:"] + _board(data)
    if "CNY" in rates and "CZK" in rates:
        lines.append(f"1 CNY = {_fmt(rates['CZK'] / rates['CNY'])} CZK")
    lines.append("Use these exact figures, name each source and when it was published. Don't add other rates.")
    return "\n".join(lines)


# ---------- answered in code ----------
ASK = re.compile(r"exchange rates|currency rates|rates board|\b(?<!crypto )currencies\b|\byuan\b|\bcny\b|\brmb\b|"
                 r"renminbi|koruna|\bczk\b|ইউয়ান|কোরুনা|এক্সচেঞ্জ রেট|বিনিময় হার", re.I)
_CURRENCY = [   # (code, how it's named)
    ("CZK", r"czech\s+(?:koruna|korun[ay]?|crowns?)|korun[ay]|koruna|\bczk\b|কোরুনা"),
    ("CNY", r"chinese\s+yuan|\byuan\b|renminbi|\brmb\b|\bcny\b|ইউয়ান"),
    ("BDT", r"bangladeshi\s+taka|\btaka\b|\bbdt\b|\btk\b|৳|টাকা"),
    ("USD", r"(?:us|u\.s\.|american)\s+dollars?|\bdollars?\b|\busd\b|\$|ডলার"),
]
# Money this can't price, or another kind of request in the same message: the model answers those.
_ELSE = re.compile(r"(?:singapore|hong kong|canadian|australian|new zealand|taiwan(?:ese)?)\s+dollars?|\beuros?\b|\beur\b|€|"
                   r"\bpounds?\b|sterling|\bgbp\b|£|rupees?|\binr\b|\bpkr\b|\byen\b|\bjpy\b|¥|riyals?|dirhams?|ringgits?|"
                   r"\bbaht\b|francs?|r[ou]?ubles?|\blira\b|dinars?|\bpesos?\b|\b(?:aud|cad|sgd|hkd|chf|sar|aed|myr|krw|thb)\b|"
                   r"bitcoin|crypto|\bgold\b|silver|\boil\b|petrol|weather|forecast|\brain\b|calendar|schedule|meeting|"
                   r"\btasks?\b|to-?do|remind|timer|alarm|\bnews\b|headline|\bstocks?\b|\bshares?\b|\bplan\b|search|"
                   r"look (?:it )?up|google|ইউরো|রুপি|পাউন্ড|ইয়েন|রিয়াল|দিরহাম|সোনা|আবহাওয়া|খবর|ক্যালেন্ডার|টাস্ক",
                   re.I)
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_AMOUNT = r"(?:(?P<n>\d[\d,]*(?:\.\d+)?)\s*|\b(?P<w>one|an?)\s+)?"
_BN_NAMES = {"USD": "মার্কিন ডলার", "BDT": "টাকা", "CNY": "ইউয়ান", "CZK": "চেক কোরুনা"}
_BN_SHORT = {"USD": "ডলার", "BDT": "টাকা", "CNY": "ইউয়ান", "CZK": "কোরুনা"}
_BN_OF = {"USD": "ডলারের", "BDT": "টাকার", "CNY": "ইউয়ানের", "CZK": "কোরুনার"}      # "the taka's" rate
_BN_MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর",
              "নভেম্বর", "ডিসেম্বর"]
_SOURCE_OF = {"en": {"European Central Bank": "the European Central Bank's", "ExchangeRate-API": "ExchangeRate-API's"},
              "bn": {"European Central Bank": "ইউরোপিয়ান সেন্ট্রাল ব্যাংকের", "ExchangeRate-API": "ExchangeRate-API-এর"}}


def _mentions(text: str) -> list[tuple[str, float | None]]:
    """(code, amount or None) for each currency named, in the order named. "500 yuan", "one koruna", "$50"."""
    text = text.translate(_BN_DIGITS)
    spans = []
    for code, name in _CURRENCY:
        for m in re.finditer(_AMOUNT + rf"(?:{name})", text, re.I):
            amount = m["n"] or m["w"]
            if not amount and m.group(0) in ("$", "৳") and (after := re.match(r"\s*(\d[\d,]*(?:\.\d+)?)", text[m.end():])):
                amount = after[1]
            value = 1.0 if amount and amount.lower() in ("one", "a", "an") else float(amount.replace(",", "")) if amount else None
            spans.append((m.start(), m.end(), code, value))
    kept, end = [], -1
    for start, stop, code, value in sorted(spans, key=lambda s: (s[0], -s[1])):
        if start >= end:        # "Czech koruna" is one mention, not two
            kept.append((code, value))
            end = stop
    return kept


def request(text: str) -> dict | None:
    """What a rates question asks for, if code can answer all of it: {"pair": (from, to) or None, "amount"}."""
    if not ASK.search(text) or _ELSE.search(text):
        return None
    named = _mentions(text)
    codes = list(dict.fromkeys(code for code, _ in named))
    amount, source = next(((value, code) for code, value in named if value is not None), (None, None))
    if len(codes) == 1 and codes[0] not in ("USD", "BDT"):
        return {"pair": (codes[0], "BDT"), "amount": amount}     # "the yuan rate": in taka, at home
    if len(codes) == 2:
        frm = source or codes[0]
        return {"pair": (frm, next(c for c in codes if c != frm)), "amount": amount}
    return {"pair": None, "amount": None}                         # the whole board


def _say(x: float) -> str:
    """A figure as it's spoken: two decimals (the screen keeps the published precision)."""
    return f"{x:,.2f}" if x >= 1 else f"{x:.4f}"


def _day(d: date, lang: str) -> str:
    return f"{d.day} {_BN_MONTHS[d.month - 1]}" if lang == "bn" else f"{d:%B} {d.day}"


def _and(items: list[str], lang: str) -> str:
    joiner = " আর " if lang == "bn" else " and "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + joiner + items[-1]


def _credit(codes: list[str], data: dict, lang: str) -> str:
    """Who published the rates behind these currencies, and when (one clause per source)."""
    groups: dict[str, list[str]] = {}
    for code in codes:
        if code != "USD" and code in data["sources"]:
            groups.setdefault(data["sources"][code]["name"], []).append(code)
    parts = []
    for source, group in groups.items():
        day = _day(data["sources"][group[0]]["day"], lang)
        if lang == "bn":
            parts.append(f"{_and([_BN_SHORT[c] for c in group[:-1]] + [_BN_OF[group[-1]]], 'bn')} রেট "
                         f"{_SOURCE_OF['bn'][source]} ({day})")
        else:
            names = _and([NAMES[c].split()[-1] for c in group], "en")
            parts.append(f"the {names} rate{'s are' if len(group) > 1 else ' is'} {_SOURCE_OF['en'][source]}, from {day}")
    if lang == "bn":
        return "; ".join(parts) + "।"
    text = "; ".join(parts) + "."
    return text[0].upper() + text[1:]


def _unavailable(lang: str) -> str:
    return ("এখন এক্সচেঞ্জ রেট পাচ্ছি না, তাই কোনো সংখ্যা বলছি না। কয়েক মিনিট পরে আবার জিজ্ঞেস করো।" if lang == "bn" else
            "I couldn't get the exchange rates right now, so I won't guess any figures. Try again in a few minutes.")


def answer(req: dict, lang: str):
    """The reply to a rates question, built here: a generator (yields its tool event) returning (spoken, screen)."""
    yield ("tool", "exchange_rates")
    span = tracing.tool_start("exchange_rates")
    try:
        data = fetch()
    except Exception:
        return _unavailable(lang), ""
    finally:
        tracing.tool_end(span)
    rates = data["rates"]
    if req["pair"]:
        frm, to = req["pair"]
        if frm not in rates or to not in rates:
            missing = NAMES[frm if frm not in rates else to]
            return ((f"{_BN_NAMES[frm if frm not in rates else to]}-এর রেট এখন পাচ্ছি না।" if lang == "bn" else
                     f"The {missing} rate isn't available right now."), "\n".join(["Exchange rates:"] + _board(data)))
        rate = rates[to] / rates[frm]
        amount = req["amount"] or 1
        shown = f"{amount:,.0f}" if amount == int(amount) else f"{amount:,}"
        line = f"{shown} {frm} = {_fmt(rate * amount)} {to}" + (f" (1 {frm} = {_fmt(rate)} {to})" if amount != 1 else "")
        lines = ["Exchange rates:"] + ([] if frm == "USD" and amount == 1 else [line]) + _board(data)   # no repeat
        if lang == "bn":
            head = (f"{shown} {_BN_NAMES[frm]} এখন {_say(rate * amount)} {_BN_NAMES[to]}, প্রতি {_BN_NAMES[frm]} "
                    f"{_say(rate)} {_BN_NAMES[to]} ধরে।" if amount != 1 else f"এক {_BN_NAMES[frm]} এখন {_say(rate)} {_BN_NAMES[to]}।")
        else:
            head = (f"{shown} {_PLURAL[frm]} is {_say(rate * amount)} {_PLURAL[to]}, at {_say(rate)} {_PLURAL[to]} per "
                    f"{NAMES[frm]}." if amount != 1 else f"One {NAMES[frm]} is {_say(rate)} {_PLURAL[to]}.")
        return f"{head} {_credit([frm, to], data, lang)}", "\n".join(lines)
    have = [c for c in ("BDT", "CNY", "CZK") if c in rates]
    if lang == "bn":
        spoken = (f"এক মার্কিন ডলার এখন {_and([f'{_say(rates[c])} {_BN_NAMES[c]}' for c in have], 'bn')}। "
                  f"{_credit(have, data, 'bn')}")
    else:
        spoken = (f"One US dollar is {_and([f'{_say(rates[c])} {NAMES[c]}' for c in have], 'en')}. "
                  f"{_credit(have, data, 'en')}")
    return spoken, "\n".join(["Exchange rates:"] + _board(data))


TOOLS = [{
    "type": "function",
    "function": {
        "name": "exchange_rates",
        "description": "Current USD, BDT, CNY and CZK exchange rates with their sources and publication times.",
        "parameters": {"type": "object", "properties": {}},
    },
}]
TOOL_MAP = {"exchange_rates": lambda args: exchange_rates()}
