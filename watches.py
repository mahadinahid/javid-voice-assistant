"""
Watches: Javid keeps an eye on something and speaks up when it happens. "Tell me when bitcoin goes above 70,000",
"alert me if ETH drops 5%", "let me know when there's news about the IVAC holiday", "tell me when it starts
raining", "বিটকয়েন ৭০ হাজার ডলারের উপরে গেলে জানিও".

Assistants answer when asked; few will wait for something to happen, and the ones that run scheduled prompts
(ChatGPT's tasks, Gemini's scheduled actions) spend a model request on every run. Here the request is parsed in
code, and the checks are code too: Binance and Yahoo Finance every PRICE_SEC, Google News every NEWS_SEC, the
forecast every RAIN_SEC. No LLM request and no Tavily credit, ever. A watch that fires becomes an alert in
timers.json (timers.add_alert), so the page rings it and says it in Javid's own voice, Telegram gets it as a
message, and the CLI says it.

request() takes a message that only creates, lists or cancels watches (another request in it goes to the model);
answer() is a generator returning (spoken, screen). check_due() runs the checks; start() runs it in a thread
(server lifespan, CLI). watches.json keeps them across restarts.
"""
import json
import os
import re
import threading
import time
import uuid

import gnews
import markets
import tracing
import weather

WATCHES_FILE = "watches.json"
MAX_ACTIVE = 12
TICK_SEC = 10
PRICE_SEC = 30
NEWS_SEC = 15 * 60
RAIN_SEC = 10 * 60
LIFETIME = {"price": 7 * 86400, "news": 3 * 86400, "rain": 86400}
NEWS_ALERTS_MAX = 5          # a news watch tells at most this many new stories, then stops
KEEP_DONE_SEC = 3600         # a fired or expired watch stays listed (as done) this long

_lock = threading.Lock()
_stop = threading.Event()
_running = {"on": False}
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_EN_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


# ---------- storage ----------
def _load() -> list[dict]:
    try:
        with open(WATCHES_FILE) as f:
            items = json.load(f)
        return items if isinstance(items, list) else []
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    tmp = WATCHES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, WATCHES_FILE)


def _prune(items: list[dict], now: float) -> list[dict]:
    return [w for w in items if not w.get("done") or now - w["done"] < KEEP_DONE_SEC]


def _active(items: list[dict]) -> list[dict]:
    return [w for w in items if not w.get("done")]


# ---------- parsing ----------
_TRIGGER = re.compile(r"\b(?:tell|let|notify|alert|ping|warn|inform|message|text|remind|wake)\s+me\b|"
                      r"\b(?:watch|monitor)\b|\bkeep (?:an |a )?(?:eye|watch|tabs) on\b|"
                      r"\bset (?:up )?(?:an? )?(?:price )?alert\b|\bgive me a (?:heads?[- ]up|shout)\b|"
                      r"জানি(?:ও|য়ো|য়ে দিও|য়ে দিবে|য়ে দেবে)|জানাবে|জানাও|সতর্ক কর|নজর রাখ", re.I)
_WHEN = re.compile(r"\b(?:when|whenever|if|as soon as|once|the moment|the minute|in case)\b|"
                   r"হলে|গেলে|নামলে|উঠলে|পড়লে|এলে|আসলে", re.I)
_LIST = re.compile(r"\bwhat (?:are|r) you (?:watching|tracking|monitoring|keeping an eye on)\b|"
                   r"\b(?:my|the|all|active|current)\s+(?:watch(?:es)?|alerts?|price alerts?)\b(?!\s+(?:for|on|if|when))|"
                   r"\blist (?:my |the |all )?(?:watches|alerts)\b|\bany (?:watches|alerts)\b|\bwhat alerts\b|"
                   r"কী কী (?:নজরে|দেখছ)|নজরদারির তালিকা", re.I)
_CANCEL = re.compile(r"\b(?:stop|cancel|remove|delete|clear|turn off|drop|forget|end)\s+(?:watching|tracking|monitoring)\b|"
                     r"\b(?:stop|cancel|remove|delete|clear|turn off|drop|forget|end)\s+(?:the |my |all |that |this |those |every )?"
                     r"(?:[\w$&.-]+\s+){0,4}?(?:watch(?:es)?|alerts?)\b|নজর রাখা বন্ধ|অ্যালার্ট বাতিল", re.I)
_ELSE = re.compile(r"\btimer\b|\balarm\b|\btasks?\b|\bto-?do\b|\bcalendar\b|\bmeeting\b|\bschedule\b|\bremember\b|"
                   r"\bconvert\b|\bplan\b|\be-?mail\b|\btranslat\w*|\bat \d{1,2}(?::\d\d)?\s*(?:am|pm)\b|"
                   r"\bin \d+ (?:minutes?|hours?)\b|টাইমার|ক্যালেন্ডার", re.I)
_NUM = r"\$?\s?(?P<n>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<u>k\b|thousand\b|m\b|million\b|হাজার|লাখ)?"
_UP = (r"(?:(?:goes|gets|is|rises|climbs|jumps|moves|trades|shoots|pumps|crosses|goes back|breaks)\s+)?"
       r"(?:above|over|past|beyond|higher than|more than|north of)|exceeds?|tops?|surpass(?:es)?|breaks? (?:above|through)")
_DOWN = (r"(?:(?:goes|gets|is|drops|falls|dips|sinks|slides|crashes|trades|moves|tanks|dumps)\s+)?"
         r"(?:below|under|beneath|lower than|less than|south of)")
_REACH = (r"hits|reaches|touches|gets to|goes to|rises to|climbs to|jumps to|drops to|falls to|dips to|crashes to|"
          r"comes to|is at|trades at|crosses|breaks")
_THRESHOLD = re.compile(rf"(?P<dir>{_UP}|{_DOWN}|{_REACH})\s+{_NUM}", re.I)
_THRESHOLD_BN = re.compile(rf"{_NUM}\s*(?:ডলার|\$)?\S*\s*(?P<dir>উপরে|ওপরে|বেশি|উপর|নিচে|নীচে|কম)", re.I)
_MOVE = re.compile(r"(?P<dir>drops?|falls?|dips?|down|loses?|crash(?:es)?|tanks?|dumps?|rises?|jumps?|gains?|up|climbs?|"
                   r"pumps?|moves?|changes?|swings?)\s+(?:by\s+|more than\s+|over\s+|at least\s+)?"
                   r"(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)|(?P<pct2>\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\s+"
                   r"(?P<dir2>drop|fall|dip|move|rise|jump|gain|swing|change|either way)", re.I)
_NEWS = re.compile(r"\b(?:news|headlines?|stor(?:y|ies)|reports?|articles?|anything new|something new|updates?)"
                   r"(?:\s+(?:comes? (?:out|up|in)|is (?:out|published)|gets published|happens|breaks|appears))?\s+"
                   r"(?:about|on|regarding|related to|involving|mentioning|for|from)\s+(?P<subject>.+)", re.I)
_NEWS_BN = re.compile(r"(?P<subject>\S.{1,60}?)\s*(?:নিয়ে|সম্পর্কে|বিষয়ে)\s*(?:কোনো\s*)?(?:নতুন\s*)?খবর")
_SUBJECT_TAIL = re.compile(r"\s*(?:\b(?:comes? out|comes? up|appears?|breaks?|is published|gets published|happens|"
                           r"drops|lands|please|for me|right away|immediately|asap|today|tonight|tomorrow|"
                           r"is out|are out|come in|comes in)\b|[?.!,])+\s*$", re.I)
_RAIN = re.compile(r"\b(?:starts?|begins?)\s+(?:to\s+)?rain(?:ing)?\b|\bit rains\b|\brain(?:s|ing)? (?:starts?|begins?)\b|"
                   r"বৃষ্টি (?:শুরু|হলে|নামলে|এলে|আসলে)", re.I)     # not "tell me if it's raining": that's now
_PLACE = re.compile(r"\bin ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)")
_COINS_BN = {"বিটকয়েন": "bitcoin", "ইথেরিয়াম": "ethereum", "ইথার": "ethereum", "সোলানা": "solana", "ডজকয়েন": "dogecoin",
             "বিএনবি": "bnb", "এক্সআরপি": "xrp"}
_WATCHISH = re.compile(r"\bprice\b|\brate\b|\bgoes (?:above|below|over|under)\b|\b(?:drops|falls|rises|crosses)\b|"
                       r"\bhits \$?\d|\d\s*(?:%|percent)", re.I)


def _number(raw: str, unit: str | None) -> float:
    n = float(raw.replace(",", ""))
    unit = (unit or "").lower()
    return n * (1000 if unit in ("k", "thousand", "হাজার") else 100000 if unit == "লাখ" else
                1_000_000 if unit in ("m", "million") else 1)


def _subject(raw: str) -> str:
    s = raw
    for _ in range(3):
        s = _SUBJECT_TAIL.sub("", s).strip()
    s = re.sub(r"^(?:the|any|a|an)\s+", "", s, flags=re.I)
    return " ".join(s.split())[:80]


def request(text: str) -> dict | None:
    """{"action": "create"|"list"|"cancel"|"cant"|"hint", ...} for a message that is only about watches, else None."""
    if not text or _ELSE.search(text):
        return None
    t = " ".join(text.translate(_BN_DIGITS).split())
    if _CANCEL.search(t):
        which = "all" if re.search(r"\b(?:all|every|everything)\b|সব", t, re.I) else t
        return {"action": "cancel", "which": which}
    if _LIST.search(t):
        return {"action": "list"}
    if not _TRIGGER.search(t):
        return None
    when = bool(_WHEN.search(t))
    watch_verb = bool(re.search(r"\b(?:watch|monitor)\b|\bkeep (?:an |a )?(?:eye|watch|tabs) on\b|নজর রাখ", t, re.I))
    if not (when or watch_verb):
        return None
    if _RAIN.search(t):
        place = _PLACE.search(text)
        return {"action": "create", "kind": "rain", "place": place[1] if place else ""}
    # "Tell me if there's any news about X" asks about now; "...when there's news", "if anything new comes out" wait.
    waits = watch_verb or re.search(r"\b(?:when|whenever|as soon as|once|the moment|the minute)\b|\bnew\b|"
                                    r"\bcomes? (?:out|up|in)\b|\bbreaks?\b|\bappears?\b|নতুন|এলে|আসলে", t, re.I)
    if waits and ((m := _NEWS.search(t)) or (m := _NEWS_BN.search(t))):
        subject = _subject(m["subject"])
        if subject and not re.search(r"https?://|www\.|t\.me/", subject) and len(subject.split()) <= 10:
            return {"action": "create", "kind": "news", "subject": subject}
    symbols = [s for s in markets.symbols_in(text) if s.lower() not in ("crypto",)] or \
        [name for bn, name in _COINS_BN.items() if bn in t]
    if symbols:
        sym = symbols[0]
        if (m := _MOVE.search(t)):
            word = (m["dir"] or m["dir2"] or "").lower()
            pct = float(m["pct"] or m["pct2"])
            direction = ("down" if re.match(r"drop|fall|dip|down|lose|crash|tank|dump", word) else
                         "up" if re.match(r"rise|jump|gain|up|climb|pump", word) else "either")
            return {"action": "create", "kind": "price", "symbol": sym, "op": "move", "pct": pct, "direction": direction}
        if (m := _THRESHOLD.search(t)) or (m := _THRESHOLD_BN.search(t)):
            word = m["dir"].lower()
            op = (">=" if re.search(r"above|over|past|beyond|higher|more|north|exceed|top|surpass|through|উপর|ওপর|বেশি", word)
                  else "<=" if re.search(r"below|under|beneath|lower|less|south|নিচে|নীচে|কম", word) else "reach")
            return {"action": "create", "kind": "price", "symbol": sym, "op": op, "target": _number(m["n"], m["u"])}
        if watch_verb or re.search(r"\balert\b", t, re.I):     # "watch bitcoin": at what level? ("tell me if bitcoin
            return {"action": "hint", "symbol": sym}            # is up today" is a question, for the model)
        return None
    if when and _WATCHISH.search(t):
        return {"action": "cant"}
    return None


# ---------- wording ----------
def _fmt(kind: str, cur: str, value: float, level: bool = False) -> str:
    """A price as said and shown: "$67,420.15", a level as "$70,000", an index in points."""
    num = f"{value:,.0f}" if level and value == int(value) else markets._price(value)
    if kind == "index":
        return f"{num} points"
    return f"${num}" if cur in ("USDT", "USDC", "FDUSD", "USD", "") else f"{num} {cur}"


def _nice(value: float) -> float:
    """A round level near `value` for an example: 88,847 -> 89,000, 112.3 -> 110."""
    if value <= 0:
        return value
    digits = len(str(int(value))) - 2
    return round(value, -digits) if digits > 0 else round(value, 2)


def _money(q: dict, value: float, level: bool = False) -> str:
    return _fmt(q["kind"], q.get("quote") or q.get("currency") or "USD", value, level)


def _name(q: dict) -> str:
    return markets._name(q) if q.get("found") else q.get("symbol", "")


def _label(w: dict) -> str:
    s = w["spec"]
    if w["kind"] == "price":
        if s["op"] == "move":
            way = {"up": "up", "down": "down", "either": "moves"}[s["direction"]]
            return f"{s['name']} {way} {s['pct']:g}%" if way != "moves" else f"{s['name']} moves {s['pct']:g}%"
        return f"{s['name']} {'above' if s['op'] == '>=' else 'below'} {s['target_text']}"
    if w["kind"] == "news":
        return f"News about {s['subject']}"
    return f"Rain in {s['place']}"


def _status(w: dict) -> str:
    s = w["spec"]
    if w["kind"] == "price" and w.get("last_value") is not None:
        return f"now {_fmt(s['kind'], s['cur'], w['last_value'])}"
    if w["kind"] == "news":
        return f"{w.get('alerts', 0)} new so far" if w.get("alerts") else "nothing new yet"
    if w["kind"] == "rain":
        return w.get("last_desc") or ""
    return ""


# ---------- creating, listing, cancelling ----------
def _create_price(req: dict, lang: str):
    yield ("tool", "market_price")
    span = tracing.tool_start("market_price")
    try:
        q = markets.quote(req["symbol"])
    finally:
        tracing.tool_end(span)
    bn = lang == "bn"
    if not q.get("found"):
        return ((f"{req['symbol']} Binance বা Yahoo Finance-এ খুঁজে পাইনি, তাই নজর রাখতে পারছি না।" if bn else
                 f"I couldn't find {req['symbol']} on Binance or Yahoo Finance, so I can't watch it."), "")
    name, price = _name(q), q["price"]
    name_bn = markets._name(q, "bn")
    spec = {"symbol": req["symbol"], "name": name, "name_bn": name_bn, "kind": q["kind"], "op": req["op"],
            "cur": q.get("quote") or q.get("currency") or "USD", "source": q["source"]}
    if req["op"] == "move":
        spec.update(pct=req["pct"], direction=req["direction"], base=price)
        want = {"up": f"rises {req['pct']:g}%", "down": f"drops {req['pct']:g}%", "either": f"moves {req['pct']:g}%"}[req["direction"]]
        want_bn = {"up": f"{req['pct']:g}% বাড়লে", "down": f"{req['pct']:g}% কমলে", "either": f"{req['pct']:g}% ওঠানামা করলে"}[req["direction"]]
    else:
        target = req["target"]
        op = req["op"] if req["op"] != "reach" else (">=" if target >= price else "<=")
        if (op == ">=" and price >= target) or (op == "<=" and price <= target):
            word = "above" if op == ">=" else "below"
            return ((f"{name_bn} এর মধ্যেই {_money(q, target, True)}-এর {'উপরে' if op == '>=' else 'নিচে'}: এখন "
                     f"{_money(q, price)}।" if bn else
                     f"{name} is already {word} {_money(q, target, True)}: it's at {_money(q, price)} now. Give me "
                     f"another level to watch for."), "")
        spec.update(op=op, target=target, target_text=_money(q, target, True))
        want = f"goes {'above' if op == '>=' else 'below'} {spec['target_text']}"
        want_bn = f"{spec['target_text']}-এর {'উপরে গেলে' if op == '>=' else 'নিচে নামলে'}"
    item = _add("price", spec, price, lang)
    if isinstance(item, str):
        return item, ""
    where, where_bn = (", here and on Telegram", ", এখানে আর Telegram-এ") if _telegram_on() else ("", "")
    spoken = (f"ঠিক আছে, {name_bn}-এর দিকে নজর রাখছি: এখন {_money(q, price)}। {want_bn} জানাব{where_bn}।" if bn else
              f"Okay, I'm watching {name}: it's at {_money(q, price)} on {q['source']} now. I'll tell you when it "
              f"{want}{where}.")
    return spoken, ""


def _telegram_on() -> bool:
    try:
        import telegram_bot
        st = telegram_bot.status()
        return bool(st["running"] and st["paired"] and not st["locked"])
    except Exception:
        return False


def _create_news(req: dict, lang: str):
    yield ("tool", "news_headlines")
    span = tracing.tool_start("news_headlines")
    try:
        items = gnews.search(req["subject"], days=3, lang=lang, limit=10, fresh=True)
    except Exception:
        items = None
    finally:
        tracing.tool_end(span)
    bn = lang == "bn"
    if items is None:
        return (("এখন Google News-এ পৌঁছাতে পারছি না; একটু পরে আবার বলো।" if bn else
                 "I can't reach Google News right now, so I can't start that watch. Try again in a minute."), "")
    spec = {"subject": req["subject"], "lang": lang, "seen": [i["title"] for i in items][:40]}
    item = _add("news", spec, None, lang)
    if isinstance(item, str):
        return item, ""
    latest = items[0] if items else None
    if bn:
        spoken = f"ঠিক আছে, {req['subject']} নিয়ে নতুন খবর এলে জানাব।" + (
            f" এখনকার সর্বশেষ: {latest['source']} — {latest['title']}।" if latest else " এখন কোনো খবর নেই।")
    else:
        spoken = (f"Okay, I'll keep an eye on the news about {req['subject']} and tell you when a new story comes out, "
                  f"for the next three days." + (f" The latest right now, from {latest['source']}: {latest['title']}."
                                                  if latest else " There's nothing about it right now."))
    return spoken, ""


def _create_rain(req: dict, lang: str):
    yield ("tool", "get_weather")
    span = tracing.tool_start("get_weather")
    try:
        loc = weather.locate(req["place"])
        fc = weather.forecast(loc)
    except Exception:
        loc = fc = None
    finally:
        tracing.tool_end(span)
    bn = lang == "bn"
    if not fc:
        return (("আবহাওয়ার তথ্য এখন পাচ্ছি না, তাই নজর রাখা শুরু করতে পারছি না।" if bn else
                 "I can't get the weather right now, so I can't start that watch."), "")
    if _raining(fc):
        return ((f"{loc['name']}-এ এখনই বৃষ্টি হচ্ছে।" if bn else f"It's already raining in {loc['name']}."), "")
    item = _add("rain", {"place": loc["name"], "lat": loc["lat"], "lon": loc["lon"], "tz": loc["tz"],
                         "country": loc.get("country", "")}, None, lang, desc=fc["now"].get("desc", ""))
    if isinstance(item, str):
        return item, ""
    now = fc["now"].get("desc") or "dry"
    return ((f"ঠিক আছে, {loc['name']}-এ বৃষ্টি শুরু হলে জানাব।" if bn else
             f"Okay, I'll tell you when it starts raining in {loc['name']}; right now it's {now}."), "")


def _same(a: dict, kind: str, spec: dict) -> bool:
    """Already watching this: the same asset and level (or move), the same news subject, rain in the same place."""
    if a["kind"] != kind:
        return False
    s = a["spec"]
    if kind == "price":
        return markets.resolve(s["symbol"])[1] == markets.resolve(spec["symbol"])[1] and s["op"] == spec["op"] and \
            s.get("target") == spec.get("target") and s.get("pct") == spec.get("pct") and s.get("direction") == spec.get("direction")
    if kind == "news":
        return s["subject"].lower() == spec["subject"].lower()
    return s["place"] == spec["place"]


def _add(kind: str, spec: dict, value, lang: str, desc: str = "") -> dict | str:
    """The new watch, or what to say instead (already watching it, or too many)."""
    now = time.time()
    bn = lang == "bn"
    with _lock:
        items = _prune(_load(), now)
        if any(_same(w, kind, spec) for w in _active(items)):
            return "এটার দিকে আগে থেকেই নজর রাখছি।" if bn else "I'm already watching that."
        if len(_active(items)) >= MAX_ACTIVE:
            return (f"এর মধ্যেই {str(MAX_ACTIVE).translate(_EN_TO_BN)}টি জিনিসে নজর রাখছি; আগে একটা বাতিল করো।" if bn else
                    f"I'm already watching {MAX_ACTIVE} things; cancel one first.")
        w = {"id": uuid.uuid4().hex[:10], "kind": kind, "spec": spec, "created": now, "expires": now + LIFETIME[kind],
             "next_check": now + {"price": PRICE_SEC, "news": NEWS_SEC, "rain": RAIN_SEC}[kind],
             "last_value": value, "last_checked": now, "last_desc": desc, "alerts": 0, "done": None, "outcome": None}
        w["label"] = _label(w)
        items.append(w)
        _save(items)
    return w


def _list(lang: str) -> tuple[str, str]:
    with _lock:
        active = _active(_prune(_load(), time.time()))
    bn = lang == "bn"
    if not active:
        return (("এখন কিছুর দিকে নজর রাখছি না। বলো, যেমন: 'বিটকয়েন ৭০ হাজার ডলারের উপরে গেলে জানিও'।" if bn else
                 "I'm not watching anything right now. Try 'tell me when bitcoin goes above 70,000' or 'let me know "
                 "when there's news about the IVAC holiday'."), "")
    labels = [w["label"] for w in active]
    spoken = (f"{str(len(active)).translate(_EN_TO_BN)}টি জিনিসে নজর রাখছি: {', '.join(labels)}।" if bn else
              f"I'm watching {len(active)} thing{'s' if len(active) > 1 else ''}: {'; '.join(labels)}.")
    screen = "Watching:\n" + "\n".join(f"- {w['label']}" + (f" ({st})" if (st := _status(w)) else "") for w in active)
    return spoken, screen


def _cancel(which: str, lang: str) -> tuple[str, str]:
    bn = lang == "bn"
    now = time.time()
    with _lock:
        items = _prune(_load(), now)
        active = _active(items)
        if which == "all":
            hits = active
        else:
            syms = {markets.resolve(s)[1] for s in markets.symbols_in(which)}
            words = set(re.findall(r"[a-z0-9ঀ-৿]{3,}", which.lower())) - {
                "stop", "cancel", "remove", "delete", "clear", "turn", "off", "drop", "forget", "watching", "watch",
                "tracking", "monitoring", "alert", "alerts", "the", "my", "that", "this", "news", "about", "please"}
            hits = [w for w in active if (w["kind"] == "price" and markets.resolve(w["spec"]["symbol"])[1] in syms)
                    or (words and words & set(re.findall(r"[a-z0-9ঀ-৿]{3,}", w["label"].lower())))]
            if not hits and len(active) == 1:
                hits = active           # "stop that alert" with one running
        if not hits:
            listed = "; ".join(w["label"] for w in active) or ("কিছু না" if bn else "nothing")
            return ((f"কোনটা বন্ধ করব বুঝিনি। নজরে আছে: {listed}।" if bn else
                     f"I couldn't tell which one to stop. I'm watching: {listed}."), "")
        ids = {w["id"] for w in hits}
        for w in items:
            if w["id"] in ids:
                w["done"], w["outcome"] = now, "cancelled"
        _save(items)
    names = "; ".join(w["label"] for w in hits)
    return ((f"বন্ধ করেছি: {names}।" if bn else f"Stopped watching {names}."), "")


def answer(req: dict, lang: str):
    """A generator (its tool events) returning (spoken, screen)."""
    bn = lang == "bn"
    action = req["action"]
    if action == "list":
        return _list(lang)
    if action == "cancel":
        return _cancel(req["which"], lang)
    if action == "cant":
        return (("এটায় এখনো নজর রাখতে পারি না। পারি ক্রিপ্টো আর শেয়ারের দাম, খবর আর বৃষ্টির দিকে।" if bn else
                 "I can't watch that yet. I can watch crypto and stock prices, the news about something, and the rain."), "")
    if action == "hint":
        q = markets.quote(req["symbol"])
        now = f" It's at {_money(q, q['price'])} now." if q.get("found") else ""
        level = _nice(q["price"] * 1.1) if q.get("found") else 100
        example = _fmt(q["kind"], q.get("quote") or q.get("currency") or "USD", level, True) if q.get("found") else "100"
        return ((f"কোন দামে জানাব বলো, যেমন: '{req['symbol']} {example}-এর উপরে গেলে জানিও'।" if bn else
                 f"At what level should I tell you?{now} Say something like 'tell me when {req['symbol']} goes above "
                 f"{example}' or 'if it drops 5%'."), "")
    maker = {"price": _create_price, "news": _create_news, "rain": _create_rain}[req["kind"]]
    return (yield from maker(req, lang))


# ---------- checking ----------
def _raining(fc: dict) -> bool:
    desc = (fc["now"].get("desc") or "").lower()
    if re.search(r"rain|drizzle|shower|thunder", desc):
        return True
    now = fc["now"].get("time")
    hour = next((h for h in fc.get("hours", []) if now and h["time"] <= now < h["time"].replace(minute=59, second=59)), None)
    return bool(hour and (hour.get("rain_mm") or 0) >= 0.3)


def _fire(w: dict, announce: dict, now: float, final: bool = True) -> None:
    import timers
    timers.add_alert(w["label"], announce)
    w["alerts"] = w.get("alerts", 0) + 1
    if final:
        w["done"], w["outcome"] = now, "fired"
    print(f"[watch] {w['label']}: {announce['en']}")


def _check_price(w: dict, now: float) -> None:
    s = w["spec"]
    q = markets.quote(s["symbol"])
    if not q.get("found"):
        return
    price = q["price"]
    w["last_value"] = price
    hit = False
    if s["op"] == ">=":
        hit = price >= s["target"]
    elif s["op"] == "<=":
        hit = price <= s["target"]
    else:
        change = (price - s["base"]) / s["base"] * 100 if s["base"] else 0
        hit = (s["direction"] == "up" and change >= s["pct"]) or (s["direction"] == "down" and change <= -s["pct"]) or \
              (s["direction"] == "either" and abs(change) >= s["pct"])
    if not hit:
        return
    value = _money(q, price)
    if s["op"] == "move":
        change = (price - s["base"]) / s["base"] * 100
        way = "up" if change > 0 else "down"
        en = f"Mahadi, {s['name']} is {way} {abs(change):.1f}% since you asked me to watch it: it's at {value} on {q['source']}."
        bn = (f"মাহাদি, {s.get('name_bn', s['name'])} {abs(change):.1f}% {'বেড়েছে' if change > 0 else 'কমেছে'}: "
              f"{q['source']}-এ এখন {value}।")
    else:
        word = "above" if s["op"] == ">=" else "below"
        en = f"Mahadi, {s['name']} just went {word} {s['target_text']}: it's at {value} on {q['source']}."
        bn = (f"মাহাদি, {s.get('name_bn', s['name'])} এইমাত্র {s['target_text']}-এর {'উপরে গেছে' if s['op'] == '>=' else 'নিচে নেমেছে'}: "
              f"{q['source']}-এ এখন {value}।")
    _fire(w, {"en": en, "bn": bn}, now)


def _check_news(w: dict, now: float) -> None:
    s = w["spec"]
    items = gnews.search(s["subject"], days=3, lang=s.get("lang", "en"), limit=10, fresh=True)
    seen = s.setdefault("seen", [])
    new = [i for i in items if not any(gnews.same_story(i["title"], t) for t in seen)]
    if not new:
        return
    s["seen"] = (seen + [i["title"] for i in new])[-60:]
    first = new[0]
    en = f"Mahadi, news about {s['subject']}: {first['source']} reports, {first['title']}."
    bn = f"মাহাদি, {s['subject']} নিয়ে নতুন খবর: {first['source']} জানাচ্ছে, {first['title']}।"
    _fire(w, {"en": en, "bn": bn}, now, final=w.get("alerts", 0) + 1 >= NEWS_ALERTS_MAX)


def _check_rain(w: dict, now: float) -> None:
    s = w["spec"]
    fc = weather.forecast({"name": s["place"], "lat": s["lat"], "lon": s["lon"], "tz": s["tz"]})
    w["last_desc"] = fc["now"].get("desc", "")
    if _raining(fc):
        _fire(w, {"en": f"Mahadi, it's started raining in {s['place']}.",
                  "bn": f"মাহাদি, {s['place']}-এ বৃষ্টি শুরু হয়েছে।"}, now)


_CHECKS = {"price": (_check_price, PRICE_SEC), "news": (_check_news, NEWS_SEC), "rain": (_check_rain, RAIN_SEC)}


def check_due(now: float | None = None) -> list[dict]:
    """Run every check that's due (the thread calls this every TICK_SEC). Returns the watches that fired."""
    now = now or time.time()
    with _lock:
        items = _prune(_load(), now)
        due = [w for w in _active(items) if w["next_check"] <= now]
    fired = []
    for w in due:
        if now >= w["expires"]:
            w["done"], w["outcome"] = now, "expired"
            continue
        check, every = _CHECKS[w["kind"]]
        before = w.get("alerts", 0)
        try:
            check(w, now)
        except Exception as e:      # a service down: try again next time
            print(f"[watch] {w['label']}: {type(e).__name__}: {str(e)[:80]}")
        w["last_checked"], w["next_check"] = now, now + every
        if w.get("alerts", 0) > before:
            fired.append(w)
    if due:
        with _lock:
            current = {w["id"]: w for w in _load()}
            for w in due:
                if w["id"] in current and not current[w["id"]].get("done"):     # not cancelled meanwhile
                    current[w["id"]] = w
            _save(_prune(list(current.values()), now))
    return fired


def active_json() -> list[dict]:
    """The Watching panel's rows."""
    with _lock:
        items = _active(_prune(_load(), time.time()))
    return [{"id": w["id"], "kind": w["kind"], "label": w["label"], "status": _status(w),
             "expires": int(w["expires"] * 1000)} for w in items]


def cancel(wid: str) -> bool:
    with _lock:
        items = _load()
        for w in items:
            if w["id"] == wid and not w.get("done"):
                w["done"], w["outcome"] = time.time(), "cancelled"
                _save(items)
                return True
    return False


def _loop() -> None:
    while not _stop.wait(TICK_SEC):
        try:
            check_due()
        except Exception as e:
            print(f"[watch] {type(e).__name__}: {str(e)[:120]}")


def start() -> None:
    """Run the checks in the background (server lifespan, CLI main). Once per process."""
    if _running["on"]:
        return
    _stop.clear()
    _running["on"] = True
    threading.Thread(target=_loop, name="watches", daemon=True).start()


def stop() -> None:
    _stop.set()
    _running["on"] = False
