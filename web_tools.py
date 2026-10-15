"""
Web search for Javid through Tavily (needs TAVILY_API_KEY in .env), and the checks that keep
web answers honest. javid.py splices TOOLS/TOOL_MAP in and calls new_turn(), check_reply(),
finalize() and sources() from its agent loop.

Rules enforced here, not left to the prompt:
  - every web answer names a source and its date (finalize appends one if the model didn't);
  - a web answer is at most MAX_SENTENCES sentences;
  - a rate or price in the answer must appear on an official domain or on two sites;
  - when sources give different figures, the answer must say they disagree;
  - read_page only opens the top READ_TOP results of a search made this turn;
  - the same query within CACHE_SEC reuses the earlier results (no credit spent);
  - at most DAILY_CREDIT_CAP Tavily credits per day.
"""
import json
import os
import re
import time
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

USAGE_FILE = "tavily_usage.jsonl"    # one line per Tavily call (or cache hit): credits spent
DAILY_CREDIT_CAP = 40
CACHE_SEC = 15 * 60
MAX_RESULTS = 5
NEWS_DAYS = 7          # default time filter for news questions
READ_TOP = 3           # read_page accepts only the top 3 results of this turn's searches
MAX_READS = 3          # read_page calls per turn
MAX_SENTENCES = 3
SNIPPET_CHARS = 300    # per result in the tool result; the full snippet is kept for the checks
PAGE_CHARS = 1500
SEARCHING = "Searching…"

# Domains whose figures count as official on their own (plus any gov/gob/gouv/mil/int domain).
OFFICIAL_DOMAINS = {
    "bb.org.bd", "bbs.gov.bd", "federalreserve.gov", "ecb.europa.eu", "europa.eu", "bankofengland.co.uk",
    "rbi.org.in", "sbp.org.pk", "boj.or.jp", "imf.org", "worldbank.org", "un.org", "who.int",
    "oecd.org", "bis.org",
}
SOURCE_NAMES = {
    "reuters.com": "Reuters", "apnews.com": "AP", "bbc.com": "BBC", "bbc.co.uk": "BBC",
    "aljazeera.com": "Al Jazeera", "bloomberg.com": "Bloomberg", "cnn.com": "CNN",
    "nytimes.com": "The New York Times", "theguardian.com": "The Guardian",
    "thedailystar.net": "The Daily Star", "prothomalo.com": "Prothom Alo", "tbsnews.net": "The Business Standard",
    "dhakatribune.com": "Dhaka Tribune", "bdnews24.com": "bdnews24", "newagebd.net": "New Age",
    "bb.org.bd": "Bangladesh Bank", "bbs.gov.bd": "Bangladesh Bureau of Statistics",
    "wikipedia.org": "Wikipedia", "britannica.com": "Britannica", "xe.com": "XE",
    "espncricinfo.com": "ESPNcricinfo",
}
_NEWSY = re.compile(r"\b(news|latest|today|tonight|yesterday|this (week|morning|evening)|headlines?|"
                    r"breaking|update|announced?|just|recent(ly)?|current(ly)?)\b", re.I)
DISAGREE = re.compile(r"disagree|differ|vary|varies|conflict|inconsisten|don'?t agree|do not agree|"
                      r"not consistent|mixed|discrepan|while others|another (source|site) says", re.I)


# ---------- Tavily client, credits, cache ----------
_client_obj = None
_client_key = ""
_last_error: dict | None = None
_cache: dict[tuple, tuple[float, dict]] = {}


def _key() -> str:
    return os.environ.get("TAVILY_API_KEY", "").strip()


def _client():
    """The Tavily client, or None without a key (regression.py swaps this for a fake)."""
    global _client_obj, _client_key
    key = _key()
    if not key:
        return None
    if _client_obj is None or key != _client_key:
        from tavily import TavilyClient
        _client_obj, _client_key = TavilyClient(api_key=key), key
    return _client_obj


def _log(kind: str, credits: int, cached: bool, what: str) -> None:
    entry = {"ts": round(time.time(), 1), "kind": kind, "credits": credits, "cached": cached, "what": what[:120]}
    try:
        with open(USAGE_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _today_log() -> list[dict]:
    start = datetime.combine(date.today(), datetime.min.time()).timestamp()
    out = []
    try:
        with open(USAGE_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("ts", 0) >= start:
                    out.append(e)
    except OSError:
        pass
    return out


def credits_today() -> int:
    return sum(e.get("credits", 0) for e in _today_log())


def _note_error(err: Exception) -> str:
    global _last_error
    msg = str(err)
    if _key():
        msg = msg.replace(_key(), "[key hidden]")
    _last_error = {"kind": type(err).__name__, "message": msg[:200], "at": time.time()}
    return type(err).__name__


def _cached(key: tuple) -> dict | None:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    _cache.pop(key, None)
    return None


def _call(kind: str, key: tuple, what: str, request) -> dict | str:
    """Cached Tavily call within the daily credit cap. Returns the response or a sentence."""
    hit = _cached(key)
    if hit is not None:
        _log(kind, 0, True, what)
        return hit
    client = _client()
    if client is None:
        return "Web search isn't set up: add TAVILY_API_KEY to .env and restart Javid."
    if credits_today() + 1 > DAILY_CREDIT_CAP:
        return (f"The daily web search budget ({DAILY_CREDIT_CAP} Tavily credits) is used up. "
                "Tell Mahadi it resets at midnight.")
    try:
        resp = request(client)
    except Exception as e:
        return f"Web search failed ({_note_error(e)}). Tell Mahadi you couldn't search right now."
    credits = (resp.get("usage") or {}).get("credits")
    _log(kind, credits if isinstance(credits, int) else 1, False, what)
    _cache[key] = (time.time(), resp)
    return resp


# ---------- sources ----------
def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def _matches(domain: str, known: str) -> bool:
    return domain == known or domain.endswith("." + known)


def source_name(domain: str) -> str:
    return next((name for d, name in SOURCE_NAMES.items() if _matches(domain, d)), domain)


def is_official(domain: str) -> bool:
    labels = domain.split(".")
    return (any(l in ("gov", "gob", "gouv", "govt", "mil") for l in labels) or labels[-1] == "int"
            or any(_matches(domain, d) for d in OFFICIAL_DOMAINS))


def _parse_date(raw) -> date | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _say_date(d: date) -> str:
    text = f"{d:%B} {d.day}"
    return text if d.year == date.today().year else f"{text}, {d.year}"


# Per-turn state: what was found, so the reply can be checked against it.
_turn: dict = {}


def new_turn() -> None:
    _turn.clear()
    _turn.update(sources=[], texts={}, top=[], reads=0, query="")


new_turn()


def used() -> bool:
    return bool(_turn["sources"])


def _add_source(r: dict) -> dict:
    url = r.get("url", "")
    for s in _turn["sources"]:
        if s["url"] == url:
            return s
    domain = _domain(url)
    published = _parse_date(r.get("published_date"))
    s = {"n": len(_turn["sources"]) + 1, "url": url, "domain": domain, "name": source_name(domain),
         "title": (r.get("title") or "").strip()[:140], "official": is_official(domain),
         "date": published.isoformat() if published else None,
         "date_text": _say_date(published) if published else f"retrieved {_say_date(date.today())}"}
    _turn["sources"].append(s)
    return s


def _remember_text(domain: str, text: str) -> None:
    _turn["texts"][domain] = _turn["texts"].get(domain, "") + "\n" + (text or "")


def sources() -> list[dict]:
    """Links for the screen: every result this turn, in the order found."""
    return [{k: s[k] for k in ("n", "name", "title", "url", "date_text", "official")}
            for s in _turn["sources"] if s["url"].startswith(("https://", "http://"))]


# ---------- tools ----------
def web_search(query: str, recency_days: int | None = None, domains: list[str] | None = None) -> str:
    query = " ".join((query or "").split())
    if not query:
        return "No query given."
    news = bool(recency_days) or bool(_NEWSY.search(query))
    days = max(1, min(int(recency_days or NEWS_DAYS), 365)) if news else None
    domains = sorted({_domain(d if "//" in d else "https://" + d) for d in domains or [] if d.strip()})
    kw = {"search_depth": "basic", "max_results": MAX_RESULTS, "include_usage": True,
          "topic": "news" if news else "general"}
    if days:
        kw["days"] = days
    if domains:
        kw["include_domains"] = domains
    key = ("search", re.sub(r"[^\w\s]", "", query.lower()), days, tuple(domains))
    resp = _call("search", key, query, lambda c: c.search(query, **kw))
    if isinstance(resp, str):
        return resp
    results = [r for r in resp.get("results", []) if r.get("url")][:MAX_RESULTS]
    if not results:
        return f"No results for '{query}'. Say you couldn't find it."
    _turn["query"] = query
    lines = [f"Results for '{query}'" + (f" (news, last {days} days)" if days else "") + ":"]
    for i, r in enumerate(results):
        s = _add_source(r)
        if i < READ_TOP and s["url"] not in _turn["top"]:
            _turn["top"].append(s["url"])
        _remember_text(s["domain"], f"{r.get('title', '')}\n{r.get('content', '')}")
        official = ", official" if s["official"] else ""
        snippet = " ".join((r.get("content") or "").split())[:SNIPPET_CHARS]
        lines.append(f"[{s['n']}] {s['name']}{official}, {s['date_text']}, {s['url']} : {s['title']} - {snippet}")
    conflicts = _conflicts()
    if conflicts:
        lines.append("Sources give different figures: " + "; ".join(conflicts) + ". Say they disagree.")
    lines.append(f"Answer in at most {MAX_SENTENCES} sentences and name the source and its date. "
                 "Only state a rate or price an official source gives or two sources agree on. "
                 "If the snippets don't answer it, read_page one of the top 3 results.")
    return "\n".join(lines)


def news_headlines(topics: list[str], days: int = 2) -> str:
    """The top news story per topic, in one tool call (a morning brief needs three searches, and one
    round each would use up the turn's tool rounds). Sources and texts are registered like web_search's."""
    topics = [" ".join(str(t).split()) for t in topics or [] if str(t).strip()][:4]
    if not topics:
        return "No topics given."
    days = max(1, min(int(days or 2), 7))
    lines = []
    for topic in topics:
        query = f"{topic} news"
        kw = {"search_depth": "basic", "max_results": 3, "include_usage": True, "topic": "news", "days": days}
        key = ("search", re.sub(r"[^\w\s]", "", query.lower()), days, ())
        resp = _call("search", key, query, lambda c, q=query, k=kw: c.search(q, **k))
        if isinstance(resp, str):
            lines.append(f"{topic}: {resp}")
            continue
        results = [r for r in resp.get("results", []) if r.get("url")]
        if not results:
            lines.append(f"{topic}: no news found in the last {days} days.")
            continue
        r = results[0]
        s = _add_source(r)
        _remember_text(s["domain"], f"{r.get('title', '')}\n{r.get('content', '')}")
        snippet = " ".join((r.get("content") or "").split())[:SNIPPET_CHARS]
        lines.append(f"{topic}: [{s['n']}] {s['title']} ({s['name']}, {s['date_text']}) - {snippet}")
    return "Top headline per topic:\n" + "\n".join(lines)


def _resolve_top(url: str) -> str | None:
    """The top result meant by `url`: exact, its result number, or the one top result on that site
    (models rewrite URLs from memory)."""
    url = (url or "").strip().strip("[]")
    if url in _turn["top"]:
        return url
    if url.isdigit():
        s = next((s for s in _turn["sources"] if s["n"] == int(url)), None)
        return s["url"] if s and s["url"] in _turn["top"] else None
    same_site = [u for u in _turn["top"] if _domain(u) == _domain(url if "//" in url else "https://" + url)]
    return same_site[0] if len(same_site) == 1 else None


def read_page(url: str) -> str:
    url = _resolve_top(url) or (url or "").strip()
    if url not in _turn["top"]:
        tops = "; ".join(_turn["top"]) or "none yet"
        return f"read_page only opens the top {READ_TOP} results of a search made this turn: {tops}."
    if _turn["reads"] >= MAX_READS:
        return f"Already read {MAX_READS} pages this turn; answer from what you have."
    _turn["reads"] += 1
    query = _turn["query"]
    resp = _call("extract", ("extract", url, query.lower()), url,
                 lambda c: c.extract(urls=[url], extract_depth="basic", query=query or None,
                                     chunks_per_source=3, include_usage=True))
    if isinstance(resp, str):
        return resp
    page = next((r for r in resp.get("results", []) if r.get("url") == url), None) or \
        (resp.get("results") or [None])[0]
    text = " ".join(((page or {}).get("raw_content") or "").split())
    if not text:
        return "Couldn't read that page. Answer from the search results or try the next one."
    s = next(s for s in _turn["sources"] if s["url"] == url)
    _remember_text(s["domain"], text)
    return f"[{s['n']}] {s['name']}, {s['date_text']} - page text: {text[:PAGE_CHARS]}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for news, current facts, rates and prices. Results are "
                           "numbered with source name and date. Your answer must name the source and "
                           "its date, e.g. 'according to Reuters, October 14'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "recency_days": {"type": "integer",
                                     "description": "For news: only results from the last N days."},
                    "domains": {"type": "array", "items": {"type": "string"},
                                "description": "Only search these sites, e.g. ['bb.org.bd']."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read the text of one of the top 3 results from this turn's web_search, "
                           "when the snippets don't answer the question.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The result's URL as listed (or its number)."}},
                "required": ["url"],
            },
        },
    },
]

TOOLS.append({
    "type": "function",
    "function": {
        "name": "news_headlines",
        "description": "The top news story for each topic (one search per topic), with source and date.",
        "parameters": {
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['Bangladesh', 'China', 'AI']"},
                "days": {"type": "integer", "description": "How recent, in days (default 2)."},
            },
            "required": ["topics"],
        },
    },
})

TOOL_MAP = {
    "web_search": lambda args: web_search(**args),
    "read_page":  lambda args: read_page(**args),
    "news_headlines": lambda args: news_headlines(**args),
}


# ---------- checking the answer ----------
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_PRE = {"us$": "USD", "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "£": "GBP", "gbp": "GBP",
        "৳": "BDT", "tk": "BDT", "tk.": "BDT", "bdt": "BDT", "₹": "INR", "inr": "INR", "¥": "JPY", "jpy": "JPY"}
_POST = {"%": "%", "percent": "%", "per cent": "%", "usd": "USD", "dollar": "USD", "dollars": "USD",
         "bdt": "BDT", "tk": "BDT", "taka": "BDT", "takas": "BDT", "eur": "EUR", "euro": "EUR", "euros": "EUR",
         "gbp": "GBP", "pound": "GBP", "pounds": "GBP", "inr": "INR", "rupee": "INR", "rupees": "INR",
         "jpy": "JPY", "yen": "JPY"}
_FIGURE = re.compile(
    r"(?<![\w$€£৳₹¥])(?P<pre>us\$|\$|€|£|৳|₹|¥|usd|eur|gbp|bdt|inr|jpy|tk\.?)\s?:?\s?(?P<n1>" + _NUM + r")(?![\d])"
    r"|(?<![\d.,])(?P<n2>" + _NUM + r")\s?(?P<post>%|per cent|percent|usd|dollars?|bdt|tk|takas?|eur|euros?|"
    r"gbp|pounds?|inr|rupees?|jpy|yen)(?![a-z])", re.I)


def figures(text: str) -> list[tuple[float, str]]:
    """Rates and prices in `text` as (value, unit): '1 USD = 122.50 BDT' -> [(1, USD), (122.5, BDT)]."""
    out = []
    for m in _FIGURE.finditer(text or ""):
        num, unit = (m["n1"], _PRE[m["pre"].lower()]) if m["n1"] else (m["n2"], _POST[m["post"].lower()])
        out.append((float(num.replace(",", "")), unit))
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 0.005 * max(abs(a), abs(b))


def _figures_by_domain() -> dict[str, list[tuple[float, str]]]:
    return {d: figures(t) for d, t in _turn["texts"].items()}


def _rivals(value: float, unit: str, by_domain: dict) -> dict[str, float]:
    """Sites that give a different figure for what looks like the same price: same unit, within
    30%, and nothing close to `value` on that site. Only currencies: percentages in news text are
    too often about different things."""
    if unit == "%":
        return {}
    out = {}
    for d, figs in by_domain.items():
        same = [v for v, u in figs if u == unit]
        if value > 0 and same and not any(_close(v, value) for v in same):
            near = [v for v in same if 0.7 < v / value < 1.3]
            if near:
                out[d] = near[0]
    return out


def _fmt(value: float, unit: str) -> str:
    n = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{n}%" if unit == "%" else f"{n} {unit}"


def _conflicts() -> list[str]:
    by_domain = _figures_by_domain()
    seen, out = set(), []
    for d, figs in by_domain.items():
        for v, u in figs:
            if v == 1 or (u, v) in seen:
                continue
            seen.add((u, v))
            rivals = _rivals(v, u, by_domain)
            if rivals:
                other, rv = next(iter(rivals.items()))
                pair = tuple(sorted([(d, v), (other, rv)]))
                text = f"{source_name(pair[0][0])} says {_fmt(pair[0][1], u)}, {source_name(pair[1][0])} says {_fmt(pair[1][1], u)}"
                if text not in out:
                    out.append(text)
    return out[:3]


_BARE = re.compile(r"(?<![\d.,])(" + _NUM + r")(?![\d])")


def _reply_figures(reply: str, by_domain: dict) -> list[tuple[float, str]]:
    """Figures in the reply: tagged ones, plus any bare number a source gives as a rate or price
    ("133.20 Bangladeshi takas" has no unit right after the number)."""
    found = figures(reply)
    source_figs = {f for figs in by_domain.values() for f in figs}
    for m in _BARE.finditer(reply):
        v = float(m.group(1).replace(",", ""))
        for sv, u in source_figs:
            if sv != 1 and _close(v, sv) and not any(u == fu and _close(v, fv) for fv, fu in found):
                found.append((v, u))
    return found


def check_reply(reply: str) -> list[str]:
    """Problems with a web answer's figures, as sentences for the model (empty = fine)."""
    by_domain = _figures_by_domain()
    flags_disagreement = bool(DISAGREE.search(reply))
    problems = []
    for value, unit in _reply_figures(reply, by_domain):
        if value == 1:
            continue        # "1 USD = ..." is the unit of the rate, not a claim
        support = sorted(d for d, figs in by_domain.items() if any(u == unit and _close(v, value) for v, u in figs))
        rivals = _rivals(value, unit, by_domain)
        if rivals and not flags_disagreement:
            others = ", ".join(f"{source_name(d)} says {_fmt(v, unit)}" for d, v in rivals.items())
            problems.append(f"{_fmt(value, unit)} is contradicted ({others}); say the sources disagree")
        elif not support:
            problems.append(f"{_fmt(value, unit)} isn't in any source; don't state it")
        elif not flags_disagreement and not any(is_official(d) for d in support) and len(support) < 2:
            problems.append(f"{_fmt(value, unit)} comes only from {source_name(support[0])}, which isn't "
                            "official; say you couldn't confirm the figure instead of stating it")
    return problems


def correction(problems: list[str]) -> str:
    return ("[Automatic check, not from Mahadi] Your answer can't be spoken yet: " + "; ".join(problems)
            + f". Rewrite it in at most {MAX_SENTENCES} sentences, naming the source and its date.")


_BOUNDARY = re.compile(r"[.!?\u0964]+[\"')\]]*\s+")   # the same rule index.html uses to split speech (। = Bangla full stop)


def sentences(text: str) -> list[str]:
    out, last = [], 0
    for m in _BOUNDARY.finditer(text):
        out.append(text[last:m.end()].strip())
        last = m.end()
    if text[last:].strip():
        out.append(text[last:].strip())
    return out


_MONTHS = r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?"
_HAS_DATE = re.compile(rf"\b({_MONTHS})\.?\s+\d{{1,2}}\b|\b\d{{1,2}}\s+({_MONTHS})\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|"
                       r"\b(today|yesterday)\b", re.I)


def finalize(reply: str, problems: list[str], max_sentences: int | None = MAX_SENTENCES) -> str:
    """The spoken web answer: unresolved problems replaced, at most max_sentences sentences (None: no cap,
    for skills, which have their own spoken budget), ending with a source name and date if the model left
    them out."""
    lead, label = [], "Source"
    if any("isn't in any source" in p or "couldn't confirm" in p for p in problems):
        reply = "I couldn't confirm that figure from an official source or two sources that agree, so I won't guess."
        label = "Checked"
    elif problems:     # only disagreements left
        lead = ["The sources disagree on this."]
    low = reply.lower()
    cited = next((s for s in _turn["sources"] if s["name"].lower() in low or s["domain"] in low), None)
    body = lead + sentences(reply)
    cap = max_sentences or len(body) + 1
    if cited and _HAS_DATE.search(reply):
        return " ".join(body[:cap])
    src = cited or _turn["sources"][0]
    return " ".join(body[:cap - 1] + [f"{label}: {src['name']}, {src['date_text']}."])


# ---------- Models panel ----------
def status() -> dict:
    log = _today_log()
    return {
        "has_key": bool(_key()), "cap": DAILY_CREDIT_CAP, "used_today": sum(e.get("credits", 0) for e in log),
        "searches": sum(e["kind"] == "search" and not e.get("cached") for e in log),
        "reads": sum(e["kind"] == "extract" and not e.get("cached") for e in log),
        "cache_hits": sum(bool(e.get("cached")) for e in log), "cache_minutes": CACHE_SEC // 60,
        "last_used": max((e["ts"] for e in log), default=None), "last_error": _last_error,
    }
