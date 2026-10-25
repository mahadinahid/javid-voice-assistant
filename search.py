"""
Web search across the engines in the provider registry (config/providers.yaml): Tavily (the default for searching and
for reading pages), Exa (semantic: "companies like X", research papers; and a fallback), Brave (news, and a second,
independent index for facts), Firecrawl (reads a page only when Tavily's extract fails or hits a JavaScript wall) and
Cohere (reranks what the engines found). web_tools.py calls it from web_search, read_page and the news fallback; the
rules for answers (a source and its date, figures confirmed by two independent sources or an official one) stay
there.

- Intent, decided in code (classify): news -> Brave + Tavily; similar (companies or papers like X, alternatives to Y)
  -> Exa; fact (a figure, a measure, a date, who/when/how tall) -> Tavily + Brave, both at once; everything else ->
  Tavily. An engine that isn't on (no key) is left out of its plan.
- Engines run in parallel. Each has a soft timeout (soft_timeout_ms in the registry, 2.5 s to start with): when it
  passes without an answer, the next engine of the plan starts too and the first usable answer wins (general: Tavily,
  then Exa, then Brave; similar: Exa, then Tavily, then Brave; news and fact: Exa after both). A slow engine is given
  up on only once another has answered, or after MAX_WAIT_SEC: Tavily's searches took 3.8 s (median of 27 traced,
  2026-10-15 to 10-24; 22% under 2.5 s), so while it's the only engine with a key a hard 2.5 s cut would fail most
  searches. An engine that errs, finds nothing, is held back by the budget guard or Javid's daily cap, is resting
  after a 429, or may not see the query (privacy rule) counts as failed and the next one starts at once. One already
  asked is billed even when its answer comes too late (the ledger says "too late").
- Results are merged by URL (one source, credited to every engine that found it), reranked (Cohere when it's on and may
  see the query; else keyword overlap, with the engines' own order and agreement as tie-breaks) and cut to the best
  3-5 (MIN_HITS..MAX_HITS).
- A page is read with Tavily's extract, or Firecrawl when that fails or gets a JavaScript wall; best_passages() keeps
  its most relevant passages within a size budget.
- Cache by normalized query: 15 min for news and for anything asked "today/latest/current" (a price moves within a
  day), 24 h for facts and similar-searches, 1 h for the rest. In memory.
"""
import concurrent.futures as cf
import html
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import providers

ENGINES = ("tavily", "exa", "brave")
LABELS = {"tavily": "Tavily", "exa": "Exa", "brave": "Brave", "firecrawl": "Firecrawl", "cohere": "Cohere"}
# intent -> (engines asked at once, then the fallbacks in order)
PLANS = {
    "news": (("brave", "tavily"), ("exa",)),
    "fact": (("tavily", "brave"), ("exa",)),
    "similar": (("exa",), ("tavily", "brave")),
    "general": (("tavily",), ("exa", "brave")),
}
INTENT_TEXT = {"news": "news", "fact": "a fact or figure", "similar": "similar companies, papers or sites",
               "general": "web"}
SOFT_TIMEOUT_SEC = 2.5          # unless the registry gives an engine its own soft_timeout_ms
NEWS_MAX_DAYS = 31              # a longer window isn't news (the model asks for 365 days on how-to questions)
MAX_WAIT_SEC = 15.0             # a search that takes longer isn't worth waiting for in a spoken answer
TTL = {"news": 15 * 60, "timely": 15 * 60, "fact": 24 * 3600, "similar": 24 * 3600, "general": 3600}
MAX_HITS, MIN_HITS = 5, 3
DAILY_CAPS = {"tavily": 40}     # Javid's own cap on top of the free allowance (credits a day, local midnight)
SEARCH_COST = {"tavily": ("search_basic", 1.0), "exa": ("search", 0.005), "brave": ("search", 0.005)}
RERANK_MODEL = "rerank-v3.5"
EXTRACT_CACHE_SEC = 15 * 60
MIN_PAGE_CHARS = 400            # less from Tavily's extract than this: the page probably needs JavaScript
PRIVATE = "the query carries personal data"

_SIMILAR = re.compile(
    r"\b(?:compan(?:y|ies)|startups?|firms?|businesses|brands?|tools?|apps?|products?|platforms?|websites?|sites|"
    r"services|papers?|studies|articles)\s+(?:like|similar to)\b|\bsimilar\s+(?:to|compan(?:y|ies)|startups?|papers?|"
    r"tools?|sites?|products?|apps?)\b|\b(?:alternatives?|competitors?|rivals)\s+(?:to|of|for)\b|"
    r"\bfind (?:me )?(?:more )?(?:ones )?like\b|\b(?:research |academic |scientific )?papers?\s+(?:on|about|regarding)\b|"
    r"\bresearch papers?\b|\b(?:studies|literature)\s+(?:on|about)\b|\barxiv\b|"
    r"\blist (?:of )?(?:the )?(?:top |best |leading )?(?:compan(?:y|ies)|startups?|firms?|papers?)\b|"
    r"\b(?:top|best|leading|biggest)\s+(?:\w+\s+){0,2}(?:compan(?:y|ies)|startups?)\b", re.I)
_PAPERS = re.compile(r"\bpapers?\b|\bstud(?:y|ies)\b|\bresearch\b|\barxiv\b|\bacademic\b|\bliterature\b", re.I)
_COMPANIES = re.compile(r"\bcompan(?:y|ies)\b|\bstartups?\b|\bfirms?\b|\bbusinesses\b|\bbrands?\b|\bcompetitors?\b", re.I)
_NEWS = re.compile(r"\bnews\b|\bheadlines?\b|\bbreaking\b|\bwhat(?:'s| is| has)?\s+(?:been\s+)?happen(?:ing|ed)\b|"
                   r"\blatest\s+(?:on|news|updates?|developments?)\b|\bupdates?\s+(?:on|about)\b|খবর|সংবাদ", re.I)
_FACT = re.compile(
    r"\b(?:rates?|prices?|costs?|how (?:much|many|tall|high|big|large|far|old|long|deep|fast|heavy|wide)|percent(?:age)?|"
    r"inflation|gdp|growth|figures?|numbers?|amount|value|worth|salary|fees?|fares?|tariffs?|scores?|population|"
    r"interest|statistics?|capital of|founded|born|died|height|distance|area of|length of|when (?:was|did|is|were|does|"
    r"will)|what year|which year|who (?:is|was|won|invented|founded|wrote|discovered|owns|leads))\b|"
    r"\bwhen\b.{0,40}\b(?:opened|started|began|launched|founded|built|completed|ended|happened|introduced)\b|"
    r"%|দাম|রেট|কত", re.I)
_TIMELY = re.compile(r"\b(?:today|tonight|now|current(?:ly)?|latest|this (?:week|month|year|morning|evening)|right now|live|"
                     r"so far)\b|আজ|এখন", re.I)
_STOP = frozenset("""the and for with from that this what when where which who how why about into onto over under near
are was were has have had not but you your our his her its they them their can could would should will just than then
there here also some any all more most less very much many one two three today tonight now new latest best top
please find search look show tell give bring get make take does did is it of in on at to by an as be or""".split())
_TAGS = re.compile(r"<[^>]+>")
_JS_WALL = re.compile(r"enable javascript|javascript (?:is )?(?:required|disabled|turned off)|please turn on javascript|"
                      r"requires javascript|you need to enable javascript|browser (?:is )?not supported", re.I)


class Unavailable(Exception):
    """The engine can't be asked (no client)."""


# ---------- intent ----------
def classify(query: str, asked: str = "", days: int | None = None) -> str:
    """news, similar, fact or general, from Mahadi's message and the model's query together."""
    text = f"{asked} {query}"
    if _SIMILAR.search(text):
        return "similar"
    if _NEWS.search(text):
        return "news"
    if _FACT.search(text):
        return "fact"
    return "news" if days and days <= NEWS_MAX_DAYS else "general"


def timely(text: str) -> bool:
    """Asked about now: today, latest, current, this week..."""
    return bool(_TIMELY.search(text or ""))


def ttl(intent: str, query: str, asked: str = "", days: int | None = None) -> int:
    """How long a search's results are reused: a fact asked "today/latest/current" keeps the news time."""
    timely = intent == "news" or bool(_TIMELY.search(f"{asked} {query}")) or bool(days and days <= 7)
    return TTL["timely"] if timely and intent != "similar" else TTL[intent]


def normalize(query: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (query or "").lower()).split())


# ---------- engines ----------
def _request(method: str, url: str, *, headers: dict, timeout: float, json_body=None, params=None) -> dict:
    """One call to an engine's HTTP API (regression.py swaps this for recorded fixtures)."""
    import httpx
    resp = httpx.request(method, url, headers=headers, json=json_body, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _key(name: str) -> str:
    names = providers.env_names(name)
    return os.environ.get(names[0], "").strip() if names else ""


def _clean(text) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", str(text or ""))).split())


def _hit(engine: str, rank: int, url: str, title, content, published, score=None) -> dict:
    return {"url": url, "title": _clean(title)[:200], "content": _clean(content), "published_date": published or None,
            "engines": [engine], "ranks": {engine: rank}, "engine_score": score}


def _done(name: str, units: float, op: str, t0: float, late: threading.Event, unit: str | None = None) -> None:
    providers.record(name, units, op=op, unit=unit, latency_ms=(time.time() - t0) * 1000,
                     detail="too late: the search had moved on" if late.is_set() else "")


def _failed(name: str, op: str, t0: float, err: Exception) -> None:
    kind = providers.error_kind(err) if not isinstance(err, (KeyError, TypeError, ValueError)) else "error"
    providers.record(name, 0, op=op, latency_ms=(time.time() - t0) * 1000, status=kind, detail=type(err).__name__)
    if kind in ("rate_limited", "quota", "payment", "auth") and (rest := providers.cooldown_seconds(name, err)):
        providers.cool(name, rest, kind)


def _tavily_search(query: str, opts: dict, late: threading.Event) -> list[dict]:
    client = opts["tavily"]()
    if client is None:
        raise Unavailable("no Tavily client")
    kw = {"search_depth": "basic", "max_results": opts["k"], "include_usage": True,
          "topic": "news" if opts["news_topic"] else "general", "timeout": providers.timeout_s("tavily", 30.0)}
    if opts["days"]:
        kw["days"] = opts["days"]
    elif opts["year"]:
        kw["time_range"] = "year"
    if opts["domains"]:
        kw["include_domains"] = list(opts["domains"])
    t0 = time.time()
    try:
        resp = client.search(query, **kw)
    except Exception as e:
        _failed("tavily", "search", t0, e)
        raise
    credits = (resp.get("usage") or {}).get("credits")
    _done("tavily", credits if isinstance(credits, (int, float)) else providers.cost("tavily", "search_basic"), "search",
          t0, late)
    return [_hit("tavily", i, r["url"], r.get("title"), r.get("content"), r.get("published_date"), r.get("score"))
            for i, r in enumerate(resp.get("results") or []) if r.get("url")]


def _exa_search(query: str, opts: dict, late: threading.Event) -> list[dict]:
    body = {"query": query, "numResults": opts["k"], "type": "auto", "contents": {"text": {"maxCharacters": 1500}}}
    if opts.get("category"):
        body["category"] = opts["category"]
    if opts["days"] or opts["year"]:
        body["startPublishedDate"] = (datetime.now(timezone.utc) - timedelta(days=opts["days"] or 365)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z")
    if opts["domains"]:
        body["includeDomains"] = list(opts["domains"])
    t0 = time.time()
    try:
        data = _request("POST", "https://api.exa.ai/search", json_body=body, timeout=providers.timeout_s("exa", 20.0),
                        headers={"x-api-key": _key("exa"), "Content-Type": "application/json"})
    except Exception as e:
        _failed("exa", "search", t0, e)
        raise
    cost = (data.get("costDollars") or {}).get("total")
    _done("exa", float(cost) if isinstance(cost, (int, float)) else providers.cost("exa", "search", 0.005), "search", t0,
          late, unit="usd")
    return [_hit("exa", i, r["url"], r.get("title"), r.get("text") or " ".join(r.get("highlights") or []),
                 r.get("publishedDate"), r.get("score"))
            for i, r in enumerate(data.get("results") or []) if r.get("url")]


def _freshness(days: int) -> str:
    return "pd" if days <= 1 else "pw" if days <= 7 else "pm" if days <= 31 else "py"


def _brave_search(query: str, opts: dict, late: threading.Event) -> list[dict]:
    news = opts["intent"] == "news"
    q = query
    if opts["domains"]:
        q += " " + (f"site:{opts['domains'][0]}" if len(opts["domains"]) == 1
                    else "(" + " OR ".join(f"site:{d}" for d in opts["domains"]) + ")")
    params = {"q": q, "count": opts["k"], "result_filter": "news,web" if news else "web", "extra_snippets": "true",
              "text_decorations": "false"}
    if opts["days"] or opts["year"]:
        params["freshness"] = _freshness(opts["days"] or 365)
    t0 = time.time()
    try:
        data = _request("GET", "https://api.search.brave.com/res/v1/web/search", params=params,
                        timeout=providers.timeout_s("brave", 20.0),
                        headers={"X-Subscription-Token": _key("brave"), "Accept": "application/json"})
    except Exception as e:
        _failed("brave", "search", t0, e)
        raise
    _done("brave", providers.cost("brave", "search", 0.005), "search", t0, late, unit="usd")
    found = ((data.get("news") or {}).get("results") or [] if news else []) + ((data.get("web") or {}).get("results") or [])
    return [_hit("brave", i, r["url"], r.get("title"),
                 " ".join([r.get("description") or "", *(r.get("extra_snippets") or [])]), r.get("page_age"))
            for i, r in enumerate(found) if r.get("url")]


_SEARCHERS = {"tavily": _tavily_search, "exa": _exa_search, "brave": _brave_search}


# ---------- who may be asked ----------
def _midnight() -> float:
    return datetime.combine(date.today(), datetime.min.time()).timestamp()


def _usable(name: str, private: str | None, cost: float) -> str | None:
    """Why `name` can't be asked this now (None: it can)."""
    if not providers.get(name):
        return "not in config/providers.yaml"
    allowed, st = providers.allow(name, cost=cost)
    if not allowed:
        return f"{st['state']}: {st['why']}" if st["why"] else st["state"]
    if name in DAILY_CAPS and providers.used(name, _midnight()) + cost > DAILY_CAPS[name]:
        return f"daily_cap: Javid's own {DAILY_CAPS[name]} a day"
    if private and not providers.personal_ok(name):
        providers.record(name, 0, op="search", status="refused", detail="personal data in the query")
        return PRIVATE
    return None


_pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="search")


def _run(intent: str, query: str, opts: dict) -> tuple[dict, dict, list]:
    """The plan for `intent`, run as described at the top: ({engine: hits} that came in time, {engine: why not} for
    the others, the engines in the order they were asked)."""
    parallel, rest = PLANS[intent]
    queue = [e for e in rest if e not in parallel]
    answers, why, asked = {}, {}, []
    pending, soft_due, soft_passed = {}, {}, set()

    def start(name: str) -> bool:
        cost = providers.cost(name, *SEARCH_COST[name]) if name in SEARCH_COST else 0.0
        if (reason := _usable(name, opts["private"], cost)):
            why[name] = reason
            return False
        late = threading.Event()
        pending[_pool.submit(_SEARCHERS[name], query, opts, late)] = (name, late)
        soft_due[name] = time.monotonic() + providers.soft_timeout_s(name, SOFT_TIMEOUT_SEC)
        asked.append(name)
        return True

    def start_next() -> None:
        while queue:
            if start(queue.pop(0)):
                return

    for name in parallel:
        start(name)
    if not pending:
        start_next()
    deadline = time.monotonic() + MAX_WAIT_SEC
    while pending:
        waits = [soft_due[n] for n, _ in pending.values() if n not in soft_passed] + [deadline]
        done, _ = cf.wait(list(pending), timeout=max(0.0, min(waits) - time.monotonic()),
                          return_when=cf.FIRST_COMPLETED)
        for fut in done:
            name, _ = pending.pop(fut)
            try:
                hits = fut.result()
            except Exception as e:
                hits = []
                why[name] = providers.error_kind(e) if not isinstance(e, Unavailable) else "nokey"
            if hits:
                answers[name] = hits
            else:
                why.setdefault(name, "no results")
                start_next()                    # failed or found nothing: the next engine, at once
        now = time.monotonic()
        for name, _ in list(pending.values()):
            if name not in soft_passed and now >= soft_due[name]:
                soft_passed.add(name)
                if not answers:
                    start_next()                # slow: the next engine starts too, the slow one keeps going
        if answers and (len(parallel) == 1 or all(n in soft_passed for n, _ in pending.values())):
            break
        if now >= deadline:
            break
    for name, late in pending.values():         # given up on: their answers come too late to use
        late.set()
        why[name] = "too slow"
    return answers, why, asked


# ---------- merge and rerank ----------
def url_key(url: str) -> str:
    """The same page however it's written: no scheme, no www, no trailing slash, no tracking parameters."""
    p = urlparse(url or "")
    host = (p.hostname or "").lower().removeprefix("www.")
    query = "&".join(sorted(kv for kv in p.query.split("&")
                            if kv and not kv.lower().startswith(("utm_", "ref=", "fbclid=", "gclid=", "ocid="))))
    return host + (p.path.rstrip("/") or "") + (f"?{query}" if query else "")


def merge(answers: dict, order: list) -> list[dict]:
    """One hit per page, credited to every engine that found it (its longest snippet, the first date given)."""
    merged = {}
    for name in order:
        for h in answers.get(name, []):
            k = url_key(h["url"])
            if k not in merged:
                merged[k] = {**h, "engines": list(h["engines"]), "ranks": dict(h["ranks"])}
                continue
            m = merged[k]
            if name not in m["engines"]:
                m["engines"].append(name)
            m["ranks"].setdefault(name, h["ranks"][name])
            if len(h["content"]) > len(m["content"]):
                m["content"] = h["content"]
            m["published_date"] = m["published_date"] or h["published_date"]
            m["title"] = m["title"] or h["title"]
    return list(merged.values())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9ঀ-৿]+", (text or "").lower()) if len(w) > 2 and w not in _STOP]


def keyword_rank(query: str, hits: list[dict]) -> list[dict]:
    """Overlap of the query's words with each title (counted twice) and snippet, plus each engine's own order and
    being found by more than one engine as tie-breaks."""
    q = set(words(query))
    out = []
    for h in hits:
        title, body = set(words(h["title"])), set(words(h["content"]))
        overlap = (2 * len(q & title) + len(q & body)) / (3 * len(q)) if q else 0.0
        prior = max(1 / (1 + r) for r in h["ranks"].values())
        out.append({**h, "score": round(overlap + 0.3 * prior + 0.15 * (len(h["engines"]) - 1), 4)})
    return sorted(out, key=lambda h: -h["score"])


def _cut(ranked: list[dict], share: float) -> list[dict]:
    """The best MAX_HITS, and beyond MIN_HITS only those scoring at least `share` of the best."""
    if not ranked:
        return ranked
    best = ranked[0]["score"] or 1e-9
    return [h for i, h in enumerate(ranked[:MAX_HITS]) if i < MIN_HITS or h["score"] >= share * best]


def _cohere(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
    t0 = time.time()
    try:
        data = _request("POST", "https://api.cohere.com/v2/rerank", timeout=providers.soft_timeout_s("cohere", 2.5),
                        headers={"Authorization": f"Bearer {_key('cohere')}", "Content-Type": "application/json"},
                        json_body={"model": RERANK_MODEL, "query": query, "documents": docs, "top_n": top_n})
    except Exception as e:
        _failed("cohere", "rerank", t0, e)
        raise
    providers.record("cohere", providers.cost("cohere", "rerank", 1.0), op="rerank", latency_ms=(time.time() - t0) * 1000)
    return [(int(r["index"]), float(r["relevance_score"])) for r in data.get("results") or []]


def rerank(query: str, hits: list[dict], private: str | None = None) -> tuple[list[dict], str]:
    """(the best 3-5, "cohere" or "keyword")."""
    if len(hits) > 1 and providers.is_on("cohere") and providers.allow("cohere", cost=1.0)[0] \
            and (not private or providers.personal_ok("cohere")):
        try:
            order = _cohere(query, [f"{h['title']}\n{h['content'][:800]}" for h in hits], min(len(hits), MAX_HITS))
            if order:
                return _cut([{**hits[i], "score": s} for i, s in order if 0 <= i < len(hits)], 0.25), "cohere"
        except Exception:
            pass                        # keyword overlap instead
    return _cut(keyword_rank(query, hits), 0.4), "keyword"


# ---------- the search ----------
_cache: dict[tuple, tuple[float, float, dict]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
        _extract_cache.clear()


def search(query: str, *, days: int | None = None, domains=(), news_topic: bool = False, asked: str = "",
           tavily=lambda: None, private: str | None = None, k: int = MAX_HITS, intent: str | None = None,
           year: bool = False) -> dict:
    """{"intent", "hits" (the best 3-5, each with its engines), "found_by", "asked", "why" (per engine that didn't
    answer), "rerank", "cached", "refused" (no engine may see the query), "failed" (none answered)}."""
    intent = intent or classify(query, asked, days)
    domains = tuple(domains or ())
    key = (normalize(query), intent, days, domains, news_topic, k, year)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < hit[1]:
            for name in hit[2]["found_by"]:
                providers.record(name, 0, op="search", status="cached")
            return {**hit[2], "cached": True}
        _cache.pop(key, None)
    category = ("research paper" if _PAPERS.search(f"{asked} {query}") else
                "company" if _COMPANIES.search(f"{asked} {query}") else None) if intent == "similar" else None
    opts = {"intent": intent, "news_topic": news_topic, "days": days, "domains": domains, "k": k, "tavily": tavily,
            "private": private, "category": category, "year": year}
    answers, why, asked_engines = _run(intent, query, opts)
    found_by = [n for n in asked_engines if n in answers]
    hits, how = rerank(query, merge(answers, found_by), private)
    result = {"intent": intent, "hits": hits, "found_by": found_by, "asked": asked_engines, "why": why,
              "rerank": how, "cached": False, "failed": not answers and any(_failure(w) for w in why.values()),
              "refused": not asked_engines and any(w == PRIVATE for w in why.values())}
    if hits:
        with _cache_lock:
            _cache[key] = (time.time(), ttl(intent, query, asked, days), result)
    return result


def _failure(why: str) -> bool:
    """A reason that means the search couldn't be made (not "nothing found", not an engine without a key)."""
    return not (why == "no results" or why == PRIVATE or why.startswith(("nokey", "disabled", "not in")))


def failure_text(result: dict, query: str) -> str:
    """Why a search gave nothing, for the model to pass on (the first engine of the plan's reason first)."""
    if not result["failed"]:
        return f"No results for '{query}'. Say you couldn't find it."
    why, (first, rest) = result["why"], PLANS[result["intent"]]
    named = [(n, why[n]) for n in (*first, *rest) if n in why
             and not why[n].startswith(("nokey", "disabled", "not in", "no results"))]
    if not named:
        return (f"No results for '{query}'. Say you couldn't find it." if "no results" in why.values()
                else "Web search isn't set up: add TAVILY_API_KEY to .env and restart Javid.")
    said = "; ".join(f"{LABELS[n]}: {r.split(': ', 1)[-1]}" for n, r in named)
    if all(r.startswith("daily_cap") for _, r in named):
        return (f"The daily web search budget ({DAILY_CAPS['tavily']} Tavily credits) is used up. "
                "Tell Mahadi it resets at midnight.")
    if all(r.startswith(("off", "cooldown", "daily_cap")) for _, r in named):
        return f"Web search is paused ({said}). Tell Mahadi."
    return f"Web search failed ({said}). Tell Mahadi you couldn't search right now."


# ---------- reading a page ----------
_extract_cache: dict[tuple, tuple[float, str, str]] = {}


def _tavily_extract(client, url: str, query: str | None) -> str:
    t0 = time.time()
    try:
        resp = client.extract(urls=[url], extract_depth="basic", query=query or None, chunks_per_source=5,
                              include_usage=True, timeout=providers.timeout_s("tavily", 30.0))
    except Exception as e:
        _failed("tavily", "extract", t0, e)
        raise
    credits = (resp.get("usage") or {}).get("credits")
    providers.record("tavily", credits if isinstance(credits, (int, float)) else providers.cost("tavily", "extract_basic"),
                     op="extract", latency_ms=(time.time() - t0) * 1000)
    page = next((r for r in resp.get("results") or [] if r.get("url") == url), None) or (resp.get("results") or [None])[0]
    return (page or {}).get("raw_content") or ""


def _firecrawl(url: str) -> str:
    t0 = time.time()
    try:
        data = _request("POST", "https://api.firecrawl.dev/v1/scrape", timeout=providers.timeout_s("firecrawl", 30.0),
                        headers={"Authorization": f"Bearer {_key('firecrawl')}", "Content-Type": "application/json"},
                        json_body={"url": url, "formats": ["markdown"], "onlyMainContent": True})
    except Exception as e:
        _failed("firecrawl", "scrape", t0, e)
        raise
    page = data.get("data") or {}
    used = (page.get("metadata") or {}).get("creditsUsed")
    providers.record("firecrawl", float(used) if isinstance(used, (int, float)) else providers.cost("firecrawl", "scrape"),
                     op="scrape", latency_ms=(time.time() - t0) * 1000)
    return page.get("markdown") or ""


def needs_js(text: str) -> bool:
    return len(" ".join((text or "").split())) < MIN_PAGE_CHARS or bool(_JS_WALL.search((text or "")[:3000]))


def read(url: str, query: str, *, tavily=lambda: None, private: str | None = None) -> tuple[str, str, str]:
    """(the page's text, the engine that read it, why it couldn't be read). Tavily's extract (sent the search query
    only if Tavily may see it), then Firecrawl when that failed or got a JavaScript wall."""
    with _cache_lock:
        hit = _extract_cache.get((url, query))
        if hit and time.time() - hit[0] < EXTRACT_CACHE_SEC:
            providers.record(hit[2], 0, op="extract", status="cached")
            return hit[1], hit[2], ""
    text, engine, why = "", "", ""
    client = tavily()
    if client is not None and not (reason := _usable("tavily", None, providers.cost("tavily", "extract_basic"))):
        try:
            text = _tavily_extract(client, url, query if not private or providers.personal_ok("tavily") else None)
            engine = "tavily"
        except Exception as e:
            why = f"Tavily: {providers.error_kind(e)}"
    elif client is not None:
        why = f"Tavily: {reason}"
    if needs_js(text) and not (reason := _usable("firecrawl", None, providers.cost("firecrawl", "scrape"))):
        try:
            scraped = _firecrawl(url)
            if len(scraped.strip()) > len(text.strip()):
                text, engine = scraped, "firecrawl"
        except Exception as e:
            why = (why + "; " if why else "") + f"Firecrawl: {providers.error_kind(e)}"
    text = text.strip()
    if text:
        with _cache_lock:
            _extract_cache[(url, query)] = (time.time(), text, engine)
    return text, engine, why


def best_passages(text: str, query: str, max_chars: int) -> str:
    """The page's passages that share the most words with the query, in page order, within `max_chars`."""
    flat = " ".join((text or "").split())
    if len(flat) <= max_chars:
        return flat
    sentences = re.split(r"(?<=[.!?।])\s+|\s+(?=#{1,6}\s)", flat)
    passages, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > 500:
            passages.append(current)
            current = ""
        current = f"{current} {s}".strip()
    if current:
        passages.append(current)
    q = set(words(query))
    scored = sorted(range(len(passages)), key=lambda i: (-len(q & set(words(passages[i]))), i))
    chosen, used = [], 0
    for i in scored:
        p = passages[i][:max_chars]
        if used + len(p) > max_chars:
            continue
        chosen.append(i)
        used += len(p) + 3
    return " … ".join(passages[i][:max_chars] for i in sorted(chosen))
