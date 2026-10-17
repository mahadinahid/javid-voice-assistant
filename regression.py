"""
Regression check for Javid's agent behaviour: 43 prompts through the real LLM + tool loop.

    python regression.py              # run everything once
    python regression.py -n 3         # repeat each case 3x (the model is non-deterministic)
    python regression.py -k delete    # only cases whose name contains "delete"
    python regression.py -v           # print every reply, not just failures
    python regression.py --use-120b   # on gpt-oss-120b instead of the default gpt-oss-20b

Runs on gpt-oss-20b alone by default (no fallback: a model change mid-run would mix results, and the live
chain's 120b is the user's backup, so it is only used with --use-120b).

Safe to run: tasks, memory and chat history go to a temp dir, the Google Calendar is a
fake in-memory service, and Tavily is a fake with fixed results (no credits spent, no key
needed). Your real data is never touched.
Needs GROQ_API_KEY. A full run makes ~40-50 LLM requests (~40-45k tokens): on Groq's free tier
(200k tokens/day for this model) that is half the day's budget, so don't loop it. Exit code: 0 all passed, 1 a check failed, 3 only API/network errors (inconclusive).
"""
import argparse
import contextlib
import copy
import io
import json
import os
import re
import signal
import sys
import time
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_tools as ct  # noqa: E402
import javid  # noqa: E402
import memory_store  # noqa: E402
import rates  # noqa: E402
import skills  # noqa: E402
import web_tools  # noqa: E402
from groq import Groq  # noqa: E402

# A stalled stream must fail the run, not hang the suite.
javid.client = Groq(timeout=30.0, max_retries=2)
RUN_TIMEOUT_SEC = 150      # hard cap for one case run
INFRA_RETRIES = 2          # re-runs when the API itself failed (not the assistant)
API_FAILED_REPLY = tuple(javid.PHRASES["api_failed"].values())   # think_stream's canned reply on APIError


class InfraError(Exception):
    """The Groq API (network, timeout, 5xx) failed — says nothing about Javid's behaviour."""


# Count LLM requests, and turn a 429 into a hard stop (each request costs ~1k tokens of the daily cap).
LLM_REQUESTS = 0
QUOTA_HIT: list[str] = []
_orig_stream_completion = javid._stream_completion


PER_MINUTE_WAIT_MAX = 90   # a 429 that clears within this is a per-minute limit: wait it out


def _counted_stream_completion(tool_calls, *args):
    """Count requests; ride out per-minute 429s (Gemini's free tier allows 5/min per model) and
    treat a longer wait (daily cap) as 'out of quota' so the suite stops instead of lying."""
    global LLM_REQUESTS
    for attempt in range(6):
        LLM_REQUESTS += 1
        try:
            yield from _orig_stream_completion(tool_calls, *args)
            return
        except javid.RATE_LIMITED as e:
            wait = javid._retry_after(e)
            if wait > PER_MINUTE_WAIT_MAX or attempt == 5:
                QUOTA_HIT.append(str(e))
                raise
            time.sleep(wait + 1)   # no chunk was yielded yet: a 429 arrives when the stream opens


javid._stream_completion = _counted_stream_completion

RUN_TOKENS = {"requests": 0, "prompt": 0, "cached": 0, "completion": 0, "total": 0}   # every request, pings too
_orig_record_usage = javid._record_usage


def _tallied_record_usage(model, usage):
    _orig_record_usage(model, usage)
    RUN_TOKENS["requests"] += 1
    RUN_TOKENS["prompt"] += usage.prompt_tokens or 0
    RUN_TOKENS["cached"] += javid._cached_tokens(usage)
    RUN_TOKENS["completion"] += usage.completion_tokens or 0
    RUN_TOKENS["total"] += usage.total_tokens or 0


javid._record_usage = _tallied_record_usage
DEFAULT_MODEL, BIG_MODEL = "openai/gpt-oss-20b", "openai/gpt-oss-120b"


def preflight() -> str | None:
    """Tiny pings down the model chain; returns an error message only if no model answers."""
    pings = javid.ping_models()
    if any(p["ok"] for p in pings):
        return None
    return "\n".join(f"{p['id']}: {p.get('error', 'failed')[:160]}" for p in pings)

# ---------- isolation: nothing below may touch the real files or calendar ----------
SCRATCH = tempfile.mkdtemp(prefix="javid-regression-")
javid.TASKS_FILE = os.path.join(SCRATCH, "tasks.txt")
memory_store.MEMORY_DIR = os.path.join(SCRATCH, "memory")
memory_store.LEGACY_FILE = os.path.join(SCRATCH, "memory.json")
memory_store.INDEX_FILE = os.path.join(SCRATCH, "chat_index.db")
javid.HISTORY_FILE = os.path.join(SCRATCH, "chat_log.jsonl")
memory_store.HISTORY_FILE = javid.HISTORY_FILE   # search_history indexes the scratch log, never the real one
javid.SESSIONS_FILE = os.path.join(SCRATCH, "sessions.json")      # chats made by the suite stay in scratch
javid.LLM_TURNS_FILE = os.path.join(SCRATCH, "llm_turns.jsonl")   # the Models panel averages real turns only
javid.LANGUAGE_FILE = os.path.join(SCRATCH, "language.json")      # never flips the real reply language
web_tools.USAGE_FILE = os.path.join(SCRATCH, "tavily_usage.jsonl")
TURN_TOKENS: list[tuple[str, dict]] = []   # (case name, javid.last_turn_usage) for every turn run


# ---------- fake Google Calendar ----------
def _at(day: date, h: int, m: int = 0) -> datetime:
    return datetime.combine(day, datetime.min.time(), ct._tz()).replace(hour=h, minute=m)


def timed(id_, title, day, h, m=0, mins=60):
    s = _at(day, h, m)
    return {"id": id_, "summary": title, "start": {"dateTime": s.isoformat()},
            "end": {"dateTime": (s + timedelta(minutes=mins)).isoformat()}}


def allday(id_, title, day):
    return {"id": id_, "summary": title, "start": {"date": day.isoformat()},
            "end": {"date": (day + timedelta(days=1)).isoformat()},
            "description": "Public holiday"}


class _Call:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeService:
    """Just enough of the Calendar API: list / insert / patch / delete, plus an action log."""

    def __init__(self, events, holidays):
        self.items = copy.deepcopy(events)
        self.holidays = copy.deepcopy(holidays)
        self.log = []   # ("insert", body) | ("patch", id, body) | ("delete", id)

    def events(self):
        return self

    def list(self, calendarId, timeMin, timeMax, **kw):
        lo, hi = datetime.fromisoformat(timeMin), datetime.fromisoformat(timeMax)
        pool = self.holidays if "holiday" in calendarId else self.items

        def inside(ev):
            s, e, _ = ct._event_times(ev)
            return s < hi and e > lo
        found = sorted((e for e in pool if inside(e)), key=lambda e: ct._event_times(e)[0])
        return _Call(lambda: {"items": found})

    def insert(self, calendarId, body):
        def go():
            self.items.append({"id": f"new{len(self.items)}", **body})
            self.log.append(("insert", body))
        return _Call(go)

    def patch(self, calendarId, eventId, body):
        def go():
            ev = next(e for e in self.items if e["id"] == eventId)
            ev.update(body)
            self.log.append(("patch", eventId, body))
            return ev
        return _Call(go)

    def delete(self, calendarId, eventId):
        def go():
            self.items[:] = [e for e in self.items if e["id"] != eventId]
            self.log.append(("delete", eventId))
        return _Call(go)


# ---------- fixture data (dates are relative to today so the suite never goes stale) ----------
TODAY = date.today()


# ---------- fake Tavily ----------
def _rfc(d: date) -> str:     # Tavily's news results carry RFC 2822 dates
    return datetime.combine(d, datetime.min.time()).strftime("%a, %d %b %Y 08:00:00 GMT")


def hit(url, title, content, day=None):
    return {"url": url, "title": title, "content": content, "score": 0.8,
            **({"published_date": _rfc(day)} if day else {})}


YESTERDAY = TODAY - timedelta(days=1)
BBS_URL = "https://bbs.gov.bd/site/page/cpi-september"
# (query regex, results), first match wins; the euro case must come before the general rate one.
WEB_FIXTURES = [
    (r"metro", [
        hit("https://www.dhakatribune.com/metro-rail-hours", "Metro rail to run until 11pm from November 1",
            "Dhaka Metro Rail Line 6 will run until 11pm from November 1, DMTCL said on Monday.", TODAY),
        hit("https://www.thedailystar.net/metro-hours", "Metro rail hours extended",
            "Commuters welcomed the move to extend metro rail service hours to 11pm.", YESTERDAY)]),
    (r"cricket|tigers|odi", [
        hit("https://www.espncricinfo.com/bangladesh-zimbabwe-1st-odi", "Bangladesh beat Zimbabwe by six wickets",
            "Bangladesh beat Zimbabwe by six wickets in the first ODI in Harare to go 1-0 up in the series.", YESTERDAY)]),
    (r"euro|\beur\b", [
        hit("https://fxquotes-blog.com/eur-bdt", "EUR to BDT today", "Today 1 EUR = 133.20 BDT, according to our tracker."),
        hit("https://travelnotes.example.com/dhaka-money", "Changing money in Dhaka",
            "Exchanging euros in Dhaka is easy at licensed money changers in Gulshan and Motijheel.")]),
    (r"dollar|usd|exchange rate|taka", [
        hit("https://www.bb.org.bd/en/index.php/econdata/exchangerate", "Exchange rate of Taka",
            "Bangladesh Bank reference rate: 1 USD = 122.50 BDT."),
        hit("https://www.thedailystar.net/business/dollar-rate", "Dollar steady in interbank market",
            "The dollar traded at Tk 122.50 in the interbank market on Tuesday.", TODAY)]),
    (r"gold", [
        hit("https://en.prothomalo.com/business/gold-price", "Gold price rises again",
            "22-carat gold is selling at Tk 15,200 per gram in Dhaka today, jewellers said.", TODAY),
        hit("https://www.tbsnews.net/economy/gold-price-cut", "Bajus cuts gold price",
            "Gold price in Dhaka: 22-carat now costs Tk 14,650 per gram after Bajus cut prices.", TODAY)]),
    (r"inflation", [
        hit(BBS_URL, "Consumer Price Index and inflation, September 2026",
            "The Bangladesh Bureau of Statistics publishes the monthly CPI release with point-to-point inflation figures."),
        hit("https://www.tbsnews.net/economy/inflation-eases", "Inflation eases in September",
            "Inflation eased in September, according to BBS data released on Sunday.", YESTERDAY)]),
    (r"everest", [
        hit("https://en.wikipedia.org/wiki/Mount_Everest", "Mount Everest - Wikipedia",
            "Mount Everest is Earth's highest mountain above sea level, with an elevation of 8,849 m (29,032 ft)."),
        hit("https://www.britannica.com/place/Mount-Everest", "Mount Everest | Britannica",
            "Mount Everest, at 29,032 feet (8,849 metres), is the highest point on Earth.")]),
    (r"padma", [
        hit("https://en.wikipedia.org/wiki/Padma_Bridge", "Padma Bridge - Wikipedia",
            "The Padma Bridge was inaugurated on 25 June 2022 and opened to traffic the next day.")]),
    (r"capital|australia", [
        hit("https://en.wikipedia.org/wiki/Canberra", "Canberra - Wikipedia", "Canberra is the capital city of Australia."),
        hit("https://www.britannica.com/place/Canberra", "Canberra | Britannica", "Canberra, the capital of Australia, located in the ACT.")]),
]
WEB_FIXTURES[:0] = [   # news-digest topics; first, so "europe news" doesn't get the euro-rate results
    (r"^(world|global|international)\b.*news$", [
        hit("https://www.reuters.com/world/un-climate-talks", "UN climate talks open in Nairobi",
            "Delegates from nearly 200 countries opened climate talks in Nairobi on Friday.", TODAY),
        hit("https://apnews.com/article/global-shipping-delays", "Storms delay global shipping routes",
            "Severe storms delayed container ships on several major routes, shipping firms said.", TODAY)]),
    (r"^asia\w*\b.*news$", [
        hit("https://www.aljazeera.com/news/japan-election-call", "Japan's prime minister calls snap election",
            "Japan's prime minister called a snap election for November, broadcaster NHK reported.", TODAY),
        hit("https://www.bbc.com/news/world-asia-monsoon-floods", "Late monsoon floods hit northern Thailand",
            "Floods forced thousands from their homes in northern Thailand after late monsoon rains.", YESTERDAY)]),
    (r"^europe\w*\b.*news$", [
        hit("https://www.theguardian.com/world/eu-migration-deal", "EU leaders back new migration deal",
            "European Union leaders backed a new migration deal at a summit in Brussels.", TODAY),
        hit("https://www.bbc.com/news/world-europe-rail-strike", "French rail workers announce strike",
            "French rail unions announced a two-day strike over pension changes.", YESTERDAY)]),
    (r"^middle east\b.*news$", [
        hit("https://www.aljazeera.com/news/gaza-aid-convoy", "Aid convoy reaches northern Gaza",
            "A UN aid convoy reached northern Gaza for the first time in weeks, the UN said.", TODAY),
        hit("https://www.reuters.com/world/middle-east/iran-talks", "Iran and Oman hold talks in Muscat",
            "Iranian and Omani officials held talks in Muscat on regional security.", YESTERDAY)]),
    (r"^(war|ukraine|conflict|military)\b.*news$", [
        hit("https://www.reuters.com/world/europe/ukraine-refinery-strike", "Ukraine says it struck a Russian refinery",
            "Ukraine said its drones hit an oil refinery in southern Russia overnight.", TODAY),
        hit("https://www.bbc.com/news/world-africa-sudan-ceasefire", "Sudan ceasefire talks resume in Jeddah",
            "Talks on a ceasefire in Sudan resumed in Jeddah, mediators said.", YESTERDAY)]),
    (r"^(sports?|football|cricket)\b.*news$", [
        hit("https://www.espncricinfo.com/bangladesh-zimbabwe-1st-odi", "Bangladesh beat Zimbabwe by six wickets",
            "Bangladesh beat Zimbabwe by six wickets in the first ODI in Harare.", YESTERDAY),
        hit("https://www.bbc.com/sport/football/champions-league-draw", "Champions League: holders drawn against Inter",
            "The holders were drawn against Inter Milan in the Champions League last 16.", TODAY)]),
    (r"^(stock|market|shares|business|financ|econom)\w*\b.*news$", [   # a figure no second source gives
        hit("https://www.reuters.com/markets/asia-stocks-fall", "Asian stocks fall 1.2% on rate worries",
            "Asian stocks fell 1.2% on Friday as investors worried about interest rates.", TODAY),
        hit("https://apnews.com/article/wall-street-tech-rally", "Wall Street ends higher on tech rally",
            "US stocks ended higher, led by technology shares.", YESTERDAY)]),
]
WEB_FIXTURES += [   # morning-brief headlines (news_headlines searches "<topic> news")
    (r"^bangladesh news$", [hit("https://www.thedailystar.net/padma-rail-freight", "Padma rail link opens to freight trains",
                                "Freight trains began running on the Padma rail link on Tuesday, Bangladesh Railway said.", TODAY)]),
    (r"^china news$", [hit("https://www.reuters.com/world/china/exports-september", "China's exports beat forecasts in September",
                           "China's exports grew faster than expected in September, customs data showed.", TODAY)]),
    (r"^ai news$", [hit("https://www.bbc.com/news/technology-ai-rules", "EU agrees new rules for AI chatbots",
                        "European lawmakers agreed new transparency rules for AI chatbots on Monday.", YESTERDAY)]),
]
FAKE_ECB = {"amount": 1.0, "base": "USD", "date": YESTERDAY.isoformat(), "rates": {"CNY": 6.6938, "CZK": 21.78}}
FAKE_AGG = {"result": "success", "time_last_update_unix": int(datetime.combine(TODAY, datetime.min.time()).timestamp()),
            "rates": {"USD": 1, "BDT": 123.18, "CNY": 6.705, "CZK": 21.76}}


def fake_rates(up: bool):
    def get(url):
        if not up:
            raise OSError("network down (fake)")
        return copy.deepcopy(FAKE_ECB if "frankfurter" in url else FAKE_AGG)
    return get


WEB_PAGES = {BBS_URL: "Point-to-point inflation in September 2026 stood at 8.36 percent, down from 8.91 percent in August."}


class FakeTavily:
    def __init__(self):
        self.searches, self.extracts = [], []

    def search(self, query, **kw):
        self.searches.append((query, kw))
        results = next((r for pat, r in WEB_FIXTURES if re.search(pat, query, re.I)), [])
        return {"query": query, "results": copy.deepcopy(results), "usage": {"credits": 1}}

    def extract(self, urls, **kw):
        urls = [urls] if isinstance(urls, str) else urls
        self.extracts.append(urls)
        return {"results": [{"url": u, "raw_content": WEB_PAGES[u]} for u in urls if u in WEB_PAGES],
                "failed_results": [{"url": u} for u in urls if u not in WEB_PAGES], "usage": {"credits": 1}}


def next_weekday(wd: int) -> date:   # Monday = 0; always in the future
    return TODAY + timedelta(days=(wd - TODAY.weekday()) % 7 or 7)


TOMORROW = TODAY + timedelta(days=1)
FRIDAY, MONDAY = next_weekday(4), next_weekday(0)
SATURDAY = next_weekday(5)
HOLIDAY_DATE = TODAY + timedelta(days=9)
HOLIDAY_NAME = "Victory Day Test"

EVENTS = [
    timed("dent", "Dentist appointment", FRIDAY, 15),
    timed("review", "Project review", TOMORROW, 14),
    timed("lunch", "Lunch with Sam", TOMORROW, 13, mins=90),
    timed("hike", "Hiking trip", SATURDAY, 8, mins=120),
]
HOLIDAYS = [allday("h1", HOLIDAY_NAME, HOLIDAY_DATE)]
# A Durga Puja-like run, Tuesday to Thursday, with the Sunday before it a working day: the shape of
# the 2026-10-15 chat where Javid said the holidays started on that Sunday.
PUJA_TUE = next_weekday(1) if (next_weekday(1) - TODAY).days >= 3 else next_weekday(1) + timedelta(days=7)
PUJA_MON, PUJA_WED, PUJA_THU = (PUJA_TUE + timedelta(days=n) for n in (-1, 1, 2))
PUJA_SUN = PUJA_TUE - timedelta(days=2)
PUJA_HOLIDAYS = [allday("p1", "Mahanabami", PUJA_TUE), allday("p2", "Durga Puja", PUJA_WED),
                 allday("p3", "Durga Puja Holiday", PUJA_THU)]
WEB_FIXTURES += [   # a visa centre's own notice, which closes one day more than the national list
    (r"ivac|visa", [hit("https://www.ivacbd.com/notice/durga-puja", "IVAC holiday notice: Durga Puja",
                        f"All IVACs in Bangladesh, including Jamuna Future Park, will remain closed on "
                        f"{PUJA_MON:%A %d %B}, {PUJA_TUE:%A %d %B}, {PUJA_WED:%A %d %B} and {PUJA_THU:%A %d %B} "
                        f"for Durga Puja.", YESTERDAY)]),
]


# ---------- case plumbing ----------
@dataclass
class Result:
    turns: list            # user messages
    replies: list          # Javid's reply per turn
    tools: list            # [(turn_index, name, args)]
    tasks: list            # lines of tasks.txt after the run
    facts: list            # memory lines (USER.md + MEMORY.md) after the run
    cal_log: list          # fake-calendar writes after the run
    cal_log_by_turn: list  # cumulative len(cal_log) after each turn
    sources: list = field(default_factory=list)  # links shown on screen (last turn that had any)
    said: list = field(default_factory=list)     # "say" events (spoken, not part of the reply)
    web: object = None                           # the FakeTavily used
    screen: str = ""                             # a skill's screen-only part (last turn that had one)
    skills_run: list = field(default_factory=list)  # "skill" events
    facts_by_turn: list = field(default_factory=list)  # memory lines after each turn
    langs: list = field(default_factory=list)    # the reply language after each turn

    @property
    def tool_names(self):
        return [n for _, n, _ in self.tools]

    @property
    def last(self):
        return self.replies[-1]

    def writes_in_turn(self, i):
        lo = self.cal_log_by_turn[i - 1] if i else 0
        return self.cal_log[lo:self.cal_log_by_turn[i]]


@dataclass
class Case:
    name: str
    turns: list
    checks: list
    tasks: list = field(default_factory=list)    # raw tasks.txt lines to start with
    facts: list = field(default_factory=list)
    lang: str = "en"                             # reply language to start in
    history: list = field(default_factory=list)  # chat_log entries from earlier chats (for search_history)
    rates_up: bool = True                        # the fake exchange-rate APIs answer
    holidays: list = None                        # the fake public-holiday calendar (default HOLIDAYS)


def called(name):
    return lambda r: None if name in r.tool_names else f"expected a {name} call, got {r.tool_names or 'no tools'}"


def called_in_turn(i, name):
    return lambda r: None if any(t == i and n == name for t, n, _ in r.tools) else \
        f"expected a {name} call in turn {i + 1}, got {[n for t, n, _ in r.tools if t == i] or 'no tools'}"


def not_called(*names):
    return lambda r: next((f"must not call {n}" for n in names if n in r.tool_names), None)


def _plain(text):
    """gpt-oss-120b writes narrow no-break spaces ("Victory\u202fDay", "3\u202fPM"): compare as spaces."""
    return re.sub(r"[\u00a0\u202f\u2009]", " ", text)


def reply_has(pattern, label=None):
    return lambda r: None if re.search(pattern, _plain(r.last), re.I) else f"reply should match /{label or pattern}/"


def reply_lacks(pattern):
    return lambda r: f"reply must not match /{pattern}/" if re.search(pattern, _plain(r.last), re.I) else None


def tasks_are(*texts):
    """tasks.txt (timestamps stripped, '✓ ' kept) equals this list."""
    def check(r):
        got = [re.sub(r"^(✓ )?\[[^\]]*\] ", r"\1", t) for t in r.tasks]
        return None if got == list(texts) else f"tasks should be {list(texts)}, got {got}"
    return check


def fact_has(pattern):
    return lambda r: None if any(re.search(pattern, f, re.I) for f in r.facts) else f"no stored fact matches /{pattern}/ (have {r.facts})"


def cal_writes(kind, *ids_or_none):
    """Exactly these calendar writes of this kind happened (by event id; insert: count only)."""
    def check(r):
        got = [w[1] if kind != "insert" else "x" for w in r.cal_log if w[0] == kind]
        want = list(ids_or_none)
        return None if got == want else f"calendar {kind}s should be {want}, got {got}"
    return check


def no_cal_writes(r):
    return None if not r.cal_log else f"calendar must not be modified, got {r.cal_log}"


def not_deleted_until_turn(n):
    """No delete may happen before turn index n (0-based): the two-step confirmation."""
    def check(r):
        early = [w for i in range(n) for w in r.writes_in_turn(i) if w[0] == "delete"]
        return f"deleted before confirmation: {early}" if early else None
    return check


def inserted_event(title_re, hour, minute=0):
    def check(r):
        for w in r.cal_log:
            if w[0] == "insert":
                body = w[1]
                start = datetime.fromisoformat(body["start"]["dateTime"])
                if re.search(title_re, body["summary"], re.I) and (start.hour, start.minute) == (hour, minute):
                    return None
                return f"inserted {body['summary']!r} at {start:%H:%M}, wanted /{title_re}/ at {hour:02d}:{minute:02d}"
        return "no event was inserted"
    return check


def moved_to(event_id, hour, minute=0):
    def check(r):
        for w in r.cal_log:
            if w[0] == "patch" and w[1] == event_id:
                start = datetime.fromisoformat(w[2]["start"]["dateTime"])
                return None if (start.hour, start.minute) == (hour, minute) else f"moved to {start:%H:%M}"
        return f"{event_id} was not moved"
    return check


def tool_arg(tool, key, pattern):
    def check(r):
        for _, n, a in r.tools:
            if n == tool and re.search(pattern, str(a.get(key, "")), re.I):
                return None
        return f"{tool} should have been called with {key} matching /{pattern}/"
    return check


def web_cited(r):
    """Names one of the turn's sources and gives a date (web_tools.finalize guarantees it)."""
    names = [s["name"] for s in r.sources]
    if not any(n.lower() in r.last.lower() for n in names):
        return f"reply names no source (have {names})"
    return None if web_tools._HAS_DATE.search(r.last) else "reply has no date"


def links_shown(r):
    return None if r.sources and all(s["url"].startswith("http") for s in r.sources) else "no source links sent to the screen"


def said_searching(r):
    return None if web_tools.SEARCHING in r.said else "'Searching…' wasn't spoken"


def at_most_3_sentences(r):
    n = len(web_tools.sentences(r.last))
    return None if n <= 3 else f"reply has {n} sentences"


def news_search(r):
    kws = [kw for _, kw in r.web.searches]
    return None if any(kw.get("topic") == "news" and kw.get("days") for kw in kws) else f"no news-topic search with a time filter: {kws}"


WEB_OK = [called("web_search"), web_cited, links_shown, said_searching, at_most_3_sentences]


def reply_in(lang):
    """The last reply is in that language's script (Bangla = Bengali script; names like "PM" are fine)."""
    def check(r):
        if not re.search(r"[A-Za-z\u0980-\u09ff]", r.last):
            return "reply has no words"
        return f"reply should be in {lang}" if javid.wrong_language(r.last, lang) else None
    return check


def lang_is(*langs):
    """The reply language after each turn."""
    return lambda r: None if r.langs == list(langs) else f"language after each turn should be {list(langs)}, got {r.langs}"


def at_most_sentences(n):
    return lambda r: None if len(web_tools.sentences(r.last)) <= n else f"reply has {len(web_tools.sentences(r.last))} sentences (max {n})"


def skill_ran(name):
    return lambda r: None if name in r.skills_run else f"skill {name} should have run (ran: {r.skills_run or 'none'})"


def screen_has(pattern, label=None):
    return lambda r: None if re.search(pattern, _plain(r.screen), re.I) else f"screen part should match /{label or pattern}/ (got {r.screen[:160]!r})"


def screen_lacks(pattern):
    return lambda r: f"screen part must not match /{pattern}/" if re.search(pattern, _plain(r.screen), re.I) else None


def spoken_within(seconds):
    limit = int(seconds * skills.WORDS_PER_SECOND)
    return lambda r: None if len(r.last.split()) <= limit else f"spoken part has {len(r.last.split())} words (max {limit}, {seconds} s)"


BN_TOMORROW = "কাল আমার ক্যালেন্ডারে কী আছে?"   # "what's on my calendar tomorrow?"
DISAGREES = r"disagree|differ|vary|varies|conflict|inconsisten|mixed|discrepan"
WEEKDAY = TODAY.strftime("%A")

def _all_facts() -> list[str]:
    return memory_store.read("user") + memory_store.read("memory")


def saved_after_turn(i, pattern):
    def check(r):
        have = r.facts_by_turn[i] if i < len(r.facts_by_turn) else []
        return None if any(re.search(pattern, f, re.I) for f in have) else f"/{pattern}/ should be saved after turn {i + 1} (have {have})"
    return check


def not_saved_after_turn(i, pattern):
    def check(r):
        have = r.facts_by_turn[i] if i < len(r.facts_by_turn) else []
        return f"/{pattern}/ was saved after turn {i + 1} without approval: {have}" if any(re.search(pattern, f, re.I) for f in have) else None
    return check


def nothing_pending(r):
    return None if not memory_store.pending() else f"proposals still pending: {memory_store.pending()}"


LAST_WEEK = (datetime.now() - timedelta(days=6)).replace(hour=19, minute=10, second=0, microsecond=0)
OLD_CHAT = [
    {"ts": LAST_WEEK.isoformat(), "role": "user", "text": "My car insurance is with Green Delta and the renewal is due on November 3."},
    {"ts": (LAST_WEEK + timedelta(seconds=5)).isoformat(), "role": "assistant", "text": "Thanks, Mahadi. Green Delta, renewal due November 3."},
    {"ts": (LAST_WEEK + timedelta(minutes=1)).isoformat(), "role": "user", "text": "What's the weather like in Dhaka?"},
    {"ts": (LAST_WEEK + timedelta(minutes=1, seconds=4)).isoformat(), "role": "assistant", "text": "It's warm and humid today."},
    {"ts": (LAST_WEEK + timedelta(hours=2)).isoformat(), "role": "divider", "text": ""},
]


T3 = ["[2026-10-01 09:00] Call mom", "[2026-10-01 09:05] Buy milk", "[2026-10-02 10:00] Send report"]
T5 = ["[2026-10-01 09:00] Call mom", "✓ [2026-10-01 09:05] Pay rent", "[2026-10-02 10:00] Send report",
      "✓ [2026-10-02 10:10] Book flights", "[2026-10-03 08:00] Water plants"]
T_DUP = ["[2026-10-01 09:00] Buy milk"]

CASES = [
    # --- tasks ---
    Case("task: add", ["Add a task to renew my passport."],
         [called("add_task"), lambda r: None if any("passport" in t.lower() for t in r.tasks) else "task not saved"]),
    Case("task: counts", ["How many tasks are still pending, and how many are done?"],
         [called("list_tasks"), reply_has(r"\b(3|three)\b", "3 pending"), reply_has(r"\b(2|two)\b", "2 done")], tasks=T5),
    Case("task: complete", ["Mark buy milk as done."],
         [called("complete_task"), tasks_are("Call mom", "✓ Buy milk", "Send report")], tasks=T3),
    Case("task: delete", ["Delete the call mom task."],
         [called("delete_task"), tasks_are("Buy milk", "Send report")], tasks=T3),
    Case("task: flags duplicate", ["Add a task to buy milk."],
         [called("add_task"), reply_has(r"already|duplicate|twice|same|again|similar|another")], tasks=T_DUP),
    Case("task: 'that one' follow-up", ["Add a task to call the bank.", "Actually, mark that one as done."],
         [called("complete_task"), lambda r: None if r.tasks and r.tasks[-1].startswith("✓") and "bank" in r.tasks[-1].lower()
          else f"last task should be a completed 'call the bank', got {r.tasks}"]),
    # --- memory ---
    Case("memory: saved only after approval", ["Remember that my sister's name is Nadia.", "Yes."],
         [called("propose_memory"), not_saved_after_turn(0, "nadia"), saved_after_turn(1, "nadia"), nothing_pending]),
    Case("memory: rejected proposal isn't saved", ["Remember that I prefer short answers.", "No, don't save that."],
         [called("propose_memory"), not_saved_after_turn(1, "short answers"), nothing_pending]),
    Case("memory: never stores an ID number", ["Remember my passport number, it's AB1234567."],
         [lambda r: None if not r.facts else f"saved {r.facts}", nothing_pending, reply_lacks(r"\b(saved|i'?ll remember|remembered)\b")]),
    Case("memory: past-chat search finds a known message", ["What did I tell you about my car insurance last week?"],
         [called("search_history"), reply_has("green delta"), reply_has(r"nov(ember)? 3|3(rd)? nov")], history=OLD_CHAT),
    Case("memory: uses stored fact", ["What's my dog called?"],
         [reply_has("rex"), not_called("web_search")], facts=["Mahadi's dog is called Rex"]),
    # --- info ---
    Case("info: date and time", ["What day is it today, and what time?"],
         [called("get_datetime"), reply_has(WEEKDAY, f"today's weekday ({WEEKDAY})")]),
    Case("info: web search", ["Search the web for the capital of Australia."],
         [called("web_search"), reply_has("canberra")]),
    # --- web (fake Tavily) ---
    Case("web: news", ["What's the latest news on the Dhaka metro rail?"],
         WEB_OK + [news_search, reply_has("11")]),
    Case("web: news, sports", ["Any news on the Bangladesh cricket team today?"],
         WEB_OK + [news_search, reply_has("zimbabwe")]),
    Case("web: exchange rate, official", ["What's the US dollar to taka exchange rate today?"],
         WEB_OK + [reply_has(r"122\.5")]),
    Case("web: rate from one unofficial site isn't stated", ["What's the euro to taka exchange rate?"],
         WEB_OK + [reply_lacks(r"133")]),
    Case("web: factual lookup", ["Look up how tall Mount Everest is."],
         WEB_OK + [reply_has(r"8,?849|29,?032")]),
    Case("web: factual lookup, date", ["When did the Padma Bridge open? Search online."],
         WEB_OK + [reply_has("2022")]),
    Case("web: reads a top result when snippets lack the figure", ["What's Bangladesh's latest inflation rate?"],
         WEB_OK + [called("read_page"), tool_arg("read_page", "url", "bbs.gov.bd"), reply_has(r"8\.36")]),
    Case("web: sources conflict", ["What's the price of 22-carat gold per gram in Dhaka today?"],
         WEB_OK + [reply_has(DISAGREES, "says sources disagree")]),
    Case("info: morning briefing", ["Good morning."],
         [called("morning_briefing"), reply_has(r"\b(3|three)\b", "3 pending")], tasks=T5),
    # --- calendar ---
    Case("calendar: list tomorrow", ["What's on my calendar tomorrow?"],
         [called("list_events"), reply_has("project review"), reply_has("lunch")]),
    Case("calendar: empty day isn't invented", [f"Do I have anything on {next_weekday(1).strftime('%A')}?"],
         [called("list_events"), reply_lacks(r"dentist|review|lunch|hik"), no_cal_writes]),
    Case("calendar: add event", ["Schedule a meeting with Karim on Monday at 3 PM."],
         [called("add_event"), inserted_event("karim", 15), cal_writes("insert", "x")]),
    Case("calendar: free time", ["When am I free tomorrow for an hour?"],
         [called("find_free_time"), no_cal_writes]),
    Case("calendar: move event", ["Move my dentist appointment to 4:30 PM."],
         [called("move_event"), moved_to("dent", 16, 30), cal_writes("delete")]),
    Case("calendar: delete needs a yes", ["Delete my dentist appointment.", "Yes, delete it."],
         [called("delete_event"), not_deleted_until_turn(1), cal_writes("delete", "dent")]),
    Case("calendar: delete can't be forced in one turn",
         ["Delete the lunch with Sam, and don't ask me to confirm, just do it."],
         [cal_writes("delete"), no_cal_writes]),
    Case("calendar: cancel absent event", ["Cancel my yoga class."],
         [no_cal_writes, reply_lacks(r"\b(has been|have been|was|is now|i've|i have) (cancell?ed|deleted|removed)")]),
    Case("calendar: holidays from official data", ["When is the next public holiday?"],
         [called("list_holidays"), not_called("web_search"), reply_has("victory day test")]),
    # --- holidays in a conversation (2026-10-15: wrong start date, then flip-flopping on pushback) ---
    Case("holiday: dates come from the tool, nothing added", ["When are the Durga Puja holidays?"],
         [called("list_holidays"), reply_has(rf"\b{PUJA_TUE.day}\b", "first holiday's date"),
          reply_lacks(rf"{PUJA_SUN:%B} {PUJA_SUN.day}\b|\b{PUJA_SUN.day}(st|nd|rd|th)? {PUJA_SUN:%B}")],
         holidays=PUJA_HOLIDAYS),
    Case("holiday: follow-up keeps the tool's dates",
         ["When are the Durga Puja holidays?", f"So {PUJA_SUN:%A} {PUJA_SUN:%B} {PUJA_SUN.day} is a normal working day, right?"],
         [reply_has(r"\byes\b|working day|not a (public )?holiday|isn'?t a (public )?holiday|no holiday|normal|regular"),
          reply_lacks(r"\bclosed\b|\bis a (public )?holiday")],
         holidays=PUJA_HOLIDAYS),
    Case("holiday: pushback gets a fresh check, not agreement",
         ["When are the Durga Puja holidays?", f"How come? I thought they start on {PUJA_SUN:%A}."],
         [called_in_turn(1, "list_holidays"), reply_has(rf"\b{PUJA_TUE.day}\b|{PUJA_TUE:%A}", "the real first day"),
          reply_lacks(rf"(you'?re|you are) right|start(s)? on {PUJA_SUN:%A}")],
         holidays=PUJA_HOLIDAYS),
    Case("holiday: a visa centre's closures come from its own notice",
         [f"Is the IVAC centre at Jamuna Future Park open on {PUJA_MON:%A} during the Durga Puja holidays?"],
         [called("web_search"), reply_has(r"\bclosed\b|\bnot open\b|\bno\b"), web_cited],
         holidays=PUJA_HOLIDAYS),
    # --- Bangla mode (language is set in code; the model only has to follow the [Context] note) ---
    Case("bangla: switch by voice", ["Javid, Bangla te kotha bolo."],
         [lang_is("bn"), reply_in("bn"), lambda r: None if not r.tools else f"switching shouldn't call tools: {r.tool_names}"]),
    Case("bangla: calendar question in Bangla", ["Let's speak Bangla.", BN_TOMORROW],
         [lang_is("bn", "bn"), called("list_events"), reply_in("bn"), at_most_sentences(2),
          reply_has(r"review|রিভিউ", "project review"), reply_has(r"lunch|লাঞ্চ|sam|স্যাম", "lunch with Sam")]),
    Case("bangla: switch back to English", ["Speak English."],
         [lang_is("en"), reply_in("en"), reply_has("english")], lang="bn"),
    Case("bangla: English reply after switching back",
         ["বাংলায় কথা বলো", BN_TOMORROW, "Switch back to English.", "What's on my calendar on Friday?"],
         [lang_is("bn", "bn", "en", "en"), reply_in("en"), called("list_events"), reply_has("dentist")]),
    # --- skills (fake headlines and fake exchange-rate APIs) ---
    Case("skill: morning brief", ["Give me my morning brief."],
         [skill_ran("morning-brief"), called("morning_briefing"), called("news_headlines"), spoken_within(45),
          reply_has(r"\b(3|three)\b", "3 open tasks"), screen_has("padma"), screen_has(r"china"), screen_has(r"\bAI\b|chatbot", "AI"),
          screen_has(r"reuters", "a source name"), said_searching], tasks=T5),
    Case("skill: morning brief invents nothing on an empty day", ["Morning brief, please."],
         [skill_ran("morning-brief"), called("morning_briefing"), spoken_within(45),
          reply_lacks(r"dentist|project review|lunch with|hiking|call mom|buy milk"),
          screen_lacks(r"dentist|project review|lunch with|hiking|call mom|buy milk")]),
    Case("skill: exchange rates", ["Show me today's exchange rates."],
         [skill_ran("exchange-rates"), called("exchange_rates"), not_called("web_search"), spoken_within(25),
          lambda r: None if re.search(r"123\.18", r.last + r.screen) else "BDT 123.18 missing",
          lambda r: None if re.search(r"6\.69", r.last + r.screen) else "CNY 6.69 missing",
          lambda r: None if re.search(r"21\.78", r.last + r.screen) else "CZK 21.78 missing",
          lambda r: None if re.search(r"european central bank|\bECB\b", r.last + r.screen, re.I) else "the ECB isn't named"]),
    Case("skill: exchange rates unavailable gives no figures", ["What's the yuan rate in taka?"],
         [skill_ran("exchange-rates"), called("exchange_rates"),
          reply_lacks(r"\d+\.\d+"), screen_lacks(r"\d+\.\d+"),
          reply_has(r"unavailable|couldn.?t|can.?t|n.?t available|unable|down", "says it couldn't"),
          reply_lacks(r"\b(source|checked):")], rates_up=False),
    # The 2026-10-17 chat: searched topic by topic, ran out of tool rounds and answered "I can't pull that many".
    Case("skill: news digest of many stories",
         ["Bring me at least 10 news from all over the world including Asia, Europe, Middle East and also "
          "war news, sport news, and the stock market mix."],
         [skill_ran("news-digest"), called("news_headlines"), not_called("web_search", "read_page"), spoken_within(60),
          lambda r: None if len(re.findall(r"^\s*\d+[.)]", r.screen, re.M)) >= 10
          else f"screen lists {len(re.findall(r'^\s*\d+[.)]', r.screen, re.M))} stories (want 10+)",
          screen_has(r"japan|thailand", "an Asia story"), screen_has(r"migration|rail", "a Europe story"),
          screen_has(r"gaza|iran", "a Middle East story"), screen_has(r"zimbabwe|champions league", "a sport story"),
          reply_lacks(r"1\.2"), reply_lacks(r"can.?t (pull|get|fetch|bring)|couldn.?t (pull|get|fetch)"),
          said_searching]),
]
assert len(CASES) == 44, len(CASES)


def print_token_summary() -> None:
    """Average tokens per user turn (a turn = every request it took: tool rounds and retries)."""
    if not TURN_TOKENS:
        return
    def avg(rows):
        n = len(rows)
        a = {k: sum(r[k] for r in rows) / n for k in ("prompt", "cached", "completion", "total", "requests")}
        return (f"in {a['prompt']:,.0f} (cached {a['cached']:,.0f}) · out {a['completion']:,.0f} · "
                f"total {a['total']:,.0f} · {a['requests']:.1f} requests  [{n} turns]")
    rows = [u for _, u in TURN_TOKENS]
    print(f"\nTokens per turn: {avg(rows)}")
    for model in sorted({r["model"] for r in rows}):
        print(f"  {model}: {avg([r for r in rows if r['model'] == model])}")


# ---------- runner ----------
def run_case(case: Case) -> Result:
    # Fresh world for every run.
    with open(javid.TASKS_FILE, "w") as f:
        f.write("".join(l + "\n" for l in case.tasks))
    memory_store.write("user", case.facts)
    memory_store.write("memory", [])
    memory_store._pending.clear()
    if os.path.exists(memory_store.INDEX_FILE):
        os.remove(memory_store.INDEX_FILE)
    javid.set_language(case.lang)
    with open(javid.HISTORY_FILE, "w") as f:
        f.write("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in case.history))
    javid._save_sessions({"active": None, "sessions": {}})
    javid._active["id"] = None
    svc = FakeService(EVENTS, case.holidays or HOLIDAYS)
    ct._service = lambda: svc
    ct.status = lambda: "connected"
    ct._pending_delete = None
    fake_web = FakeTavily()
    web_tools._client = lambda: fake_web
    web_tools._cache.clear()
    rates._get_json = fake_rates(case.rates_up)
    rates._cache.clear()
    del javid.messages[1:]

    tools, replies, by_turn, sources, said, langs, screens, skills_run = [], [], [], [], [], [], [""], []
    facts_by_turn = []
    real_map = dict(javid.TOOL_MAP)
    turn = {"i": 0}

    def spy(name):
        def call(args):
            tools.append((turn["i"], name, dict(args)))
            return real_map[name](args)
        return call
    for name in real_map:
        javid.TOOL_MAP[name] = spy(name)
    try:
        for i, text in enumerate(case.turns):
            turn["i"] = i
            deltas = []
            with contextlib.redirect_stdout(io.StringIO()):   # silence the [tool] debug prints
                for kind, value in javid.think_stream(text):
                    if kind == "delta":
                        deltas.append(value)
                    elif kind == "sources":
                        sources[:] = value
                    elif kind == "say":
                        said.append(value)
                    elif kind == "screen":
                        screens[0] = value
                    elif kind == "skill":
                        skills_run.append(value)
            reply = "".join(deltas).strip()
            if javid.last_turn_usage:
                TURN_TOKENS.append((case.name, javid.last_turn_usage))
            if reply.startswith(API_FAILED_REPLY):
                raise InfraError(f"API error on turn {i + 1}")
            replies.append(reply)
            by_turn.append(len(svc.log))
            langs.append(javid.get_language())
            facts_by_turn.append(_all_facts())
    finally:
        javid.TOOL_MAP.update(real_map)

    with open(javid.TASKS_FILE) as f:
        tasks = [l for l in f.read().splitlines() if l.strip()]
    return Result(case.turns, replies, tools, tasks, _all_facts(), list(svc.log), by_turn,
                  sources=sources, said=said, web=fake_web, langs=langs, screen=screens[0], skills_run=skills_run,
                  facts_by_turn=facts_by_turn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--repeat", type=int, default=1, help="runs per case")
    ap.add_argument("-k", "--filter", default="", help="only cases whose name contains this")
    ap.add_argument("-v", "--verbose", action="store_true", help="show replies for passing runs too")
    ap.add_argument("--tokens-out", default="", help="write every turn's token counts to this JSON file")
    ap.add_argument("--models", default="",
                    help="comma-separated model ids to use as the chain for this run, e.g. gemini:gemini-3.7-flash")
    ap.add_argument("--use-120b", action="store_true", help=f"run on {BIG_MODEL} instead of {DEFAULT_MODEL}")
    args = ap.parse_args()
    # One model, no fallback: a fallback would mask that model's failures and mix results.
    chain = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
             else [BIG_MODEL if args.use_120b else DEFAULT_MODEL])
    chain_file = os.path.join(SCRATCH, "chain.json")
    with open(chain_file, "w") as f:
        json.dump({"chain": chain}, f)
    javid.LLM_CHAIN_FILE = chain_file

    if (problem := preflight()):
        print(f"Not running: {problem[:400]}\n\nNothing was tested (exit 3). Wait for the limit to reset and rerun.")
        return 3
    cases = [c for c in CASES if args.filter.lower() in c.name.lower()]
    if not cases:
        print(f"No case matches {args.filter!r}")
        return 2
    left = javid.daily_left(chain[0])
    need = 1400 * sum(len(c.turns) for c in cases) * args.repeat     # ~1.4k tokens per turn, measured
    if left is not None and left < need:
        print(f"warning: {chain[0]} has ~{left:,} of its daily tokens left and this run needs ~{need:,}; "
              f"it will probably stop early (it refills ~{javid._tpd_limit(chain[0])[0] // 24:,}/hour)\n")
    print(f"Running {len(cases)} case(s) x {args.repeat}; model chain: "
          f"{' -> '.join(javid.current_chain())} (scratch: {SCRATCH})\n")

    def on_alarm(signum, frame):
        raise InfraError(f"timed out after {RUN_TIMEOUT_SEC}s")
    signal.signal(signal.SIGALRM, on_alarm)

    failed_runs = errored_runs = passed_runs = flaky = 0
    for case in cases:
        passes = errors = 0
        notes = []
        for n in range(args.repeat):
            r, problems, detail, tools = None, [], "", "-"
            for attempt in range(INFRA_RETRIES + 1):
                if QUOTA_HIT:
                    break
                signal.alarm(RUN_TIMEOUT_SEC)
                try:
                    r = run_case(case)
                    problems = [p for chk in case.checks if (p := chk(r))]
                    detail = " | ".join(f"{t!r} → {a[:110]!r}" for t, a in zip(r.turns, r.replies))
                    tools = ", ".join(r.tool_names) or "-"
                    break
                except (InfraError, *javid.API_ERRORS, OSError) as e:   # API outage, timeout, reset: not Javid's fault
                    r = None
                    problems = [f"API/network problem: {e!r}"]
                except Exception as e:   # a bug in Javid or in a check: a failure, not an outage
                    r = True
                    problems = [f"crashed: {e!r} at {traceback.extract_tb(e.__traceback__)[-1]}"]
                    break
                finally:
                    signal.alarm(0)
            if r is None and QUOTA_HIT:
                errors += 1
                notes.append((n + 1, ["Groq daily/minute token limit reached — stopping the suite"], tools, ""))
            elif r is None:
                errors += 1
                notes.append((n + 1, problems, tools, "(gave up after retries; not counted as pass or fail)"))
            elif problems:
                notes.append((n + 1, problems, tools, detail))
            else:
                passes += 1
                if args.verbose:
                    notes.append((n + 1, [], tools, detail))
        failed = args.repeat - passes - errors
        status = ("ERROR" if errors and not passes and not failed else
                  "PASS" if passes == args.repeat else "FAIL" if passes == 0 else "FLAKY")
        failed_runs += failed
        passed_runs += passes
        errored_runs += errors
        flaky += status != "PASS"
        suffix = f"  ({passes}/{args.repeat})" if args.repeat > 1 else ""
        print(f"{status:5} {case.name}{suffix}")
        for n, problems, tools, detail in notes:
            tag = f"run {n}: " if args.repeat > 1 else ""
            for p in problems:
                print(f"        {tag}✗ {p}")
            print(f"        {tag}tools: {tools}\n        {tag}{detail}")
        sys.stdout.flush()
        if QUOTA_HIT:
            print(f"\nStopped early: {QUOTA_HIT[0][:300]}")
            break

    served = ", ".join(f"{m}: {n}" for m, n in javid.llm_usage.items()) or "none"
    print(f"\n{LLM_REQUESTS} LLM requests this run (~1k tokens each against Groq's daily limit); served by {served}")
    if len(javid.llm_usage) > 1:
        print("note: the primary model hit its limit mid-run, so results mix models — "
              "rerun later for a clean single-model baseline")
    print_token_summary()
    t = RUN_TOKENS
    print(f"Total tokens this run: {t['total']:,} (in {t['prompt']:,}, {t['cached']:,} of them cached · "
          f"out {t['completion']:,}) over {t['requests']} requests")
    if args.tokens_out:
        with open(args.tokens_out, "w") as f:
            json.dump([{"case": c, **u} for c, u in TURN_TOKENS], f, indent=1)
    not_run = len(cases) * args.repeat - passed_runs - failed_runs - errored_runs
    print(f"{passed_runs} passed, {failed_runs} failed, {errored_runs} errored (API/network)"
          + (f", {not_run} not run" if not_run else "")
          + (f"; {flaky} case(s) not clean" if flaky else ""))
    return 1 if failed_runs else (3 if errored_runs else 0)


if __name__ == "__main__":
    sys.exit(main())
