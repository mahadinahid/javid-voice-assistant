"""
Regression check for Javid's agent behaviour: 89 cases, 75 of them prompts through the real LLM + tool loop.

    python regression.py              # run everything once
    python regression.py -n 3         # repeat each case 3x (the model is non-deterministic)
    python regression.py -k delete    # only cases whose name contains "delete"
    python regression.py -v           # print every reply, not just failures
    python regression.py --use-120b   # on gpt-oss-120b instead of the default gpt-oss-20b
    python regression.py --offline    # only the 14 scripted cases: no LLM request, no tokens, a few seconds

The scripted ("offline") cases fix the model's output to test what code decides whatever the model says: the
garble guard, questions about Javid itself (about.py), and the voice every spoken sentence gets. Each case's
line ends with its average time per turn and tokens: the web, weather and price services are fakes, so the
time is the model's (plus Javid's own work), not what a real search adds.

Runs on gpt-oss-20b alone by default (no fallback: a model change mid-run would mix results, and the live
chain's 120b is the user's backup, so it is only used with --use-120b).

Safe to run: tasks, memory, chat history and timers go to a temp dir, the Google Calendar is a
fake in-memory service, and Tavily, the weather services and the price APIs are fakes with fixed
results (no credits spent, no key needed). Your real data is never touched.
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
import markets  # noqa: E402
import memory_store  # noqa: E402
import rates  # noqa: E402
import skills  # noqa: E402
import timers  # noqa: E402
import weather  # noqa: E402
import web_tools  # noqa: E402
from groq import Groq  # noqa: E402

class _SuiteGroq(Groq):
    def _should_retry(self, response):
        # A 429 comes straight to _counted_stream_completion, which waits it out and leaves the wait out of the
        # turn's measured time; the SDK's own retry would sleep where the timer can't see it. 5xx still retry.
        return response.status_code != 429 and super()._should_retry(response)


# A stalled stream must fail the run, not hang the suite.
javid.client = _SuiteGroq(timeout=30.0, max_retries=2)
RUN_TIMEOUT_SEC = 150      # hard cap for one case run
INFRA_RETRIES = 2          # re-runs when the API itself failed (not the assistant)
API_FAILED_REPLY = tuple(javid.PHRASES["api_failed"].values())   # think_stream's canned reply on APIError


class InfraError(Exception):
    """The Groq API (network, timeout, 5xx) failed — says nothing about Javid's behaviour."""


# Count LLM requests, and turn a 429 into a hard stop (each request costs ~1k tokens of the daily cap).
LLM_REQUESTS = 0
QUOTA_HIT: list[str] = []
RATE_WAITS = [0.0]      # seconds slept riding out per-minute 429s: left out of a turn's measured time
_orig_stream_completion = javid._stream_completion


PER_MINUTE_WAIT_MAX = 90   # a 429 that clears within this is a per-minute limit: wait it out


def _counted_stream_completion(tool_calls, *args):
    """Count requests; ride out per-minute 429s (Gemini's free tier allows 5/min per model) and
    treat a longer wait (daily cap) as 'out of quota' so the suite stops instead of lying."""
    global LLM_REQUESTS
    if _script is not None:      # an offline case: the model's part is scripted, no request is made
        yield from _scripted_completion(tool_calls)
        return
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
            RATE_WAITS[0] += wait + 1


javid._stream_completion = _counted_stream_completion

# ---------- scripted model: offline cases test what code decides, whatever the model says (no tokens) ----------
_script: list | None = None     # the running offline case's model outputs, used in order
SCRIPTED_REQUESTS = 0


def call(tool, **args):
    """A scripted tool call (a plain string in a script is a scripted reply)."""
    return {"tool": tool, "args": args}


def _scripted_completion(tool_calls):
    global SCRIPTED_REQUESTS
    if not _script:
        raise AssertionError("the model was asked more often than the case's script allows")
    SCRIPTED_REQUESTS += 1
    step = _script.pop(0)
    if isinstance(step, dict):
        tool_calls[0] = {"id": f"call_s{SCRIPTED_REQUESTS}", "name": step["tool"], "arguments": json.dumps(step["args"])}
    else:
        yield step


# ---------- voice: speak each reply the way index.html does, recording the voice edge-tts was asked for ----------
TTS_CALLS: list[tuple[str, str]] = []     # (text, voice) for every edge-tts request


class FakeCommunicate:
    """edge_tts.Communicate without the network. Any option besides text and voice (rate, pitch) would
    change how the voice sounds, so it fails the case."""

    def __init__(self, text, voice, **options):
        if options:
            raise AssertionError(f"edge-tts asked for {options}: the locked voice is used as it is")
        TTS_CALLS.append((text, voice))

    async def stream(self):
        yield {"type": "audio", "data": b"ID3 fake mp3"}


@contextlib.contextmanager
def fake_edge_tts():
    real = javid.edge_tts.Communicate
    javid.edge_tts.Communicate = FakeCommunicate
    try:
        yield
    finally:
        javid.edge_tts.Communicate = real


_speech_client = None


def page_speech(said: list[str], reply: str) -> list[str]:
    """What index.html sends to /speak for one turn: each "say" event, then the reply sentence by sentence
    (web_tools.sentences is feedSpeech's rule), skipping chunks with no letters or digits (speakChunk)."""
    return [c for c in [*said, *web_tools.sentences(reply)] if re.search(r"[a-z0-9ঀ-৿]", c, re.I)]


def speak_like_page(chunks: list[str], page_lang: str | None) -> list[tuple[str, str]]:
    """POST each chunk to the server's /speak, as the page does; (chunk, voice used) for each."""
    global _speech_client
    if _speech_client is None:
        from fastapi.testclient import TestClient   # no `with`: the server's startup (warm-up, pings) doesn't run
        import server
        _speech_client = TestClient(server.app)
    out = []
    with fake_edge_tts():
        for chunk in chunks:
            javid._tts_cache.clear()     # every sentence reaches edge-tts, so every one's voice is recorded
            before = len(TTS_CALLS)
            res = _speech_client.post("/speak", json={"text": chunk, **({"lang": page_lang} if page_lang else {})})
            if res.status_code != 200 or len(TTS_CALLS) != before + 1:
                raise AssertionError(f"/speak gave {res.status_code} for {chunk!r} ({len(TTS_CALLS) - before} edge-tts calls)")
            out.append((chunk, TTS_CALLS[-1][1]))
    return out

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
timers.TIMERS_FILE = os.path.join(SCRATCH, "timers.json")          # never rings or lists the real timers
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
WEB_FIXTURES[:0] = [   # the research agent's searches about electric cars in Bangladesh
    (r"electric|\bevs?\b|charging|battery|vehicle", [
        hit("https://www.thedailystar.net/business/electric-vehicle-policy", "Govt drafts electric vehicle policy",
            "Bangladesh's draft electric vehicle policy would cut import duties on EVs and set up charging rules.", TODAY),
        hit("https://www.tbsnews.net/economy/ev-charging-stations", "Few charging stations hold back EV adoption",
            "Dhaka has only a few dozen public charging stations, so most EV owners charge at home.", YESTERDAY),
        hit("https://www.dhakatribune.com/business/ev-running-costs", "EVs cheaper to run, costly to buy",
            "Owners say electric cars cost far less to run than petrol cars, but the purchase price and battery "
            "replacement are high.", YESTERDAY),
        hit("https://en.prothomalo.com/bangladesh/ev-load-shedding", "Load-shedding worries EV owners",
            "Power cuts during summer load-shedding make home charging unreliable, EV owners say.", TODAY)]),
]
WEB_FIXTURES[:0] = [   # the market agent's crypto headlines
    (r"^(crypto|bitcoin|btc|ethereum|altcoin)\w*\b.*news$", [
        hit("https://www.reuters.com/markets/crypto-etf-inflows", "Bitcoin steadies as ETF inflows return",
            "Bitcoin steadied as inflows into spot bitcoin ETFs resumed after a week of outflows.", TODAY),
        hit("https://www.coindesk.com/markets/altcoins-solana-rally", "Solana leads altcoin gains",
            "Solana led gains among large altcoins as on-chain activity picked up.", YESTERDAY)]),
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


def fake_rates(up: bool, ecb_up: bool = True):
    def get(url):
        if not up or (not ecb_up and "frankfurter" in url):
            raise OSError("network down (fake)")
        return copy.deepcopy(FAKE_ECB if "frankfurter" in url else FAKE_AGG)
    return get


# Fake weather in Open-Meteo's shape: thunderstorms likely this afternoon (70%, peak 4 PM), rain tomorrow (80%).
def fake_open_meteo() -> dict:
    days = [TODAY + timedelta(days=i) for i in range(7)]
    hours = [datetime.combine(d, datetime.min.time()) + timedelta(hours=h) for d in days for h in range(24)]

    def chance(t):
        if t.date() == TODAY:
            return 70 if t.hour == 16 else 40 if 13 <= t.hour <= 19 else 10
        if t.date() == days[1]:
            return 80 if t.hour == 15 else 50 if 11 <= t.hour <= 18 else 20
        return 5
    return {
        "timezone": "Asia/Dhaka", "utc_offset_seconds": 21600,
        "current": {"time": datetime.now().strftime("%Y-%m-%dT%H:00"), "temperature_2m": 31.4,
                    "apparent_temperature": 36.2, "relative_humidity_2m": 70, "weather_code": 2, "wind_speed_10m": 12.3},
        "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
                   "temperature_2m": [26 + 7 * (1 - abs(t.hour - 14) / 14) for t in hours],
                   "precipitation_probability": [chance(t) for t in hours],
                   "precipitation": [1.2 if chance(t) >= 50 else 0.0 for t in hours],
                   "weather_code": [95 if chance(t) >= 70 else 61 if chance(t) >= 50 else 2 for t in hours]},
        "daily": {"time": [d.isoformat() for d in days],
                  "weather_code": [95, 61, 3, 2, 1, 1, 2], "temperature_2m_max": [33.1, 31.8, 31, 30, 31, 32, 33],
                  "temperature_2m_min": [26.2, 25.1, 25, 24, 25, 26, 26],
                  "precipitation_probability_max": [70, 80, 20, 10, 5, 5, 10],
                  "precipitation_sum": [6.2, 9.5, 0.4, 0, 0, 0, 0.1], "uv_index_max": [8.1, 6.0, 7, 7, 8, 8, 8],
                  "sunrise": [f"{d}T05:58" for d in days], "sunset": [f"{d}T17:32" for d in days]},
    }


FAKE_PLACES = {"sylhet": {"name": "Sylhet", "latitude": 24.8998, "longitude": 91.8714, "country": "Bangladesh",
                          "country_code": "BD", "admin1": "Sylhet Division", "timezone": "Asia/Dhaka"}}


def fake_weather_get(url, timeout=None):
    if "geocoding" in url:
        return {"results": [copy.deepcopy(v) for k, v in FAKE_PLACES.items() if k in url.lower()]}
    if "open-meteo" in url:
        return fake_open_meteo()
    raise OSError("MET Norway isn't faked")


# Fake live prices: Binance's ticker and Yahoo's chart shapes. An unknown symbol is a 400 (Binance) / 404 (Yahoo).
FAKE_TICKERS = {"BTCUSDT": ("82152.34", "1.363", "83158.00", "80684.36"), "ETHUSDT": ("2396.45", "0.771", "2446.27", "2329.00"),
                "BNBUSDT": ("612.30", "-0.420", "620.10", "605.80"), "SOLUSDT": ("104.24", "2.070", "106.22", "100.43")}


FAKE_STOCKS = {"AAPL": ("Apple Inc.", 342.94, 343.46, -0.151, 346.0, 340.12, "NasdaqGS"),
               "^GSPC": ("S&P 500", 7697.39, 7666.72, 0.4, 7702.1, 7651.8, "SNP"),
               "^IXIC": ("NASDAQ Composite", 26454.51, 26231.55, 0.85, 26470.0, 26150.2, "Nasdaq GIDS"),
               "^DJI": ("Dow Jones Industrial Average", 51970.08, 52001.28, -0.06, 52110.4, 51890.6, "DJI")}


def fake_market_get(url):
    import urllib.error
    import urllib.parse
    if "binance" in url:
        sym = url.rsplit("=", 1)[1]
        if sym not in FAKE_TICKERS:
            raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)
        last, pct, high, low = FAKE_TICKERS[sym]
        return {"symbol": sym, "lastPrice": last, "priceChangePercent": pct, "highPrice": high, "lowPrice": low,
                "quoteVolume": "1159623859.5", "closeTime": int(time.time() * 1000)}
    if "yahoo" in url and (sym := next((k for k in FAKE_STOCKS if f"/{urllib.parse.quote(k)}?" in url), None)):
        name, price, prev, pct, high, low, exchange = FAKE_STOCKS[sym]
        return {"chart": {"result": [{"meta": {"symbol": sym, "shortName": name, "currency": "USD",
                "regularMarketPrice": price, "chartPreviousClose": prev, "regularMarketChangePercent": pct,
                "regularMarketDayHigh": high, "regularMarketDayLow": low, "regularMarketTime": int(time.time()),
                "fullExchangeName": exchange, "exchangeTimezoneName": "America/New_York"}}]}}
    raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


WEB_PAGES = {BBS_URL: "Point-to-point inflation in September 2026 stood at 8.36 percent, down from 8.91 percent in August."}


class FakeTavily:
    def __init__(self, down: bool = False):
        self.searches, self.extracts, self.down = [], [], down

    def search(self, query, **kw):
        self.searches.append((query, kw))
        if self.down:
            raise ConnectionError("Tavily unreachable (fake)")
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
# A weekday with no fake events (it was always Tuesday, which on a Monday is tomorrow: lunch and the review).
EMPTY_DAY = next(d for d in map(next_weekday, (1, 2, 3, 0, 6)) if d not in (TOMORROW, FRIDAY, SATURDAY, HOLIDAY_DATE))
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
    agents_run: list = field(default_factory=list)  # "agent" events (an agent took the job)
    timers: list = field(default_factory=list)   # timers/reminders running after the run
    voices: list = field(default_factory=list)   # per turn: [(sentence, voice)] as the page had them spoken
    seconds: list = field(default_factory=list)  # per turn: time until the reply was complete, minus 429 waits
    tokens: int = 0                              # LLM tokens the run spent (all turns)
    model_requests: int = 0                      # model requests the run made, real or scripted
    tool_events: list = field(default_factory=list)  # [(turn_index, name)] from "tool" events: code answers too

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
    ecb_up: bool = True                          # the ECB's rates answer (else only the aggregator does)
    web_down: bool = False                       # every fake Tavily search fails
    events: list = None                          # the fake calendar's events (default EVENTS)
    script: list = None                          # offline: the model's outputs, in order (no LLM, no tokens)
    speak: bool = False                          # speak each reply through /speak as the page does (fake edge-tts)
    page_lang: str = None                        # the language a (stale) page sends to /speak with each sentence


def _ran(r, name, turn=None):
    """The tool ran: called by the model (spied, with its args) or by code that answers on its own (a "tool" event)."""
    return any((turn is None or t == turn) and n == name for t, n, _ in r.tools) or \
        any((turn is None or t == turn) and n == name for t, n in r.tool_events)


def called(name):
    return lambda r: None if _ran(r, name) else f"expected a {name} call, got {r.tool_names or [n for _, n in r.tool_events] or 'no tools'}"


def called_in_turn(i, name):
    return lambda r: None if _ran(r, name, i) else \
        f"expected a {name} call in turn {i + 1}, got {[n for t, n, _ in r.tools if t == i] or 'no tools'}"


def not_called(*names):
    return lambda r: next((f"must not call {n}" for n in names if _ran(r, n)), None)


def _plain(text):
    """gpt-oss writes narrow no-break spaces ("Victory\u202fDay", "3\u202fPM") and no-break hyphens
    ("fxquotes\u2011blog.com"): compare them as spaces and hyphens."""
    return re.sub(r"[\u00a0\u202f\u2009]", " ", text).replace("\u2011", "-")


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
    if not any(n.lower() in _plain(r.last).lower() for n in names):
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


def agent_ran(name):
    return lambda r: None if name in r.agents_run else f"expected the {name} agent, got {r.agents_run or 'none'}"


def timer_is(seconds):
    def check(r):
        items = [t for t in r.timers if t["kind"] == "timer"]
        if len(items) != 1:
            return f"expected one timer, got {[(t['kind'], t['label'], t['seconds']) for t in r.timers]}"
        return None if abs(items[0]["seconds"] - seconds) <= 2 else f"the timer is {items[0]['seconds']}s, not {seconds}s"
    return check


def reminder_at(day, hour, minute=0):
    def check(r):
        items = [t for t in r.timers if t["kind"] == "reminder"]
        if len(items) != 1:
            return f"expected one reminder, got {[(t['kind'], t['label']) for t in r.timers]}"
        due = datetime.fromtimestamp(items[0]["due"] / 1000)
        return None if (due.date(), due.hour, due.minute) == (day, hour, minute) else f"the reminder is at {due:%a %d %H:%M}"
    return check


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


# ---------- checks for the offline (scripted) cases, voices and the skill/agent cases ----------
def model_requests(n):
    return lambda r: None if r.model_requests == n else f"expected {n} model request(s), the run made {r.model_requests}"


def one_voice(voice):
    """Every sentence spoken, in every turn, was synthesized in this voice, and every turn said something."""
    def check(r):
        if len(r.voices) != len(r.turns) or not all(r.voices):
            return f"not every turn was spoken: sentences per turn {[len(v) for v in r.voices]}"
        wrong = [(c, v) for turn in r.voices for c, v in turn if v != voice]
        n = sum(map(len, r.voices))
        return f"{len(wrong)} of {n} sentences weren't {voice}: {wrong[:3]}" if wrong else None
    return check


def voices_by_turn(*voices):
    """Turn i was spoken entirely in voices[i]."""
    def check(r):
        got = [sorted({v for _, v in turn}) for turn in r.voices]
        return None if got == [[v] for v in voices] else f"voices per turn should be {list(voices)}, got {got}"
    return check


def cached_phrases_in_locked_voices(r):
    """Every fixed phrase is regenerated, sentence by sentence, in its language's locked voice, and the CLI's
    speak() goes through the same synthesize()."""
    bn = javid._BN_LETTERS.search
    with fake_edge_tts():
        TTS_CALLS.clear()
        javid.warm_phrases()
        warmed = dict(TTS_CALLS)
        cached = list(javid._tts_cache)
        real_run = javid.subprocess.run
        javid.subprocess.run = lambda *a, **kw: None      # afplay
        try:
            javid.speak("Goodbye, Mahadi. Talk soon.")
        finally:
            javid.subprocess.run = real_run
        cli_voice = TTS_CALLS[-1][1]
    javid._tts_cache.clear()
    want = {s: javid.VOICES[lang] for texts in javid.PHRASES.values() for lang, t in texts.items()
            for s in web_tools.sentences(t) if len(s) <= javid._TTS_CACHE_MAX_CHARS}
    if missing := [s for s in want if s not in warmed]:
        return f"not regenerated: {missing[:3]}"
    if wrong := [(s, warmed[s]) for s, v in want.items() if warmed[s] != v]:
        return f"phrases in the wrong voice: {wrong[:3]}"
    if stray := [(v, t) for v, t in cached if v != (javid.EDGE_VOICE_BN if bn(t) else javid.EDGE_VOICE)]:
        return f"cache holds clips in another voice: {stray[:3]}"
    want_cli = javid.VOICES[javid.get_language()]
    return None if cli_voice == want_cli else f"the CLI's speak() used {cli_voice}, not {want_cli}"


def figure_in(pattern, label):
    """The figure appears in the spoken or the screen part."""
    return lambda r: None if re.search(pattern, _plain(r.last + "\n" + r.screen)) else f"{label} is missing"


def mentions_only_if(name_re, also_re, label):
    """If the reply or screen names `name_re`, it must also say `also_re` (e.g. the ECB only as unavailable)."""
    def check(r):
        text = _plain(r.last + "\n" + r.screen)
        return label if re.search(name_re, text, re.I) and not re.search(also_re, text, re.I) else None
    return check


def numbered_lines_at_least(n):
    def check(r):
        found = len(re.findall(r"^\s*(?:\d+|[০-৯]+)[.)]", r.screen, re.M))
        return None if found >= n else f"screen lists {found} numbered stories (want {n}+)"
    return check


def tool_count_at_most(name, n):
    return lambda r: None if r.tool_names.count(name) <= n else f"{name} ran {r.tool_names.count(name)} times (max {n})"


ADVICE = (r"\byou should (?:buy|sell)\b|\b(?:good|great|right|bad) time to (?:buy|sell)\b|\bwill (?:rise|fall|go up|"
          r"go down|climb|drop|hit|reach)\b|\bprice target\b|\bi (?:recommend|suggest) (?:buying|selling)\b")
COULD_NOT = r"couldn.?t|could not|can.?t|unable|unavailable|not available|no (?:reliable )?(?:results|sources|news|information)|didn.?t find|down|failed"
TODAY_EVENTS = [timed("standup", "Team standup", TODAY, 10), timed("dinner", "Dinner with Rafi", TODAY, 20)]

# The reply that started the garble guard (chat_log, 2026-10-13), to "why does your voice change for tasks?".
GARBLED_2026_10_13 = ("I’m designed to keep the conversation friendly friendly… Actually…\xa0…\xa0…... ... …\xa0…\xa0…\xa0... "
                      "...\xa0...\xa0…\xa0...…\xa0…\n\nSorry…\xa0\xa0…\xa0…\n\n...\xa0…\n\n...")
# Ellipsis fragments restarting themselves, the shape reported from gpt-oss-20b (that reply wasn't logged).
FRAGMENT_LOOP = "Sure... I... I'm... I'm not... I'm not sure... I'm not sure what... I'm not sure what you mean..."
TRAILS_OFF = ("You have two pending tasks, Mahadi: reply to the email and the meeting next Friday, and nothing on your "
              "calendar today... and... ... ... ...")
NEWS_SCREEN = ("Headlines (6 stories):\n1. World - UN climate talks open in Nairobi (Reuters, today)\n"
               "2. World - Storms delay global shipping routes (AP, today)\n3. Asia - Japan's prime minister calls snap "
               "election (Al Jazeera, today)\n4. Asia - Late monsoon floods hit northern Thailand (BBC, yesterday)\n"
               "5. Europe - EU leaders back new migration deal (The Guardian, today)\n6. Europe - French rail workers "
               "announce strike (BBC, yesterday)")
PLAN_SPOKEN = ("SPOKEN: Tomorrow you have lunch with Sam at 1 PM and the project review at 2:30 PM, with time for your "
               "tasks in the morning. Shall I add the suggested blocks to your calendar?")
PLAN_SCREEN = (f"{TOMORROW:%A}, {TOMORROW:%B} {TOMORROW.day} · thunderstorms likely in the afternoon\n"
               "9:00 AM - 9:30 AM  Call mom (suggested)\n10:00 AM - 11:00 AM  Send report (suggested)\n"
               "1:00 PM - 2:30 PM  Lunch with Sam (calendar)\n2:30 PM - 3:30 PM  Project review (calendar)")
# Garbled but with every event and the weekday in it, so the planner's own must-show check doesn't catch it.
PLAN_GARBLED = (f"{TOMORROW:%A} ... ... ...\n1:00 PM - 2:30 PM  Lunch with Sam ... ... ...\n"
                "2:30 PM - 3:30 PM  Project review ... ... ...")

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
    Case("calendar: empty day isn't invented", [f"Do I have anything on {EMPTY_DAY:%A}?"],
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
    # The 2026-10-17 chat: searched topic by topic, ran out of tool rounds and answered "I can't pull that many".
    # Its "stock market mix" once sent it to the market agent instead (2026-10-19): news requests aren't market briefs.
    Case("skill: news digest of many stories",
         ["Bring me at least 10 news from all over the world including Asia, Europe, Middle East and also "
          "war news, sport news, and the stock market mix."],
         [skill_ran("news-digest"), called("news_headlines"), not_called("web_search", "read_page", "delegate"), spoken_within(60),
          lambda r: None if len(re.findall(r"^\s*\d+[.)]", r.screen, re.M)) >= 10
          else f"screen lists {len(re.findall(r'^\s*\d+[.)]', r.screen, re.M))} stories (want 10+)",
          screen_has(r"japan|thailand", "an Asia story"), screen_has(r"migration|rail", "a Europe story"),
          screen_has(r"gaza|iran", "a Middle East story"), screen_has(r"zimbabwe|champions league", "a sport story"),
          reply_lacks(r"1\.2"), reply_lacks(r"can.?t (pull|get|fetch|bring)|couldn.?t (pull|get|fetch)"),
          said_searching]),
    # ---------- tools added 2026-10-19 ----------
    # Javid made up "sunny, 28°C in India" in a greeting (2026-10-17) when it had no weather tool.
    Case("weather: today's forecast from the tool", ["What's the weather like today?"],
         [called("get_weather"), not_called("web_search"), reply_has(r"\b3[13]\b|thunder", "the tool's figures"),
          reply_lacks(r"\bsunny\b|\bsnow")]),
    Case("weather: rain tomorrow", ["Will it rain tomorrow?"],
         [called("get_weather"), tool_arg("get_weather", "day", r"tomorrow|" + TOMORROW.isoformat()), reply_has(r"\b80\b")]),
    Case("timer: set by duration", ["Set a timer for 10 minutes for the tea."],
         [called("set_timer"), timer_is(600), reply_has(r"\b10\b|\bten\b")]),
    Case("timer: reminder at a clock time", ["Remind me to call mom at 7pm tomorrow."],
         [called("set_timer"), not_called("add_task"), reminder_at(TOMORROW, 19), reply_has(r"\b7\b|seven")]),
    Case("calc: exact arithmetic", ["What's 17.5% of 2,340?"], [called("calculate"), reply_has(r"409\.5")]),
    Case("calc: unit conversion", ["How many miles is 42 km?"], [called("calculate"), reply_has(r"26\.1|26\.09")]),
    Case("market: crypto price from Binance", ["What's the bitcoin price right now?"],
         [called("market_price"), not_called("web_search"), reply_has(r"82,?152", "the Binance price"),
          reply_has(r"binance")]),
    Case("agent: research hands back a cited report", ["Research the pros and cons of electric cars in Bangladesh."],
         [called("delegate"), agent_ran("research"), called("web_search"), spoken_within(40),
          screen_has(r"\[\d\]", "a source number"), screen_has(r"charg|battery|import dut|load.?shedding|running cost",
                                                              "a finding from the results"), links_shown]),
    Case("agent: planner builds the day from real data", ["Plan my day tomorrow."],
         [called("delegate"), agent_ran("planner"), called("list_events"), called("list_tasks"), no_cal_writes,
          spoken_within(40), screen_has(r"project review", "tomorrow's review"), screen_has(r"lunch", "tomorrow's lunch"),
          screen_has(r"call mom|send report|water plants", "a pending task"),
          screen_lacks(r"pay rent|book flights")], tasks=T5),
    # The planner ends by asking whether to add its blocks: the "yes" must reach add_event (the agent's
    # follow-up tools) with the plan from the previous turn's screen part, not run the planner again.
    Case("agent: planner's blocks added after a yes", ["Plan my day tomorrow.", "Yes, add them."],
         [agent_ran("planner"), called_in_turn(1, "add_event"),
          lambda r: None if r.writes_in_turn(1) else "no calendar writes after the yes",
          lambda r: None if r.tool_names.count("delegate") == 1 else f"the planner ran {r.tool_names.count('delegate')} times"],
         tasks=T5),
    # ---------- audit 2026-10-20: three more cases per skill and agent ----------
    # morning-brief
    Case("skill: morning brief with today's events", ["Morning brief, please."],
         [skill_ran("morning-brief"), called("morning_briefing"), spoken_within(45), screen_has("standup"),
          screen_has(r"dinner|rafi", "the dinner"), reply_has(r"standup|dinner|rafi", "an event, spoken"),
          reply_lacks(r"dentist|project review|lunch with|hiking")], tasks=T3, events=TODAY_EVENTS),
    Case("skill: morning brief with the news down invents no headlines", ["Give me my morning brief."],
         [skill_ran("morning-brief"), called("morning_briefing"), reply_has(r"\b(3|three)\b", "3 pending"),
          reply_has(COULD_NOT, "says the news is unavailable"),
          lambda r: None if not re.search(r"padma|exports|chatbot|reuters|\bbbc\b", r.last + r.screen, re.I)
          else "a headline appeared with the news down"], tasks=T3, web_down=True),
    Case("skill: morning brief in Bangla mode", ["আমার morning brief দাও"],
         [skill_ran("morning-brief"), called("morning_briefing"), reply_in("bn"), spoken_within(45),
          reply_has(r"\b3\b|৩|তিন", "3 open tasks"), screen_has(r"padma|পদ্মা", "the Bangladesh headline")],
         tasks=T3, lang="bn"),
    # news-digest
    Case("skill: news digest of two named regions", ["Give me world news headlines, three from Asia and three from Europe."],
         [skill_ran("news-digest"), called("news_headlines"), not_called("web_search", "read_page", "delegate"),
          tool_arg("news_headlines", "topics", "asia"), tool_arg("news_headlines", "topics", "europe"),
          screen_has(r"japan|thailand", "an Asia story"), screen_has(r"migration|rail", "a Europe story"),
          screen_lacks(r"gaza|iran|zimbabwe"), spoken_within(60)]),
    Case("skill: news digest follow-up about one story",
         ["Give me 5 news stories from around the world.", "Tell me more about the Japan story."],
         [skill_ran("news-digest"), called_in_turn(0, "news_headlines"), reply_has("japan"),
          reply_has(r"snap election|election", "the election story")]),
    Case("skill: news digest in Bangla mode", ["বিশ্বের খবর দাও"],
         [skill_ran("news-digest"), called("news_headlines"), not_called("web_search"), reply_in("bn"),
          numbered_lines_at_least(4), spoken_within(60)], lang="bn"),
    # planner agent
    Case("agent: planner for a named day", ["Plan my day on Friday."],
         [agent_ran("planner"), called("list_events"), screen_has("dentist"), screen_has("friday"),
          screen_lacks(r"project review|lunch with sam|hiking"), no_cal_writes, spoken_within(40)], tasks=T3),
    Case("agent: planner, what to do first", ["What should I do first today?"],
         [agent_ran("planner"), called("list_tasks"), screen_has(r"call mom|send report|water plants", "a pending task"),
          screen_lacks(r"pay rent|book flights"), reply_has(r"call mom|send report|water plants", "a pending task, spoken"),
          no_cal_writes], tasks=T5),
    Case("agent: planner's blocks not added after a no",
         ["Plan my day tomorrow.", "No thanks, leave my calendar as it is."],
         [agent_ran("planner"), no_cal_writes, tool_count_at_most("delegate", 1),
          reply_lacks(r"\b(?:i'?ve|i have|have been|has been|are now|is now) (?:added|scheduled|put)")], tasks=T5),
    # research agent
    Case("agent: research deep dive cites its sources", ["Do a deep dive on EV charging in Dhaka."],
         [agent_ran("research"), called("web_search"), tool_count_at_most("web_search", 4), tool_count_at_most("read_page", 2),
          screen_has(r"\[\d\]", "a source number"), screen_has(r"charg"), reply_lacks(r"\[\d\]"), links_shown,
          spoken_within(40)]),
    Case("agent: research with no results invents nothing", ["Research the history of the Ahsan Manzil palace."],
         [agent_ran("research"), called("web_search"), reply_has(COULD_NOT, "says it found nothing"),
          screen_lacks(r"\[\d\]"), lambda r: None if not re.search(r"\b18[4-9]\d\b|nawab|abdul ghani|\bpink\b",
                                                                   r.last + r.screen, re.I) else "facts from memory, not sources"]),
    Case("agent: research on contested prices flags the disagreement", ["Research gold prices in Dhaka: the pros and cons of buying gold now."],
         [agent_ran("research"), called("web_search"), spoken_within(40),
          lambda r: None if not re.search(r"15,?200|14,?650", r.last + r.screen) or re.search(DISAGREES, r.last + r.screen, re.I)
          else "states one site's gold price as fact"]),
    # ---------- exchange rates and market briefs: answered in code (rates.py, markets.py), no model request ----------
    Case("rates: the board, in code", ["Show me today's exchange rates."],
         [called("exchange_rates"), not_called("web_search", "use_skill"), model_requests(0), spoken_within(25),
          figure_in(r"123\.18", "BDT 123.18"), figure_in(r"6\.69", "CNY 6.69"), figure_in(r"21\.78", "CZK 21.78"),
          figure_in(r"(?i)european central bank", "the ECB named"), figure_in(r"(?i)exchangerate-api", "ExchangeRate-API named")],
         script=[]),
    Case("rates: unavailable gives no figures", ["What's the yuan rate in taka?"],
         [called("exchange_rates"), reply_lacks(r"\d+\.\d+"), screen_lacks(r"\d+\.\d+"),
          reply_has(COULD_NOT, "says it couldn't"), model_requests(0)], rates_up=False, script=[]),
    # The skill once said "1 Czech koruna is 5.6556 taka, based on the European Central Bank's reference rate".
    Case("rates: koruna in taka credits both sources", ["How many taka is one Czech koruna today?"],
         [figure_in(r"5\.66", "1 CZK = 5.66 BDT"), reply_has("european central bank"), reply_has("exchangerate-api"),
          model_requests(0)], script=[]),
    Case("rates: an amount is converted in code", ["How much is 500 yuan in taka?"],
         [reply_has(r"9,201\.05"), screen_has(r"500 CNY = 9,201\.05 BDT"), model_requests(0)], script=[]),
    Case("rates: in Bangla mode, in Bangla", ["আজকের exchange rates দেখাও"],
         [reply_in("bn"), figure_in(r"123\.18", "BDT 123.18"), model_requests(0)], lang="bn", script=[]),
    Case("rates: with the ECB down, the aggregator's figures, credited to it", ["Show me the exchange rates."],
         [figure_in(r"6\.705", "CNY 6.705 from ExchangeRate-API"), reply_has("exchangerate-api"),
          lambda r: None if not re.search(r"6\.69|21\.78", r.last + r.screen) else "quotes ECB figures it never got",
          mentions_only_if(r"european central bank|\bECB\b", COULD_NOT, "names the ECB as a source while it was down"),
          model_requests(0)], ecb_up=False, script=[]),
    Case("rates: another currency goes to the model", ["What are the exchange rates for the euro and the yuan?"],
         [called("exchange_rates"), model_requests(2)],
         script=[call("exchange_rates"), "One yuan is 18.40 taka, Mahadi; the euro isn't on my rates board."]),
    Case("rates: a second request in the same message goes to the model", ["Show me the exchange rates and tomorrow's weather."],
         [called("exchange_rates"), called("get_weather"), model_requests(3)],
         script=[call("exchange_rates"), call("get_weather", day="tomorrow"),
                 "One US dollar is 123.18 taka, and tomorrow looks rainy, Mahadi."]),
    Case("market: crypto brief, in code", ["Give me a crypto market update."],
         [called("market_price"), called("news_headlines"), not_called("delegate"), model_requests(0), spoken_within(40),
          screen_has(r"82,?152", "BTC from Binance"), screen_has("not financial advice"), links_shown, said_searching],
         script=[]),
    Case("market: no buy/sell advice", ["Analyze bitcoin for me. Should I buy now?"],
         [screen_has(r"82,?152", "BTC from Binance"), reply_has("don't give buy or sell advice"), reply_lacks(ADVICE),
          screen_lacks(ADVICE), model_requests(0)], script=[]),
    Case("market: the coins named, and only those", ["Market update on my coins: BTC and SOL."],
         [screen_has(r"104\.24", "SOL"), screen_has(r"82,?152", "BTC"), screen_lacks(r"2,?396\.45|612\.30"),
          model_requests(0)], script=[]),
    Case("market: a stock from Yahoo Finance", ["Give me a stock market update on Apple."],
         [screen_has(r"342\.94", "AAPL"), reply_has("yahoo finance"), screen_has("not financial advice"), model_requests(0)],
         script=[]),
    Case("market: an unknown coin is said, not skipped", ["Market update on my coins: BTC and FOOCOIN."],
         [reply_has(r"couldn't find FOOCOIN"), screen_has(r"82,?152", "BTC"), model_requests(0)], script=[]),
    Case("market: a headline's one-site figure is left out", ["How's the stock market today?"],
         [screen_has(r"7,?697\.39", "the S&P 500"), screen_has("wall street", "the figure-free headline"),
          screen_lacks(r"1\.2%"), reply_lacks(r"1\.2%"), model_requests(0)], script=[]),
    Case("market: in Bangla mode, in Bangla", ["ক্রিপ্টো market update দাও"],
         [reply_in("bn"), screen_has(r"82,?152", "BTC"), model_requests(0)], lang="bn", script=[]),
    Case("market: a second request in the same message goes to the model",
         ["Give me a crypto market update and set a timer for 10 minutes."],
         [called("market_price"), called("set_timer"), model_requests(3)],
         script=[call("market_price", symbols=["BTC", "ETH", "BNB", "SOL"]), call("set_timer", duration="10 minutes"),
                 "Bitcoin is at $82,152.34 on Binance, Mahadi, and your 10-minute timer is running."]),
    # ---------- voice: every sentence Javid says, through /speak, in the locked voice ----------
    Case("voice: 5 chat replies, one voice",
         ["Hello Javid.", "How are you doing today?", "Tell me a fun fact about the moon.",
          "What's a good way to stay focused while working?", "Thanks, that's helpful."],
         [one_voice(javid.EDGE_VOICE)], speak=True),
    Case("voice: 5 task replies, one voice",
         ["Add a task to renew my passport.", "What's on my task list?", "Mark buy milk as done.",
          "Delete the call mom task.", "How many tasks are still pending?"],
         [one_voice(javid.EDGE_VOICE), called("add_task"), called("complete_task"), called("delete_task")],
         tasks=T3, speak=True),
    Case("voice: fixed phrases and the phrase cache use the locked voices", ["Let's speak Bangla.", "Speak English."],
         [voices_by_turn(javid.EDGE_VOICE_BN, javid.EDGE_VOICE), model_requests(0), cached_phrases_in_locked_voices],
         script=[], speak=True),
    Case("voice: Bangla mode is Pradeep only, English words too", ["কাল আমার কী আছে?"],
         [one_voice(javid.EDGE_VOICE_BN)], lang="bn", speak=True,
         script=["ঠিক আছে, মাহাদি। OK. Project review কাল দুপুর ২টায়।"]),
    Case("voice: a taka sign stays in Andrew's voice, whatever a stale tab sends", ["Hello Javid."],
         [one_voice(javid.EDGE_VOICE)], speak=True, page_lang="bn",
         script=["The dollar is ৳122.50 today, Mahadi. That's the Bangladesh Bank rate."]),
    # ---------- garble guard (scripted: the model's output is fixed, so these never depend on luck) ----------
    Case("garble: the 2026-10-13 ellipsis reply is retried", ["Tell me a fun fact."],
         [reply_has("three hearts"), model_requests(2)], script=[GARBLED_2026_10_13, "Octopuses have three hearts, Mahadi."]),
    Case("garble: a fragment loop is retried", ["Tell me a fun fact."],
         [reply_has("three hearts"), model_requests(2)], script=[FRAGMENT_LOOP, "Octopuses have three hearts, Mahadi."]),
    Case("garble: a reply trailing off into dots is retried", ["What's up?"],
         [reply_lacks(r"\.\.\.|…"), model_requests(2)], script=[TRAILS_OFF, "Not much, Mahadi. What can I do for you?"]),
    Case("garble: twice garbled gets the apology, not the dots", ["Tell me a fun fact."],
         [reply_has("lost my train of thought"), reply_lacks(r"\.\.\.|…"), model_requests(2)],
         script=[GARBLED_2026_10_13, FRAGMENT_LOOP]),
    Case("garble: a skill's garbled spoken part isn't diluted by its screen part", ["Give me world news headlines."],
         [skill_ran("news-digest"), reply_lacks(r"\.\.\.|…"), reply_has("snap election"), model_requests(4)],
         script=[call("use_skill", name="news-digest"), call("news_headlines", topics=["World", "Asia", "Europe"], per_topic=2),
                 "SPOKEN: I found... ... ... ... ...\nSCREEN:\n" + NEWS_SCREEN,
                 "SPOKEN: I found six stories, Mahadi. Reuters reports that UN climate talks opened in Nairobi, Al "
                 "Jazeera says Japan's prime minister called a snap election, and The Guardian says EU leaders backed "
                 "a migration deal. The full list is on your screen.\nSCREEN:\n" + NEWS_SCREEN]),
    # The screen part: Javid's own loop only ever saw an agent's spoken part, so a garbled screen went straight up.
    Case("garble: an agent's garbled screen part is asked for again", ["Plan my day tomorrow."],
         [agent_ran("planner"), screen_has("project review"), screen_lacks(r"\.\.\.|…"), no_cal_writes, model_requests(2)],
         tasks=T3, script=[PLAN_SPOKEN + "\nSCREEN:\n" + PLAN_GARBLED, PLAN_SPOKEN + "\nSCREEN:\n" + PLAN_SCREEN]),
    # ---------- questions about Javid itself: answered from about.py, never the model ----------
    Case("self: why the voice changes (the 2026-10-13 question)",
         ["I wanted to ask about one thing. When you are talking as a Javid, the voice is one and when you are "
          "talking about the task, the voice changes. Why is that?"],
         [reply_has("andrew"), reply_has("pradeep"), reply_has("no other voice"), model_requests(0)], script=[]),
    Case("self: weather sources (the 2026-10-15 question)",
         ["What type of sources are you using to find the weather? Because there are reliable and authentic "
          "thousands of websites."],
         [reply_has("open-meteo"), reply_lacks(r"reuters|bloomberg|trained"), model_requests(0)], script=[]),
    Case("self: model and design, with the full picture on screen", ["How do you work? Which model do you run on?"],
         [reply_has(r"gpt-oss-20b"), screen_has("tavily"), screen_has("edge-tts"), screen_has("open-meteo"),
          model_requests(0)], script=[]),
    Case("self: in Bangla mode, in Bangla", ["তুমি কীভাবে কাজ করো?"],
         [reply_in("bn"), model_requests(0)], lang="bn", script=[]),
    Case("self: ordinary questions still go to the model",
         ["How do you work out 17.5% of 2,340?", "So, what's your opinion on that?"],
         [called("calculate"), lambda r: None if "409.5" in r.replies[0] else "first reply lacks 409.5",
          reply_has("round number"), model_requests(3)],
         script=[call("calculate", expression="17.5% of 2,340"), "It's 409.5, Mahadi.",
                 "It's a tidy result, not quite a round number."]),
]
assert len(CASES) == 96, len(CASES)


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
    global _script
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
    svc = FakeService(EVENTS if case.events is None else case.events, case.holidays or HOLIDAYS)
    ct._service = lambda: svc
    ct.status = lambda: "connected"
    ct._pending_delete = None
    fake_web = FakeTavily(down=case.web_down)
    web_tools._client = lambda: fake_web
    web_tools._cache.clear()
    rates._get_json = fake_rates(case.rates_up, case.ecb_up)
    rates._cache.clear()
    weather._get_json = fake_weather_get
    weather._cache.clear()
    weather._places.clear()
    weather._skip_until["open-meteo"] = 0.0
    markets._get_json = fake_market_get
    markets._cache.clear()
    if os.path.exists(timers.TIMERS_FILE):
        os.remove(timers.TIMERS_FILE)
    del javid.messages[1:]

    tools, replies, by_turn, sources, said, langs, screens, skills_run, agents_run = [], [], [], [], [], [], [""], [], []
    facts_by_turn, voices, seconds, tokens, tool_events = [], [], [], 0, []
    _script = copy.deepcopy(case.script)
    requests_before = LLM_REQUESTS + SCRIPTED_REQUESTS
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
            deltas, turn_said = [], []
            started, waited = time.monotonic(), RATE_WAITS[0]
            with contextlib.redirect_stdout(io.StringIO()):   # silence the [tool] debug prints
                for kind, value in javid.think_stream(text):
                    if kind == "delta":
                        deltas.append(value)
                    elif kind == "sources":
                        sources[:] = value
                    elif kind == "say":
                        said.append(value)
                        turn_said.append(value)
                    elif kind == "tool":
                        tool_events.append((i, value))
                    elif kind == "screen":
                        screens[0] = value
                    elif kind == "skill":
                        skills_run.append(value)
                    elif kind == "agent":
                        agents_run.append(value)
            reply = "".join(deltas).strip()
            seconds.append(time.monotonic() - started - (RATE_WAITS[0] - waited))
            if javid.last_turn_usage:
                TURN_TOKENS.append((case.name, {**javid.last_turn_usage, "seconds": round(seconds[-1], 2)}))
                tokens += javid.last_turn_usage["total"]
            if reply.startswith(API_FAILED_REPLY):
                raise InfraError(f"API error on turn {i + 1}")
            replies.append(reply)
            if case.speak:
                voices.append(speak_like_page(page_speech(turn_said, reply), case.page_lang))
            by_turn.append(len(svc.log))
            langs.append(javid.get_language())
            facts_by_turn.append(_all_facts())
    finally:
        javid.TOOL_MAP.update(real_map)
        _script = None

    with open(javid.TASKS_FILE) as f:
        tasks = [l for l in f.read().splitlines() if l.strip()]
    return Result(case.turns, replies, tools, tasks, _all_facts(), list(svc.log), by_turn,
                  sources=sources, said=said, web=fake_web, langs=langs, screen=screens[0], skills_run=skills_run,
                  facts_by_turn=facts_by_turn, agents_run=agents_run, timers=timers.active_json(), voices=voices,
                  seconds=seconds, tokens=tokens, model_requests=LLM_REQUESTS + SCRIPTED_REQUESTS - requests_before,
                  tool_events=tool_events)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--repeat", type=int, default=1, help="runs per case")
    ap.add_argument("-k", "--filter", default="", help="only cases whose name contains this")
    ap.add_argument("-v", "--verbose", action="store_true", help="show replies for passing runs too")
    ap.add_argument("--tokens-out", default="", help="write every turn's token counts to this JSON file")
    ap.add_argument("--models", default="",
                    help="comma-separated model ids to use as the chain for this run, e.g. gemini:gemini-3.7-flash")
    ap.add_argument("--use-120b", action="store_true", help=f"run on {BIG_MODEL} instead of {DEFAULT_MODEL}")
    ap.add_argument("--offline", action="store_true", help="only the scripted cases (no LLM requests, no tokens)")
    args = ap.parse_args()
    # One model, no fallback: a fallback would mask that model's failures and mix results.
    chain = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
             else [BIG_MODEL if args.use_120b else DEFAULT_MODEL])
    chain_file = os.path.join(SCRATCH, "chain.json")
    with open(chain_file, "w") as f:
        json.dump({"chain": chain}, f)
    javid.LLM_CHAIN_FILE = chain_file

    cases = [c for c in CASES if args.filter.lower() in c.name.lower() and (c.script is not None or not args.offline)]
    if not cases:
        print(f"No case matches {args.filter!r}")
        return 2
    live = [c for c in cases if c.script is None]
    if live and (problem := preflight()):
        print(f"Not running: {problem[:400]}\n\nNothing was tested (exit 3). Wait for the limit to reset and rerun.")
        return 3
    left = javid.daily_left(chain[0])
    need = 1400 * sum(len(c.turns) for c in live) * args.repeat     # ~1.4k tokens per turn, measured
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
        notes, timing = [], []      # timing: (seconds per turn, tokens) of each completed run
        for n in range(args.repeat):
            r, problems, detail, tools = None, [], "", "-"
            for attempt in range(INFRA_RETRIES + 1):
                if QUOTA_HIT:
                    break
                signal.alarm(RUN_TIMEOUT_SEC)
                try:
                    r = run_case(case)
                    timing.append((sum(r.seconds) / max(len(r.seconds), 1), r.tokens))
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
        if timing:
            suffix += (f"  [{sum(t for t, _ in timing) / len(timing):.1f} s/turn, "
                       f"{sum(n for _, n in timing) / len(timing):,.0f} tokens{', offline' if case.script is not None else ''}]")
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
