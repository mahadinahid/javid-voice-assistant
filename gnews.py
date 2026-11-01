"""
Google News, read through its public RSS feeds: the headlines Javid reads out (news.py), the first source of
the news_headlines tool and of market briefs (web_tools.headline_items), news watches (watches.py), and
web_search's fallback when Tavily is down. No key and no credits, about 0.5-1 s per feed.

Why: news is what Mahadi asks for most, and through Tavily it often went wrong (2026-10-20/22): "World News
Tonight tops viewership" from detpress.com read out as world news, "Good Morning America" from a local TV
site, "I couldn't reach the news sites right now" during a Tavily ConnectionError, and "why not just go to
Google?" three times in one evening. Google's own feeds have editors' ranking for the world, business,
technology and sport sections, and its search takes site: filters, so each topic here searches a fixed list
of established outlets. Bangla mode uses the Bangladesh edition in Bengali, minus West Bengal's papers.

Every item: {"title", "source", "domain", "link", "published" (aware datetime or None), "cluster" (other outlets'
headlines for the same story, from the feed), "topic"}. A link is Google News' redirect to the article.
"""
import html
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import certifi

BASE = "https://news.google.com/rss"
EDITIONS = {"en": "hl=en-US&gl=US&ceid=US:en", "bn": "hl=bn&gl=BD&ceid=BD:bn"}
TIMEOUT_SEC = 6
CACHE_SEC = 10 * 60         # a repeat of the same feed within this reuses it (watches ask for fresh=True)
MAX_AGE_HOURS = 48          # older items are dropped (section pages carry evergreen links with old dates)
SEARCH_DAYS = 2
PER_OUTLET = 2              # at most this many items from one outlet in a topic (more when many are asked for)

# Established outlets per kind of story: a topic's search only returns these sites.
WORLD_SITES = ["reuters.com", "apnews.com", "bbc.com", "aljazeera.com", "theguardian.com", "nytimes.com",
               "washingtonpost.com", "cnn.com", "npr.org", "dw.com", "france24.com", "ft.com", "bloomberg.com"]
ASIA_SITES = ["reuters.com", "apnews.com", "bbc.com", "aljazeera.com", "scmp.com", "asia.nikkei.com", "thehindu.com",
              "straitstimes.com", "channelnewsasia.com", "japantimes.co.jp", "thediplomat.com", "nytimes.com"]
BD_SITES = ["thedailystar.net", "tbsnews.net", "dhakatribune.com", "bdnews24.com", "newagebd.net", "en.prothomalo.com",
            "thefinancialexpress.com.bd", "unb.com.bd", "bssnews.net", "reuters.com", "aljazeera.com", "bbc.com", "dw.com"]
TECH_SITES = ["techcrunch.com", "theverge.com", "wired.com", "arstechnica.com", "technologyreview.com", "venturebeat.com",
              "reuters.com", "bloomberg.com", "cnbc.com", "engadget.com", "axios.com", "ft.com"]
BIZ_SITES = ["reuters.com", "bloomberg.com", "cnbc.com", "wsj.com", "ft.com", "marketwatch.com", "apnews.com",
             "economist.com", "barrons.com", "axios.com"]
CRYPTO_SITES = ["coindesk.com", "theblock.co", "decrypt.co", "cointelegraph.com", "blockworks.co", "reuters.com",
                "bloomberg.com", "cnbc.com", "forbes.com"]
SPORT_SITES = ["espncricinfo.com", "cricbuzz.com", "espn.com", "bbc.com", "skysports.com", "theguardian.com",
               "reuters.com", "apnews.com", "icc-cricket.com", "thedailystar.net"]
CRICKET_SITES = ["espncricinfo.com", "cricbuzz.com", "icc-cricket.com", "bbc.com/sport/cricket", "skysports.com/cricket",
                 "theguardian.com/sport/cricket", "espn.com/cricket", "reuters.com/sports/cricket",
                 "thedailystar.net/sports/cricket", "en.prothomalo.com/sports/cricket"]

# key: (English name, Bangla name, English feed, Bangla feed). A feed is ("top",), ("section", NAME) or
# ("search", query, sites); a Bangla feed of None uses the English one.
TOPICS = {
    "top": ("Top stories", "প্রধান খবর", ("top",), ("top",)),
    "world": ("World", "বিশ্ব", ("section", "WORLD"), ("section", "WORLD")),
    "bangladesh": ("Bangladesh", "বাংলাদেশ", ("search", "Bangladesh", BD_SITES), ("section", "NATION")),
    "asia": ("Asia", "এশিয়া", ("search", "(Asia OR China OR Japan OR India OR Korea OR Pakistan OR Indonesia OR "
                                        "Philippines OR Vietnam OR Thailand OR Myanmar OR Taiwan)", ASIA_SITES),
             ("search", "এশিয়া OR চীন OR জাপান OR ভারত OR কোরিয়া OR পাকিস্তান OR মিয়ানমার", None)),
    "europe": ("Europe", "ইউরোপ", ("search", "(Europe OR EU OR Ukraine OR Russia OR France OR Germany OR Britain OR "
                                           "Italy OR Spain OR Poland)", WORLD_SITES),
               ("search", "ইউরোপ OR ইউক্রেন OR রাশিয়া OR ফ্রান্স OR জার্মানি OR যুক্তরাজ্য", None)),
    "middle east": ("Middle East", "মধ্যপ্রাচ্য", ("search", '("Middle East" OR Israel OR Gaza OR Iran OR Lebanon OR '
                                                            'Syria OR Yemen OR Saudi OR Iraq OR Gulf)', WORLD_SITES),
                    ("search", "মধ্যপ্রাচ্য OR ইসরায়েল OR গাজা OR ইরান OR সৌদি", None)),
    "africa": ("Africa", "আফ্রিকা", ("search", '(Africa OR Nigeria OR Kenya OR Sudan OR Ethiopia OR Congo OR Egypt OR '
                                             '"South Africa")', WORLD_SITES), ("search", "আফ্রিকা", None)),
    "americas": ("The Americas", "আমেরিকা মহাদেশ", ("search", '("Latin America" OR Brazil OR Mexico OR Argentina OR '
                                                              'Venezuela OR Colombia OR Canada OR Chile)', WORLD_SITES),
                 ("search", "লাতিন আমেরিকা OR ব্রাজিল OR মেক্সিকো OR কানাডা", None)),
    "usa": ("The US", "যুক্তরাষ্ট্র", ("search", '("United States" OR Trump OR Congress OR "White House")', WORLD_SITES),
            ("search", "যুক্তরাষ্ট্র OR ট্রাম্প", None)),
    "war": ("War and conflict", "যুদ্ধ", ("search", "(war OR ceasefire OR airstrike OR missile OR troops OR offensive)",
                                          WORLD_SITES), ("search", "যুদ্ধ OR যুদ্ধবিরতি OR হামলা", None)),
    "ai": ("AI", "এআই", ("search", '(AI OR "artificial intelligence" OR OpenAI OR Anthropic OR ChatGPT OR Gemini OR '
                                   'Nvidia OR chatbot)', TECH_SITES),
           ("search", "এআই OR কৃত্রিম বুদ্ধিমত্তা OR চ্যাটজিপিটি", None)),
    "tech": ("Technology", "প্রযুক্তি", ("section", "TECHNOLOGY"), ("section", "TECHNOLOGY")),
    "business": ("Business", "ব্যবসা", ("section", "BUSINESS"), ("section", "BUSINESS")),
    "markets": ("Markets", "শেয়ারবাজার", ("search", '("stock market" OR stocks OR "Wall Street" OR "S&P 500" OR Nasdaq '
                                                   'OR Dow OR shares)', BIZ_SITES),
                ("search", "শেয়ারবাজার OR পুঁজিবাজার", None)),
    "crypto": ("Crypto", "ক্রিপ্টো", ("search", "(bitcoin OR crypto OR cryptocurrency OR ethereum OR Binance OR "
                                              "stablecoin)", CRYPTO_SITES), ("search", "ক্রিপ্টো OR বিটকয়েন", None)),
    "sport": ("Sport", "খেলা", ("search", '(cricket OR "Premier League" OR "Champions League" OR "La Liga" OR FIFA OR '
                                         'soccer OR T20 OR ODI)', SPORT_SITES), ("section", "SPORTS")),
    "cricket": ("Cricket", "ক্রিকেট", ("search", "cricket", CRICKET_SITES), ("search", "ক্রিকেট", None)),
    "football": ("Football", "ফুটবল", ("search", '(soccer OR "Premier League" OR "Champions League" OR "La Liga" OR '
                                               '"Serie A" OR Bundesliga OR FIFA OR UEFA)', SPORT_SITES),
                 ("search", "ফুটবল", None)),
    "science": ("Science", "বিজ্ঞান", ("section", "SCIENCE"), ("section", "SCIENCE")),
    "health": ("Health", "স্বাস্থ্য", ("section", "HEALTH"), ("section", "HEALTH")),
    "entertainment": ("Entertainment", "বিনোদন", ("section", "ENTERTAINMENT"), ("section", "ENTERTAINMENT")),
    "climate": ("Climate", "জলবায়ু", ("search", '(climate OR "climate change" OR emissions OR heatwave OR COP31)',
                                       WORLD_SITES), ("search", "জলবায়ু", None)),
}

# Countries asked about as a whole ("news from Japan"): a search of the world outlets for the name.
COUNTRIES = {
    "china": ("China", "চীন"), "india": ("India", "ভারত"), "pakistan": ("Pakistan", "পাকিস্তান"),
    "japan": ("Japan", "জাপান"), "korea": ("Korea", "কোরিয়া"), "myanmar": ("Myanmar", "মিয়ানমার"),
    "nepal": ("Nepal", "নেপাল"), "sri lanka": ("Sri Lanka", "শ্রীলঙ্কা"), "afghanistan": ("Afghanistan", "আফগানিস্তান"),
    "russia": ("Russia", "রাশিয়া"), "ukraine": ("Ukraine", "ইউক্রেন"), "israel": ("Israel", "ইসরায়েল"),
    "iran": ("Iran", "ইরান"), "gaza": ("Gaza", "গাজা"), "palestine": ("Palestine", "ফিলিস্তিন"),
    "saudi arabia": ("Saudi Arabia", "সৌদি আরব"), "turkey": ("Turkey", "তুরস্ক"), "uk": ("Britain", "যুক্তরাজ্য"),
    "france": ("France", "ফ্রান্স"), "germany": ("Germany", "জার্মানি"), "canada": ("Canada", "কানাডা"),
    "australia": ("Australia", "অস্ট্রেলিয়া"), "malaysia": ("Malaysia", "মালয়েশিয়া"),
    "singapore": ("Singapore", "সিঙ্গাপুর"), "indonesia": ("Indonesia", "ইন্দোনেশিয়া"), "brazil": ("Brazil", "ব্রাজিল"),
}
for _key, (_en, _bn) in COUNTRIES.items():
    TOPICS.setdefault(_key, (_en, _bn, ("search", f'"{_en}"', WORLD_SITES), ("search", _bn, None)))

# The words for each topic in Mahadi's messages and in the model's news_headlines topics, lowercase, most
# specific first. Bengali has no \b: Python's word boundary breaks at Bengali vowel signs.
_ADJECTIVES = {"china": "chinese|beijing", "india": "indian|delhi", "pakistan": "pakistani", "japan": "japanese|tokyo",
               "korea": "korean|(?:south |north )korea", "russia": "russian|moscow|kremlin", "ukraine": "ukrainian|kyiv",
               "israel": "israeli", "iran": "iranian|tehran", "palestine": "palestinian", "saudi arabia": "saudi",
               "turkey": "turkish|türkiye|turkiye", "uk": "britain|british|england|london", "france": "french|paris",
               "germany": "german|berlin", "canada": "canadian", "australia": "australian", "malaysia": "malaysian",
               "indonesia": "indonesian", "brazil": "brazilian", "afghanistan": "afghan|taliban", "nepal": "nepali|nepalese",
               "myanmar": "burma|rohingya", "sri lanka": "sri lankan"}
ALIASES = [
    ("middle east", r"\bmiddle[- ]east\w*|\bmid-?east\b|\bmideast\b|মধ্যপ্রাচ্য"),
    ("americas", r"\blatin america\w*|\bsouth america\w*|\bthe americas\b|লাতিন আমেরিকা"),
    ("usa", r"\bunited states\b|\bamerica\b|\bamerican\b|\busa\b|\bwhite house\b|যুক্তরাষ্ট্র|মার্কিন"),
    ("bangladesh", r"\bbangladesh\w*|\bdhaka\b|\blocal\b|\bnational\b|\bdomestic\b|বাংলাদেশ|দেশের|ঢাকা"),
    ("world", r"\bworld\b|\bglobal\b|\binternational\b|\bforeign\b|বিশ্ব|আন্তর্জাতিক"),
    ("asia", r"\basia\w*|\basian\b|এশিয়া"),
    ("europe", r"\beurope\w*|\beu\b|ইউরোপ"),
    ("africa", r"\bafrica\w*|আফ্রিকা"),
    ("war", r"\bwars?\b|\bconflicts?\b|\bmilitary\b|যুদ্ধ"),
    ("ai", r"\bai\b|\ba\.i\.|\bartificial intelligence\b|\bchat ?gpt\b|\bopenai\b|\bllms?\b|এআই|কৃত্রিম বুদ্ধিমত্তা"),
    ("crypto", r"\bcrypto\w*|\bbitcoin\b|\bbinance\b|ক্রিপ্টো|বিটকয়েন"),
    ("markets", r"\bstock ?markets?\b|\bshare ?markets?\b|\bstocks?\b|\bshares\b|\bwall street\b|\bmarkets?\b|"
                r"শেয়ার ?বাজার|পুঁজিবাজার"),
    ("tech", r"\btech\b|\btechnolog\w*|প্রযুক্তি"),
    ("business", r"\bbusiness\w*|\beconom\w*|\bfinanc\w*|ব্যবসা|অর্থনীতি"),
    ("cricket", r"\bcricket\w*|ক্রিকেট"),
    ("football", r"\bfootball\b|\bsoccer\b|\bpremier league\b|\bchampions league\b|ফুটবল"),
    ("sport", r"\bsports?\b|\bsporting\b|খেলা"),
    ("science", r"\bscien\w*|বিজ্ঞান"),
    ("health", r"\bhealth\w*|স্বাস্থ্য"),
    ("entertainment", r"\bentertainment\b|\bcelebrit\w*|\bmovies?\b|\bfilms?\b|বিনোদন"),
    ("climate", r"\bclimate\b|জলবায়ু"),
] + [(k, rf"\b{k}\b" + (f"|\\b(?:{_ADJECTIVES[k]})\\b" if k in _ADJECTIVES else "") + f"|{bn}")
     for k, (_, bn) in COUNTRIES.items()]
# A Bengali word is matched whole, its case ending too ("বিশ্বের" is বিশ্ব + "-এর").
ALIASES = [(k, re.compile(re.sub(r"([ঀ-৿][ঀ-৿ ]*[ঀ-৿])", r"\1[\\u0980-\\u09ff]*", p), re.I))
           for k, p in ALIASES]


def key_for(topic: str) -> str | None:
    """The TOPICS key a topic phrase names ("Middle East", "sport news", "বিশ্বের খবর"), or None."""
    t = " ".join((topic or "").lower().split())
    if t in TOPICS:
        return t
    rest = re.sub(r"\b(?:news|headlines?|stories|latest|top|the|market)\b|খবর", " ", t).strip()
    for key, pattern in ALIASES:
        m = pattern.search(rest)
        if m and not rest[:m.start()].strip(" -,") and not rest[m.end():].strip(" -,"):
            return key
    return None

# West Bengal's Bengali papers share the Bangladesh edition: Bangla mode keeps Bangladesh's own outlets.
_INDIAN_BENGALI = ("abplive.com", "anandabazar.com", "eisamay.com", "news18.com", "bartamanpatrika.com",
                   "zeenews.india.com", "tv9bangla.com", "hindustantimes.com", "indianexpress.com", "oneindia.com",
                   "kolkata24x7.com", "thewall.in", "sangbadpratidin.in", "aajkaal.in", "uttarbangasambad.com")

# Not news stories: section and video pages, podcasts, opinion, listicles and market clickbait.
_JUNK_PREFIX = re.compile(r"^(?:watch|video|videos|live|live updates|podcast|quiz|photos?|gallery|in pictures|"
                          r"opinion|editorial|letters?|listen|explainer|analysis|review|sponsored)\s*[:|–—-]", re.I)
_LABEL = re.compile(r"^([^|:]{2,40})\s\|\s")       # "Asian Angle | ...", "Fixtures, Schedule | ..."
_JUNK = re.compile(r"\blatest news\b|\bnews (?:&|and) updates\b|\bbreaking news,|\bfantasy\b|\bcheat sheet\b|"
                   r"\bbest bets\b|\bbetting\b|\bpicks? (?:and|&) predictions?\b|\bhoroscope|\bcrossword|\bwordle\b|"
                   r"\bshould you buy\b|\b(?:stocks?|cryptos?|coins?) to (?:buy|watch|consider)\b|\brock-solid\b|"
                   r"\bmillionaire\b|\bif i could tell\b|\bmost upside\b|\btop \d+ (?:stocks?|cryptos?|coins?)\b|"
                   r"\bdeals? of the day\b|\bhighlights \||^\w+ videos\b|\bmatch highlights\b|"
                   r"\blatest (?:\w+ )?updates\b|\bmatches, stats\b|^\d+(?:st|nd|rd|th) (?:match|test|odi|t20i?)\b|"
                   r"\bquizzes\b|\bpoints table$|\bauction tracker\b|\bstats (?:&|and) head-to-head\b|"
                   r"টিভিতে আজকের খেলা|আজকের রাশিফল|খেলার সূচি", re.I)
_STRIP_PREFIX = re.compile(r"^(?:exclusive|breaking|update|updated)\s*:\s*", re.I)

_ctx = ssl.create_default_context(cafile=certifi.where())
_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()
_state = {"last_ok": None, "last_error": None, "fetches": 0}


def _get(url: str) -> bytes:
    """The feed's bytes (regression.py swaps this for a fake)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Javid personal assistant)"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=_ctx) as r:
        return r.read()


def _domain(url: str) -> str:
    return (urllib.parse.urlparse(url or "").hostname or "").removeprefix("www.")


def _matches(domain: str, known: str) -> bool:
    return domain == known or domain.endswith("." + known)


def feed_url(feed: tuple, lang: str = "en", days: int = SEARCH_DAYS) -> str:
    edition = EDITIONS["bn" if lang == "bn" else "en"]
    if feed[0] == "top":
        return f"{BASE}?{edition}"
    if feed[0] == "section":
        return f"{BASE}/headlines/section/topic/{feed[1]}?{edition}"
    query, sites = feed[1], feed[2] if len(feed) > 2 else None
    q = f"{query} when:{max(1, int(days))}d"
    if sites:
        q += " (" + " OR ".join(f"site:{s}" for s in sites) + ")"
    return f"{BASE}/search?q={urllib.parse.quote(q)}&{edition}"


def _clean_title(title: str, source: str) -> str:
    title = " ".join(html.unescape(title or "").split())
    for suffix in (f" - {source}", f" – {source}", f" | {source}"):
        if source and title.endswith(suffix):
            title = title[: -len(suffix)]
    if " | " in title and len(title.rsplit(" | ", 1)[1].split()) <= 4:
        title = title.rsplit(" | ", 1)[0]       # "...great purge | Barney Ronay": the author or section
    return title.strip()


def _cluster(description: str) -> list[dict]:
    """Other outlets' headlines for the same story, from the item's description list."""
    out = []
    for m in re.finditer(r"<li>\s*<a [^>]*>(.*?)</a>(?:&nbsp;|\s)*<font[^>]*>(.*?)</font>", description or "", re.S):
        out.append({"title": " ".join(html.unescape(re.sub(r"<[^>]+>", "", m[1])).split()),
                    "source": " ".join(html.unescape(m[2]).split())})
    return out


def _published(raw: str | None) -> datetime | None:
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    items = []
    for it in root.findall("./channel/item"):
        src = it.find("source")
        source = " ".join((src.text or "").split()) if src is not None else ""
        title = _clean_title(it.findtext("title") or "", source)
        if not title:
            continue
        if re.fullmatch(r"[A-Za-z][\w-]*\.(?:com|net|org)", source):
            source = source.rsplit(".", 1)[0]       # "Bloomberg.com", "DW.com", "bdnews24.com"
        items.append({"title": title, "source": source,
                      "domain": _domain(src.get("url", "") if src is not None else ""),
                      "link": (it.findtext("link") or "").strip(), "published": _published(it.findtext("pubDate")),
                      "cluster": _cluster(html.unescape(it.findtext("description") or ""))})
    return items


def is_junk(item: dict, now: datetime | None = None) -> bool:
    title = item["title"]
    if len(title.split()) < 4 or _JUNK_PREFIX.match(title) or _JUNK.search(title):
        return True
    if (label := _LABEL.match(title)) and len(label[1].split()) <= 4:
        return True
    now = now or datetime.now(timezone.utc)
    return bool(item["published"] and now - item["published"] > timedelta(hours=MAX_AGE_HOURS))


def fetch(url: str, fresh: bool = False) -> list[dict]:
    """A feed's items, cached for CACHE_SEC. Raises on a network or parse error."""
    with _lock:
        hit = _cache.get(url)
    if hit and not fresh and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    try:
        items = parse(_get(url))
    except Exception as e:
        _state["last_error"] = {"kind": type(e).__name__, "message": str(e)[:160], "at": time.time()}
        raise
    _state["last_ok"] = time.time()
    _state["fetches"] += 1
    with _lock:
        _cache[url] = (time.time(), items)
    return items


def _rank(items: list[dict], editorial: bool, lang: str, limit: int) -> list[dict]:
    """Junk and old items out; Bangla mode keeps Bangladesh's outlets; at most PER_OUTLET per outlet. A section
    feed keeps Google's editorial order; a search is reordered so fresher items come first among close ones."""
    now = datetime.now(timezone.utc)
    kept = [i for i in items if not is_junk(i, now)
            and not (lang == "bn" and (i["domain"].endswith(".in") or any(_matches(i["domain"], d) for d in _INDIAN_BENGALI)))]
    if not editorial:
        def score(pair):
            n, item = pair
            age = (now - item["published"]).total_seconds() / 3600 if item["published"] else 24
            return n + 1.5 * age
        kept = [i for _, i in sorted(enumerate(kept), key=score)]
    out, per = [], {}
    for item in kept:
        outlet = item["source"] or item["domain"]
        if per.get(outlet, 0) >= max(PER_OUTLET, limit // 5):
            continue
        per[outlet] = per.get(outlet, 0) + 1
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _words(title: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9ঀ-৿]+", title.lower()) if len(w) > 3}


def same_story(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def name(key: str, lang: str = "en") -> str:
    entry = TOPICS.get(key)
    return (entry[1] if lang == "bn" else entry[0]) if entry else key


def topic_feed(key: str, lang: str = "en") -> tuple:
    entry = TOPICS[key]
    return (entry[3] or entry[2]) if lang == "bn" else entry[2]


# A sport's name searched on general sports sites also matches their menus: on 2026-11-01 half of "cricket news" was
# tennis, boxing and Formula 1 from pages with "Cricket" in the navigation. So cricket searches its own outlets and the
# big sites' cricket sections (CRICKET_SITES), plus the general sports sites with only the headlines that name it
# kept: words no other sport uses (not "batter", baseball's too; not "spins", Formula 1's; not "21st century").
_CRICKET_WORDS = re.compile(
    r"\bcricket|\bT20I?s?\b|\bODIs?\b|\b(?:IPL|BPL|PSL|WPL|WTC)\b|\bBig Bash\b|\bAshes\b|\bwickets?\b|"
    r"(?<!st )(?<!nd )(?<!rd )(?<!th )\bcentur(?:y|ies)\b|\bbowl(?:er|ers|ing|ed)\b|\bbats(?:man|men)\b|"
    r"\bspinners?\b|\bpace(?:man|men)\b|\bseamers?\b|\ball-rounders?\b|\btri-series\b|"
    r"\b(?:first|second|third|fourth|fifth) Test\b|\bTest (?:series|match|squad|side|captain)\b", re.I)
WIDER = {"cricket": (("search", "cricket", SPORT_SITES), _CRICKET_WORDS)}   # topic: (extra feed, headlines kept)

# A region's or country's stories in the world section come first: Google's editors ranked those, while a search
# ranks by match (its first "Asia" story was posh British schools opening in India).
_FROM_WORLD = {"asia", "europe", "middle east", "africa", "americas", "usa", "war", *COUNTRIES}


def _terms(query: str) -> re.Pattern:
    """The words a search query looks for, as a pattern for headlines ("(Asia OR China ...)" -> asia|china|...)."""
    words = [a or b for a, b in re.findall(r'"([^"]+)"|([^\s()"]+)', query) if (a or b) not in ("OR", "AND")]
    return re.compile("|".join(rf"\b{re.escape(w)}\b" if w.isascii() else re.escape(w) for w in words), re.I)


def headlines(topics: list[str], per_topic: int = 3, lang: str = "en", fresh: bool = False) -> list[dict]:
    """[{"topic", "items", "error"}] per topic, fetched in parallel. A topic is a TOPICS key, else free words
    searched among the world and business outlets. The same story under two topics is kept under the first."""
    topics = list(dict.fromkeys(t for t in topics if t))
    if not topics:
        return []

    def one(topic):
        if topic in TOPICS:
            feed = topic_feed(topic, lang)
        else:
            feed = ("search", f'"{topic}"' if " " in topic else topic, None if lang == "bn" else WORLD_SITES + BIZ_SITES)
        try:
            raw = fetch(feed_url(feed, lang), fresh)
        except Exception as e:
            return {"topic": topic, "items": [], "error": type(e).__name__}
        if topic in WIDER and lang == "en":
            wider, about = WIDER[topic]
            try:
                raw = raw + [i for i in fetch(feed_url(wider, lang), fresh) if about.search(i["title"])
                             and not any(same_story(i["title"], r["title"]) for r in raw)]
            except Exception:
                pass        # its own outlets still answer
        items = _rank(raw, feed[0] != "search", lang, per_topic + 6)
        if topic in _FROM_WORLD and feed[0] == "search":
            try:
                pattern = dict(ALIASES)[topic] if topic in COUNTRIES else _terms(feed[1])
                world = [i for i in _rank(fetch(feed_url(topic_feed("world", lang), lang), fresh), True, lang, 40)
                         if pattern.search(i["title"])]
                items = world + [i for i in items if not any(same_story(i["title"], w["title"]) for w in world)]
            except Exception:
                pass        # the search alone still answers
        return {"topic": topic, "items": items[:per_topic + 6], "error": None}

    with ThreadPoolExecutor(max_workers=min(len(topics), 8)) as pool:
        groups = list(pool.map(one, topics))
    seen: list[str] = []
    for g in groups:
        unique = []
        for item in g["items"]:
            if any(same_story(item["title"], s) for s in seen):
                continue
            seen.append(item["title"])
            unique.append({**item, "topic": g["topic"]})
        g["items"] = unique[:per_topic]
    return groups


def search(query: str, days: int = SEARCH_DAYS, lang: str = "en", sites: list[str] | None = None,
           limit: int = 8, fresh: bool = False) -> list[dict]:
    """Free-text news search (a watch's subject, web_search's fallback). Raises on a network error."""
    query = " ".join((query or "").split())
    if not query:
        return []
    items = fetch(feed_url(("search", query, sites), lang, days), fresh)
    return _rank(items, False, lang, limit)


def status() -> dict:
    return dict(_state)
