"""
Live prices: market_price for the model, and market briefs answered in code. Crypto comes from Binance's public
market-data API (data-api.binance.vision: spot tickers, no key or account; Mahadi trades on Binance), stocks and
indices from Yahoo Finance's chart endpoint (unofficial, no key, wants a browser User-Agent). Figures are
formatted here and the model only relays them. They also count as a trusted source for web_tools' figure
checks, so an answer that quotes them next to news headlines isn't flagged as unconfirmed.

A market brief ("crypto market update", "how's bitcoin", "my coins": BRIEF_ASK) is built by brief(): prices for
the coins or stocks named (BTC, ETH, BNB and SOL if none; the S&P 500, Nasdaq and Dow for "the stock market"),
one news search for what's behind them, spoken and screen parts in English or Bangla, no LLM request. It used
to be the market agent, which only copied these figures into sentences. A headline whose figure only one news
site gives is left out, and there are no predictions or buy/sell calls. brief_request() leaves to the model a
message that also asks for something else (a timer, the price in taka).
"""
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import certifi

import tracing
import web_tools

BINANCE_URL = "https://data-api.binance.vision/api/v3/ticker/24hr?symbol="
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}?range=1d&interval=1d"
TIMEOUT_SEC = 8
CACHE_SEC = 30          # prices move: only a repeat within half a minute reuses one
MAX_SYMBOLS = 8
QUOTES = ("USDT", "USDC", "FDUSD", "BTC", "ETH", "BNB", "EUR", "TRY")

CRYPTO = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "ether": "ETH", "eth": "ETH", "bnb": "BNB",
          "binance coin": "BNB", "solana": "SOL", "sol": "SOL", "xrp": "XRP", "ripple": "XRP", "dogecoin": "DOGE",
          "doge": "DOGE", "cardano": "ADA", "ada": "ADA", "toncoin": "TON", "ton": "TON", "tron": "TRX", "trx": "TRX",
          "polkadot": "DOT", "dot": "DOT", "avalanche": "AVAX", "avax": "AVAX", "chainlink": "LINK", "link": "LINK",
          "polygon": "POL", "matic": "POL", "pol": "POL", "litecoin": "LTC", "ltc": "LTC", "shiba inu": "SHIB",
          "shiba": "SHIB", "shib": "SHIB", "pepe": "PEPE", "sui": "SUI", "near": "NEAR", "aptos": "APT", "apt": "APT",
          "arbitrum": "ARB", "arb": "ARB", "optimism": "OP", "uniswap": "UNI", "uni": "UNI", "bitcoin cash": "BCH",
          "bch": "BCH", "stellar": "XLM", "xlm": "XLM", "cosmos": "ATOM", "atom": "ATOM", "filecoin": "FIL",
          "fil": "FIL", "internet computer": "ICP", "icp": "ICP", "hedera": "HBAR", "hbar": "HBAR",
          "injective": "INJ", "inj": "INJ", "render": "RENDER", "fetch": "FET", "fet": "FET", "bonk": "BONK",
          "dogwifhat": "WIF", "wif": "WIF", "ethena": "ENA", "ena": "ENA", "bittensor": "TAO", "tao": "TAO",
          "jupiter": "JUP", "jup": "JUP", "sei": "SEI", "trump": "TRUMP", "floki": "FLOKI", "aave": "AAVE"}
STOCKS = {"apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "tesla": "TSLA", "amazon": "AMZN",
          "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META", "netflix": "NFLX",
          "amd": "AMD", "intel": "INTC", "coinbase": "COIN", "microstrategy": "MSTR", "palantir": "PLTR",
          "tsmc": "TSM", "berkshire": "BRK-B", "s&p 500": "^GSPC", "s&p": "^GSPC", "sp500": "^GSPC",
          "nasdaq": "^IXIC", "dow": "^DJI", "dow jones": "^DJI", "nikkei": "^N225", "ftse": "^FTSE",
          "dax": "^GDAXI", "sensex": "^BSESN", "nifty": "^NSEI", "hang seng": "^HSI"}

_cache: dict = {}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Javid personal assistant)"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=ctx) as r:
        return json.loads(r.read().decode())


def _price(p: float) -> str:
    if p >= 100:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.10f}".rstrip("0")


def _clock(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def _money(p: float, quote: str) -> str:
    return f"${_price(p)}" if quote in ("USDT", "USDC", "FDUSD", "USD") else f"{_price(p)} {quote}"


def _move(pct: float, span: str) -> str:
    return f"{'up' if pct > 0 else 'down' if pct < 0 else 'flat'}{f' {abs(pct):.2f}%' if pct else ''} {span}"


def resolve(symbol: str) -> tuple[str, str, str]:
    """(kind, code, label): ('crypto', 'BTCUSDT', 'BTC/USDT'), ('stock', 'AAPL', 'AAPL') or ('either', 'XYZ', 'XYZ')."""
    s = " ".join(symbol.lower().replace("$", "").split())
    if s in CRYPTO:
        return "crypto", CRYPTO[s] + "USDT", f"{CRYPTO[s]}/USDT"
    if m := re.fullmatch(r"([a-z0-9]{2,12}?)\s*[/\-]?\s*(" + "|".join(q.lower() for q in QUOTES) + r")", s):
        base = CRYPTO.get(m[1], m[1].upper())
        return "crypto", base + m[2].upper(), f"{base}/{m[2].upper()}"
    if s in STOCKS:
        return "stock", STOCKS[s], STOCKS[s]
    return "either", s.upper(), s.upper()


def _binance(code: str, label: str) -> dict:
    d = _get_json(BINANCE_URL + code)
    quote = next(q for q in QUOTES if code.endswith(q))
    base = code[:-len(quote)]
    return {"found": True, "kind": "crypto", "symbol": base, "label": label, "name": _COIN_NAMES.get(base, base),
            "quote": quote, "price": float(d["lastPrice"]), "pct": float(d["priceChangePercent"]),
            "high": float(d["highPrice"]), "low": float(d["lowPrice"]), "volume": float(d["quoteVolume"]),
            "at": datetime.fromtimestamp(d["closeTime"] / 1000), "source": "Binance", "domain": "binance.com"}


def _yahoo(code: str) -> dict:
    d = _get_json(YAHOO_URL.format(urllib.parse.quote(code)))
    meta = d["chart"]["result"][0]["meta"]
    price, prev = meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose")
    pct = meta.get("regularMarketChangePercent")
    if pct is None and prev:
        pct = (price - prev) / prev * 100
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    return {"found": True, "kind": "index" if code.startswith("^") else "stock", "symbol": meta.get("symbol", code),
            "name": meta.get("shortName") or meta.get("longName") or code, "currency": meta.get("currency", ""),
            "price": price, "pct": pct or 0, "low": meta.get("regularMarketDayLow"), "high": meta.get("regularMarketDayHigh"),
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName", ""),
            "at": datetime.fromtimestamp(meta["regularMarketTime"], ZoneInfo(tz)), "source": "Yahoo Finance",
            "domain": "finance.yahoo.com"}


def _line(q: dict) -> str:
    """One quote as market_price gives it to the model."""
    if q["kind"] == "crypto":
        quote = q["quote"]
        return (f"{q['label']}: {_money(q['price'], quote)}, {_move(q['pct'], 'in 24 hours')}; "
                f"24h high {_money(q['high'], quote)}, low {_money(q['low'], quote)}; "
                f"24h volume {_money(q['volume'], quote).split('.')[0]}. Binance spot, {_clock(q['at'])}")
    value = f"{_price(q['price'])} points" if q["kind"] == "index" else f"{_price(q['price'])} {q['currency']}".strip()
    rng = f"; day range {_price(q['low'])} to {_price(q['high'])}" if q["low"] and q["high"] else ""
    at = q["at"]
    return (f"{q['name']} ({q['symbol']}): {value}, {_move(q['pct'], 'from the previous close')}{rng}. "
            f"{q['exchange']} via Yahoo Finance, as of {at:%b} {at.day}, {_clock(at)} {at.tzname()}")


def _quote(symbol: str) -> dict:
    """One symbol's quote (found=False, with the reason in "line", if it couldn't be found)."""
    key = symbol.lower().strip()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    kind, code, label = resolve(symbol)
    out = None
    if kind in ("crypto", "either"):
        try:
            out = _binance(code if kind == "crypto" else code + "USDT", label if kind == "crypto" else f"{code}/USDT")
        except Exception as e:
            if kind == "crypto":
                out = {"found": False, "symbol": label.split("/")[0], "unknown": "400" in str(e),
                       "line": f"{label}: not available on Binance ({'not listed' if '400' in str(e) else type(e).__name__})."}
    if out is None:
        try:
            out = _yahoo(code)
        except Exception as e:
            out = {"found": False, "symbol": symbol.strip(), "unknown": "404" in str(e),
                   "line": f"{symbol}: no price found ({'unknown symbol' if '404' in str(e) else type(e).__name__}); "
                           "for a stock, ask for its ticker (e.g. AAPL)."}
    if out["found"]:
        out["line"] = _line(out)
        _cache[key] = (time.time(), out)
    return out


def quotes(symbols: list[str]) -> list[dict]:
    """Quotes for up to MAX_SYMBOLS symbols, fetched in parallel; found ones become trusted figures for the turn."""
    symbols = list(dict.fromkeys(s.strip() for s in symbols or [] if str(s).strip()))[:MAX_SYMBOLS]
    if not symbols:
        return []
    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        out = list(pool.map(_quote, symbols))
    for q in out:
        if q["found"]:
            web_tools.note_data(q["domain"], q["line"])     # a trusted figure if the answer also cites news
    return out


def market_price(symbols: list[str] | str) -> str:
    if isinstance(symbols, str):
        symbols = re.split(r"[,;]| and ", symbols)
    found = quotes(symbols)
    if not found:
        return "No symbols given: say which coins or stocks, e.g. ['BTC', 'ETH'] or ['AAPL']."
    return ("Live prices:\n" + "\n".join(f"- {q['line']}" for q in found)
            + "\nUse these exact figures and say where and when they're from. Don't predict prices or say "
              "whether to buy or sell.")


# ---------- market briefs, answered in code ----------
BRIEF_ASK = re.compile(
    r"^(?!.*\b(?:news|headlines?|stories)\b).*?(?:\b(?:crypto|coin|stock|share)s? market\b|"
    r"\bmarket (?:update|overview|report|analysis|summary|brief|wrap)\b|"
    r"\b(?:analy[sz]e|analysis of|outlook for|outlook on|how is|how's|hows)\s+(?:the\s+)?(?:crypto|bitcoin|btc|"
    r"ethereum|eth|solana|bnb|xrp|market|stock market)\b|\bmy (?:coins|crypto|portfolio|watchlist)\b|"
    r"ক্রিপ্টো মার্কেট|শেয়ার ?বাজার|মার্কেট আপডেট)", re.I | re.S)
# Another request in the same message (or a price in taka), or a question about what Javid can do ("if I want you
# to be my trading assistant for the stock market, can you help me with that?"): the model answers those.
_ELSE = re.compile(r"weather|forecast|\brain\b|calendar|schedule|meeting|appointment|\btasks?\b|to-?do|remind|timer|"
                   r"alarm|exchange rates?|\byuan\b|koruna|\btaka\b|\bplan (?:my|the|our)\b|\bsearch\b|look (?:it )?up|"
                   r"translate|e-?mail|\bbe my\b|\bif i want you\b|\bassistant\b|\bhelp me (?:out )?with (?:that|this|it)\b|"
                   r"আবহাওয়া|ক্যালেন্ডার|টাস্ক|টাকা|টাইমার", re.I)
_ADVICE_ASK = re.compile(r"\bshould i (?:buy|sell|hold|invest|get)\b|\b(?:good|right|bad) time to (?:buy|sell|invest)\b|"
                         r"\bworth (?:buying|it)\b|\bbuy or sell\b|কিনব|বিক্রি করব|কেনা উচিত", re.I)
_STOCK_WORDS = re.compile(r"\b(?:stock|share|equity|equities)s?\b|wall street|\bindices\b|\bindex\b|শেয়ার", re.I)
CRYPTO_DEFAULT = ["BTC", "ETH", "BNB", "SOL"]
INDEX_DEFAULT = ["S&P 500", "Nasdaq", "Dow Jones"]
# Names that are also English words count only when written as a ticker: "NEAR", not "near the top".
_AMBIGUOUS = {"near", "link", "dot", "fetch", "render", "trump", "uni", "ton", "atom", "fil", "apt", "arb", "sei",
              "jup", "wif", "ena", "inj", "pol", "ada", "tao", "sui", "op"}
_NOT_TICKERS = {"AI", "US", "USA", "UK", "EU", "OK", "ETF", "ETFS", "CEO", "CFO", "GDP", "USD", "USDT", "USDC", "BDT",
                "EUR", "AM", "PM", "IPO", "NFT", "DSE", "CSE", "IT", "TV", "PC", "API", "FYI", "ASAP", "ATH", "DCA",
                "YTD", "LOL", "BD", "NY", "NYC", "SEC", "FED", "FOMC", "CPI", "IMF", "UN", "UTC", "GMT", "EST", "EDT"}
_COIN_NAMES = {**{code: max((k for k, c in CRYPTO.items() if c == code), key=len).title() for code in set(CRYPTO.values())},
               "BNB": "BNB", "XRP": "XRP", "FET": "Fetch.ai"}
_COIN_NAMES_BN = {"BTC": "বিটকয়েন", "ETH": "ইথেরিয়াম", "SOL": "সোলানা", "DOGE": "ডজকয়েন"}
# A daily-recap title says nothing when read out ("Stock Market News for Oct 15, 2026"): screen only.
_GENERIC_TITLE = re.compile(r"\bnews (?:for|on|today)\b|\blive(?: updates| blog)?\b|^\W*(?:stock|crypto|market)\w* "
                            r"(?:news|today|update)s?\b", re.I)
_COMPANY_SUFFIX = re.compile(r",?\s+(?:inc\.?|incorporated|corporation|corp\.?|company|co\.|ltd\.?|limited|plc|n\.v\.|"
                             r"s\.a\.|holdings?)$", re.I)


def symbols_in(text: str) -> list[str]:
    """The coins and stocks a message names, in order: known names ("bitcoin", "apple", "s&p 500") and tickers."""
    lower, spans = text.lower(), []
    for name in sorted({*CRYPTO, *STOCKS}, key=len, reverse=True):
        for m in re.finditer(rf"(?<![\w&^$]){re.escape(name)}(?![\w&])", lower):
            if name in _AMBIGUOUS and not text[m.start():m.end()].isupper():
                continue
            if not any(m.start() < e and s < m.end() for s, e, _ in spans):
                spans.append((m.start(), m.end(), name))
    if sum(c.isupper() for c in text) < 0.6 * sum(c.isalpha() for c in text):    # not when it's all in capitals
        for m in re.finditer(r"\b[A-Z]{2,10}\b", text):
            if m[0] not in _NOT_TICKERS and not any(m.start() < e and s < m.end() for s, e, _ in spans):
                spans.append((m.start(), m.end(), m[0]))
    named, seen = [], set()
    for _, _, name in sorted(spans):
        if (code := resolve(name)[1]) not in seen:
            seen.add(code)
            named.append(name)
    return named


def brief_request(text: str) -> dict | None:
    """What a market brief should cover, if code can answer the whole message; else None."""
    if not BRIEF_ASK.search(text) or _ELSE.search(text):
        return None
    symbols = symbols_in(text) or (INDEX_DEFAULT if _STOCK_WORDS.search(text) else CRYPTO_DEFAULT)
    return {"symbols": symbols[:MAX_SYMBOLS], "advice": bool(_ADVICE_ASK.search(text))}


def _name(q: dict, lang: str = "en") -> str:
    if q["kind"] == "crypto":
        return _COIN_NAMES_BN.get(q["symbol"], q["name"]) if lang == "bn" else q["name"]
    return _COMPANY_SUFFIX.sub("", q["name"]).strip()


def _value(q: dict) -> str:
    if q["kind"] == "index":
        return f"{_price(q['price'])} points"
    return _money(q["price"], q.get("quote") or q.get("currency") or "")


def _change(pct: float, lang: str) -> str:
    if lang == "bn":
        return f"{abs(pct):.2f}% {'বেড়েছে' if pct > 0 else 'কমেছে'}" if round(pct, 2) else "প্রায় একই আছে"
    return f"{'up' if pct > 0 else 'down'} {abs(pct):.2f}%" if round(pct, 2) else "flat"


def _said(q: dict, lang: str, first: bool) -> str:
    """One asset in the spoken part: "Bitcoin is at $82,152.34, up 1.36% in 24 hours"."""
    span = {"crypto": ("in 24 hours", "২৪ ঘণ্টায়"), "stock": ("from the previous close", "আগের ক্লোজ থেকে"),
            "index": ("", "")}[q["kind"]]
    if lang == "bn":
        return f"{_name(q, 'bn')} এখন {_value(q).replace('points', 'পয়েন্ট')}, {span[1] + ' ' if span[1] else ''}{_change(q['pct'], 'bn')}"
    the = "the " if q["kind"] == "index" else ""
    tail = f" {span[0]}" if span[0] and (first or q["kind"] != "crypto") else ""
    return f"{the}{_name(q)} is at {_value(q)}, {_change(q['pct'], 'en')}{tail}"


def _and(items: list[str], lang: str) -> str:
    if len(items) == 1:
        return items[0]
    return "; ".join(items[:-1]) + (" আর " if lang == "bn" else "; and ") + items[-1]


def _spoken(found: list[dict], missing: list[dict], news: list[dict], advice: bool, lang: str) -> str:
    out, said = [], 0
    for source in dict.fromkeys(q["source"] for q in found):
        group = [q for q in found if q["source"] == source][:max(0, 4 - said)]
        if not group:
            continue
        items = [_said(q, lang, i == 0) for i, q in enumerate(group)]
        said += len(group)
        out.append(f"{source}-এ {_and(items, 'bn')}।" if lang == "bn" else f"On {source}, {_and(items, 'en')}.")
    if len(found) > said:
        out.append("বাকিগুলো স্ক্রিনে আছে।" if lang == "bn" else "The rest are on your screen.")
    if unknown := [q["symbol"] for q in missing if q["unknown"]]:
        out.append(f"{', '.join(unknown)} খুঁজে পাইনি।" if lang == "bn" else f"I couldn't find {', '.join(unknown)}.")
    if failed := [q["symbol"] for q in missing if not q["unknown"]]:     # a network or service error
        out.append(f"{', '.join(failed)}-এর দাম এখন পাচ্ছি না।" if lang == "bn" else
                   f"I couldn't get prices for {', '.join(failed)} right now.")
    if told := next((n for n in news if not _GENERIC_TITLE.search(n["title"])), None):
        out.append(f"খবরে {told['name']} বলছে: {told['title']}।" if lang == "bn" else
                   f"In the news, {told['name']} says: {told['title'].rstrip('.')}.")
    if advice:
        out.append("কেনা-বেচার পরামর্শ আমি দিই না, সিদ্ধান্তটা তোমার।" if lang == "bn" else
                   "I don't give buy or sell advice, so that call is yours.")
    return " ".join(out)


def _screen(found: list[dict], missing: list[dict], news: list[dict]) -> str:
    lines = []
    for source in dict.fromkeys(q["source"] for q in found):
        group = [q for q in found if q["source"] == source]
        at = group[0]["at"]
        when = _clock(at) if source == "Binance" else f"{at:%b} {at.day}, {_clock(at)} {at.tzname()}"
        lines.append(f"{'Crypto' if source == 'Binance' else 'Stocks and indices'} ({source}, {when}):")
        for q in group:
            if q["kind"] == "crypto":
                lines.append(f"- {q['symbol']}  {_value(q)}  {_change(q['pct'], 'en')} in 24h  "
                             f"(high {_money(q['high'], q['quote'])} · low {_money(q['low'], q['quote'])})")
            else:
                rng = f"  (day range {_price(q['low'])} to {_price(q['high'])})" if q["low"] and q["high"] else ""
                label = _name(q) if q["kind"] == "index" else f"{q['symbol']} {_name(q)}"
                lines.append(f"- {label}  {_value(q)}  {_change(q['pct'], 'en')} from the previous close{rng}")
    if unknown := [q["symbol"] for q in missing if q["unknown"]]:
        lines.append(f"Not found: {', '.join(unknown)}")
    if failed := [q["symbol"] for q in missing if not q["unknown"]]:
        lines.append(f"Unavailable right now: {', '.join(failed)}")
    if found:
        lines.append("News:")
        lines += [f"- {n['title']} ({n['name']}, {n['date_text']})" for n in news] or ["- No headlines found right now"]
        lines.append("Not financial advice.")
    return "\n".join(lines)


def _topics(found: list[dict]) -> list[str]:
    """What to search the news for: the crypto market, the companies named, or the stock market."""
    topics = ["crypto market"] if any(q["kind"] == "crypto" for q in found) else []
    topics += [_name(q) for q in found if q["kind"] == "stock"]
    if not topics:
        topics = ["stock market"]
    return topics[:2]


def brief(req: dict, lang: str):
    """A market brief built here: a generator (yields its tool events) returning (spoken, screen)."""
    yield ("tool", "market_price")
    span = tracing.tool_start("market_price")
    try:
        got = quotes(req["symbols"])
    finally:
        tracing.tool_end(span)
    found = [q for q in got if q["found"]]
    missing = [q for q in got if not q["found"]]
    news = []
    if found:
        yield ("tool", "news_headlines")
        span = tracing.tool_start("news_headlines")
        try:
            web_tools.news_headlines(_topics(found), per_topic=2)
        finally:
            tracing.tool_end(span)
        # A headline's own figure ("stocks fall 1.2%") is one news site's: only headlines without one.
        news = sorted((s for s in web_tools.sources() if not web_tools.check_reply(f"{s['title']} ({s['name']}, {s['date_text']})")),
                      key=lambda s: s["date"] or "", reverse=True)[:3]     # newest first
    return _spoken(found, missing, news, req["advice"], lang), _screen(found, missing, news)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "market_price",
        "description": "Live prices: crypto from Binance (24h change, high/low) and stocks or indices from Yahoo "
                       "Finance. Use it for any price of a coin, stock or index, not web_search.",
        "parameters": {"type": "object", "properties": {
            "symbols": {"type": "array", "items": {"type": "string"},
                        "description": "Coins or tickers, e.g. ['BTC', 'ETH'], ['solana'], ['AAPL', 'S&P 500']."}},
            "required": ["symbols"]},
    },
}]
TOOL_MAP = {"market_price": lambda args: market_price(**args)}
