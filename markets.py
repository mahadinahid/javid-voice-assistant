"""
Live prices for market_price and the market agent. Crypto comes from Binance's public market-data API
(data-api.binance.vision: spot tickers, no key or account; Mahadi trades on Binance), stocks and indices from
Yahoo Finance's chart endpoint (unofficial, no key, wants a browser User-Agent). Figures are formatted here and
the model only relays them. They also count as a trusted source for web_tools' figure checks, so a market
answer that quotes them next to news headlines isn't flagged as unconfirmed.
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


def _binance(code: str, label: str) -> str:
    d = _get_json(BINANCE_URL + code)
    quote = next(q for q in QUOTES if code.endswith(q))
    at = datetime.fromtimestamp(d["closeTime"] / 1000)
    return (f"{label}: {_money(float(d['lastPrice']), quote)}, {_move(float(d['priceChangePercent']), 'in 24 hours')}; "
            f"24h high {_money(float(d['highPrice']), quote)}, low {_money(float(d['lowPrice']), quote)}; "
            f"24h volume {_money(float(d['quoteVolume']), quote).split('.')[0]}. Binance spot, {_clock(at)}")


def _yahoo(code: str) -> str:
    d = _get_json(YAHOO_URL.format(urllib.parse.quote(code)))
    meta = d["chart"]["result"][0]["meta"]
    price, prev = meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose")
    pct = meta.get("regularMarketChangePercent")
    if pct is None and prev:
        pct = (price - prev) / prev * 100
    name = meta.get("shortName") or meta.get("longName") or code
    index = code.startswith("^")
    value = f"{_price(price)} points" if index else f"{_price(price)} {meta.get('currency', '')}".strip()
    tz = meta.get("exchangeTimezoneName") or "America/New_York"
    at = datetime.fromtimestamp(meta["regularMarketTime"], ZoneInfo(tz))
    rng = ""
    if meta.get("regularMarketDayLow") and meta.get("regularMarketDayHigh"):
        rng = f"; day range {_price(meta['regularMarketDayLow'])} to {_price(meta['regularMarketDayHigh'])}"
    return (f"{name} ({meta.get('symbol', code)}): {value}, {_move(pct or 0, 'from the previous close')}{rng}. "
            f"{meta.get('fullExchangeName') or meta.get('exchangeName', '')} via Yahoo Finance, as of "
            f"{at:%b} {at.day}, {_clock(at)} {at.tzname()}")


def _quote(symbol: str) -> tuple[str, str]:
    """(line, domain) for one symbol; the line says so if it couldn't be found."""
    key = symbol.lower().strip()
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    kind, code, label = resolve(symbol)
    out = None
    if kind in ("crypto", "either"):
        try:
            out = (_binance(code if kind == "crypto" else code + "USDT", label if kind == "crypto" else f"{code}/USDT"),
                   "binance.com")
        except Exception as e:
            if kind == "crypto":
                out = (f"{label}: not available on Binance ({'not listed' if '400' in str(e) else type(e).__name__}).", "")
    if out is None:
        try:
            out = (_yahoo(code), "finance.yahoo.com")
        except Exception as e:
            out = (f"{symbol}: no price found ({'unknown symbol' if '404' in str(e) else type(e).__name__}); "
                   "for a stock, ask for its ticker (e.g. AAPL).", "")
    if out[1]:
        _cache[key] = (time.time(), out)
    return out


def market_price(symbols: list[str] | str) -> str:
    if isinstance(symbols, str):
        symbols = re.split(r"[,;]| and ", symbols)
    symbols = list(dict.fromkeys(s.strip() for s in symbols or [] if str(s).strip()))[:MAX_SYMBOLS]
    if not symbols:
        return "No symbols given: say which coins or stocks, e.g. ['BTC', 'ETH'] or ['AAPL']."
    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        quotes = list(pool.map(_quote, symbols))
    for line, domain in quotes:
        if domain:
            web_tools.note_data(domain, line)     # a trusted figure if the answer also cites news
    return ("Live prices:\n" + "\n".join(f"- {line}" for line, _ in quotes)
            + "\nUse these exact figures and say where and when they're from. Don't predict prices or say "
              "whether to buy or sell.")


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
