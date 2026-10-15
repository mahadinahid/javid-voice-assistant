"""
Exchange rates for the exchange-rates skill: CNY, BDT, CZK and USD, with when each rate was published.

USD->CNY and USD->CZK come from the European Central Bank's reference rates (official; via the Frankfurter
API, which gives the ECB's date: the ECB sets them around 16:00 CET). The ECB doesn't quote taka, so BDT
comes from ExchangeRate-API (a market-data aggregator, updated once a day; it reports the update time).
Cross rates are computed here, not by the model. No API key needed; results are cached for CACHE_SEC.
"""
import json
import ssl
import time
import urllib.request
from datetime import date, datetime, timezone

import certifi   # macOS python.org builds have no CA bundle for urllib; groq/httpx already depend on certifi

ECB_URL = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=CNY,CZK"
AGG_URL = "https://open.er-api.com/v6/latest/USD"
CACHE_SEC = 15 * 60
TIMEOUT_SEC = 8
CURRENCIES = ("USD", "BDT", "CNY", "CZK")
NAMES = {"USD": "US dollar", "BDT": "taka", "CNY": "yuan", "CZK": "Czech koruna"}

_cache: dict = {}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Javid/1.0"})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC, context=ctx) as r:
        return json.loads(r.read().decode())


def fetch() -> dict:
    """{"rates": {code: per 1 USD}, "sources": {code: (name, official, published text)}}; raises if nothing came back."""
    hit = _cache.get("rates")
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    rates, sources, errors = {"USD": 1.0}, {}, []
    try:
        ecb = _get_json(ECB_URL)
        day = date.fromisoformat(ecb["date"])
        for code in ("CNY", "CZK"):
            rates[code] = float(ecb["rates"][code])
            sources[code] = ("European Central Bank", True, f"{day:%B} {day.day} reference rate (set about 16:00 CET)")
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
                sources[code] = ("ExchangeRate-API", False, f"updated {at:%B} {at.day}, {at:%H:%M} UTC")
    except Exception as e:
        errors.append(f"ExchangeRate-API: {type(e).__name__}")
    if len(rates) == 1:
        raise RuntimeError("; ".join(errors) or "no rates")
    data = {"rates": rates, "sources": sources, "errors": errors}
    _cache["rates"] = (time.time(), data)
    return data


def _fmt(x: float) -> str:
    return f"{x:,.2f}" if x >= 10 else f"{x:,.4f}".rstrip("0").rstrip(".")


def exchange_rates() -> str:
    try:
        data = fetch()
    except Exception as e:
        return (f"Couldn't get exchange rates right now ({e}). Tell Mahadi the rates are unavailable; "
                "don't give any figures.")
    rates, sources = data["rates"], data["sources"]
    lines = ["Per 1 US dollar:"]
    for code in ("BDT", "CNY", "CZK"):
        if code in rates:
            name, official, when = sources[code]
            lines.append(f"- {_fmt(rates[code])} {code} ({NAMES[code]}): {name}{', official' if official else ''}, {when}")
        else:
            lines.append(f"- {code}: unavailable right now")
    if "BDT" in rates:
        lines.append("In taka: " + "; ".join(f"1 {c} = {_fmt(rates['BDT'] / rates[c])} BDT" for c in ("CNY", "CZK") if c in rates))
    if "CNY" in rates and "CZK" in rates:
        lines.append(f"1 CNY = {_fmt(rates['CZK'] / rates['CNY'])} CZK")
    if data["errors"]:
        lines.append("Some sources failed: " + "; ".join(data["errors"]))
    lines.append("Use these exact figures, name each source and when it was published. Don't add other rates.")
    return "\n".join(lines)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "exchange_rates",
        "description": "Current USD, BDT, CNY and CZK exchange rates with their sources and publication times.",
        "parameters": {"type": "object", "properties": {}},
    },
}]
TOOL_MAP = {"exchange_rates": lambda args: exchange_rates()}
