"""
Weather for get_weather and the morning briefing.

Open-Meteo (free, no key) gives current conditions, hourly chance of rain, daily high/low, UV and sunrise/
sunset. When it fails - its forecast API answered "The service is overloaded" all afternoon on 2026-10-19 -
MET Norway's locationforecast is used instead (free, no key, asks for an identifying User-Agent). MET gives
rain amounts, not chances, so its day high/low and rain are worked out here from its hourly series. Places
are looked up with Open-Meteo's geocoding API; home (Dhaka) is built in, so the default needs no lookup.
Figures are computed and worded here: Javid invented "sunny, 28°C in India" in a morning greeting
(2026-10-17) when it had no weather tool.
"""
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import certifi   # python.org macOS builds have no CA bundle for urllib (see rates.py)

import calendar_tools   # _parse_day: the same day words as the calendar ("tomorrow", "friday", "কাল")

HOME = {"name": "Dhaka", "country": "Bangladesh", "lat": 23.7104, "lon": 90.4074, "tz": "Asia/Dhaka"}
_HOME_WORDS = {"", "home", "here", "my city", "my area", "dhaka", "ঢাকা", "bangladesh", "বাংলাদেশ"}
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search?"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast?"
MET_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact?"
USER_AGENT = "Javid/1.0 (personal voice assistant; github.com/mahadinahid/javid-voice-assistant)"
CACHE_SEC = 10 * 60
TIMEOUT_SEC = 6
MAX_DAYS = 7
SKIP_AFTER_FAIL_SEC = 10 * 60   # after Open-Meteo fails, go straight to MET for this long

_cache: dict = {}      # (lat, lon) -> (time, forecast)
_places: dict = {}     # lowercased place -> location
_skip_until = {"open-meteo": 0.0}

# WMO weather codes (Open-Meteo)
WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers",
       81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
       95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail"}
# MET Norway symbol codes are run together ("lightrainshowersandthunder_day"): split them into words.
_MET_WORDS = [("clearsky", "clear sky"), ("partlycloudy", "partly cloudy"), ("fair", "fair"), ("cloudy", "cloudy"),
              ("fog", "fog"), ("lights", "light "), ("light", "light "), ("heavy", "heavy "),
              ("rainshowers", "rain showers"), ("sleetshowers", "sleet showers"), ("snowshowers", "snow showers"),
              ("rain", "rain"), ("sleet", "sleet"), ("snow", "snow"), ("andthunder", " and thunder")]
# How much a condition matters when one has to stand for a whole day.
_SEVERITY = [("thunder", 6), ("heavy", 5), ("rain", 4), ("sleet", 4), ("snow", 4), ("drizzle", 3), ("shower", 3),
             ("fog", 2), ("overcast", 1), ("cloudy", 1)]


def _get_json(url: str, timeout: float = TIMEOUT_SEC) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        data = json.loads(r.read().decode())
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data.get("reason") or "error")    # Open-Meteo: {"error": true, "reason": "...overloaded"}
    return data


def _met_desc(code: str | None) -> str:
    base = re.sub(r"_(day|night|polartwilight)$", "", code or "")
    out, rest = "", base
    while rest:
        for key, words in _MET_WORDS:
            if rest.startswith(key):
                out, rest = out + words, rest[len(key):]
                break
        else:
            return base
    return " ".join(out.split())


def _severity(desc: str) -> int:
    return max((n for word, n in _SEVERITY if word in desc), default=0)


# ---------- places ----------
def locate(place: str) -> dict:
    """{"name", "country", "lat", "lon", "tz"}; raises LookupError for an unknown place."""
    name = " ".join((place or "").split()).strip(" .")
    key = name.lower()
    if key in _HOME_WORDS or key.removesuffix(", bangladesh") == "dhaka":
        return HOME
    if key in _places:
        return _places[key]
    city, _, hint = name.partition(",")
    lang = "bn" if re.search(r"[\u0980-\u09ff]", city) else "en"     # Bengali names only match in Bengali
    data = _get_json(GEO_URL + urllib.parse.urlencode({"name": city.strip(), "count": 5, "language": lang,
                                                         "format": "json"}))
    results = data.get("results") or []
    if not results:
        raise LookupError(f"Couldn't find a place called '{name}'.")
    hint = hint.strip().lower()
    pick = (next((r for r in results if hint and hint in f"{r.get('country', '')} {r.get('admin1', '')}".lower()), None)
            or next((r for r in results if not hint and r.get("country_code") == "BD"), None)   # home country first
            or results[0])
    loc = {"name": pick["name"], "country": pick.get("country", ""), "lat": pick["latitude"],
           "lon": pick["longitude"], "tz": pick.get("timezone") or "UTC"}
    _places[key] = loc
    return loc


# ---------- forecasts, in one shape whichever service answered ----------
def _open_meteo(loc: dict, timeout: float) -> dict:
    q = {"latitude": loc["lat"], "longitude": loc["lon"], "timezone": loc["tz"], "forecast_days": MAX_DAYS,
         "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
         "hourly": "temperature_2m,precipitation_probability,precipitation,weather_code",
         "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                  "precipitation_sum,uv_index_max,sunrise,sunset"}
    d = _get_json(OPEN_METEO_URL + urllib.parse.urlencode(q), timeout)
    cur, hr, dl = d["current"], d["hourly"], d["daily"]
    hours = [{"time": datetime.fromisoformat(t), "temp": hr["temperature_2m"][i],
              "rain_chance": hr["precipitation_probability"][i], "rain_mm": hr["precipitation"][i],
              "desc": WMO.get(hr["weather_code"][i], "")} for i, t in enumerate(hr["time"])]
    days = [{"date": date.fromisoformat(t), "high": dl["temperature_2m_max"][i], "low": dl["temperature_2m_min"][i],
             "desc": WMO.get(dl["weather_code"][i], ""), "rain_chance": dl["precipitation_probability_max"][i],
             "rain_mm": dl["precipitation_sum"][i], "uv": dl["uv_index_max"][i],
             "sunrise": datetime.fromisoformat(dl["sunrise"][i]) if dl["sunrise"][i] else None,
             "sunset": datetime.fromisoformat(dl["sunset"][i]) if dl["sunset"][i] else None}
            for i, t in enumerate(dl["time"])]
    now = {"time": datetime.fromisoformat(cur["time"]), "temp": cur.get("temperature_2m"),
           "feels": cur.get("apparent_temperature"), "humidity": cur.get("relative_humidity_2m"),
           "wind": cur.get("wind_speed_10m"), "desc": WMO.get(cur.get("weather_code"), "")}
    return {"source": "Open-Meteo", "now": now, "hours": hours, "days": days}


def _met_no(loc: dict, timeout: float) -> dict:
    d = _get_json(MET_URL + urllib.parse.urlencode({"lat": round(loc["lat"], 4), "lon": round(loc["lon"], 4)}), timeout)
    tz = ZoneInfo(loc["tz"])
    hours = []
    for e in d["properties"]["timeseries"]:
        t = datetime.fromisoformat(e["time"].replace("Z", "+00:00")).astimezone(tz).replace(tzinfo=None)
        inst = e["data"]["instant"]["details"]
        nxt = e["data"].get("next_1_hours") or e["data"].get("next_6_hours") or {}
        wind = inst.get("wind_speed")
        hours.append({"time": t, "temp": inst.get("air_temperature"), "humidity": inst.get("relative_humidity"),
                      "wind": None if wind is None else wind * 3.6, "rain_chance": None,
                      "rain_mm": (nxt.get("details") or {}).get("precipitation_amount"),
                      "desc": _met_desc((nxt.get("summary") or {}).get("symbol_code"))})
    if not hours:
        raise RuntimeError("MET Norway sent no forecast")
    days = []
    for d_ in sorted({h["time"].date() for h in hours})[:MAX_DAYS]:
        hs = [h for h in hours if h["time"].date() == d_ and h["temp"] is not None]
        if not hs:
            continue
        rain = sum(h["rain_mm"] or 0 for h in hs)
        wet = [h for h in hs if (h["rain_mm"] or 0) >= 0.1]
        if rain >= 1 and wet:      # a wet day is described by its worst hour, a dry one by its usual sky
            desc = max(wet, key=lambda h: _severity(h["desc"]))["desc"]
        else:
            daytime = [h["desc"] for h in hs if 7 <= h["time"].hour <= 18 and h["desc"]] or [h["desc"] for h in hs]
            desc = max(set(daytime), key=daytime.count) if daytime else ""
        days.append({"date": d_, "high": max(h["temp"] for h in hs), "low": min(h["temp"] for h in hs),
                     "desc": desc, "rain_chance": None, "rain_mm": round(rain, 1), "uv": None,
                     "sunrise": None, "sunset": None, "partial": len(hs) < 4 or hs[0]["time"].hour > 6})
    h0 = hours[0]
    now = {"time": h0["time"], "temp": h0["temp"], "feels": None, "humidity": h0["humidity"], "wind": h0["wind"],
           "desc": h0["desc"]}
    return {"source": "MET Norway", "now": now, "hours": hours, "days": days}


def forecast(loc: dict, timeout: float = TIMEOUT_SEC) -> dict:
    key = (round(loc["lat"], 3), round(loc["lon"], 3))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SEC:
        return hit[1]
    errors = []
    if time.time() >= _skip_until["open-meteo"]:
        try:
            fc = _open_meteo(loc, timeout)
            _cache[key] = (time.time(), fc)
            return fc
        except Exception as e:
            errors.append(f"Open-Meteo: {type(e).__name__}: {str(e)[:60]}")
            _skip_until["open-meteo"] = time.time() + SKIP_AFTER_FAIL_SEC
            print(f"[weather] Open-Meteo failed ({str(e)[:60]}); using MET Norway")
    try:
        fc = _met_no(loc, timeout)
    except Exception as e:
        errors.append(f"MET Norway: {type(e).__name__}")
        raise RuntimeError("; ".join(errors)) from None
    _cache[key] = (time.time(), fc)
    return fc


# ---------- wording ----------
def _deg(x) -> str:
    return "unknown" if x is None else f"{round(x)}°C"


def _clock(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")     # "4 PM", "5:32 PM"


def _uv(uv) -> str:
    if uv is None:
        return ""
    word = "low" if uv < 3 else "moderate" if uv < 6 else "high" if uv < 8 else "very high" if uv < 11 else "extreme"
    return f"UV index {round(uv)} ({word})"


def _rain(day: dict, hours: list[dict], after: datetime | None) -> str:
    hs = [h for h in hours if h["time"].date() == day["date"] and (after is None or h["time"] >= after)]
    if day["rain_chance"] is not None:       # Open-Meteo: a chance, and when it peaks
        chance = day["rain_chance"]
        peak = max((h for h in hs if h["rain_chance"] is not None), key=lambda h: h["rain_chance"], default=None)
        if chance >= 30 and peak and peak["rain_chance"] >= 30:
            return f"{chance}% chance of rain, most likely around {_clock(peak['time'])}"
        return f"{chance}% chance of rain"
    mm = sum(h["rain_mm"] or 0 for h in hs)  # MET Norway: an amount
    if mm < 0.2:
        return "no rain expected"
    peak = max(hs, key=lambda h: h["rain_mm"] or 0)
    return f"about {mm:.1f} mm of rain expected, mostly around {_clock(peak['time'])}"


def _day_label(d: date, today: date) -> str:
    if d == today:
        return "Today"
    if d == today + timedelta(days=1):
        return f"Tomorrow ({d:%A}, {d:%B} {d.day})"
    return f"{d:%A}, {d:%B} {d.day}"


def _day_line(day: dict, fc: dict, today: date, now: datetime) -> str:
    rest = fc["source"] == "MET Norway" and day["date"] == today and day.get("partial")
    label = "Rest of today" if rest else _day_label(day["date"], today)
    parts = [f"high {_deg(day['high'])}, low {_deg(day['low'])}", day["desc"],
             _rain(day, fc["hours"], now if day["date"] == today else None), _uv(day["uv"])]
    if day["date"] == today and day["sunset"]:
        parts.append(f"sunset {_clock(day['sunset'])}")
    return f"{label}: " + "; ".join(p for p in parts if p) + "."


def get_weather(place: str = "", day: str = "today", days: int = 1) -> str:
    try:
        loc = locate(place)
    except LookupError as e:
        return f"{e} Ask Mahadi which city he means."
    except Exception as e:
        return (f"Couldn't look up '{place}' right now ({type(e).__name__}). Tell Mahadi the weather is "
                "unavailable; don't guess it.")
    try:
        start = calendar_tools._parse_day((day or "today").replace("tonight", "today"))
    except ValueError:
        start = date.today()
    count = max(1, min(int(days or 1), MAX_DAYS))
    try:
        fc = forecast(loc)
    except Exception as e:
        return (f"Couldn't get the weather right now ({e}). Tell Mahadi the weather is unavailable; "
                "don't guess any figures.")
    local_now = datetime.now(ZoneInfo(loc["tz"])).replace(tzinfo=None)
    today = local_now.date()
    where = loc["name"] + (f", {loc['country']}" if loc["country"] else "")
    lines = [f"Weather for {where} ({fc['source']}; local time {_clock(local_now)}):"]
    if start <= today:
        n = fc["now"]
        bits = [_deg(n["temp"]) + (f", feels like {_deg(n['feels'])}" if n["feels"] is not None else ""), n["desc"],
                f"humidity {round(n['humidity'])}%" if n["humidity"] is not None else "",
                f"wind {round(n['wind'])} km/h" if n["wind"] is not None else ""]
        lines.append("Now: " + ", ".join(b for b in bits if b) + ".")
    by_date = {d["date"]: d for d in fc["days"]}
    for i in range(count):
        d = max(start, today) + timedelta(days=i)
        if d in by_date:
            lines.append(_day_line(by_date[d], fc, today, local_now))
        else:
            lines.append(f"{_day_label(d, today)}: no forecast that far ahead.")
    lines.append("Say these figures as given (Celsius); don't add any others.")
    return "\n".join(lines)


def brief_line(timeout: float = 4) -> str | None:
    """Home weather in one line for the morning briefing; None if no service answered."""
    try:
        fc = forecast(HOME, timeout)
    except Exception:
        return None
    local_now = datetime.now(ZoneInfo(HOME["tz"])).replace(tzinfo=None)
    today = next((d for d in fc["days"] if d["date"] == local_now.date()), None)
    now = fc["now"]
    line = f"Weather in {HOME['name']} ({fc['source']}): {_deg(now['temp'])} now, {now['desc'] or 'no sky report'}"
    if today:
        line += f"; today's high {_deg(today['high'])}, low {_deg(today['low'])}; {_rain(today, fc['hours'], local_now)}"
    return line + "."


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather now and the forecast (high/low, rain, UV) for a place; Mahadi's home (Dhaka) "
                       "if no place is named. Call it for any weather question and use only its figures.",
        "parameters": {"type": "object", "properties": {
            "place": {"type": "string", "description": "City, e.g. 'Sylhet' or 'London, UK'. Empty = home."},
            "day": {"type": "string", "description": "'today', 'tomorrow', a weekday or YYYY-MM-DD (default today)."},
            "days": {"type": "integer", "description": "How many days from that day, 1-7 (default 1)."},
        }},
    },
}]
TOOL_MAP = {"get_weather": lambda args: get_weather(**args)}
