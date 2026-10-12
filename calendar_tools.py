"""
Google Calendar for Javid — read today's agenda, add events, find free time.

Setup (once): put an OAuth "Desktop app" client file at credentials.json, then
connect from the web UI sidebar (or call connect()). The token is saved to
token.json and refreshed automatically after that.
"""

import os
import re
from datetime import date, datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
CALENDAR_ID = "primary"
WORK_START, WORK_END = 9, 18   # hours searched by find_free_time

NOT_CONNECTED = "Google Calendar isn't connected yet. Mahadi can connect it from the sidebar."


class NotConnected(Exception):
    pass


# ---------- auth ----------
def _load_creds() -> Credentials | None:
    if not os.path.exists(TOKEN_FILE):
        return None
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            return None  # revoked or expired for good; needs a fresh sign-in
        _save(creds)
        return creds
    return None


def _save(creds: Credentials) -> None:
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())


def status() -> str:
    """'connected', 'needs_auth' (credentials.json present), or 'no_credentials'."""
    if _load_creds():
        return "connected"
    return "needs_auth" if os.path.exists(CREDENTIALS_FILE) else "no_credentials"


def connect() -> str:
    """Open the Google sign-in page in the browser and wait for consent."""
    if not os.path.exists(CREDENTIALS_FILE):
        return "no_credentials"
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=300)
    _save(creds)
    return status()


def _service():
    creds = _load_creds()
    if not creds:
        raise NotConnected
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# ---------- helpers ----------
def _tz():
    return datetime.now().astimezone().tzinfo


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _parse_day(day: str) -> date:
    """'today', 'tomorrow', a weekday ('friday', 'next tuesday'), or YYYY-MM-DD.

    Date math happens here, not in the model — small LLMs get weekdays wrong.
    A weekday means its next occurrence; said on that same weekday, a week ahead.
    """
    text = (day or "today").strip().lower()
    for filler in ("on ", "this ", "next ", "coming "):
        text = text.removeprefix(filler)
    today = date.today()
    if text == "today":
        return today
    if text == "tomorrow":
        return today + timedelta(days=1)
    for i, name in enumerate(_WEEKDAYS):
        if text in (name, name[:3]):
            ahead = (i - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
    return date.fromisoformat(text)


def _parse_time(text: str) -> tuple[int, int]:
    """'15:00', '3pm', '3:30 pm', '9' -> (hour, minute)."""
    m = re.fullmatch(r"\s*(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?m?\.?\s*", text.lower())
    if not m:
        raise ValueError(f"unrecognised time '{text}'")
    hour, minute, ampm = int(m[1]), int(m[2] or 0), m[3]
    if ampm == "p" and hour < 12:
        hour += 12
    if ampm == "a" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"unrecognised time '{text}'")
    return hour, minute


def _day_bounds(d: date, days: int = 1) -> tuple[datetime, datetime]:
    start = datetime.combine(d, datetime.min.time(), _tz())
    return start, start + timedelta(days=days)


def _fetch(start: datetime, end: datetime) -> list[dict]:
    result = _service().events().list(
        calendarId=CALENDAR_ID,
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        singleEvents=True,
        orderBy="startTime",
        maxResults=50,
    ).execute()
    return result.get("items", [])


def _event_times(ev: dict) -> tuple[datetime, datetime, bool]:
    """(start, end, all_day) in local time."""
    s, e = ev["start"], ev["end"]
    if "date" in s:  # all-day
        return (datetime.combine(date.fromisoformat(s["date"]), datetime.min.time(), _tz()),
                datetime.combine(date.fromisoformat(e["date"]), datetime.min.time(), _tz()), True)
    return (datetime.fromisoformat(s["dateTime"]).astimezone(_tz()),
            datetime.fromisoformat(e["dateTime"]).astimezone(_tz()), False)


def _fmt_time(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def _guard(fn):
    """Turn auth/API failures into sentences Javid can say."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotConnected:
            return NOT_CONNECTED
        except ValueError as e:
            return f"I couldn't understand that date or time: {e}"
        except HttpError as e:
            return f"Google Calendar returned an error: {e.reason}"
        except Exception as e:
            return f"Calendar is unavailable right now ({type(e).__name__})."
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------- tools ----------
@_guard
def list_events(day: str = "today", days: int = 1) -> str:
    d = _parse_day(day)
    days = max(1, min(int(days), 14))
    events = _fetch(*_day_bounds(d, days))
    span = d.strftime("%A, %B %d") + (f" plus the next {days - 1} days" if days > 1 else "")
    if not events:
        return f"No events on {span}."
    lines = [f"{len(events)} event(s) on {span}:"]
    for ev in events:
        start, end, all_day = _event_times(ev)
        when = "all day" if all_day else f"{_fmt_time(start)} to {_fmt_time(end)}"
        prefix = start.strftime("%a %b %d, ") if days > 1 else ""
        where = f" at {ev['location']}" if ev.get("location") else ""
        lines.append(f"- {prefix}{when}: {ev.get('summary', '(no title)')}{where}")
    return "\n".join(lines)


@_guard
def add_event(title: str, day: str, time: str, duration_minutes: int = 60, location: str = "") -> str:
    hour, minute = _parse_time(time)
    begin = datetime.combine(_parse_day(day), datetime.min.time(), _tz()).replace(hour=hour, minute=minute)
    end = begin + timedelta(minutes=int(duration_minutes))
    body = {
        "summary": title,
        "start": {"dateTime": begin.isoformat()},
        "end": {"dateTime": end.isoformat()},
    }
    if location:
        body["location"] = location
    _service().events().insert(calendarId=CALENDAR_ID, body=body).execute()

    clashes = [ev.get("summary", "(no title)") for ev in _fetch(begin, end)
               if ev.get("summary") != title and not _event_times(ev)[2]]  # all-day events don't clash
    note = f" Heads up: it overlaps with {', '.join(clashes)}." if clashes else ""
    return (f"Added '{title}' on {begin.strftime('%A, %B %d')} "
            f"from {_fmt_time(begin)} to {_fmt_time(end)}.{note}")


@_guard
def find_free_time(day: str = "today", duration_minutes: int = 30) -> str:
    d = _parse_day(day)
    need = timedelta(minutes=int(duration_minutes))
    window_start = datetime.combine(d, datetime.min.time(), _tz()).replace(hour=WORK_START)
    window_end = window_start.replace(hour=WORK_END)
    now = datetime.now(_tz())
    if d == now.date():
        # round up to the next quarter hour so we never suggest the past
        now = (now + timedelta(minutes=14)).replace(second=0, microsecond=0)
        now -= timedelta(minutes=now.minute % 15)
        window_start = max(window_start, now)
    if window_start >= window_end:
        return f"No working hours left on {d.strftime('%A')}."

    busy = sorted((s, e) for s, e, all_day in map(_event_times, _fetch(window_start, window_end))
                  if not all_day)
    slots, cursor = [], window_start
    for s, e in busy:
        if s - cursor >= need:
            slots.append((cursor, s))
        cursor = max(cursor, e)
    if window_end - cursor >= need:
        slots.append((cursor, window_end))

    if not slots:
        return f"No free {duration_minutes}-minute slot between {WORK_START}:00 and {WORK_END}:00 on {d.strftime('%A')}."
    return f"Free on {d.strftime('%A, %B %d')}: " + "; ".join(
        f"{_fmt_time(s)} to {_fmt_time(e)}" for s, e in slots)


# Google's public holiday calendars: en.<code>#holiday@group.v.calendar.google.com
HOLIDAY_CALENDARS = {
    "bangladesh": "bd", "india": "indian", "pakistan": "pk", "usa": "usa",
    "united states": "usa", "uk": "uk", "united kingdom": "uk", "saudi arabia": "saudiarabian",
    "malaysia": "malaysia", "japan": "japanese", "germany": "german",
    "australia": "australian", "canada": "canadian",
}
HOME_COUNTRY = "bangladesh"
HOME_WEEKEND = {4, 5}   # Friday, Saturday (Monday = 0)


def _breaks(holidays: dict[date, str]) -> list[str]:
    """Runs of 3+ consecutive days off (holidays plus weekend) that include a holiday."""
    out, seen = [], set()
    for d in sorted(holidays):
        if d in seen:
            continue
        start = end = d
        off = lambda x: x in holidays or x.weekday() in HOME_WEEKEND
        while off(start - timedelta(days=1)):
            start -= timedelta(days=1)
        while off(end + timedelta(days=1)):
            end += timedelta(days=1)
        span = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        seen.update(span)
        if len(span) >= 3:
            names = list(dict.fromkeys(holidays[x] for x in span if x in holidays))
            out.append(f"{start.strftime('%A, %B %d')} to {end.strftime('%A, %B %d')}: "
                       f"{len(span)} days off in a row ({', '.join(names)} plus the weekend)")
    return out


@_guard
def list_holidays(country: str = HOME_COUNTRY, days: int = 60, include_observances: bool = False) -> str:
    """Official public holidays from Google's holiday calendar — not a web guess."""
    key = (country or HOME_COUNTRY).strip().lower()
    code = HOLIDAY_CALENDARS.get(key)
    if not code:
        return f"I don't have a holiday calendar for {country}. Known: {', '.join(sorted(HOLIDAY_CALENDARS))}."
    start = datetime.combine(date.today(), datetime.min.time(), _tz())
    end = start + timedelta(days=max(1, min(int(days), 366)))
    items = _service().events().list(
        calendarId=f"en.{code}#holiday@group.v.calendar.google.com",
        timeMin=start.isoformat(), timeMax=end.isoformat(),
        singleEvents=True, orderBy="startTime",
    ).execute().get("items", [])

    rows, public = [], {}
    for ev in items:
        kind = "observance" if (ev.get("description") or "").startswith("Observance") else "public holiday"
        if kind == "observance" and not include_observances:
            continue
        d = date.fromisoformat(ev["start"]["date"]) if "date" in ev["start"] else _event_times(ev)[0].date()
        rows.append(f"- {d.strftime('%A, %B %d, %Y')}: {ev.get('summary')} ({kind})")
        if kind == "public holiday":
            public.setdefault(d, ev.get("summary"))
    if not rows:
        return f"No public holidays in {country.title()} in the next {days} days."
    text = f"Holidays in {country.title()} (next {days} days, from Google's official calendar):\n" + "\n".join(rows)
    if public:
        first = min(public)
        text += f"\nNext public holiday: {first.strftime('%A, %B %d')} ({public[first]})."
    if key == HOME_COUNTRY and (breaks := _breaks(public)):
        text += "\nLong breaks:\n" + "\n".join(f"- {b}" for b in breaks)
    return text


# ---------- web UI ----------
def agenda(day: str = "today") -> dict:
    st = status()
    if st != "connected":
        return {"status": st, "events": []}
    try:
        events = _fetch(*_day_bounds(_parse_day(day)))
    except Exception as e:
        return {"status": "error", "message": str(e)[:120], "events": []}
    out = []
    for ev in events:
        start, end, all_day = _event_times(ev)
        out.append({
            "title": ev.get("summary", "(no title)"),
            "start": "All day" if all_day else _fmt_time(start),
            "end": "" if all_day else _fmt_time(end),
            "past": end <= datetime.now(_tz()),
        })
    return {"status": "connected", "events": out}


def briefing_line() -> str | None:
    """Today's events for the morning briefing, or None if not connected."""
    if status() != "connected":
        return None
    return list_events("today")


# ---------- tool schemas ----------
_DAY_HELP = ("Pass the user's words, don't compute dates: 'today', 'tomorrow', a weekday "
             "like 'friday' or 'next tuesday', or an explicit date YYYY-MM-DD.")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_holidays",
            "description": "Official public holidays for a country from Google's holiday calendar. "
                           "Always use this (not web_search) for questions about holidays or days off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": f"Country name; default '{HOME_COUNTRY}'."},
                    "days": {"type": "integer", "description": "How far ahead to look, in days (default 60).", "default": 60},
                    "include_observances": {"type": "boolean", "description": "Also list non-holiday observances.", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List Google Calendar events for a day (and optionally the following days).",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": _DAY_HELP},
                    "days": {"type": "integer", "description": "How many days to include, 1-14. Use 7 for 'this week'.", "default": 1},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": "Create a Google Calendar event. Ask first if the day or time is unclear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "day": {"type": "string", "description": _DAY_HELP},
                    "time": {"type": "string", "description": "Start time as the user said it, e.g. '3pm', '9:30 am', '15:00'."},
                    "duration_minutes": {"type": "integer", "default": 60},
                    "location": {"type": "string"},
                },
                "required": ["title", "day", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_free_time",
            "description": f"Find free slots between {WORK_START}:00 and {WORK_END}:00 on a day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": _DAY_HELP},
                    "duration_minutes": {"type": "integer", "default": 30},
                },
                "required": ["day"],
            },
        },
    },
]

TOOL_MAP = {
    "list_holidays":  lambda args: list_holidays(**args),
    "list_events":    lambda args: list_events(**args),
    "add_event":      lambda args: add_event(**args),
    "find_free_time": lambda args: find_free_time(**args),
}
