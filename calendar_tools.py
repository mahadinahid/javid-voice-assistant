"""
Google Calendar for Javid — read the agenda, add / move / delete events, find free time.

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
    _lock_down(CREDENTIALS_FILE)
    _lock_down(TOKEN_FILE)
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


def _lock_down(path: str) -> None:
    """Keep the OAuth files owner-only (they hold a refresh token / client secret)."""
    try:
        if os.stat(path).st_mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


def _save(creds: Credentials) -> None:
    # Create with 0600 so the token is never briefly world-readable, then fix a file from before.
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())
    os.chmod(TOKEN_FILE, 0o600)


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


# ---------- finding, moving and deleting events ----------
# Events are found by name in Python (never by model-supplied IDs). Zero or several
# matches are reported back instead of guessed at. Deleting takes two user turns.
_GENERIC = {"my", "the", "a", "an", "event", "appointment", "meeting", "with", "at", "on",
            "for", "to", "of", "and"}
SEARCH_DAYS = 30       # how far ahead an event is looked for when no day is given

_turn = 0              # bumped once per user message (javid.think_stream)
_pending_delete = None  # {"id": event id, "turn": turn it was asked about}


def new_turn() -> None:
    global _turn
    _turn += 1


def _when(start: datetime, end: datetime, all_day: bool) -> str:
    day = start.strftime("%A, %B %d")
    return f"{day}, all day" if all_day else f"{day} from {_fmt_time(start)} to {_fmt_time(end)}"


def _describe(ev: dict) -> str:
    return f"'{ev.get('summary', '(no title)')}' on {_when(*_event_times(ev))}"


def _find_events(title: str, day: str = "", time: str = "") -> list[dict]:
    if day:
        start, end = _day_bounds(_parse_day(day))
    else:
        start, end = _day_bounds(date.today(), SEARCH_DAYS)
    words = [w for w in re.findall(r"[a-z0-9']+", (title or "").lower()) if w not in _GENERIC]
    hits = [ev for ev in _fetch(start, end)
            if all(w in ev.get("summary", "").lower() for w in words)]
    if time:
        hour, minute = _parse_time(time)
        exact = bool(re.search(r"[ap]\.?m", time.lower()))   # "3" alone could be 3 AM or 3 PM

        def at_time(ev):
            s, _, all_day = _event_times(ev)
            return not all_day and s.minute == minute and (
                s.hour == hour if exact or hour > 12 else s.hour % 12 == hour % 12)
        hits = [ev for ev in hits if at_time(ev)]
    return hits


def _pick(title: str, day: str, time: str) -> tuple[dict | None, str | None]:
    """The one event meant, or (None, a sentence explaining why not)."""
    hits = _find_events(title, day, time)
    if not hits and (day or time):
        # The model often fills optional day/time with a guess ("today") the user never
        # said. A wrong hint shouldn't hide the event, so retry on the title alone;
        # callers show the event's real day and time before anything is changed.
        hits = _find_events(title)
    if not hits:
        return None, f"I couldn't find an event matching '{title}' in the next {SEARCH_DAYS} days."
    if len(hits) > 1:
        same = [ev for ev in hits if ev.get("summary", "").strip().lower() == (title or "").strip().lower()]
        if len(same) == 1:
            return same[0], None
        rows = "\n".join(f"- {_describe(ev)}" for ev in hits[:6])
        return None, (f"{len(hits)} events match '{title}':\n{rows}\n"
                      "Nothing was changed. Ask which one; its day or time tells them apart.")
    return hits[0], None


def _side_notes(ev: dict) -> str:
    notes = []
    if ev.get("recurringEventId"):
        notes.append("This was one occurrence of a repeating event; the others are untouched.")
    if any(not a.get("self") for a in ev.get("attendees", [])):
        notes.append("It has guests, and they were not notified.")
    return (" " + " ".join(notes)) if notes else ""


@_guard
def move_event(title: str, new_day: str = "", new_time: str = "", day: str = "", time: str = "") -> str:
    """Reschedule an event to another day and/or time, keeping its length."""
    if not (new_day or new_time):
        return "Tell me where to move it: a new day, a new time, or both."
    ev, problem = _pick(title, day, time)
    if problem:
        return problem
    start, end, all_day = _event_times(ev)
    target = _parse_day(new_day) if new_day else start.date()
    if all_day:
        if new_time:
            return f"{_describe(ev)} is an all-day event, so it can only move to another day."
        new_start = datetime.combine(target, datetime.min.time(), _tz())
        new_end = new_start + (end - start)
        body = {"start": {"date": target.isoformat()}, "end": {"date": new_end.date().isoformat()}}
    else:
        hour, minute = _parse_time(new_time) if new_time else (start.hour, start.minute)
        new_start = datetime.combine(target, datetime.min.time(), _tz()).replace(hour=hour, minute=minute)
        new_end = new_start + (end - start)
        body = {"start": {"dateTime": new_start.isoformat()}, "end": {"dateTime": new_end.isoformat()}}
        for key in ("start", "end"):
            if ev[key].get("timeZone"):
                body[key]["timeZone"] = ev[key]["timeZone"]
    if new_start == start:
        return f"{_describe(ev)} is already then. Nothing changed."

    def at(t: datetime) -> str:
        return t.strftime("%A, %B %d") + (" (all day)" if all_day else f" at {_fmt_time(t)}")

    _service().events().patch(calendarId=CALENDAR_ID, eventId=ev["id"], body=body).execute()
    clashes = [e.get("summary", "(no title)") for e in _fetch(new_start, new_end)
               if e["id"] != ev["id"] and not _event_times(e)[2]] if not all_day else []
    note = f" Heads up: it overlaps with {', '.join(clashes)}." if clashes else ""
    ends = "" if all_day else f", ending at {_fmt_time(new_end)}"
    if day and _parse_day(day) != start.date():
        note += f" Note: it was on {start.strftime('%A, %B %d')}, not {day}."
    return (f"Moved '{ev.get('summary', '(no title)')}' from {at(start)} to {at(new_start)}{ends}."
            f"{note}{_side_notes(ev)}")


@_guard
def delete_event(title: str, day: str = "", time: str = "", confirm: bool = False) -> str:
    """Delete an event — only after the user has confirmed on their next message."""
    global _pending_delete
    ev, problem = _pick(title, day, time)
    if problem:
        return problem
    confirmed = (confirm and _pending_delete is not None and _pending_delete["id"] == ev["id"]
                 and _pending_delete["turn"] + 1 == _turn)
    if not confirmed:
        _pending_delete = {"id": ev["id"], "turn": _turn}
        return (f"Found {_describe(ev)}. NOT deleted yet. Ask the user, using exactly this day and "
                f"time: \"I found {_describe(ev)}. Delete it?\" Only after they answer yes, call "
                "delete_event again with the same arguments and confirm=true.")
    _service().events().delete(calendarId=CALENDAR_ID, eventId=ev["id"]).execute()
    _pending_delete = None
    return f"Deleted {_describe(ev)}.{_side_notes(ev)}"


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
            "name": "move_event",
            "description": "Reschedule an existing calendar event to another day and/or time (keeps its "
                           "length). Identify it by title, plus its current day or time if there could "
                           "be several.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Part of the event's title, as the user said it."},
                    "day": {"type": "string", "description": "Its CURRENT day. Leave out unless the user said it. " + _DAY_HELP},
                    "time": {"type": "string", "description": "Its CURRENT start time. Leave out unless the user said it."},
                    "new_day": {"type": "string", "description": "The day to move it to. " + _DAY_HELP},
                    "new_time": {"type": "string", "description": "The new start time as the user said it, e.g. '4pm'."},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Delete a calendar event. The first call only finds it; the user must confirm "
                           "(their next message), then call again with confirm=true and the same arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Part of the event's title, as the user said it."},
                    "day": {"type": "string", "description": "The event's day. Leave out unless the user said it. " + _DAY_HELP},
                    "time": {"type": "string", "description": "Its start time. Leave out unless the user said it."},
                    "confirm": {"type": "boolean", "description": "true only after the user said yes.", "default": False},
                },
                "required": ["title"],
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
    "move_event":     lambda args: move_event(**args),
    "delete_event":   lambda args: delete_event(**args),
    "find_free_time": lambda args: find_free_time(**args),
}
