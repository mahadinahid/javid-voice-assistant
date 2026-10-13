"""
Regression check for Javid's agent behaviour: 20 prompts through the real LLM + tool loop.

    python regression.py              # run everything once
    python regression.py -n 3         # repeat each case 3x (the model is non-deterministic)
    python regression.py -k delete    # only cases whose name contains "delete"
    python regression.py -v           # print every reply, not just failures

Safe to run: tasks, memory and chat history go to a temp dir, the Google Calendar is a
fake in-memory service, and web search is stubbed. Your real data is never touched.
Needs GROQ_API_KEY. A full run makes ~40-50 LLM requests (~100k tokens): on Groq's free tier
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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_tools as ct  # noqa: E402
import javid  # noqa: E402
from groq import Groq  # noqa: E402

# A stalled stream must fail the run, not hang the suite.
javid.client = Groq(timeout=30.0, max_retries=2)
RUN_TIMEOUT_SEC = 150      # hard cap for one case run
INFRA_RETRIES = 2          # re-runs when the API itself failed (not the assistant)
API_FAILED_REPLY = "I couldn't complete that request"   # think_stream's canned reply on APIError


class InfraError(Exception):
    """The Groq API (network, timeout, 5xx) failed — says nothing about Javid's behaviour."""


# Count LLM requests, and turn a 429 into a hard stop (each request costs ~2.3k+ tokens of the daily cap).
LLM_REQUESTS = 0
QUOTA_HIT: list[str] = []
_orig_stream_completion = javid._stream_completion


PER_MINUTE_WAIT_MAX = 90   # a 429 that clears within this is a per-minute limit: wait it out


def _counted_stream_completion(tool_calls):
    """Count requests; ride out per-minute 429s (Gemini's free tier allows 5/min per model) and
    treat a longer wait (daily cap) as 'out of quota' so the suite stops instead of lying."""
    global LLM_REQUESTS
    for attempt in range(6):
        LLM_REQUESTS += 1
        try:
            yield from _orig_stream_completion(tool_calls)
            return
        except javid.RATE_LIMITED as e:
            wait = javid._retry_after(e)
            if wait > PER_MINUTE_WAIT_MAX or attempt == 5:
                QUOTA_HIT.append(str(e))
                raise
            time.sleep(wait + 1)   # no chunk was yielded yet: a 429 arrives when the stream opens


javid._stream_completion = _counted_stream_completion


def preflight() -> str | None:
    """Tiny pings down the model chain; returns an error message only if no model answers."""
    pings = javid.ping_models()
    if any(p["ok"] for p in pings):
        return None
    return "\n".join(f"{p['id']}: {p.get('error', 'failed')[:160]}" for p in pings)

# ---------- isolation: nothing below may touch the real files or calendar ----------
SCRATCH = tempfile.mkdtemp(prefix="javid-regression-")
javid.TASKS_FILE = os.path.join(SCRATCH, "tasks.txt")
javid.MEMORY_FILE = os.path.join(SCRATCH, "memory.json")
javid.HISTORY_FILE = os.path.join(SCRATCH, "chat_log.jsonl")


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


# ---------- case plumbing ----------
@dataclass
class Result:
    turns: list            # user messages
    replies: list          # Javid's reply per turn
    tools: list            # [(turn_index, name, args)]
    tasks: list            # lines of tasks.txt after the run
    facts: list            # memory facts after the run
    cal_log: list          # fake-calendar writes after the run
    cal_log_by_turn: list  # cumulative len(cal_log) after each turn

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


def called(name):
    return lambda r: None if name in r.tool_names else f"expected a {name} call, got {r.tool_names or 'no tools'}"


def not_called(*names):
    return lambda r: next((f"must not call {n}" for n in names if n in r.tool_names), None)


def reply_has(pattern, label=None):
    return lambda r: None if re.search(pattern, r.last, re.I) else f"reply should match /{label or pattern}/"


def reply_lacks(pattern):
    return lambda r: f"reply must not match /{pattern}/" if re.search(pattern, r.last, re.I) else None


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


WEEKDAY = TODAY.strftime("%A")

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
    Case("memory: remember", ["Remember that my sister's name is Nadia."],
         [called("remember"), fact_has("nadia")]),
    Case("memory: uses stored fact", ["What's my dog called?"],
         [reply_has("rex"), not_called("web_search")], facts=["Mahadi's dog is called Rex"]),
    # --- info ---
    Case("info: date and time", ["What day is it today, and what time?"],
         [called("get_datetime"), reply_has(WEEKDAY, f"today's weekday ({WEEKDAY})")]),
    Case("info: web search", ["Search the web for the capital of Australia."],
         [called("web_search"), reply_has("canberra")]),
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
]
assert len(CASES) == 20, len(CASES)


# ---------- runner ----------
def run_case(case: Case) -> Result:
    # Fresh world for every run.
    with open(javid.TASKS_FILE, "w") as f:
        f.write("".join(l + "\n" for l in case.tasks))
    with open(javid.MEMORY_FILE, "w") as f:
        json.dump({"facts": case.facts}, f)
    open(javid.HISTORY_FILE, "w").close()
    svc = FakeService(EVENTS, HOLIDAYS)
    ct._service = lambda: svc
    ct.status = lambda: "connected"
    ct._pending_delete = None
    del javid.messages[1:]

    tools, replies, by_turn = [], [], []
    real_map = dict(javid.TOOL_MAP)
    turn = {"i": 0}

    def spy(name):
        def call(args):
            tools.append((turn["i"], name, dict(args)))
            return real_map[name](args)
        return call
    for name in real_map:
        javid.TOOL_MAP[name] = spy(name)
    javid.TOOL_MAP["web_search"] = lambda args: (
        tools.append((turn["i"], "web_search", dict(args))) or
        "Canberra - Wikipedia: Canberra is the capital city of Australia.\n"
        "Australia - Britannica: Canberra is the capital of Australia, located in the ACT.")
    try:
        for i, text in enumerate(case.turns):
            turn["i"] = i
            with contextlib.redirect_stdout(io.StringIO()):   # silence the [tool] debug prints
                reply = javid.think(text)
            if reply.startswith(API_FAILED_REPLY):
                raise InfraError(f"API error on turn {i + 1}")
            replies.append(reply)
            by_turn.append(len(svc.log))
    finally:
        javid.TOOL_MAP.update(real_map)

    with open(javid.TASKS_FILE) as f:
        tasks = [l for l in f.read().splitlines() if l.strip()]
    return Result(case.turns, replies, tools, tasks, javid.load_memory()["facts"], list(svc.log), by_turn)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--repeat", type=int, default=1, help="runs per case")
    ap.add_argument("-k", "--filter", default="", help="only cases whose name contains this")
    ap.add_argument("-v", "--verbose", action="store_true", help="show replies for passing runs too")
    ap.add_argument("--models", default="",
                    help="comma-separated model ids to use as the chain for this run, e.g. gemini:gemini-3.7-flash")
    args = ap.parse_args()
    if args.models:   # test one provider in isolation: no fallback to Groq masking its failures
        chain_file = os.path.join(SCRATCH, "chain.json")
        with open(chain_file, "w") as f:
            json.dump({"chain": [m.strip() for m in args.models.split(",") if m.strip()]}, f)
        javid.LLM_CHAIN_FILE = chain_file

    if (problem := preflight()):
        print(f"Not running: {problem[:400]}\n\nNothing was tested (exit 3). Wait for the limit to reset and rerun.")
        return 3
    cases = [c for c in CASES if args.filter.lower() in c.name.lower()]
    if not cases:
        print(f"No case matches {args.filter!r}")
        return 2
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
                except Exception as e:   # API outage, timeout, connection reset: not Javid's fault
                    r = None
                    problems = [f"API/network problem: {e!r}"]
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
    print(f"\n{LLM_REQUESTS} LLM requests this run (each is 2.3k+ tokens against Groq's daily limit); served by {served}")
    if len(javid.llm_usage) > 1:
        print("note: the primary model hit its limit mid-run, so results mix models — "
              "rerun later for a clean single-model baseline")
    not_run = len(cases) * args.repeat - passed_runs - failed_runs - errored_runs
    print(f"{passed_runs} passed, {failed_runs} failed, {errored_runs} errored (API/network)"
          + (f", {not_run} not run" if not_run else "")
          + (f"; {flaky} case(s) not clean" if flaky else ""))
    return 1 if failed_runs else (3 if errored_runs else 0)


if __name__ == "__main__":
    sys.exit(main())
