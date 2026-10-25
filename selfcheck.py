"""
Javid answering for itself from what it recorded, never from the model: a self-check ("what's wrong?", "why
can't you hear me?", "evaluate the bugs in your system") and where the last answer came from ("where did you get
that?", "did you get that from Google?").

Why: on 2026-10-22 Mahadi asked "Can you evaluate the bug problems in your system so that I can fix the problem?",
then "What's wrong? Why cannot you listen and answer?". The model can't see Javid's state, so the first got "I'm not
able to review my own code" and the second a figure-check fallback about a Reuters story. The real cause was in
the server log: fifteen sentences dropped as hands-free clips without "Javid" after the follow-up window. Javid
now keeps what it needs to answer that itself: what happened to each clip it heard (note_clip), each voice
synthesis (note_tts) and each transcription failure (note_stt_error), in memory only; plus what it already
tracks (model cooldowns and quota, web search errors and credits, Google News, the calendar, Telegram, latency).

request() names the kind of question; answer() is a generator (it yields the self-check's "tool" event) that
returns (spoken, screen). The live probes (Groq's model list, one Google News feed) are free and run in parallel.
"""
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

WINDOW_SEC = 15 * 60          # "recently" for clips and transcription errors
VOICE_WINDOW_SEC = 3600
PROBE_SEC = 4.0

_clips: deque = deque(maxlen=300)    # (ts, mode, outcome, transcript): every /talk clip
_tts: deque = deque(maxlen=300)      # (ts, ok, kind): every edge-tts synthesis (cache hits aren't)
_stt: deque = deque(maxlen=50)       # (ts, kind): transcriptions that failed

# outcome: "answered", "ignored" (hands-free without "Javid", outside the conversation window), "noise",
# "not_caught" (one word over push-to-talk), "ack" (a bare "Javid").
def note_clip(mode: str, outcome: str, transcript: str) -> None:
    _clips.append((time.time(), mode, outcome, (transcript or "")[:120]))


def note_tts(ok: bool, kind: str = "") -> None:
    _tts.append((time.time(), ok, kind))


def note_stt_error(kind: str) -> None:
    _stt.append((time.time(), kind))


# ---------- which question ----------
_END = r"(?=\s*(?:[?.!,]|$))"      # the phrase ends its sentence: "what's wrong?" but not "what's wrong with the economy"
_NEG = r"(?:can'?t|cannot|couldn'?t|don'?t|didn'?t|won'?t|aren'?t)"
_FAIL_VERB = (r"(?:hear|listen|answer|respond|reply|understand|work|get|pull|bring|find|fetch|search|reach|load|speak|"
              r"talk|catch|open|connect)")
_DIAGNOSE = re.compile(
    r"\bwhat(?:'?s| is)\s+(?:wrong|the (?:problem|issue|matter))(?:\s+with\s+(?:you|javid|yourself|the (?:system|app|"
    rf"mic|microphone|hands-?free|voice)))?{_END}|"
    rf"\bwhy\b.{{0,20}}\b{_NEG}\s+(?:you|javid)\s+(?:just\s+)?(?:\w+\s+)?{_FAIL_VERB}|"            # why can't you hear me
    rf"\bwhy\b.{{0,15}}\b(?:you|javid)\b.{{0,12}}\b(?:{_NEG}|not)\s+(?:\w+\s+){{0,2}}{_FAIL_VERB}|"  # why you cannot pull...
    rf"\b(?:you'?re|you are|javid is|you)\s+(?:{_NEG}|isn'?t|not|never)\s+(?:\w+\s+)?(?:hear|listen|answer|respond|work)\w*\b|"
    r"\bself[- ]?(?:check|test|diagnos\w*)\b|\brun (?:a |your )?(?:diagnostics?|health check|system check|self[- ]?check)\b|"
    rf"\bis (?:everything|anything|something) (?:ok|okay|fine|alright|all right|working|wrong|broken){_END}|"
    r"\b(?:system|health) (?:status|check|report)\b|\bstatus report\b|"
    r"\bwhy (?:are|were|is) (?:you|javid|it|everything) (?:so )?slow\b|\b(?:you'?re|you are) (?:so |too |very )?slow\b|"
    rf"\bare you (?:broken|ok|okay|alright|working){_END}|"
    r"কী সমস্যা|সমস্যা কী|শুনতে পাচ্ছ না|শুনছ না|উত্তর দিচ্ছ না|কাজ করছ না|নিজেকে পরীক্ষা", re.I)
# "Evaluate the bug problems in your system" (2026-10-22): only about Javid's own system, not "fix the problem with my car".
_FIX = re.compile(r"\b(?:check|evaluate|diagnose|test|inspect|examine|find|fix|debug)\s+(?:the\s+|your\s+|my\s+)?(?:own\s+)?"
                  r"(?:system|bugs?|bug problems?|problems?|issues?|errors?|health|status)\b", re.I)
_ABOUT_YOU = re.compile(r"\byour\b|\byourself\b|\bjavid\b|\bsystem\b|\bthe app\b|তোমার|নিজেকে", re.I)
_PROVENANCE = re.compile(
    r"\bwhere did (?:you|that|this|it|those|these|they)\s+(?:\w+\s+)?(?:get|find|come|hear|read|learn|take|pull|bring)\w*\b|"
    r"\b(?:what'?s|what is|what was|what were|which (?:is|was|were)|give me|show me|tell me|name)\s+(?:your|the)\s+"
    rf"sources?(?:\s+(?:for|of|on|behind)\s+(?:that|this|it|those|these))?{_END}|"
    rf"\bsources? (?:for|of|on|behind) (?:that|this|it|those|these){_END}|^\W*sources?\W*$|"
    rf"\bhow (?:do|did) you know (?:that|this|it|those|these){_END}|"
    rf"\bhow did you (?:find (?:that|this|it) out|get (?:that|this|it)){_END}|"
    r"\b(?:who|which (?:site|website|outlet|paper|newspaper|source|channel)) (?:said|says|reported|reports|wrote|published) "
    rf"(?:that|this|it|those|these){_END}|"
    r"\bdid you\s+(?:just\s+)?(?:get|bring|find|pull|take|read|search|look|fetch)\s+(?:\w+\s+){0,4}?(?:out\s+)?(?:from|on|in|up on|at)\s+"
    r"(?:the\s+)?(?:google|internet|web|online)\b|\bis (?:that|this|it) from (?:the\s+)?(?:google|internet|web)\b|"
    r"\bdid you (?:just )?make (?:that|this|it|those|these) up\b|\bare you making (?:that|this|it) up\b|"
    rf"\bdid you (?:look|search) (?:that|this|it) up{_END}|\bdid you (?:check|verify|confirm) (?:that|this|it){_END}|"
    r"কোথা থেকে (?:পেলে|জানলে|আনলে)|সোর্স কী|উৎস কী|কোন সোর্স", re.I)
_FOCUS = [("hearing", re.compile(r"hear|listen|answer|respond|reply|catch|understand|mic|শুন|উত্তর", re.I)),
          ("news", re.compile(r"news|headline|search|internet|web|google|online|খবর|সার্চ", re.I)),
          ("speed", re.compile(r"slow|late|delay|lag|ধীর|দেরি", re.I)),
          ("voice", re.compile(r"voice|speak|talk|sound|silent|কণ্ঠ|ভয়েস", re.I))]


def request(text: str) -> str | None:
    """"diagnose", "provenance", or None. A question about Google as a design choice ("why don't you search
    Google?") is about.py's."""
    text = text or ""
    if _PROVENANCE.search(text):
        return "provenance"
    if re.search(r"\bgoogle\b(?!.*\bnews\b)", text, re.I):
        return None
    if _DIAGNOSE.search(text) or (_FIX.search(text) and _ABOUT_YOU.search(text)):
        return "diagnose"
    return None


# ---------- the self-check ----------
def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def _ago(ts: float) -> str:
    mins = int((time.time() - ts) // 60)
    return "just now" if mins < 1 else f"{mins} min ago" if mins < 60 else f"at {_clock(ts)}"


def _probe_groq() -> tuple[bool, str]:
    import javid
    t = time.time()
    try:
        javid.client.with_options(timeout=PROBE_SEC, max_retries=0).models.list()
        return True, f"{(time.time() - t) * 1000:.0f} ms"
    except Exception as e:
        return False, type(e).__name__


def _probe_news() -> tuple[bool, str]:
    import gnews
    t = time.time()
    try:
        items = gnews.fetch(gnews.feed_url(("section", "WORLD")), fresh=True)
        return bool(items), f"{(time.time() - t) * 1000:.0f} ms" if items else "no stories"
    except Exception as e:
        return False, type(e).__name__


def _probe_calendar() -> str:
    import calendar_tools
    try:
        return calendar_tools.status()
    except Exception as e:
        return f"error ({type(e).__name__})"


def _hearing(now: float) -> dict:
    recent = [c for c in _clips if now - c[0] < WINDOW_SEC]
    count = {k: sum(c[2] == k for c in recent) for k in ("answered", "ignored", "noise", "not_caught", "ack")}
    errors = [s for s in _stt if now - s[0] < WINDOW_SEC]
    ignored = [c for c in recent if c[2] == "ignored"]
    line = (f"{len(recent)} clips in the last 15 min: {count['answered']} answered"
            + (f", {count['ignored']} ignored (no \"Javid\", after the 30-second window)" if count["ignored"] else "")
            + (f", {count['noise']} only noise" if count["noise"] else "")
            + (f", {count['not_caught']} too short" if count["not_caught"] else "")) if recent else "No clips in the last 15 min"
    if ignored:
        line += f"\n  Last ignored: \"{ignored[-1][3]}\" ({_ago(ignored[-1][0])})"
    if errors:
        line += f"\n  Speech recognition failed {len(errors)} time(s) ({errors[-1][1]}, {_ago(errors[-1][0])})"
    problem = bn = None
    if errors:
        problem = (f"Speech recognition failed {len(errors)} time{'s' if len(errors) > 1 else ''} in the last 15 "
                   f"minutes, so I couldn't hear those clips; Groq's Whisper service didn't answer.")
        bn = f"গত ১৫ মিনিটে {_bn(len(errors))} বার কথা চেনার সার্ভিস (Whisper) কাজ করেনি, তাই সেগুলো শুনতে পাইনি।"
    elif len(ignored) >= 1:
        last = ignored[-1]
        problem = (f"I ignored {len(ignored)} thing{'s' if len(ignored) > 1 else ''} I heard in the last 15 minutes, "
                   f"because {'they' if len(ignored) > 1 else 'it'} didn't start with 'Javid' and came after the "
                   f"30-second window; the last was \"{_short(last[3])}\". Say 'Javid' first, or hold the talk button.")
        bn = (f"গত ১৫ মিনিটে {_bn(len(ignored))}টি কথা শুনেও উত্তর দিইনি, কারণ সেগুলো 'জাভেদ' দিয়ে শুরু হয়নি আর "
              f"৩০ সেকেন্ডের সময় পেরিয়ে গিয়েছিল। আগে 'জাভেদ' বলো, অথবা টক বোতাম চেপে ধরো।")
    n = count["answered"] + count["ack"]
    ok_said = ("I haven't heard anything in the last 15 minutes; if you spoke, check that the microphone is on "
               "and the page shows it listening." if not recent else
               f"I heard you {n} time{'s' if n != 1 else ''} in the last 15 minutes and answered every one." if n else
               "In the last 15 minutes I only heard noise." if not problem else "")
    return {"name": "Hearing", "ok": problem is None, "line": f"Whisper ({_stt_model()}) · {line}",
            "problem": problem, "problem_bn": bn, "ok_said": ok_said}


def _bn(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))


def _short(text: str, words: int = 8) -> str:
    w = text.split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def _stt_model() -> str:
    import javid
    return javid.STT_MODEL


def _brain(groq_probe: tuple[bool, str]) -> dict:
    import javid
    st = javid.llm_status()
    models = st["models"]
    primary = models[0] if models else None
    serving = next((m for m in models if m["state"] == "serving"), None)
    parts, problem, bn = [], None, None
    for m in models[:3]:
        left = (f", {round(100 * m['remaining_24h'] / m['tpd_limit'])}% of today's tokens left"
                if m.get("tpd_limit") and m.get("remaining_24h") is not None else "")
        state = {"serving": "answering", "ready": "ready", "cooling": f"resting {m['retry_in_s'] // 60 + 1} min",
                 "off": "at its free limit", "reserve": "in reserve", "nokey": "no key",
                 "missing": "not on this key", "disabled": "turned off"}.get(m["state"], m["state"])
        parts.append(f"{m['name']} ({state}{left}{'' if m.get('tools') else ', chat only'})")
    if not serving:
        problem = "None of my language models can answer right now: they're all resting or out of today's quota."
        bn = "এখন কোনো ল্যাঙ্গুয়েজ মডেলই উত্তর দিতে পারছে না: সবগুলো বিশ্রামে বা আজকের কোটা শেষ।"
    elif primary and serving["id"] != primary["id"]:
        why = {"cooling": f"is resting for about {primary['retry_in_s'] // 60 + 1} minutes after a limit or an error",
               "off": "is at 95 percent of its free allowance, so it's off until that refills",
               "reserve": "is past 80 percent of its free allowance, so it only answers when the others can't",
               "nokey": "has no key", "disabled": "is turned off in the providers file"}.get(primary["state"], "isn't available")
        problem = f"My main model, {primary['name']}, {why}, so {serving['name']} is answering."
        bn = f"আমার মূল মডেল {primary['name']} এখন পাওয়া যাচ্ছে না, তাই {serving['name']} উত্তর দিচ্ছে।"
    # The tool gate (javid.tools_allowed): a chat-only model can talk, but tasks, calendar, memory and searches need one
    # that passed its tool tests.
    if serving and not serving.get("tools") and not any(
            m.get("tools") and m["state"] in ("serving", "ready", "reserve") for m in models):
        problem = ((problem + " " if problem else "") + f"{serving['name']} can only chat: none of the models I let use "
                   "tools can answer now, so tasks, the calendar, memory and searches wait until one is back.")
        bn = ((bn + " " if bn else "") + f"{serving['name']} শুধু কথা বলতে পারে: টুল ব্যবহারের অনুমতি থাকা কোনো মডেল এখন "
              "উত্তর দিতে পারছে না, তাই কাজ, ক্যালেন্ডার, মেমরি আর সার্চ অপেক্ষায়।")
    if not groq_probe[0]:
        problem = (problem + " " if problem else "") + f"Groq didn't answer my check just now ({groq_probe[1]})."
        bn = (bn + " " if bn else "") + "Groq এখন সাড়া দিচ্ছে না।"
    return {"name": "Brain", "ok": problem is None,
            "line": " · ".join(parts) + f" · Groq {'reachable, ' + groq_probe[1] if groq_probe[0] else 'unreachable: ' + groq_probe[1]}",
            "problem": problem, "problem_bn": bn, "ok_said": f"{serving['name']} is answering." if serving else ""}


def _voice(now: float) -> dict:
    import javid
    import tracing
    recent = [v for v in _tts if now - v[0] < VOICE_WINDOW_SEC]
    failed = [v for v in recent if not v[1]]
    stats = tracing.stats()
    chat = stats.get("chat", {}).get("ttfa_median")
    line = (f"{javid.EDGE_VOICE} / {javid.EDGE_VOICE_BN} via edge-tts · last hour: {len(recent)} sentences"
            + (f", {len(failed)} failed tries ({failed[-1][2]}, {_ago(failed[-1][0])})" if failed else ", none failed"))
    if chat:
        line += f" · chat replies start speaking after {chat / 1000:.1f} s (median)"
    problem = bn = None
    if len(failed) >= 2:
        problem = (f"The voice service failed {len(failed)} times in the last hour; I retry three times, then show the "
                   "reply as text, never another voice.")
        bn = (f"গত এক ঘণ্টায় ভয়েস সার্ভিস {_bn(len(failed))} বার কাজ করেনি; আমি তিনবার চেষ্টা করি, তারপর লেখা দেখাই, "
              "অন্য কোনো ভয়েস না।")
    return {"name": "Voice", "ok": problem is None, "line": line, "problem": problem, "problem_bn": bn,
            "ok_said": "the voice is working"}


def _news(now: float, probe: tuple[bool, str]) -> dict:
    import web_tools
    web = web_tools.status()
    err = web.get("last_error")
    recent_err = err if err and now - err["at"] < 3 * 3600 else None
    line = (f"Google News {'reachable, ' + probe[1] if probe[0] else 'unreachable: ' + probe[1]} · Tavily web search: "
            f"{web['used_today']} of {web['cap']} credits today"
            + (f", {web['month']['used']:,} of {web['month']['amount']:,.0f} this month"
               if (web.get("month") or {}).get("amount") and web["month"].get("used") is not None else "")
            + (f", last error {recent_err['kind']} {_ago(recent_err['at'])}" if recent_err else "")
            + ("" if web["has_key"] else ", no key"))
    problem = bn = None
    if not probe[0]:
        problem = f"I can't reach Google News right now ({probe[1]}), so headlines may fail; check the internet connection."
        bn = "এখন Google News-এ পৌঁছাতে পারছি না, তাই খবর আসতে না-ও পারে; ইন্টারনেট সংযোগ দেখো।"
    elif recent_err:
        problem = (f"The web search service, Tavily, failed {_ago(recent_err['at'])} ({recent_err['kind']}). News now "
                   "comes from Google News, which is working, so ask me again.")
        bn = "ওয়েব সার্চ সার্ভিস Tavily একটু আগে কাজ করেনি; খবর এখন Google News থেকে আসে, সেটা চলছে, আবার জিজ্ঞেস করো।"
    elif web["used_today"] >= web["cap"]:
        problem = "Today's web search budget is used up; news still works through Google News."
        bn = "আজকের ওয়েব সার্চের বাজেট শেষ; খবর তবুও Google News থেকে আসবে।"
    return {"name": "News and web", "ok": problem is None, "line": line, "problem": problem, "problem_bn": bn,
            "ok_said": "news and web search are working"}


def _calendar(state: str) -> dict:
    ok = state == "connected"
    problem = None if ok else ("Your Google Calendar needs a fresh sign-in: press Connect Google Calendar in the Today panel."
                               if state == "needs_auth" else "Your Google Calendar isn't set up."
                               if state == "no_credentials" else f"The calendar check failed ({state}).")
    return {"name": "Calendar", "ok": ok, "line": {"connected": "Google Calendar connected", "needs_auth": "needs a fresh sign-in",
                                                    "no_credentials": "not set up"}.get(state, state),
            "problem": problem, "problem_bn": None if ok else "তোমার Google Calendar সংযুক্ত নেই: Today প্যানেলে Connect চাপো।",
            "ok_said": "your calendar is connected"}


def _telegram() -> dict:
    try:
        import telegram_bot
        st = telegram_bot.status()
    except Exception as e:
        return {"name": "Telegram", "ok": True, "line": f"status unavailable ({type(e).__name__})", "problem": None,
                "problem_bn": None, "ok_said": ""}
    if not st["configured"]:
        line = "not set up"
    else:
        line = ("running" if st["running"] else "stopped") + (", paired" if st["paired"] else ", not paired") + \
               (", locked from the phone" if st["locked"] else "")
    return {"name": "Telegram", "ok": True, "line": line, "problem": None, "problem_bn": None, "ok_said": ""}


def diagnose(text: str, lang: str):
    """The self-check: a generator (its "tool" event) returning (spoken, screen)."""
    import tracing
    yield ("tool", "self_check")
    span = tracing.tool_start("self_check")
    now = time.time()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            groq_f, news_f, cal_f = pool.submit(_probe_groq), pool.submit(_probe_news), pool.submit(_probe_calendar)

            def get(f, default):
                try:
                    return f.result(timeout=PROBE_SEC + 1)
                except Exception as e:
                    return default(e)
            groq = get(groq_f, lambda e: (False, type(e).__name__))
            news_probe = get(news_f, lambda e: (False, type(e).__name__))
            cal = get(cal_f, lambda e: f"no answer ({type(e).__name__})")
        checks = [_hearing(now), _brain(groq), _voice(now), _news(now, news_probe), _calendar(cal), _telegram()]
    finally:
        tracing.tool_end(span)
    focus = next((name for name, p in _FOCUS if p.search(text)), None)
    order = {"hearing": "Hearing", "news": "News and web", "speed": "Voice", "voice": "Voice"}.get(focus)
    checks.sort(key=lambda c: (c["name"] != order, c["ok"]))      # the asked-about part first, then problems
    problems = [c for c in checks if not c["ok"]]
    bn = lang == "bn"
    if problems:
        said = [c["problem_bn"] if bn else c["problem"] for c in problems[:3]]
        if focus and order and all(c["name"] != order for c in problems):
            first = next(c for c in checks if c["name"] == order)
            if first["ok_said"] and not bn:
                said.insert(0, f"{first['ok_said'][0].upper()}{first['ok_said'][1:]}")
        lead = (f"নিজেকে পরীক্ষা করে {_bn(len(problems))}টি সমস্যা পেয়েছি, মাহাদি।" if bn else
                f"I checked myself and found {len(problems)} thing{'s' if len(problems) > 1 else ''}, Mahadi.")
        spoken = " ".join([lead] + said)
    else:
        hearing = next(c for c in checks if c["name"] == "Hearing")
        spoken = ("নিজেকে পরীক্ষা করেছি, মাহাদি: শোনা, মডেল, ভয়েস, খবর আর ক্যালেন্ডার সব ঠিক আছে।" if bn else
                  "I checked myself, Mahadi: hearing, my model, the voice, the news and your calendar are all "
                  f"working. {hearing['ok_said']}".strip())
    spoken += " পুরো রিপোর্ট স্ক্রিনে আছে।" if bn else " The full report is on your screen."
    screen = "\n".join([f"Self-check · {_clock(now)}:"] +
                       [f"{'✓' if c['ok'] else '⚠'} {c['name']}: {c['line']}" for c in checks])
    lat = tracing.stats()
    if any(v.get("ttfa_median") for v in lat.values()):
        screen += "\nLatency, first audio (median of recent requests): " + " · ".join(
            f"{k} {v['ttfa_median'] / 1000:.1f} s" for k, v in lat.items() if v.get("ttfa_median"))
    return spoken, screen


# ---------- where the last answer came from ----------
_TOOL_SOURCES = {
    "market_price": ("live market data from Binance and Yahoo Finance", "Binance আর Yahoo Finance-এর লাইভ দাম"),
    "exchange_rates": ("the European Central Bank's reference rates and ExchangeRate-API",
                       "ইউরোপিয়ান সেন্ট্রাল ব্যাংক আর ExchangeRate-API-এর রেট"),
    "get_weather": ("the Open-Meteo forecast", "Open-Meteo-র আবহাওয়ার পূর্বাভাস"),
    "list_events": ("your Google Calendar", "তোমার Google Calendar"), "find_free_time": ("your Google Calendar", "তোমার Google Calendar"),
    "add_event": ("your Google Calendar", "তোমার Google Calendar"), "move_event": ("your Google Calendar", "তোমার Google Calendar"),
    "delete_event": ("your Google Calendar", "তোমার Google Calendar"),
    "list_holidays": ("Google's public holiday calendar for Bangladesh", "Google-এর বাংলাদেশের ছুটির ক্যালেন্ডার"),
    "list_tasks": ("your task list", "তোমার কাজের তালিকা"), "add_task": ("your task list", "তোমার কাজের তালিকা"),
    "complete_task": ("your task list", "তোমার কাজের তালিকা"), "delete_task": ("your task list", "তোমার কাজের তালিকা"),
    "morning_briefing": ("your tasks, calendar and the weather", "তোমার কাজ, ক্যালেন্ডার আর আবহাওয়া"),
    "calculate": ("my calculator, in code, so the arithmetic is exact", "আমার কোডের ক্যালকুলেটর"),
    "get_datetime": ("your Mac's clock", "তোমার ম্যাকের ঘড়ি"),
    "search_history": ("our past chats", "আমাদের আগের চ্যাট"),
    "set_timer": ("my timers", "আমার টাইমার"), "timers": ("my timers", "আমার টাইমার"),
    "self_check": ("my own self-check", "আমার নিজের পরীক্ষা"),
    "watch": ("my watch list", "আমার নজরদারির তালিকা"),
}


def _and(items: list[str], lang: str) -> str:
    joiner = " আর " if lang == "bn" else " and "
    return items[0] if len(items) < 2 else ", ".join(items[:-1]) + joiner + items[-1]


def provenance(text: str, lang: str, previous: dict | None):
    """Where the previous answer came from, from what that turn recorded (javid.py keeps its tools and sources on
    the reply): a generator returning (spoken, screen). The sources are shown again as links."""
    import javid
    import web_tools
    yield from ()       # a generator like the self-check, though nothing here is fetched
    bn = lang == "bn"
    if not previous or not previous.get("content"):
        return ("এই চ্যাটে আগে কোনো উত্তর নেই, তাই উৎস বলার কিছু নেই।" if bn else
                "There's no earlier answer in this chat to give a source for."), ""
    tools = [t for t in dict.fromkeys(previous.get("tools") or []) if t not in ("use_skill", "delegate")]
    sources = previous.get("sources") or []
    asked_google = bool(re.search(r"\bgoogle\b", text, re.I))
    parts, lines = [], []
    if sources:
        names = list(dict.fromkeys(s["name"] for s in sources))[:3]
        via_google = any(s.get("via") == "Google News" for s in sources)
        dates = sorted({s["date_text"] for s in sources if s.get("date_text") and not s["date_text"].startswith("retrieved")})
        when = f", {_and(dates[:2], lang)}" if dates and not bn else ""
        if via_google:
            parts.append(("Google News-এ " + _and(names, lang) + "-এর শিরোনাম") if bn else
                         f"Google News: headlines from {_and(names, lang)}{when}")
        else:
            engines = list(dict.fromkeys(e for s in sources for e in s.get("engines") or [])) or ["Tavily"]
            parts.append((_and(engines, lang) + " ওয়েব সার্চে " + _and(names, lang) + "-এর খবর") if bn else
                         f"a web search through {_and(engines, lang)}: {_and(names, lang)}{when}")
        web_tools.restore_sources(sources)
        lines = [f"{s['n']}. {s['name']}: {s['title']} ({s.get('date_text', '')})" for s in sources]
    seen = set()
    for t in tools:
        if t in _TOOL_SOURCES and _TOOL_SOURCES[t][0] not in seen:
            seen.add(_TOOL_SOURCES[t][0])
            parts.append(_TOOL_SOURCES[t][1 if bn else 0])
    model = javid.next_model().split(" on ")[0]
    if not parts:
        spoken = (f"ওটা কোনো সার্চ বা টুল থেকে না, ল্যাঙ্গুয়েজ মডেলের ({model}) নিজের জানা থেকে, তাই যাচাই করা নয়। "
                  "চাইলে বলো, খুঁজে দেখি।" if bn else
                  f"That one didn't come from a search or a tool: it was the language model's own knowledge ({model}), "
                  "so it isn't checked. Say 'look it up' and I'll search.")
        return spoken, ""
    if bn:
        spoken = f"ওটা এসেছে {_and(parts, 'bn')}।" + (" লিংকগুলো স্ক্রিনে আছে।" if sources else "")
    else:
        lead = ("Yes, that came from " if asked_google and any(s.get("via") == "Google News" for s in sources) else
                "No, not from Google: that came from " if asked_google else "That came from ")
        spoken = lead + _and(parts, "en") + "." + (" The links are on your screen." if sources else "")
    screen = ("Sources of the last answer:\n" + "\n".join(lines)) if lines else ""
    return spoken, screen


def answer(kind: str, text: str, lang: str, previous: dict | None):
    if kind == "provenance":
        return provenance(text, lang, previous)
    return diagnose(text, lang)
