"""
News requests answered in code, with no LLM request: "show me the news", "world news", "10 news from Asia, Europe
and the Middle East", "AI news", "how about the cricket news?", "বিশ্বের খবর দাও". The headlines come from Google
News (gnews.py, Tavily for a topic it couldn't get), are read out with their outlets' names, and the full list
goes on screen with links.

It used to be the news-digest skill: the model chose topics, called news_headlines and rewrote the results into
sentences, and in Mahadi's own chats (2026-10-20/22) that went wrong in every way: TV listings from unknown sites
read out as world news, "Can you show me the news briefing of today?" answered with his empty task list, a
digest replaced by "I couldn't confirm a specific growth figure", and "I couldn't reach the news sites".

request() takes only a message that is all news request: topics it knows (gnews.ALIASES), a count, and filler
words. Anything else in it (a specific story, "why", "tell me more about the third one", a timer, the weather)
leaves the message to the model, which still has news_headlines, web_search and the news-digest skill.
Headlines are quoted as their outlets wrote them; one whose figure no second outlet gives is shown, not spoken
(the same rule as web answers: a figure needs an official source or two).
"""
import math
import re
from datetime import datetime, timezone

import gnews
import skills
import tracing
import web_tools

SPOKEN_SECONDS = 50
MAX_STORIES = 20
SINGLE_TOPIC = 5            # stories when one topic is asked for and no number
PER_TOPIC = 2               # stories per topic when several are asked for
SPOKEN_SINGLE = 4           # headlines read out for one topic
DEFAULT_MIX = {"en": ["world", "bangladesh", "business", "ai", "sport"], "bn": ["bangladesh", "world", "sport"]}
SUBTOPIC = {"cricket": "sport", "football": "sport", "ai": "tech", "crypto": "markets"}   # "sports news in cricket"
_CONTINENTS = {"asia", "europe", "middle east", "africa", "americas"}

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_EN_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_NEWS = re.compile(r"\bnews\b|\bheadlines?\b|\btop stories\b|\bwhat'?s (?:happening|going on) (?:in|around) the world\b|"
                   r"খবর|সংবাদ|শিরোনাম", re.I)
# Not a request for headlines: "I've got good news", "newspaper", "the news app".
_NOT_NEWS = re.compile(r"\b(?:good|bad|great|sad|happy|fake|old|exciting) news\b|\bnewspapers?\b|\bnewsletters?\b|"
                       r"\bnews (?:app|channel|site|website|anchor|reader|feed setting)s?\b", re.I)
# Another request in the same message: the model answers the whole thing.
_ELSE = re.compile(r"\bweather\b|\bforecast\b|\brain\b|\btemperature\b|\btimer\b|\balarm\b|\bremind|\btasks?\b|\bto-?do\b|"
                   r"\bcalendar\b|\bmeeting\b|\bevents?\b|\bschedule\b|\bremember\b|\bprice of\b|\bhow much\b|"
                   r"\bexchange rates?\b|\bconvert\b|\bplan\b|\bset\b|\badd\b|\bdelete\b|\bcancel\b|\bemail\b|"
                   r"\btranslat\w*|\bwhat time\b|\bwhat day\b|আবহাওয়া|টাইমার|মনে করিয়ে|ক্যালেন্ডার|অ্যালার্ম", re.I)
# A question about the news, not a request for it: the model, with web_search and the screen of the last reply.
_QUESTION = re.compile(r"\bwhy\b(?! not\b)|\bexplain\w*|\bwhat do you think\b|\byour (?:opinion|view|take|thoughts?)\b|"
                       r"\banaly[sz]\w*|\bsummar\w*|\btell me more\b|\bmore (?:about|on|details?)\b|\bdetails?\b|"
                       r"\bthe (?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th)) (?:one|story|headline|item|news)\b|"
                       r"\b(?:that|this|those|these) (?:story|stories|news|headlines?|one)\b|\bis (?:it|that|this) true\b|"
                       r"\bdid you\b|\bwhere did\b|\bhow (?:does|do|did|will|would|can|could|is|are)\b(?! you\b)|"
                       r"\bwhat happened\b|\bwhen (?:is|was|will|did|does)\b|\bshould i\b|\bimpact\b|\baffects?\b|"
                       r"\bmean for\b|\bopen\b|\bclosed?\b|কেন|ব্যাখ্যা|বিস্তারিত|মতামত", re.I)
# "From all over the world" frames the request; it isn't the World topic unless nothing else is named.
_FRAMING = re.compile(r"\b(?:from |in |of )?(?:all over|around|across|throughout|everywhere in|all around) the (?:world|globe)\b|"
                      r"\bworld ?wide\b|\bglobally\b|\bfrom everywhere\b|সারা (?:বিশ্বের|বিশ্ব|দুনিয়ার|দুনিয়া)", re.I)
_TIME = re.compile(r"\b(?:in |over |from |of )?(?:the )?(?:last|past) (?:\d+ |few |couple of )?(?:hours?|days?|week|24 hours)\b|"
                   r"\b(?:this|last) (?:week|morning|evening|night)\b|\btoday'?s?\b|\btonight'?s?\b|\byesterday'?s?\b|"
                   r"\bright now\b|\bat the moment\b|\bcurrently\b|\brecently\b|আজকের|আজ|এখনকার", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "a dozen": 12, "dozen": 12}
_COUNT = re.compile(r"(?<![\w./])(\d{1,2})(?:\s*(?:টি|টা))?(?![\d.:%/])|\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
                    + r")\b", re.I)
_UNION = re.compile(r",|&|\band\b|\bor\b|\bplus\b|\balso\b|\bincluding\b|\bas well as\b|\bthen\b|\bwith\b|আর|এবং|ও", re.I)
_FILLER = set("""
a an the me my us our you your i i'd i'm im we some any all of from about on in for to with and or also plus
including as well mix mixed bit few at least more most give show bring tell get fetch pull find read share send
play hear let's lets let can could would will please pls javid hey hi hello okay ok so then now new newest latest
recent current top main major biggest important breaking key daily quick brief briefing summary roundup round-up
round up update updates updated headline headlines news stories story what what's whats is are was were up with
happening going go out there internet online web google search engine development developments industry sector
situation front side related good evening morning afternoon night friend dear buddy bro need want wanna like love
know check see look how hows how's just mean asking asked ask talking talk speak say first here move moving
particularly significant things thing big list rundown quickly again too only same different other others another
sports-wise wise like lot lots little yeah yes well whole entire bunch kind sort type types stuff something anything
everything every each region regions part parts country countries continent today's around over across do don't
time number many much story's events happenings affairs current-affairs world's it why not
খবর সংবাদ শিরোনাম দাও দেখাও শোনাও বলো বল কী কি আছে সব কিছু একটু প্লিজ এর র থেকে সম্পর্কে নিয়ে সর্বশেষ নতুন
গুরুত্বপূর্ণ প্রধান জাভেদ মাহাদি আমাকে আমায় দিন দেখান শোনান বলুন টপ তাজা এখন আপডেট গুলো গুলি আর এবং ও
""".split())


def _norm(text: str) -> str:
    t = (text or "").translate(_BN_DIGITS).replace("’", "'").replace("‘", "'").lower()
    return " " + " ".join(t.split()) + " "


def _count(t: str) -> tuple[int | None, str]:
    """The number of stories asked for, and the text without it. Numbers that each go with a topic add up
    ("three from Asia and three from Europe" is six); otherwise the largest counts ("top 10... the first 10")."""
    found = list(_COUNT.finditer(t))
    if not found:
        return None, t
    starts = {s for s, _, _ in _spans(t)}
    values = [int(m[1]) if m[1] else _NUMBER_WORDS[m[2].lower()] for m in found]
    attached = [v for m, v in zip(found, values)
                if (gap := re.match(r"\s*(?:(?:news|stories|headlines|from|in|of|about|on|the)\s+){0,3}", t[m.end():]))
                and m.end() + gap.end() in starts]
    n = sum(attached) if len(attached) > 1 else max(values)
    for m in reversed(found):
        t = t[:m.start()] + " " + t[m.end():]
    return (max(1, min(n, MAX_STORIES)) if n else None), t


def _spans(t: str) -> list[tuple[int, int, str]]:
    """(start, end, key) of each topic named, in order, longest match first where two overlap."""
    found = []
    for key, pattern in gnews.ALIASES:
        for m in pattern.finditer(t):
            found.append((m.start(), m.end(), key))
    found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out = []
    for s in found:
        if not out or s[0] >= out[-1][1]:
            out.append(s)
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9'ঀ-৿-]+", text.lower())


def request(text: str) -> dict | None:
    """{"topics", "count", "google"} for a message that only asks for news headlines, else None."""
    if not _NEWS.search(text or "") or _NOT_NEWS.search(text) or _ELSE.search(text) or _QUESTION.search(text):
        return None
    t = _norm(text)
    framed = bool(_FRAMING.search(t))
    t = _TIME.sub(" ", _FRAMING.sub(" , ", t))
    count, t = _count(t)
    spans = _spans(t)
    topics = []
    for i, (start, _end, key) in enumerate(spans):
        if i and not _UNION.search(t[spans[i - 1][1]:start]):
            # Two topics with no "and" between them name one subject ("Bangladesh cricket team", "AI news from
            # China"): that's a specific search, unless one is part of the other ("sports news in cricket").
            prev = topics[-1]
            if SUBTOPIC.get(key) == prev:
                topics[-1] = key
                continue
            if SUBTOPIC.get(prev) == key or prev == key:
                continue
            return None
        if key not in topics:
            topics.append(key)
    rest = t
    for start, end, _ in reversed(spans):
        rest = rest[:start] + " " + rest[end:]
    if [w for w in _words(_NEWS.sub(" ", rest)) if w not in _FILLER and not w.isdigit()]:
        return None         # something else is asked about ("the Dhaka metro rail", "the IVAC holiday")
    if "world" in topics and any(k in _CONTINENTS for k in topics):
        topics.remove("world")      # "world news, three from Asia and three from Europe": the regions are the world
    if not topics:
        topics = ["world"] if framed else []
    return {"topics": topics, "count": count, "google": bool(re.search(r"\bgoogle\b", t))}


# ---------- the answer ----------
_LEADS = {"world": "In world news", "bangladesh": "In Bangladesh", "asia": "In Asia", "europe": "In Europe",
          "middle east": "In the Middle East", "africa": "In Africa", "americas": "In the Americas", "usa": "In the US",
          "war": "On the wars", "ai": "In AI", "tech": "In tech", "business": "In business",
          "markets": "On the markets", "crypto": "In crypto", "sport": "In sport", "cricket": "In cricket",
          "football": "In football", "science": "In science", "health": "In health",
          "entertainment": "In entertainment", "climate": "On the climate"}
_SINGLE = {"world": "Here are today's top world stories", "bangladesh": "Here's the latest from Bangladesh",
           "ai": "Here's the latest in AI", "sport": "Here's the latest in sport", "markets": "Here's the latest from the markets",
           "crypto": "Here's the latest in crypto", "business": "Here's the latest business news",
           "tech": "Here's the latest in tech", "war": "Here's the latest on the wars",
           "cricket": "Here's the latest in cricket", "football": "Here's the latest in football"}
_ABBREVIATIONS = [("U.S.", "US"), ("U.K.", "UK"), ("U.N.", "UN"), ("E.U.", "EU"), ("A.I.", "AI")]


def _say(title: str) -> str:
    """A headline as one spoken sentence: no "Exclusive:" tag, no dotted abbreviations (the page splits speech
    at ". "), a full stop at the end."""
    title = gnews._STRIP_PREFIX.sub("", title).strip()
    for dotted, plain in _ABBREVIATIONS:
        title = title.replace(dotted, plain)
    return title if title[-1:] in ".!?।" else title + ("।" if re.search(r"[\u0985-\u09b9]", title) else ".")


def _ago(published: datetime | None, lang: str) -> str:
    if not published:
        return ""
    mins = max(0, int((datetime.now(timezone.utc) - published).total_seconds() // 60))
    if lang == "bn":
        n = (str(mins) + " মিনিট" if mins < 60 else str(mins // 60) + " ঘণ্টা").translate(_EN_TO_BN)
        return "এইমাত্র" if mins < 5 else f"{n} আগে"
    if mins < 5:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    if mins < 24 * 60:
        return f"{mins // 60} h ago"
    local = published.astimezone()
    return "yesterday" if (datetime.now().date() - local.date()).days == 1 else f"{local:%b} {local.day}"


def _pick(groups: list[dict], total: int) -> list[dict]:
    """Round-robin over the topics up to `total` stories, so ten stories over six topics give each one or two."""
    picked, depth = [], 0
    while len(picked) < total and any(len(g["items"]) > depth for g in groups):
        for g in groups:
            if depth < len(g["items"]) and len(picked) < total:
                picked.append(g["items"][depth])
        depth += 1
    return picked


def _speakable(item: dict) -> bool:
    """No figure that only this one outlet gives (web_tools.check_reply on the headline)."""
    return not web_tools.check_reply(item["title"])


def _spoken(groups: list[dict], picked: list[dict], req: dict, lang: str) -> str:
    shown = [g for g in groups if any(i in picked for i in g["items"])]
    failed = [g for g in groups if not g["items"]]
    if not picked:
        return ("এখন খবর আনতে পারছি না, মাহাদি। এক মিনিট পরে আবার বলো।" if lang == "bn" else
                "I couldn't reach the news right now, Mahadi. Try again in a minute.")
    out = []
    if len(shown) == 1:
        key = shown[0]["topic"]
        items = [i for i in picked if _speakable(i)][:SPOKEN_SINGLE]
        if lang == "bn":
            out.append(f"{gnews.name(key, 'bn')} নিয়ে আজকের খবর, মাহাদি।" if key not in ("top", "bangladesh")
                       else "আজকের প্রধান খবর, মাহাদি।")
            out += [f"{i['name']}: {_say(i['title'])}" for i in items]
        else:
            lead = _SINGLE.get(key) or f"Here's the latest {gnews.name(key)} news"
            out.append(lead + (" from Google News." if req.get("google") else ", Mahadi."))
            out += [f"{i['name']}: {_say(i['title'])}" for i in items]
    else:
        n = len(picked)
        out.append(f"{str(n).translate(_EN_TO_BN)}টি খবর পেয়েছি, মাহাদি।" if lang == "bn" else
                   f"I found {n} stories{' on Google News' if req.get('google') else ''}, Mahadi.")
        for g in shown:
            item = next((i for i in g["items"] if i in picked and _speakable(i)), None)
            if not item:
                continue
            if lang == "bn":
                out.append(f"{gnews.name(g['topic'], 'bn')}: {item['name']} জানাচ্ছে, {_say(item['title'])}")
            else:
                lead = _LEADS.get(g["topic"]) or f"In {gnews.name(g['topic'])}"
                out.append(f"{lead}, {item['name']} reports: {_say(item['title'])}")
    # The headlines are cut to the time budget; what's missing and where the rest is are always said.
    tail = []
    if failed:
        names = ", ".join(gnews.name(g["topic"], lang) for g in failed)
        tail.append(f"{names}-এর খবর পাইনি।" if lang == "bn" else f"I couldn't get the {names} news.")
    tail.append("পুরো তালিকা স্ক্রিনে আছে।" if lang == "bn" else "The full list is on your screen.")
    budget = SPOKEN_SECONDS - sum(len(t.split()) for t in tail) / skills.WORDS_PER_SECOND
    return " ".join([skills.trim_spoken(" ".join(out), budget)] + tail)


def _screen(groups: list[dict], picked: list[dict], lang: str) -> str:
    if not picked:
        return ""
    clock = datetime.now().strftime("%I:%M %p").lstrip("0")
    shown = [g for g in groups if any(i in picked for i in g["items"])]
    head = (f"খবর · Google News, {clock}" if lang == "bn" else f"Headlines · Google News, {clock}")
    lines = [f"{head} ({len(picked)}):" if lang != "bn" else f"{head} ({str(len(picked)).translate(_EN_TO_BN)}টি):"]
    n = 0
    for g in shown:
        if len(shown) > 1:
            lines.append(f"{gnews.name(g['topic'], lang)}:")
        for item in (i for i in g["items"] if i in picked):
            n += 1
            outlets = len(item.get("cluster") or [])
            extra = " · ".join(x for x in (_ago(item.get("published"), lang) or item.get("date_text", ""),
                                            (f"{str(outlets).translate(_EN_TO_BN)}টি মাধ্যম" if lang == "bn" else f"{outlets} outlets")
                                            if outlets > 1 else "") if x)
            lines.append(f"{n}. {item['title']} ({item['name']}{', ' + extra if extra else ''})")
    return "\n".join(lines)


def answer(req: dict, lang: str):
    """A generator (yields its tool event) returning (spoken, screen)."""
    topics = req["topics"] or DEFAULT_MIX["bn" if lang == "bn" else "en"]
    total = req["count"] or (SINGLE_TOPIC if len(topics) == 1 else PER_TOPIC * len(topics))
    per_topic = max(1, min(math.ceil(total / len(topics)) + (1 if len(topics) > 1 else 0), 8))
    yield ("tool", "news_headlines")
    span = tracing.tool_start("news_headlines")
    try:
        groups = web_tools.headline_items(topics, per_topic=per_topic, lang=lang)
    finally:
        tracing.tool_end(span)
    picked = _pick(groups, total)
    # The links under the reply are the stories on screen, in the screen's order (topic by topic).
    web_tools.keep_sources([i["url"] for g in groups for i in g["items"] if i in picked])
    return _spoken(groups, picked, req, lang), _screen(groups, picked, lang)
