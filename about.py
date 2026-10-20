"""
Javid's description of itself, answered in code: what it is built from, and which model, voice and data
sources it uses. The model doesn't know Javid's design and guesses. Asked why its voice changed when it
talked about tasks (2026-10-13), gpt-oss-120b answered with a loop of ellipses; asked what sources it uses
for the weather (2026-10-15), gpt-oss-20b said Reuters, Bloomberg and "the trusted domains I've been trained
on", none of which was ever true.

answer(text, lang, facts) gives (spoken, screen) when a message asks about Javid itself, else None and the
model answers. Every pattern needs the question to be about Javid ("you", "your", "Javid") plus a design
topic, so "how do you work out 17% of 2,340" or "what's your opinion on this news" still go to the model.
The live facts (the model chain, the voices, the skills and agents) come from javid.py as `facts`, so what
Javid says about itself can't drift from the code.
"""
import re

_YOU = r"(?:\byou\b|\byour\b|\byourself\b|\bjavid\b|তুমি|তোমার|তোমাকে|তোমায়|আপনি|আপনার|আপনাকে|জাভেদ)"
_NOT_THAT = r"(?!.*\b(?:for|on|behind|of|in)\s+(?:that|this|it|those|these)\b)"   # "your source for that?" is about one answer


def _re(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I | re.S)


# (topic, pattern), most specific first; a message gets the first two topics that match.
_TOPICS = [
    ("voice", _re(rf"^(?=.*{_YOU})(?!.*\bvoice\s*(?:note|message|memo|mail|call|assistant|recorder|typing|search|"
                  r"command)s?\b).*(?:\bvoices?\b|\bsound(?:s|ing)?\s+(?:different|weird|strange|robotic|odd)\b|"
                  r"\baccent\b|text.to.speech|\btts\b|edge.?tts|কণ্ঠ|গলা|ভয়েস|আওয়াজ)")),
    ("hearing", _re(r"\bhow\s+(?:do|can|does)\s+(?:you|javid)\s+(?:hear|understand|listen|recogni[sz]e)\b|"
                    r"\bspeech.to.text\b|\bspeech recognition\b|\bwhisper\b|\bwake.?word\b|"
                    r"\b(?:are|do)\s+you\s+(?:always|constantly|still)\s+(?:listen|hear|record)\w*|"
                    r"\byou\b.{0,20}\blistening\s+(?:all the time|always|constantly)\b")),
    ("model", _re(r"\b(?:what|which)\s+(?:ai\s+|llm\s+|language\s+|large language\s+)?(?:model|llm|ai|brain|engine)\b"
                  r".{0,30}\b(?:are you|do you (?:use|run)|powers? you|is behind you|you (?:use|run on|are)|runs? you)\b|"
                  r"\bare\s+you\s+(?:chat ?gpt|gpt[\w.-]*|gemini|claude|llama|qwen|deepseek|grok|copilot|siri|alexa|"
                  r"an?\s+(?:ai|llm|bot|robot|chatbot|language model|real person|human))\b|"
                  r"\bwhat\s+(?:are you|do you)\s+(?:running|run|built|based)\s+on\b|"
                  r"\bwhat(?:'s| is)\s+(?:your|javid'?s)\s+(?:model|llm|ai model|brain|engine)\b|"
                  r"\bwho\s+(?:writes|generates|makes)\s+your\s+(?:answers|replies|responses)\b|"
                  r"কোন\s+(?:মডেল|এআই)|তুমি কি (?:চ্যাটজিপিটি|এআই)")),
    ("search", _re(r"\b(?:why|how come)\b.{0,20}\b(?:don'?t|didn'?t|can'?t|won'?t|not|couldn'?t)\s+you\s+(?:just\s+)?"
                   r"(?:search|google|use google|look)\b.{0,30}\bgoogle\b|"
                   r"\bdo\s+you\s+(?:use|search(?:\s+(?:on|in|with))?)\s+(?:the\s+)?google\b|"
                   r"\b(?:which|what)\s+search\s+engine\b|\bhow\s+do\s+you\s+search\b|\bwhere\s+do\s+you\s+search\b|"
                   r"\bcan(?:'t)?\s+you\s+(?:search|use)\s+(?:on\s+|in\s+|with\s+)?(?:the\s+)?google\W*$")),
    ("sources", _re(rf"^{_NOT_THAT}.*(?:\b(?:what|which)\s+(?:kinds?|types?|sorts?)\s+of\s+sources?\b|"
                    r"\b(?:what|which)\s+sources?\s+(?:do|are|did)\s+you\b|"
                    r"\bwhere\s+do\s+you\s+get\s+(?:your\s+|the\s+|all\s+)?(?:news|weather|prices?|rates?|data|"
                    r"information|info|facts|forecasts?|figures)\b|"
                    r"\bwhere\s+does\s+(?:your|the)\s+(?:news|weather|price|rate|data|information|info|forecast)\w*\s+"
                    r"come\s+from\b|"
                    r"\bhow\s+do\s+you\s+(?:get|find|know|check)\s+(?:the\s+)?(?:weather|prices?|exchange rates?|"
                    r"rates|news|holidays|forecast)\b|"
                    r"\b(?:your|javid'?s)\s+(?:data\s+)?sources\b|"
                    r"\bwhat\s+(?:do\s+you|does\s+javid)\s+use\s+(?:for|to\s+(?:get|find|check))\s+(?:the\s+)?"
                    r"(?:weather|prices?|rates|news|holidays)\b|"
                    r"(?:তুমি|তোমার).{0,30}(?:সোর্স|উৎস|কোথা থেকে))")),
    ("memory", _re(r"\bhow\s+(?:does|do)\s+your\s+memory\b|\bhow\s+do\s+you\s+remember\b|"
                   r"\bwhat\s+do\s+you\s+(?:store|save|keep|record|log)\b|"
                   r"\bwhere\s+(?:is|are|do\s+you\s+(?:keep|store|save))\s+(?:my|our)\s+(?:data|chats?|conversations?|"
                   r"memory|memories|information|info|tasks|history)\b|"
                   r"\bdo\s+you\s+(?:store|save|keep|record|log)\s+(?:my|our|what i|everything|the)\b|"
                   r"\b(?:is|are)\s+(?:my|our)\s+(?:data|information|info|chats?|conversations?)\s+"
                   r"(?:safe|private|secure|stored|saved|shared)\b|\bprivacy\b|\bwho\s+can\s+(?:see|read|access)\s+(?:my|our)\b")),
    ("creator", _re(r"\bwho\s+(?:made|built|created|developed|designed|programmed|wrote|coded|owns|trained)\s+(?:you|javid)\b|"
                    r"\bwho(?:'s| is)\s+(?:your|javid'?s)\s+(?:creator|developer|maker|author|owner|programmer)\b|"
                    r"তোমাকে\s+কে\s+(?:বানিয়েছে|তৈরি করেছে|বানাল)")),
    ("capabilities", _re(r"\bwhat\s+(?:else\s+)?can\s+you\s+do\b(?!\s+(?:about|for|to|with)\b)|"
                         r"\bwhat\s+can\s+you\s+help(?:\s+me)?\s+with\b|"
                         r"\bwhat\s+are\s+(?:your|you)\s+(?:capabilities|skills|abilities|features|functions|tools|agents|"
                         r"limits|limitations)\b|"
                         r"\bwhat\s+(?:tools|skills|agents|features|abilities)\s+do\s+you\s+have\b|\bwhat\s+can'?t\s+you\s+do\b|"
                         r"\bwhat\s+are\s+you\s+(?:able|capable)\s+(?:to\s+do|of)\b|^\W*how\s+can\s+you\s+help(?:\s+me)?\W*$|"
                         r"তুমি\s+(?:কী|কি)\s+(?:কী|কি)\s+করতে\s+পার")),
    ("design", _re(r"\bhow\s+(?:do|does)\s+(?:you|javid)\s+work\b(?!\s+out)|"
                   r"\bhow\s+(?:are|were)\s+(?:you|javid)\s+(?:built|made|designed|programmed|coded|created|set up)\b|"
                   r"\b(?:your|javid'?s)\s+(?:design|architecture|internals|source code|code ?base|tech(?:nology)?\s+stack|"
                   r"system prompt)\b|"
                   r"\bwhat\s+are\s+you\s+(?:built|made|written|coded|programmed)\s+(?:with|in|from|on)\b|"
                   r"\bwhat\s+(?:language|framework)\s+are\s+you\s+(?:written|built|made|coded)\s+in\b|"
                   r"\bexplain\s+(?:your|how you)\s+(?:design|work|architecture)\b|"
                   r"তুমি\s+(?:কীভাবে|কিভাবে)\s+কাজ\s+কর|তোমার\s+ডিজাইন")),
    ("identity", _re(r"^\W*(?:(?:hey|hi|hello|so|ok(?:ay)?)\W+)?(?:javid\W+)?(?:who|what)\s+(?:are|r)\s+(?:you|u)\W*$|"
                     r"\btell\s+me\s+(?:a\s+bit\s+|a\s+little\s+)?about\s+yourself\b|\bintroduce\s+yourself\b|"
                     r"^\W*(?:who|what)\s+is\s+javid\W*$|\bwhat(?:'s| is)\s+your\s+name\b|^\W*তুমি\s+কে\W*$")),
]

# What a "sources" question is about: each gets its own answer ("what sources do you use for the weather?").
_SOURCE_KINDS = [
    ("weather", _re(r"weather|forecast|\brain|temperature|আবহাওয়া|বৃষ্টি")),
    ("prices", _re(r"\bprices?\b|crypto|bitcoin|\bcoins?\b|\bstocks?\b|\bshares?\b|market|দাম")),
    ("rates", _re(r"exchange|\brates?\b|currenc|taka|dollar|yuan|koruna|রেট|টাকা|ডলার")),
    ("news", _re(r"\bnews\b|headline|খবর")),
    ("holidays", _re(r"holiday|ছুটি")),
    ("calendar", _re(r"calendar|\btasks?\b|ক্যালেন্ডার")),
]

_SPOKEN = {
    "voice": {
        "en": "In English I always speak with Microsoft's Andrew neural voice, and in Bangla mode with Pradeep, both "
              "through edge-tts; Bengali words are always Pradeep, because Andrew can't pronounce them. There's no "
              "other voice: if that service fails, you get the reply as text with a voice-unavailable note, never a "
              "different voice.",
        "bn": "ইংরেজিতে আমি সবসময় Microsoft-এর Andrew নিউরাল ভয়েসে কথা বলি, আর বাংলা মোডে Pradeep-এর ভয়েসে, দুটোই "
              "edge-tts দিয়ে। অন্য কোনো ভয়েস নেই: সার্ভিসটা কাজ না করলে উত্তর লেখা হিসেবে দেখাই, অন্য ভয়েসে কখনো না।"},
    "hearing": {
        "en": "I hear you through Groq's Whisper speech recognition. I only listen when you press talk, or in "
              "hands-free mode, where I answer only what starts with 'Javid', plus a few seconds right after I reply.",
        "bn": "আমি Groq-এর Whisper দিয়ে তোমার কথা শুনি। তুমি টক বোতাম চাপলে শুনি, আর হ্যান্ডস-ফ্রি মোডে শুধু 'জাভেদ' "
              "দিয়ে শুরু হওয়া কথার উত্তর দিই।"},
    "model": {
        "en": "My replies are written by a language model, {model} right now{backup}. Your tasks, calendar, "
              "weather, prices and sums come from tools in code, not from the model's memory.",
        "bn": "আমার উত্তর লেখে একটা ল্যাঙ্গুয়েজ মডেল, এখন {model}{backup}। তোমার কাজ, ক্যালেন্ডার, আবহাওয়া, দাম আর "
              "হিসাব আসে কোডের টুল থেকে, মডেলের মনে রাখা থেকে না।"},
    "search": {
        "en": "I search the web through Tavily, a search service, not Google itself, and I name the source and date. "
              "I only state a price or rate that an official site or two separate sites give, so tell me what to "
              "look up and I'll search.",
        "bn": "আমি Google না, Tavily নামের একটা সার্চ সার্ভিস দিয়ে ওয়েবে খুঁজি, আর উৎস ও তারিখ বলি। কোনো দাম বা রেট "
              "বলি শুধু সরকারি সাইট বা দুটো আলাদা সাইটে পেলে; কী খুঁজব বলো।"},
    "sources": {
        "en": "Weather comes from Open-Meteo, live prices from Binance and Yahoo Finance, exchange rates from the "
              "European Central Bank and ExchangeRate-API, and holidays from Google's public holiday calendar. News "
              "and other web answers come from a Tavily web search, and I name the source and date.",
        "bn": "আবহাওয়া আসে Open-Meteo থেকে, দাম Binance আর Yahoo Finance থেকে, এক্সচেঞ্জ রেট ইউরোপিয়ান সেন্ট্রাল "
              "ব্যাংক আর ExchangeRate-API থেকে, আর ছুটির দিন Google-এর ছুটির ক্যালেন্ডার থেকে। খবর আর অন্য ওয়েবের "
              "উত্তর আসে Tavily সার্চ থেকে, উৎস আর তারিখসহ।"},
    "memory": {
        "en": "I only remember what you approve: I propose a line, and it's saved to a file on your Mac once you "
              "say yes, never passwords, keys or ID numbers. Our chats stay on your Mac too, and each message goes "
              "to the model's provider, {provider} right now, to be answered.",
        "bn": "তুমি অনুমতি দিলেই শুধু মনে রাখি: আমি একটা লাইন প্রস্তাব করি, তুমি হ্যাঁ বললে সেটা তোমার ম্যাকের ফাইলে "
              "সেভ হয়; পাসওয়ার্ড বা আইডি নম্বর কখনো না। আমাদের চ্যাটও তোমার ম্যাকে থাকে, আর প্রতিটা মেসেজ উত্তরের জন্য "
              "মডেলের প্রোভাইডার {provider}-এর কাছে যায়।"},
    "creator": {
        "en": "Mahadi built me, as his own project: a Python program that runs on his Mac. The words come from a "
              "language model, {model}, and everything else is code.",
        "bn": "মাহাদি নিজেই আমাকে বানিয়েছে, তার ম্যাকে চলা একটা পাইথন প্রোগ্রাম হিসেবে। কথাগুলো লেখে একটা ল্যাঙ্গুয়েজ "
              "মডেল, {model}, আর বাকি সব কোড।"},
    "capabilities": {
        "en": "I keep your tasks and calendar, set timers and reminders, check the weather, live prices and "
              "exchange rates, do sums and conversions, search the web and the news, and remember what you approve. "
              "For bigger jobs I have {agents} agents; I can't send email or messages, make calls or buy anything.",
        "bn": "আমি তোমার কাজের তালিকা আর ক্যালেন্ডার রাখি, টাইমার দিই, আবহাওয়া, দাম আর এক্সচেঞ্জ রেট দেখি, হিসাব করি, "
              "ওয়েব আর খবর খুঁজি, আর তুমি যা অনুমতি দাও তা মনে রাখি। বড় কাজের জন্য {agents} এজেন্ট আছে; ইমেইল বা "
              "মেসেজ পাঠানো, কল করা বা কেনাকাটা পারি না।"},
    "design": {
        "en": "Each message goes to a language model, {model} right now, which calls tools written in code for "
              "tasks, calendar, weather, prices, sums and web search, and the reply is checked in code before I say "
              "it. The full picture is on your screen.",
        "bn": "প্রতিটা মেসেজ যায় একটা ল্যাঙ্গুয়েজ মডেলের কাছে, এখন {model}, যে কোডে লেখা টুল দিয়ে কাজ, ক্যালেন্ডার, "
              "আবহাওয়া, দাম, হিসাব আর ওয়েব সার্চ করে; বলার আগে কোড উত্তরটা যাচাই করে। পুরো বিবরণ তোমার স্ক্রিনে।"},
    "identity": {
        "en": "I'm Javid, your personal voice assistant, Mahadi: a program on your Mac that keeps your tasks and "
              "calendar, checks the weather, prices and news, and remembers what you approve.",
        "bn": "আমি জাভেদ, তোমার নিজের ভয়েস অ্যাসিস্ট্যান্ট, মাহাদি: তোমার ম্যাকে চলা একটা প্রোগ্রাম, যে তোমার কাজ আর "
              "ক্যালেন্ডার রাখে, আবহাওয়া, দাম আর খবর দেখে, আর তুমি যা অনুমতি দাও তা মনে রাখে।"},
}

_SOURCE_SPOKEN = {
    "weather": {"en": "Weather comes from Open-Meteo, a forecast service, with MET Norway as the backup when it's "
                      "down; Dhaka is home unless you name a place.",
                "bn": "আবহাওয়া আসে Open-Meteo থেকে, সেটা বন্ধ থাকলে MET Norway থেকে; জায়গা না বললে ঢাকার আবহাওয়া বলি।"},
    "prices": {"en": "Crypto prices come live from Binance's public market data, and stocks and indices from Yahoo Finance.",
               "bn": "ক্রিপ্টোর দাম সরাসরি Binance থেকে, আর শেয়ার ও সূচকের দাম Yahoo Finance থেকে আসে।"},
    "rates": {"en": "The yuan and koruna are the European Central Bank's reference rates, and the taka comes from "
                    "ExchangeRate-API, because the ECB doesn't quote it.",
              "bn": "ইউয়ান আর কোরুনার রেট ইউরোপিয়ান সেন্ট্রাল ব্যাংকের, আর টাকার রেট ExchangeRate-API থেকে, কারণ ECB "
                    "টাকার রেট দেয় না।"},
    "news": {"en": "News comes from a Tavily web search of the last few days' news, and I name each story's source and date.",
             "bn": "খবর আসে Tavily দিয়ে গত কয়েক দিনের খবর খুঁজে, প্রতিটার উৎস আর তারিখসহ।"},
    "holidays": {"en": "Public holidays come from Google's public holiday calendar for Bangladesh; an office's own "
                       "closures, like a visa centre's, come from its notice on the web.",
                 "bn": "সরকারি ছুটির দিন আসে Google-এর বাংলাদেশের ছুটির ক্যালেন্ডার থেকে; কোনো অফিসের নিজের ছুটি তার "
                       "নোটিশ থেকে।"},
    "calendar": {"en": "Your calendar is your own Google Calendar, and your tasks are a text file on your Mac.",
                 "bn": "তোমার ক্যালেন্ডার তোমার নিজের Google Calendar, আর কাজের তালিকা তোমার ম্যাকের একটা ফাইলে।"},
}

_DESIGN_SCREEN = """How Javid works:
- Ears: Groq Whisper speech recognition ({stt}); hands-free answers only what starts with "Javid".
- Brain: {model}, backups {backups}. It writes the reply and picks tools; facts come from the tools.
- Tools in code: tasks, Google Calendar, timers, weather (Open-Meteo, MET Norway), live prices (Binance, Yahoo Finance), exchange rates (ECB, ExchangeRate-API), calculator, web and news search (Tavily), public holidays (Google), memory, past chats.
- Skills: {skills}. Agents: {agents}, each with its own tools and steps.
- Checks in code: dates, counts and sums worked out in Python; web figures need an official site or two sources; holiday dates must match the official list; deleting an event needs a yes; memory needs your approval; garbled replies are retried.
- Voice: {voice_en} in English, {voice_bn} in Bangla mode, through edge-tts. No other voice.
- Your data: tasks, memory and chats are files on Mahadi's Mac; each message goes to the model's provider to be answered."""

_CAPABILITIES_SCREEN = """What Javid can do:
- Tasks: add, list, complete, delete
- Google Calendar: events, free time, add, move, delete (asks first)
- Timers and reminders
- Weather, live crypto and stock prices, exchange rates
- Sums and unit conversions
- Web search, news, public holidays
- Memory you approve, and search of past chats
Skills: {skills}
Agents: {agents}
Can't: send email or messages, make calls, buy anything"""

_SCREENS = {"design": _DESIGN_SCREEN, "capabilities": _CAPABILITIES_SCREEN}


def topics(text: str) -> list[str]:
    """The design topics a message asks about (at most two), or [] when it isn't about Javid itself."""
    return [name for name, pattern in _TOPICS if pattern.search(text)][:2]


def _and(items: list[str], lang: str) -> str:
    joiner = " আর " if lang == "bn" else " and "
    return items[0] if len(items) < 2 else ", ".join(items[:-1]) + joiner + items[-1]


def answer(text: str, lang: str, facts: dict) -> tuple[str, str] | None:
    """(spoken, screen) from the fixed self-description, or None (not a question about Javid)."""
    found = topics(text)
    if not found:
        return None
    lang = lang if lang in ("en", "bn") else "en"
    values = {**facts, "agents": _and(facts["agents"], lang), "skills": ", ".join(facts["skills"]),
              "backups": ", ".join(facts["backups"]) or "none"}
    first = facts["backups"][:1]
    values["backup"] = "" if not first else (f"; সেটা ব্যস্ত থাকলে {first[0]} কাজ নেয়" if lang == "bn" else
                                              f"; if it's busy or out of its daily quota, {first[0]} takes over")
    parts, screen = [], ""
    for topic in found:
        if topic == "sources" and (kinds := [k for k, p in _SOURCE_KINDS if p.search(text)]):
            parts += [_SOURCE_SPOKEN[k][lang] for k in kinds]
        else:
            parts.append(_SPOKEN[topic][lang].format(**values))
        screen = screen or _SCREENS.get(topic, "").format(**{**values, "agents": _and(facts["agents"], "en")})
    return " ".join(dict.fromkeys(parts)), screen
