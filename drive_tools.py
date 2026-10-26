"""
Google Drive for Javid (2026-10-26): its long outputs, the ones with a screen part (a research report, a morning brief,
the headlines, a plan), saved as Google Docs in a "Javid Reports" folder with their title, date and sources.

Setup (once): connect from the sidebar ("Connect Google Drive") or call connect(). It uses Javid's own OAuth client
(credentials.json, the one the calendar uses) with the drive.file scope only: Javid can see and change only the files
it made itself. The token is token_drive.json (mode 0600), apart from the calendar's token.json, and before any use it
must be for Javid's own client id and carry drive.file and nothing more: no other app's token (Hermes's, say) and no
broader one is ever used.

Saving is decided in code (javid.think_stream asks answer(), request() and wants_save()), never by the model:
- a message that says "save" ("save that to Drive", "research X and save it") saves at once: the word is the go-ahead;
- any other request to put a report in Drive ("put that in my Drive") gets a question, and only a yes on the very next
  message saves it;
- on a channel whose writes need a button (Telegram) the save waits for its Approve button (approvals.py).
Never uploaded: Javid's memory files and chat history (a request to save them is refused), and a document holding a
key, a token, a Google sign-in secret or a line of Javid's memory (checked on the finished document, before upload).
"""
import html
import json
import os
import re
import time
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
CREDENTIALS_FILE = "credentials.json"     # Javid's own OAuth client, shared with the calendar
TOKEN_FILE = "token_drive.json"           # Drive's own token: drive.file only
STATE_FILE = "drive.json"                 # the folder's id, so it's reused
CALENDAR_TOKEN_FILE = "token.json"        # the calendar's token: its secrets mustn't end up in a document either
FOLDER_NAME = "Javid Reports"
FOLDER_MIME = "application/vnd.google-apps.folder"
DOC_MIME = "application/vnd.google-apps.document"
CONFIRM_SEC = 300                         # how long a "save it?" waits (and only for the very next message)
MEMORY_LINE_MIN = 20                      # a memory line this long found in a report stops the upload

# What Javid says, in each language (a Bangla-mode save is said in Bangla).
SAY = {
    "saved": {"en": "Saved it to your Google Drive, in the Javid Reports folder.",
              "bn": "তোমার Google Drive-এর Javid Reports ফোল্ডারে সেভ করেছি।"},
    "dropped": {"en": "Okay, I won't save it.", "bn": "ঠিক আছে, সেভ করব না।"},
    "refused": {"en": "I never upload my memory, our chat history or anything with a key in it. I can save a report "
                      "or a brief.",
                "bn": "মেমরি, চ্যাটের ইতিহাস বা চাবি আছে এমন কিছু আমি কখনো আপলোড করি না। রিপোর্ট বা ব্রিফ সেভ করতে পারি।"},
    "nothing": {"en": "There's no report to save yet: ask for research, a morning brief or the headlines first.",
                "bn": "এখনো সেভ করার মতো রিপোর্ট নেই: আগে রিসার্চ, মর্নিং ব্রিফ বা খবর চাও।"},
    "approve": {"en": "Tap Approve below to save “{title}” to your Drive.",
                "bn": "“{title}” Drive-এ সেভ করতে নিচে Approve চাপো।"},
    "ask": {"en": "Should I save “{title}” to your Google Drive? Say yes or no.",
            "bn": "“{title}” কি তোমার Google Drive-এ সেভ করব? হ্যাঁ বা না বলো।"},
    "not_connected": {"en": "Google Drive isn't connected yet: connect it from the sidebar first.",
                      "bn": "Google Drive এখনো যুক্ত হয়নি: আগে সাইডবার থেকে যুক্ত করো।"},
    "key": {"en": "I didn't save it: it contains something that looks like a key.",
            "bn": "সেভ করিনি: এতে চাবির মতো কিছু আছে।"},
    "sign_in": {"en": "I didn't save it: it contains a Google sign-in secret.",
                "bn": "সেভ করিনি: এতে Google সাইন-ইনের গোপন তথ্য আছে।"},
    "memory": {"en": "I didn't save it: it contains a line of my memory, and memory never goes to Drive.",
               "bn": "সেভ করিনি: এতে আমার মেমরির একটা লাইন আছে, আর মেমরি কখনো Drive-এ যায় না।"},
    "reconnect": {"en": "I couldn't save it: Google wants Drive connected again, from the sidebar.",
                  "bn": "সেভ করতে পারিনি: Google চায় সাইডবার থেকে Drive আবার যুক্ত করো।"},
    "failed": {"en": "I couldn't save it: Google Drive answered with an error. Try again in a moment.",
               "bn": "সেভ করতে পারিনি: Google Drive একটা ত্রুটি দেখিয়েছে। একটু পরে আবার চেষ্টা করো।"},
    "offline": {"en": "I couldn't reach Google Drive just now. Try again in a moment.",
                "bn": "এখন Google Drive-এ পৌঁছাতে পারছি না। একটু পরে আবার চেষ্টা করো।"},
}


def say(key: str, lang: str = "en", **values) -> str:
    return SAY[key]["bn" if lang == "bn" else "en"].format(**values)


class NotConnected(Exception):
    pass


# ---------- auth: Javid's own client, drive.file only ----------
def _client_id() -> str | None:
    """The client id in credentials.json: the only one a Drive token may belong to."""
    try:
        with open(CREDENTIALS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return (data.get("installed") or data.get("web") or {}).get("client_id")


def _token_info() -> dict | None:
    try:
        with open(TOKEN_FILE) as f:
            info = json.load(f)
        return info if isinstance(info, dict) else None
    except (OSError, ValueError):
        return None


def token_problem(info: dict | None) -> str | None:
    """Why this Drive token must not be used, or None: missing, another client's (not Javid's own), or with more access
    than drive.file."""
    if not info:
        return "not connected"
    cid = _client_id()
    if not cid or info.get("client_id") != cid:
        return "the Drive token isn't for Javid's own Google client"
    if set(info.get("scopes") or []) != set(SCOPES):
        return "the Drive token has other access than drive.file"
    return None


def _lock_down(path: str) -> None:
    try:
        if os.stat(path).st_mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


def _write_private(path: str, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)     # never briefly readable by others
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(path, 0o600)


def _load_creds() -> Credentials | None:
    _lock_down(TOKEN_FILE)
    info = _token_info()
    if token_problem(info):
        return None
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            return None          # revoked or expired for good: needs a fresh sign-in
        _write_private(TOKEN_FILE, creds.to_json())
        return creds
    return None


def status() -> str:
    """'connected', 'needs_auth' (credentials.json there), 'no_credentials', or 'wrong_token' (one not to be used)."""
    if _load_creds():
        return "connected"
    if os.path.exists(TOKEN_FILE) and token_problem(_token_info()) not in (None, "not connected"):
        return "wrong_token"
    return "needs_auth" if os.path.exists(CREDENTIALS_FILE) else "no_credentials"


def connect() -> str:
    """Google's sign-in page in the browser, asking for drive.file only; waits up to 5 minutes for consent."""
    if not os.path.exists(CREDENTIALS_FILE):
        return "no_credentials"
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, timeout_seconds=300)
    _write_private(TOKEN_FILE, creds.to_json())
    return status()


def _service():
    """The Drive API client (regression.py swaps this for a fake)."""
    creds = _load_creds()
    if not creds:
        raise NotConnected
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------- the report: title, date, body, sources ----------
_SAVE_CLAUSE = re.compile(r"[,;]?\s*(?:and\s+|then\s+)*(?:also\s+)?(?:please\s+)?(?:save|upload|put|store|keep|"
                          r"back\s*up|send)\b.*$", re.I)
_RESEARCH_VERB = re.compile(r"^\W*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:do\s+)?(?:an?\s+|some\s+)?"
                            r"(?:research|deep[\s-]dive|investigate|look\s+into)\s*(?:on|into|about|of)?\s*", re.I)


def last_report(messages: list) -> dict | None:
    """The latest long output in this chat (an assistant reply with a screen part) and the request it answered."""
    for i in range(len(messages) - 1, 0, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls") and (m.get("screen") or "").strip():
            asked = next((x["content"] for x in reversed(messages[1:i]) if x.get("role") == "user"), "")
            return {"spoken": m.get("content") or "", "screen": m["screen"], "sources": m.get("sources") or [],
                    "agent": m.get("agent"), "skill": m.get("skill"), "asked": asked}
    return None


def title_of(report: dict) -> str:
    """"Research: <topic>", "Morning brief", "Headlines", "Plan: <day>", else the screen part's first line."""
    first = next((line.strip() for line in report["screen"].splitlines() if line.strip()), "")
    if report.get("agent") == "research":
        topic = _RESEARCH_VERB.sub("", _SAVE_CLAUSE.sub("", report.get("asked") or "")).strip(" .?!।")
        return "Research: " + ((topic[:1].upper() + topic[1:])[:80] if topic else "report")
    if report.get("agent") == "planner":
        return "Plan: " + first.split("·")[0].strip()[:60]
    if report.get("skill") == "morning-brief":
        return "Morning brief"
    if first.lower().startswith("headlines") or report.get("skill") == "news-digest":
        return "Headlines"
    return (first.rstrip(":")[:70] or "Javid report").strip()


def _body_html(screen: str) -> str:
    """The screen part as the page shows it: headings (a short line ending in ":"), bullets, numbered items, lines."""
    out, items = [], []

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")
            items.clear()
    for raw in screen.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        text = html.escape(line)
        if re.match(r"^[-•*]\s+", line):
            items.append(html.escape(re.sub(r"^[-•*]\s+", "", line)))
            continue
        flush()
        if len(line) <= 60 and line.endswith(":"):
            out.append(f"<h2>{html.escape(line.rstrip(':'))}</h2>")
        else:
            out.append(f"<p>{text}</p>")
    flush()
    return "\n".join(out)


def document(report: dict, when: datetime | None = None) -> dict:
    """{title, name (its Drive name: title — date), html (what's uploaded), text (the same, for the checks)}."""
    when = when or datetime.now()
    title = title_of(report)
    sources = [s for s in report.get("sources") or [] if str(s.get("url", "")).startswith("http")]
    saved = f"Saved by Javid on {when:%A %d %B %Y, %H:%M}"
    rows = [(s.get("name") or s.get("domain") or "source", s["url"], s.get("date_text") or s.get("date") or "")
            for s in sources]
    src_html = ("<h2>Sources</h2><ol>" + "".join(
        f'<li><a href="{html.escape(url, quote=True)}">{html.escape(name)}</a>{", " + html.escape(day) if day else ""}'
        f"{' · ' + html.escape(s.get('title', '')) if s.get('title') else ''}</li>"
        for (name, url, day), s in zip(rows, sources)) + "</ol>") if rows else ""
    page = (f"<html><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title></head><body>"
            f"<h1>{html.escape(title)}</h1><p><i>{saved}</i></p>{_body_html(report['screen'])}{src_html}</body></html>")
    text = "\n".join([title, saved, report["screen"]] + [f"{n}: {u} {d}" for n, u, d in rows])
    return {"title": title, "name": f"{title} — {when:%Y-%m-%d}", "html": page, "text": text}


# ---------- never uploaded: keys, sign-in secrets, memory ----------
def _sign_in_secrets() -> list[str]:
    """The OAuth secrets on this machine (client secret, refresh and access tokens): none may appear in a document."""
    found = []
    for path in (CREDENTIALS_FILE, TOKEN_FILE, CALENDAR_TOKEN_FILE):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for part in (data, data.get("installed") or {}, data.get("web") or {}):
            for k in ("client_secret", "refresh_token", "token"):
                if isinstance(part.get(k), str) and len(part[k]) >= 12:
                    found.append(part[k])
    return found


def problem(doc: dict) -> str | None:
    """Why this document mustn't be uploaded (a SAY key: "key", "sign_in", "memory"), or None."""
    import javid
    import memory_store
    text = doc["text"]
    if javid._redact(text) != text:
        return "key"
    if any(secret in text for secret in _sign_in_secrets()):
        return "sign_in"
    flat = " ".join(text.lower().split())
    for line in memory_store.read("user") + memory_store.read("memory"):
        line = " ".join(line.lower().split())
        if len(line) >= MEMORY_LINE_MIN and line in flat:
            return "memory"
    return None


# ---------- upload ----------
def _folder(svc) -> str:
    """The "Javid Reports" folder's id: the one Javid made before (kept in drive.json, or found by name among the files
    drive.file lets it see), else a new one."""
    try:
        with open(STATE_FILE) as f:
            fid = json.load(f).get("folder_id")
    except (OSError, ValueError, AttributeError):
        fid = None
    if fid:
        try:
            if not svc.files().get(fileId=fid, fields="id, trashed").execute().get("trashed"):
                return fid
        except HttpError as e:
            if _status(e) not in (403, 404):     # gone, or no longer Javid's: look it up or make it again
                raise
    found = svc.files().list(q=f"name = '{FOLDER_NAME}' and mimeType = '{FOLDER_MIME}' and trashed = false",
                             spaces="drive", fields="files(id, name)", pageSize=10).execute().get("files") or []
    fid = found[0]["id"] if found else svc.files().create(
        body={"name": FOLDER_NAME, "mimeType": FOLDER_MIME}, fields="id").execute()["id"]
    _write_private(STATE_FILE, json.dumps({"folder_id": fid}))
    return fid


def _status(err) -> int | None:
    return getattr(err, "status_code", None) or getattr(getattr(err, "resp", None), "status", None)


def save(report: dict) -> dict:
    """Upload the report as a Google Doc in Javid Reports. {"ok", "said" (a SAY key), "link", "name"}."""
    doc = document(report)
    why = problem(doc)
    if why:
        return {"ok": False, "said": why, "name": doc["name"]}
    try:
        svc = _service()
        folder = _folder(svc)
        made = svc.files().create(
            body={"name": doc["name"], "mimeType": DOC_MIME, "parents": [folder], "description": "Saved by Javid"},
            media_body=MediaInMemoryUpload(doc["html"].encode("utf-8"), mimetype="text/html", resumable=False),
            fields="id, name, webViewLink").execute()
    except NotConnected:
        return {"ok": False, "said": "not_connected", "name": doc["name"]}
    except HttpError as e:
        print(f"[drive] upload failed: HTTP {_status(e)}")
        return {"ok": False, "said": "reconnect" if _status(e) in (401, 403) else "failed", "name": doc["name"]}
    except OSError as e:
        print(f"[drive] upload failed: {type(e).__name__}")
        return {"ok": False, "said": "offline", "name": doc["name"]}
    return {"ok": True, "said": "saved", "link": made.get("webViewLink"), "name": doc["name"]}


def link_source(result: dict) -> list[dict]:
    """The saved document as a link under the reply (the page's and Telegram's source list)."""
    if not result.get("link"):
        return []
    return [{"n": 1, "url": result["link"], "domain": "docs.google.com", "name": "Google Drive",
             "title": result["name"], "official": False, "date": None, "date_text": "saved just now"}]


# ---------- what Mahadi said: save now, ask first, or refuse ----------
_DRIVE = r"(?:google\s+)?drive|ড্রাইভ"
_SAVE = re.compile(r"\bsave\b|সেভ|সংরক্ষণ", re.I)
_THING = r"(?:that|it|this|these|them|the\s+(?:report|brief|morning\s+brief|plan|list|headlines|news|research|summary|answer|results?))"
_ONLY_SAVE = re.compile(rf"^\W*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?save\s+{_THING}"
                        rf"(?:\s+(?:to|in|on|into)\s+(?:my\s+)?(?:{_DRIVE}))?(?:\s+(?:for\s+me|please|now))*\W*$", re.I)
_TO_DRIVE = re.compile(rf"\b(?:put|upload|store|keep|back\s*up|send|add|copy|save|move)\b.{{0,60}}\b(?:{_DRIVE})\b|"
                       rf"(?:{_DRIVE}).{{0,20}}(?:রাখো|রাখ|আপলোড|সেভ)", re.I)
_NOT_ALLOWED = re.compile(r"\b(?:chats?|conversations?|history|memor(?:y|ies)|transcripts?|logs?|keys?|passwords?|"
                          r"tokens?|secrets?|\.env|credentials?)\b|চ্যাট|মেমরি|ইতিহাস", re.I)
_AND_SAVE = re.compile(rf"\b(?:and|then|,)\s+(?:also\s+)?(?:please\s+)?save\b|\bsave\b.{{0,40}}\b(?:{_DRIVE})\b|"
                       r"ড্রাইভ.{0,20}সেভ|সেভ.{0,20}ড্রাইভ", re.I)


def request(text: str, has_report: bool) -> str | None:
    """What a message about saving to Drive asks for, if it is only that: "save" (do it now), "ask" (it named Drive
    but didn't say save), "refuse" (chat history, memory or keys) or None (not a Drive request: the turn goes on)."""
    t = " ".join((text or "").split())
    to_drive = bool(_TO_DRIVE.search(t))
    only_save = bool(_ONLY_SAVE.match(_POLITE.sub("", t)))
    if not (to_drive or (only_save and has_report)):
        return None
    if len(t) > 140 or not _starts_with_saving(t):   # "plan my day and save it": the plan first (wants_save)
        return None
    if _NOT_ALLOWED.search(t):
        return "refuse"
    return "save" if _SAVE.search(t) else "ask"


_POLITE = re.compile(r"^(?:\W|please|pls|ok(?:ay)?|now|hey|hi|javid|so|and|also|then|(?:can|could|would|will)\s+you)*", re.I)


def _starts_with_saving(t: str) -> bool:
    """The message is about saving and nothing else: it opens with the saving ("save that...", "put it in my Drive",
    "can you upload the brief to Drive"), or it's a short Bengali "ড্রাইভে সেভ করো"."""
    if re.search(r"ড্রাইভ", t) and len(t.split()) <= 8:
        return True
    m = _SAVE_CLAUSE.search(t)
    return bool(m) and not _POLITE.sub("", t[:m.start()]).strip(" ,.!?")


def wants_save(text: str) -> str | None:
    """A request for a long output that also asks to keep it: "research X and save it" ("save") or "...and put it in
    my Drive" ("ask"). None: it didn't ask."""
    t = " ".join((text or "").split())
    if _NOT_ALLOWED.search(_SAVE_CLAUSE.search(t).group(0) if _SAVE_CLAUSE.search(t) else ""):
        return None
    if _AND_SAVE.search(t):
        return "save"
    return "ask" if _TO_DRIVE.search(t) else None


# ---------- "save it?": only the very next message answers ----------
_turn = 0
_pending: dict = {}


def new_turn() -> None:
    global _turn
    _turn += 1


def ask(report: dict, lang: str = "en") -> str:
    """Hold this report for a yes on the next message; what Javid says."""
    _pending.clear()
    _pending.update(report=report, turn=_turn, at=time.time())
    return say("ask", lang, title=title_of(report))


def answer(text: str) -> tuple[str, dict] | None:
    """A yes ("save") or no ("drop") to the question asked on the previous message, decided here; any other message
    drops the question. None: nothing was asked, or this isn't an answer."""
    import memory_store
    if not _pending:
        return None
    p = dict(_pending)
    _pending.clear()
    if p["turn"] != _turn - 1 or time.time() - p["at"] > CONFIRM_SEC:
        return None
    if memory_store._NO.match(text.strip()):
        return "drop", p["report"]
    if memory_store._YES.match(text.strip()):
        return "save", p["report"]
    return None
