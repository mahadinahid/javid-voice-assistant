"""
Telegram: talk to Javid from your phone. A bot made with @BotFather (its token as TELEGRAM_BOT_TOKEN in .env)
polls Telegram's Bot API (long polling: no public URL or open port needed), and only the paired account can
use it. The server starts it (server.py lifespan) when the token is set.

Setup from the phone: the token can be pasted on the local page http://localhost:8000/telegram (set_token: checked
with Telegram, saved to .env, the bot starts without a restart). Until an account is paired, the bot has a one-time
code; the page shows it as a QR code of the bot's link (https://t.me/<bot>?start=<code>), so scanning it with the
phone opens Telegram and tapping Start sends "/start <code>", which pairs that account (telegram.json). Typing
"/pair <code>" works too. Anyone else is told it's private and never reaches Javid. API keys and the model order
can't be changed from Telegram.

Each message is one Javid turn in the bot's own chat (a "Telegram" chat in the web sidebar, so phone and web
contexts don't mix; memory, tasks, calendar and past-chat search are shared), under javid.hold_turn() so it
never runs at the same time as a web turn, and the web's open chat is reopened afterwards. A voice message is
transcribed like /talk and also gets a voice-note reply in the locked voice: javid.synthesize, sentence groups
by voice as the page speaks them, converted to OGG/Opus by ffmpeg (an MP3 audio file if ffmpeg is missing; if
edge-tts is down, the text says the voice is unavailable - never another voice). A skill's or agent's screen
part and the source links follow as their own messages; a memory proposal comes with Save / Don't save buttons.
Timers and reminders are pushed when they come due. Commands: /help, /new, /tasks, /timers, /lang, /voice,
/status.
"""
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time

import httpx

import javid
import memory_store
import timers
import tracing
import web_tools

API = "https://api.telegram.org"
STATE_FILE = "telegram.json"
POLL_SEC = 30               # long-poll wait per getUpdates request
TIMER_CHECK_SEC = 5
MAX_TEXT = 4000             # Telegram's limit is 4096 characters a message
MAX_PAIR_TRIES = 5          # wrong codes before a new one is made
COMMANDS = [("help", "What I can do here"), ("new", "Start a new Telegram chat"), ("tasks", "Your task list"),
            ("timers", "Running timers and reminders"), ("lang", "Reply language: /lang en or /lang bn"),
            ("voice", "Voice replies: only to voice messages, or always"), ("status", "Model, language, tasks")]

_stop = threading.Event()
_state_lock = threading.Lock()
_pair = {"code": None, "tries": 0}
_told: set[int] = set()     # strangers already told the bot is private (once each)
_http: httpx.Client | None = None
_bot = {"username": ""}
_running = {"on": False}
_TOKEN = re.compile(r"\d{5,15}:[A-Za-z0-9_-]{30,60}")
_PAIR = re.compile(r"/(?:pair|start)(?:@\w+)?\s+(\d{6})")    # typed, or sent by the QR link's Start button


def token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def _safe(text) -> str:
    """Error text without the token (it's part of every API URL)."""
    text = str(text)
    return text.replace(token(), "<token>") if token() else text


# ---------- state (telegram.json: owner, chat, update offset, voice setting) ----------
def _load() -> dict:
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(state: dict) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.chmod(tmp, 0o600)
    os.replace(tmp, STATE_FILE)


def _update_state(**changes) -> dict:
    with _state_lock:
        state = {**_load(), **changes}
        _save(state)
        return state


def status() -> dict:
    """For the /telegram page (this machine only): set up? running? paired? and while it isn't paired, the code
    and the bot link that carries it (the page shows it as a QR code)."""
    paired = bool(_load().get("owner"))
    code = None if paired else _pair["code"]
    return {"configured": bool(token()), "running": _running["on"],
            "bot": f"@{_bot['username']}" if _bot["username"] else None, "paired": paired, "pair_code": code,
            "pair_link": f"https://t.me/{_bot['username']}?start={code}" if code and _bot["username"] else None}


def set_token(value: str) -> dict:
    """The /telegram page's token box: check the token with Telegram (getMe), save it to .env and start the bot
    (no restart). Raises ValueError with a message safe to show; nothing is saved unless Telegram accepts it."""
    value = value.strip()
    if not _TOKEN.fullmatch(value):
        raise ValueError("That doesn't look like a bot token: digits, a colon, then about 35 letters and digits.")
    try:
        me = _api("getMe", _token=value)
    except RuntimeError as e:
        if str(e)[:3] in ("401", "404"):
            raise ValueError("Telegram rejected this token. Nothing was saved.") from None
        raise ValueError("Couldn't reach Telegram to check it. Nothing was saved.") from None
    if (old := token()) and old.split(":")[0] != value.split(":")[0]:
        _update_state(offset=0)            # another bot: its updates are numbered from scratch
    javid._write_env("TELEGRAM_BOT_TOKEN", value)
    os.environ["TELEGRAM_BOT_TOKEN"] = value
    _bot["username"] = me.get("username", "")
    if _running["on"]:
        _hello()                           # the polling thread picks the new token up on its next call
    else:
        start()
    return status()


# ---------- Bot API ----------
def _client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=httpx.Timeout(POLL_SEC + 15, connect=10))
    return _http


def _api(method: str, files: dict | None = None, _token: str | None = None, **params):
    """One Bot API call: its result, or RuntimeError with Telegram's description (the token never in it).
    _token: a token being checked before it's saved."""
    url = f"{API}/bot{_token or token()}/{method}"
    try:
        if files:   # multipart: every other field is a string
            data = {k: v if isinstance(v, str) else json.dumps(v) for k, v in params.items()}
            r = _client().post(url, data=data, files=files)
        else:
            r = _client().post(url, json=params)
        body = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise RuntimeError(_safe(f"{type(e).__name__}: {e}").replace(_token or "\0", "<token>")) from None
    if not body.get("ok"):
        raise RuntimeError(f"{r.status_code} {body.get('description', '')}")
    return body["result"]


def _download(file_id: str) -> bytes:
    path = _api("getFile", file_id=file_id)["file_path"]
    try:
        r = _client().get(f"{API}/file/bot{token()}/{path}")
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(_safe(e)) from None
    return r.content


def _chunks(text: str) -> list[str]:
    """Text in pieces Telegram accepts, cut at line breaks where it can."""
    out = []
    while len(text) > MAX_TEXT:
        cut = text.rfind("\n", 0, MAX_TEXT)
        cut = cut if cut > MAX_TEXT // 2 else MAX_TEXT
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return out + ([text] if text.strip() else [])


def send(chat_id: int, text: str, buttons: list[list[dict]] | None = None) -> int | None:
    """Plain text (no markdown: replies aren't written for it), the buttons on the last piece; its message id."""
    msg_id = None
    parts = _chunks(text)
    for i, part in enumerate(parts):
        extra = {"reply_markup": {"inline_keyboard": buttons}} if buttons and i == len(parts) - 1 else {}
        msg_id = _api("sendMessage", chat_id=chat_id, text=part, disable_web_page_preview=True, **extra)["message_id"]
    return msg_id


def _typing(chat_id: int, action: str = "typing") -> None:
    try:
        _api("sendChatAction", chat_id=chat_id, action=action)
    except RuntimeError:
        pass        # only a nicety


# ---------- voice ----------
def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
                                           if os.path.exists(p)), None)


def _to_ogg(mp3: bytes) -> bytes | None:
    """edge-tts MP3 as OGG/Opus, the format Telegram shows as a voice note; None without ffmpeg."""
    if not (ffmpeg := _ffmpeg()):
        return None
    try:
        out = subprocess.run([ffmpeg, "-loglevel", "error", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "32k",
                              "-ac", "1", "-ar", "48000", "-f", "ogg", "pipe:1"], input=mp3, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 and out.stdout else None


def _speech(text: str) -> bytes:
    """The reply in the locked voice, as the page speaks it: sentence by sentence (web_tools.sentences), each
    with javid.voice_for's voice, consecutive sentences in the same voice synthesized together. Raises if
    edge-tts fails (no other voice)."""
    groups: list[tuple[str, list[str]]] = []
    for sentence in web_tools.sentences(text):
        if not re.search(r"[a-z0-9ঀ-৿]", sentence, re.I):
            continue
        voice = javid.voice_for(sentence)
        if groups and groups[-1][0] == voice:
            groups[-1][1].append(sentence)
        else:
            groups.append((voice, [sentence]))
    return b"".join(javid.synthesize(" ".join(group))[0] for _, group in groups)   # MP3 frames join cleanly


def send_voice(chat_id: int, text: str) -> None:
    try:
        mp3 = _speech(text)
    except RuntimeError:
        send(chat_id, "🔇 Voice unavailable right now; the reply is above as text.")
        return
    if not mp3:
        return
    if (ogg := _to_ogg(mp3)):
        _api("sendVoice", files={"voice": ("javid.ogg", ogg, "audio/ogg")}, chat_id=str(chat_id))
    else:
        _api("sendAudio", files={"audio": ("javid.mp3", mp3, "audio/mpeg")}, chat_id=str(chat_id), title="Javid")


# ---------- turns ----------
def _telegram_chat() -> str:
    """The bot's own chat id (made, or made again if it was deleted in the web sidebar). Call under hold_turn."""
    sid = _load().get("session")
    if not sid or not javid.has_session(sid):
        sid = javid.create_session("Telegram")
        _update_state(session=sid)
    return sid


def run_turn(chat_id: int, text: str, voice: bool, trace: tracing.Trace | None = None) -> None:
    """One Javid turn for Telegram, its reply sent as messages (and a voice note if `voice`)."""
    reply, screen, links, proposals = "", "", [], []
    last_action = 0.0
    with javid.hold_turn():
        before = javid.active_session()
        sid = _telegram_chat()
        if sid != before:
            javid.open_session(sid)
        tracing.activate(trace)
        try:
            for kind, value in javid.think_stream(text):
                if kind == "delta":
                    reply += value
                elif kind == "screen":
                    screen = value
                elif kind == "sources":
                    links = value
                elif kind == "memory_proposal":
                    proposals.append(value)
                elif kind in ("tool", "say", "agent") and time.time() - last_action > 4:
                    last_action = time.time()
                    _typing(chat_id, "record_voice" if voice else "typing")
        finally:
            tracing.activate(None)
            if before and before != sid and javid.has_session(before):
                javid.open_session(before)      # the web page's chat stays open
    if trace:
        trace.mark("reply_out")
    send(chat_id, reply.strip() or javid.phrase("stuck"))
    if screen:
        send(chat_id, screen)
    if links:
        send(chat_id, "Sources:\n" + "\n".join(f"{s['n']}. {s['name']}: {s['url']}" for s in links[:5]))
    for p in proposals:
        send(chat_id, f"Save to memory?\n“{p['text']}”", [[{"text": "✅ Save", "callback_data": f"mem:yes:{p['id']}"},
                                                         {"text": "✖ Don't save", "callback_data": f"mem:no:{p['id']}"}]])
    if voice and reply.strip():
        _typing(chat_id, "upload_voice")
        send_voice(chat_id, reply.strip())


def _voice_turn(chat_id: int, msg: dict, trace: tracing.Trace) -> None:
    media = msg.get("voice") or msg.get("audio") or {}
    _typing(chat_id, "typing")
    audio = _download(media["file_id"])
    ext = "ogg" if msg.get("voice") else (media.get("mime_type", "audio/mpeg").split("/")[-1] or "mp3")
    text = javid.transcribe_turn(audio, f"voice.{ext}")
    trace.mark("stt_done")
    trace.label = text[:60]
    if not text.strip() or javid.is_noise(text):
        send(chat_id, javid.phrase("not_caught"))
        return
    send(chat_id, f"🎙 “{text.strip()}”")
    run_turn(chat_id, text, voice=True, trace=trace)


# ---------- commands ----------
def _help() -> str:
    return ("Talk to me here like on the web page: type, or send a voice message and I'll answer with my voice "
            "too. Commands:\n" + "\n".join(f"/{name} - {what}" for name, what in COMMANDS))


def _status_text() -> str:
    state = _load()
    lines = [f"Model: {javid.next_model()}",
             f"Reply language: {'Bangla' if javid.get_language() == 'bn' else 'English'}",
             f"Voice replies: {'always' if state.get('voice') == 'always' else 'to voice messages'}",
             javid.list_tasks().splitlines()[0],
             timers.timers_tool("list").splitlines()[0]]
    return "\n".join(lines)


def _command(chat_id: int, text: str) -> None:
    name, _, arg = text[1:].partition(" ")
    name = name.split("@")[0].lower()
    arg = arg.strip().lower()
    if name in ("start", "help"):
        send(chat_id, _help())
    elif name == "new":
        with javid.hold_turn():
            _update_state(session=javid.create_session("Telegram"))
        send(chat_id, "New chat started. The old one is still in the web sidebar.")
    elif name == "tasks":
        send(chat_id, javid.list_tasks())
    elif name == "timers":
        send(chat_id, timers.timers_tool("list"))
    elif name == "lang":
        if arg in ("en", "english", "bn", "bangla", "bengali"):
            lang = "bn" if arg.startswith("b") else "en"
            javid.set_language(lang)
            send(chat_id, javid.phrase("switched", lang))
        else:
            send(chat_id, f"Reply language: {'Bangla' if javid.get_language() == 'bn' else 'English'}. "
                          "Change it with /lang en or /lang bn.")
    elif name == "voice":
        always = _load().get("voice") != "always"
        _update_state(voice="always" if always else "auto")
        send(chat_id, "I'll send a voice note with every reply." if always else
                      "I'll send voice notes only when you send a voice message.")
    elif name == "status":
        send(chat_id, _status_text())
    else:
        send(chat_id, f"I don't know /{name}.\n\n{_help()}")


# ---------- updates ----------
def _new_pair_code() -> str:
    _pair["code"], _pair["tries"] = f"{secrets.randbelow(10**6):06d}", 0
    print(f"[telegram] not paired: scan the QR code at http://localhost:8000/telegram with your phone, or send "
          f"/pair {_pair['code']} to @{_bot['username'] or 'your bot'}")
    return _pair["code"]


def _stranger(chat_id: int, user_id: int, text: str) -> None:
    """Someone who isn't the owner: pair them if they have the code, else say once that the bot is private."""
    if not _load().get("owner"):
        m = _PAIR.fullmatch(text.strip())
        if m and _pair["code"] and secrets.compare_digest(m[1], _pair["code"]):
            _update_state(owner=user_id, chat=chat_id)
            _pair["code"] = None
            print(f"[telegram] paired with Telegram user {user_id}")
            send(chat_id, f"Paired. Hi Mahadi, it's Javid.\n\n{_help()}")
            return
        if m:
            _pair["tries"] += 1
            if _pair["tries"] >= MAX_PAIR_TRIES:
                _new_pair_code()
            send(chat_id, "That code isn't right. Scan the QR code on Javid's Telegram page again.")
            return
        send(chat_id, "To use me, scan the QR code at http://localhost:8000/telegram on Javid's Mac, "
                      "or send /pair followed by the code shown there.")
        return
    if user_id not in _told:
        _told.add(user_id)
        send(chat_id, "This is a private assistant.")


def handle_update(update: dict) -> None:
    """One update from getUpdates: a message or a button press."""
    if (query := update.get("callback_query")):
        _button(query)
        return
    msg = update.get("message") or {}
    chat, user = msg.get("chat") or {}, msg.get("from") or {}
    if chat.get("type") != "private" or not user.get("id"):
        return
    chat_id, text = chat["id"], (msg.get("text") or "").strip()
    if user["id"] != _load().get("owner"):
        _stranger(chat_id, user["id"], text)
        return
    if text.startswith("/"):
        _command(chat_id, text)
        return
    trace = tracing.start("telegram", "voice" if msg.get("voice") or msg.get("audio") else "text", text[:60])
    trace.mark("speech_end", trace.stages.get("request_received"))
    try:
        if msg.get("voice") or msg.get("audio"):
            _voice_turn(chat_id, msg, trace)
        elif text:
            _typing(chat_id)
            run_turn(chat_id, text, voice=_load().get("voice") == "always", trace=trace)
        else:
            send(chat_id, "I can read text and listen to voice messages.")
    finally:
        tracing.finish_local(trace)


def _button(query: dict) -> None:
    """Save / Don't save on a memory proposal (the owner's only)."""
    if (query.get("from") or {}).get("id") != _load().get("owner"):
        _api("answerCallbackQuery", callback_query_id=query["id"], text="This is a private assistant.")
        return
    kind, _, rest = (query.get("data") or "").partition(":")
    choice, _, pid = rest.partition(":")
    note = "Already decided."
    if kind == "mem":
        try:
            if choice == "yes":
                memory_store.approve(pid)
                note = "✓ Saved"
            else:
                note = "✕ Not saved" if memory_store.reject(pid) else note
        except KeyError:
            pass
        except ValueError as e:        # a secret, refused on approval too
            note = str(e)[:180]
    _api("answerCallbackQuery", callback_query_id=query["id"], text=note)
    msg = query.get("message") or {}
    if msg.get("message_id"):
        try:
            _api("editMessageText", chat_id=msg["chat"]["id"], message_id=msg["message_id"],
                 text=f"{msg.get('text', '')}\n\n{note}")
        except RuntimeError:
            pass


# ---------- timers ----------
def push_due() -> None:
    """Timers and reminders that came due, sent to the owner once each (the page still rings them too)."""
    state = _load()
    if not state.get("chat"):
        return
    lang = javid.get_language()
    for t in timers.take_due_for("telegram"):
        late = time.time() - t["due"]
        text = t["announce"].get(lang) or t["announce"]["en"]
        send(state["chat"], f"⏰ {text}" + (f" (missed: it was due {timers._when(t['due'])})" if late > 120 else ""))


# ---------- the loops ----------
def _poll() -> None:
    backoff = 5
    while not _stop.is_set():
        try:
            if not _bot["username"]:
                _hello()
            updates = _api("getUpdates", offset=_load().get("offset", 0), timeout=POLL_SEC,
                           allowed_updates=["message", "callback_query"])
            backoff = 5
        except RuntimeError as e:
            if str(e).startswith("401"):
                print("[telegram] the bot token was rejected; Telegram is off until it's fixed in .env")
                _running["on"] = False
                return
            print(f"[telegram] {str(e)[:120]}; retrying in {backoff}s")
            _stop.wait(backoff)
            backoff = min(backoff * 2, 120)
            continue
        for update in updates:
            _update_state(offset=update["update_id"] + 1)    # first: a message that crashes isn't retried forever
            try:
                handle_update(update)
            except Exception as e:
                print(f"[telegram] update {update['update_id']}: {_safe(repr(e))[:160]}")
                chat = ((update.get("message") or {}).get("chat") or {}).get("id")
                if chat and (update["message"].get("from") or {}).get("id") == _load().get("owner"):
                    try:
                        send(chat, "Something went wrong on my end.")
                    except RuntimeError:
                        pass


def _timer_loop() -> None:
    while not _stop.wait(TIMER_CHECK_SEC):
        try:
            push_due()
        except Exception as e:
            print(f"[telegram] timers: {_safe(repr(e))[:120]}")


def _hello() -> None:
    """Who the bot is, and its command menu. Raises RuntimeError (e.g. offline at startup: _poll calls it again)."""
    _bot["username"] = _api("getMe").get("username", "")
    _api("setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS])
    if not _load().get("owner"):
        _new_pair_code()
    else:
        print(f"[telegram] @{_bot['username']} is listening")


def start() -> bool:
    """Start polling if TELEGRAM_BOT_TOKEN is set (server lifespan). False if it isn't, or Telegram rejects it;
    offline at startup, it keeps trying."""
    if not token():
        return False
    try:
        _hello()
    except RuntimeError as e:
        if str(e).startswith("401"):
            print("[telegram] not started: Telegram rejected TELEGRAM_BOT_TOKEN")
            return False
        print(f"[telegram] can't reach Telegram yet ({str(e)[:80]}); will keep trying")
    timers.take_due_for("telegram", baseline=True)     # what already rang before Telegram was on isn't re-sent
    _stop.clear()
    _running["on"] = True
    threading.Thread(target=_poll, name="telegram-poll", daemon=True).start()
    threading.Thread(target=_timer_loop, name="telegram-timers", daemon=True).start()
    return True


def stop() -> None:
    _stop.set()
    _running["on"] = False
