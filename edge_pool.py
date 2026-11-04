"""Pre-opened edge-tts connections (2026-11-04): the same voice, a little sooner.

edge_tts.Communicate opens a new WebSocket for every sentence: TCP, TLS and the upgrade with its token took 0.5-0.67 s
here, every time. Measured 2026-11-04 (24 requests, fresh and pre-opened interleaved): when the service was quick, a
sentence's first audio came 0.17-0.22 s after it was sent on a socket opened beforehand, against 0.68-0.90 s through
Communicate; when it was slow (5-7 s), both were; requests that came back without audio were as common either way. So
while a turn runs (`want`), POOL_SIZE sockets are kept open and ready. Each carries one sentence and is then closed (a
second request on the same socket failed in testing), and none is kept past MAX_AGE_SEC (one left 25 s was dropped).

The request is the one Communicate sends, built with edge-tts's own helpers: the same voice (TTSConfig), SSML,
speech.config and audio format; only when the socket opens changes. Not hedging: one request per sentence. A sentence
too long for one request, no socket ready, a socket the service has quietly closed (no answer within ACK_SEC) and an
edge-tts whose internals have moved all go through Communicate, as before (`speak` returns None).
"""
import asyncio
import threading
import time
from xml.sax.saxutils import escape

try:
    import aiohttp
    from edge_tts.communicate import (_SSL_CTX, DRM, SEC_MS_GEC_VERSION, WSS_HEADERS, WSS_URL, connect_id,
                                      date_to_string, get_headers_and_data, mkssml, remove_incompatible_characters,
                                      ssml_headers_plus_data)
    from edge_tts.data_classes import TTSConfig
    from edge_tts.exceptions import NoAudioReceived, UnexpectedResponse
    AVAILABLE = True
except Exception as e:      # an edge-tts upgrade moved them: every sentence goes through Communicate
    print(f"[tts] pre-opened edge-tts sockets off: {type(e).__name__}")
    AVAILABLE = False

ENABLED = True              # regression.py turns it off: its fake edge-tts is Communicate
POOL_SIZE = 2               # the reply's first two sentences; a taken socket is replaced while wanted
MAX_AGE_SEC = 8.0           # fine after 3 and 10 s in testing, dropped after 25 s
WANT_SEC = 20.0             # a want() keeps sockets ready this long
ACK_SEC = 1.5               # the service answers a request with turn.start in ~0.12 s: silence means a dead socket
OPEN_PAUSE_SEC = 5.0        # after a socket couldn't be opened, wait this long before opening another
MAX_BYTES = 4096            # Communicate splits longer texts into several requests; those go through it
TICK_SEC = 0.25

_CONFIG = ("X-Timestamp:{ts}\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n"
           '{{"context":{{"synthesis":{{"audio":{{"metadataoptions":{{'
           '"sentenceBoundaryEnabled":"true","wordBoundaryEnabled":"false"}},'
           '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}}}}}\r\n')     # Communicate's, for SentenceBoundary

_start_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_pool: list = []            # ready sockets, oldest first; touched only on _loop's thread
_state = {"want_until": 0.0, "opening": 0, "pause_until": 0.0, "failing": False, "used": 0, "dead": 0}


class DeadSocket(Exception):
    """The pre-opened socket was closed or silent before the service took the request: not the service failing it."""


class Cancelled(Exception):
    """Nobody is waiting for this sentence any more (the page hung up): stop, and don't count it as a failure."""


class _Socket:
    def __init__(self, session, ws):
        self.session, self.ws, self.opened = session, ws, time.monotonic()

    def stale(self) -> bool:
        return self.ws.closed or time.monotonic() - self.opened > MAX_AGE_SEC

    async def close(self) -> None:
        try:
            await self.ws.close()
        finally:
            await self.session.close()


async def _connect() -> _Socket:
    """A new socket with speech.config sent, as Communicate opens one (regression.py swaps this for a fake)."""
    session = aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(
        total=None, connect=None, sock_connect=10, sock_read=60))
    try:
        ws = await session.ws_connect(f"{WSS_URL}&ConnectionId={connect_id()}&Sec-MS-GEC={DRM.generate_sec_ms_gec()}"
                                      f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}", compress=15,
                                      headers=DRM.headers_with_muid(WSS_HEADERS), ssl=_SSL_CTX)
        await ws.send_str(_CONFIG.format(ts=date_to_string()))
        return _Socket(session, ws)
    except aiohttp.ClientResponseError as e:
        if e.status == 403:
            DRM.handle_client_response_error(e)     # the token is time-based: Communicate corrects the clock the same way
        await session.close()
        raise
    except BaseException:
        await session.close()
        raise


async def _open_one() -> None:
    try:
        sock = await _connect()
    except Exception as e:
        _state["pause_until"] = time.monotonic() + OPEN_PAUSE_SEC
        if not _state["failing"]:
            print(f"[tts] couldn't pre-open an edge-tts socket ({type(e).__name__}); sentences connect as before")
        _state["failing"] = True
        return
    finally:
        _state["opening"] -= 1
    _state["failing"] = False
    if time.monotonic() < _state["want_until"]:
        _pool.append(sock)
    else:
        await sock.close()


async def _keeper() -> None:
    """Closes stale sockets, and opens new ones while they're wanted."""
    while True:
        try:
            for sock in [s for s in _pool if s.stale() or time.monotonic() >= _state["want_until"]]:
                _pool.remove(sock)
                asyncio.ensure_future(sock.close())
            now = time.monotonic()
            if ENABLED and now < _state["want_until"] and now >= _state["pause_until"]:
                while len(_pool) + _state["opening"] < POOL_SIZE:
                    _state["opening"] += 1
                    asyncio.ensure_future(_open_one())
        except Exception as e:      # never let the keeper die
            print(f"[tts] pre-open keeper: {type(e).__name__}")
        await asyncio.sleep(TICK_SEC)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _start_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="edge-pool", daemon=True).start()
            asyncio.run_coroutine_threadsafe(_keeper(), _loop)
    return _loop


def want(seconds: float = WANT_SEC) -> None:
    """A reply is coming: keep sockets ready for the next `seconds`."""
    if not (ENABLED and AVAILABLE):
        return
    _state["want_until"] = max(_state["want_until"], time.monotonic() + seconds)
    _ensure_loop()


def _take() -> _Socket | None:
    while _pool:
        sock = _pool.pop(0)
        if not sock.stale():
            return sock
        asyncio.ensure_future(sock.close())
    return None


async def _request(sock: _Socket, text: str, voice: str, first_sec: float, stall_sec: float,
                   cancelled=None) -> bytes:
    """One sentence on a ready socket. Raises DeadSocket if the service never took it; otherwise like a Communicate
    try: TimeoutError (no first audio within first_sec, or stall_sec between messages after it), NoAudioReceived."""
    tc = TTSConfig(voice, "+0%", "+0%", "+0Hz", "SentenceBoundary")
    ssml = mkssml(tc, escape(remove_incompatible_characters(text)))
    try:
        await sock.ws.send_str(ssml_headers_plus_data(connect_id(), date_to_string(), ssml))
    except Exception as e:
        raise DeadSocket(type(e).__name__) from None
    loop = asyncio.get_running_loop()
    sent = loop.time()
    audio, answered, last = bytearray(), False, sent
    while True:
        if cancelled and cancelled():
            raise Cancelled
        limit = (sent + ACK_SEC if not answered else sent + first_sec if not audio else last + stall_sec)
        try:        # short waits, to look at `cancelled` (a timed-out receive keeps the socket and its messages)
            msg = await sock.ws.receive(timeout=max(0.001, min(TICK_SEC, limit - loop.time())))
        except TimeoutError:
            if loop.time() < limit:
                continue                    # just a tick, to look at `cancelled`
            if not answered:
                raise DeadSocket("silent") from None
            raise TimeoutError("edge-tts sent no audio in time") from None
        last = loop.time()
        if msg.type == aiohttp.WSMsgType.TEXT:
            answered = True
            raw = msg.data.encode("utf-8")
            headers, _ = get_headers_and_data(raw, raw.find(b"\r\n\r\n"))
            if headers.get(b"Path") == b"turn.end":
                break
        elif msg.type == aiohttp.WSMsgType.BINARY:
            answered = True
            if len(msg.data) < 2:
                raise UnexpectedResponse("binary message without its header length")
            headers, data = get_headers_and_data(msg.data, int.from_bytes(msg.data[:2], "big"))
            if headers.get(b"Path") != b"audio":
                raise UnexpectedResponse("binary message that isn't audio")
            kind = headers.get(b"Content-Type")
            if kind not in (b"audio/mpeg", None) or (kind is None and data):
                raise UnexpectedResponse("unexpected audio message")
            audio += data
        else:                               # closed or failed
            if not answered:
                raise DeadSocket(msg.type.name)
            break
    if not audio:
        raise NoAudioReceived("No audio was received.")
    return bytes(audio)


async def _speak(text: str, voice: str, first_sec: float, stall_sec: float, cancelled) -> bytes | None:
    sock = _take()
    if sock is None:
        return None
    _state["used"] += 1
    try:
        return await _request(sock, text, voice, first_sec, stall_sec, cancelled)
    except DeadSocket:
        _state["dead"] += 1
        return None
    finally:
        await sock.close()


def speak(text: str, voice: str, first_sec: float, stall_sec: float, cancelled=None) -> bytes | None:
    """The sentence's audio through a ready socket, or None when there was none (or it was dead): the caller then
    uses Communicate. Blocks the calling thread; raises like a Communicate try, or Cancelled once `cancelled()`."""
    if not (ENABLED and AVAILABLE) or len(escape(text).encode("utf-8")) > MAX_BYTES:
        return None
    want()      # a reply is being spoken: keep the next sentence's socket ready too
    future = asyncio.run_coroutine_threadsafe(_speak(text, voice, first_sec, stall_sec, cancelled), _ensure_loop())
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise


def status() -> dict:
    return {"ready": len(_pool), "used": _state["used"], "dead": _state["dead"], "failing": _state["failing"]}
