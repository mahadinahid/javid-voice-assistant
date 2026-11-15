"""
The web server's lock (SECURITY.md, item 1 of the 2026-10-21 audit; built 2026-11-14).

The server listens on 127.0.0.1 only, but anything that can reach that port could use it: another website open in
the browser (a cross-site POST of an audio clip to /talk is a "simple" request, sent without a preflight), a page
reaching 127.0.0.1 through DNS rebinding, or another program or account on this Mac. So every request is checked
here, in one place, before any route runs:

  1. Host must be this server's own (localhost:8000 or 127.0.0.1:8000): a DNS-rebinding page's requests carry its
     own host name.
  2. An Origin header, when the browser sends one, must be the page's own origin: another website's requests
     carry that site's origin, whatever the route.
  3. Every route but the two pages themselves (GET / and GET /telegram, static files with nothing private in them)
     must carry Javid's key in the X-Javid-Token header. The key is JAVID_WEB_TOKEN in .env (made here on first
     start, written owner-only); the server prints a link carrying it (#key=...) once, the page's head script keeps
     it in the browser's storage and sends it with every request. A program without the key gets 401 and nothing
     else, whatever it asks for.

`_local_page_only` in server.py stays on the routes it covers (Host and Origin required, not just checked when
present): the second lock on the ones that change keys, memory, themes and the Telegram lock.
"""
import os
import re
import secrets

import javid

ENV_NAME = "JAVID_WEB_TOKEN"
HEADER = "x-javid-token"
PORT = int(os.environ.get("JAVID_PORT") or 8000)      # JAVID_PORT: a test server on another port (CLAUDE.md, Testing safely)
HOSTS = (f"localhost:{PORT}", f"127.0.0.1:{PORT}")
BASE_URL = f"http://localhost:{PORT}"
PUBLIC = {("GET", "/"), ("GET", "/telegram")}          # the pages themselves: a browser can't send a header to fetch them
_FORMAT = re.compile(r"^[A-Za-z0-9_\-]{32,128}$")


def token() -> str:
    """Javid's key: JAVID_WEB_TOKEN from the environment (.env). Missing or malformed, a new one is made and written
    to .env (owner-only, through javid._write_env), so a key exists from the first start and survives restarts."""
    key = os.environ.get(ENV_NAME, "").strip()
    if not _FORMAT.match(key):
        key = secrets.token_urlsafe(32)
        javid._write_env(ENV_NAME, key)
        os.environ[ENV_NAME] = key
    return key


def link() -> str:
    """The page's address with the key in the fragment (a fragment never leaves the browser: it isn't sent to the
    server or logged). The head script moves it into storage and takes it off the address bar."""
    return f"{BASE_URL}/#key={token()}"


def refuse(method: str, path: str, headers) -> tuple[int, str] | None:
    """Why this request may not run: (status, detail), or None when it may. `headers` is any case-insensitive
    mapping (Starlette's), read with .get."""
    host = headers.get("host", "")
    if host not in HOSTS:
        return 403, "Only from this machine."
    origin = headers.get("origin")
    if origin is not None and origin != f"http://{host}":
        return 403, "Only the Javid page may use this."
    if (method.upper(), path) in PUBLIC:
        return None
    sent = headers.get(HEADER, "")
    if not secrets.compare_digest(sent.encode("utf-8", "replace"), token().encode()):
        return 401, ("Javid's key is missing or wrong. Open the link the server printed, or run "
                     "`python server.py --open`.")
    return None
