"""
Javid's outside providers (config/providers.yaml): which are on, what each has used of its free allowance, the
budget guard, cooldowns, and the usage ledger behind them (data/providers.db, SQLite).

javid.py asks state() before each model request and records each one (and each transcription); web_tools.py does the
same for Tavily; the Providers panel shows status(). Enforced here:
- On = enabled in the registry and every env key it names set; anything else is off and never called.
- Each call is one ledger row: provider, model, units in the provider's own unit (credits, requests, usd, neurons,
  characters... never converted to tokens), latency, status. Usage is summed over the window of the provider's reset
  rule (the 1st of the month at 00:00 UTC, midnight Pacific...), so a counter starts again when its window does.
  Providers that report their own quota (Groq's response headers, ElevenLabs' API) give it through a reporter.
- Budget guard: at RESERVE_AT (80 %) of an allowance a provider is "reserve", used only when the others fail; at
  OFF_AT (95 %) it is off until its reset; a call whose cost would go past the allowance is refused; an allowance of
  0 (no free tier) keeps it off.
- A 429, 402 or quota error puts the provider (or the model, for per_model providers) in a cooldown until its
  retry-after time. Cooldowns live in the same database, so a restart or the CLI respects them.
- personal_ok (default false) says whether a provider may receive calendar, memory, chat history or tasks. The router
  that applies it is in javid.py (model requests) and web_tools.py (search queries); PrivacyBlocked is raised when a
  request needs it and no provider has it.
"""
import contextlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import yaml

REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "providers.yaml")
DB_FILE = os.path.join("data", "providers.db")
# A new ledger starts from the usage Javid logged before it existed (once; regression.py empties this).
IMPORT_FILES = {"llm": "llm_usage.jsonl", "tavily": "tavily_usage.jsonl"}
KINDS = ("llm", "search", "extract", "rerank", "tts", "storage")
RESETS = ("daily_utc", "daily_pacific", "monthly_utc", "rolling_24h", "headers", "api")
RESERVE_AT = 0.80
OFF_AT = 0.95
RELOAD_SEC = 2.0            # how often the registry file's modification time is checked
PACIFIC = ZoneInfo("America/Los_Angeles")
BALANCE_SEC = 600           # ElevenLabs' balance is asked for again after this long
FAILED = ("error", "rate_limited", "quota", "payment", "auth", "missing", "unavailable", "refused")
# What Javid uses each provider for: facts about the code, not settings, so they live here.
USES = {
    "groq": "chat models and speech recognition (Whisper)",
    "gemini": "backup chat model",
    "deepseek": "backup chat model (paid)",
    "ifm": "last backup chat model",
    "tavily": "web search (the default) and page reads; news comes from Google News",
    "exa": "web search: companies or papers like X, and a fallback",
    "brave": "web search: news, and a second index for facts",
    "firecrawl": "reads a page Tavily can't (JavaScript)",
    "cohere": "reranks web search results",
    "elevenlabs": "balance only: Javid's voice is edge-tts",
}


class PrivacyBlocked(Exception):
    """The request needs calendar, memory, chat history or tasks, and no provider that could take it has
    personal_ok: true."""


# ---------- registry ----------
_lock = threading.RLock()
_reg = {"path": None, "mtime": None, "checked": 0.0, "entries": {}, "errors": []}


def _as_list(value) -> list[str]:
    return [str(v).strip() for v in (value if isinstance(value, list) else [value]) if str(v or "").strip()]


def _num(value, what: str):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, not {value!r}") from None


def _normalize(raw: dict) -> dict:
    name = str(raw.get("name") or "").strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", name):
        raise ValueError(f"an entry has no usable name ({raw.get('name')!r})")
    if raw.get("kind") not in KINDS:
        raise ValueError(f"{name}: kind must be one of {', '.join(KINDS)}")
    env = _as_list(raw.get("env"))
    if not env or not all(re.fullmatch(r"[A-Z][A-Z0-9_]*", v) for v in env):
        raise ValueError(f"{name}: env must name its .env variable(s)")
    reset = raw.get("reset") or "monthly_utc"
    if reset not in RESETS:
        raise ValueError(f"{name}: reset must be one of {', '.join(RESETS)}")
    a = raw.get("allowance") or {}
    if not isinstance(a, dict):
        raise ValueError(f"{name}: allowance must be a mapping")
    source = a.get("from")
    if source not in (None, "headers", "api"):
        raise ValueError(f"{name}: allowance from must be headers or api")
    allowance = {"amount": _num(a.get("amount"), f"{name}: allowance amount"),
                 "unit": str(a.get("unit") or ("tokens" if source == "headers" else "calls")),
                 "per": str(a.get("per") or ""), "per_model": a.get("per_model") is True, "from": source,
                 "verified": a.get("verified") is True}
    rate = raw.get("rate_limit")
    if rate is not None:
        if not isinstance(rate, dict) or not _num(rate.get("amount"), f"{name}: rate_limit amount"):
            raise ValueError(f"{name}: rate_limit needs an amount")
        rate = {"amount": float(rate["amount"]), "unit": str(rate.get("unit") or "requests"), "per": "minute"}
    costs = raw.get("costs") or {}
    if not isinstance(costs, dict):
        raise ValueError(f"{name}: costs must be a mapping")
    return {
        "name": name, "label": str(raw.get("label") or name), "kind": raw["kind"], "env": env,
        "enabled": raw.get("enabled", True) is not False,           # only an explicit false turns it off
        "priority": int(_num(raw.get("priority", 100), f"{name}: priority") or 0),
        "allowance": allowance, "reset": reset, "rate_limit": rate,
        "costs": {str(k): float(_num(v, f"{name}: cost {k}") or 0) for k, v in costs.items()},
        "timeout_ms": int(min(300000, max(1000, _num(raw.get("timeout_ms", 30000), f"{name}: timeout_ms") or 30000))),
        # search.py: how long an engine may take before the next one starts too (it isn't cut off then)
        "soft_timeout_ms": (int(min(60000, max(50, _num(raw["soft_timeout_ms"], f"{name}: soft_timeout_ms"))))
                            if raw.get("soft_timeout_ms") is not None else None),
        "personal_ok": raw.get("personal_ok") is True,               # only an explicit true allows personal data
    }


def _registry() -> dict[str, dict]:
    """The registry, read again when the file changed. A broken file keeps the last good registry, and the problem
    is printed once and shown on the Providers panel."""
    with _lock:
        now = time.time()
        if _reg["path"] == REGISTRY_FILE and now - _reg["checked"] < RELOAD_SEC:
            return _reg["entries"]
        _reg["checked"] = now
        try:
            mtime = os.stat(REGISTRY_FILE).st_mtime
        except OSError:
            mtime = None
        if _reg["path"] == REGISTRY_FILE and mtime == _reg["mtime"]:
            return _reg["entries"]
        entries, errors = {}, []
        try:
            with open(REGISTRY_FILE, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            items = data.get("providers") if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise ValueError("it has no 'providers:' list")
            for raw in items:
                try:
                    entry = _normalize(raw if isinstance(raw, dict) else {})
                    if entry["name"] in entries:
                        raise ValueError(f"{entry['name']} is listed twice")
                    entries[entry["name"]] = entry
                except ValueError as e:
                    errors.append(str(e))
        except (OSError, ValueError, yaml.YAMLError) as e:
            errors.append(f"{os.path.basename(REGISTRY_FILE)}: {str(e).splitlines()[0][:160]}")
            if _reg["path"] == REGISTRY_FILE and _reg["entries"]:
                entries = _reg["entries"]
        for msg in errors:
            if msg not in _reg["errors"]:
                print(f"[providers] {msg}")
        _reg.update(path=REGISTRY_FILE, mtime=mtime, entries=entries, errors=errors)
        return entries


def reload() -> None:
    """Read the registry again on the next call (tests point REGISTRY_FILE at a scratch copy)."""
    with _lock:
        _reg["path"] = None


def get(name: str) -> dict | None:
    return _registry().get(name)


def entries(kind: str | None = None) -> list[dict]:
    return sorted((e for e in _registry().values() if kind in (None, e["kind"])),
                  key=lambda e: (KINDS.index(e["kind"]), e["priority"], e["name"]))


def errors() -> list[str]:
    _registry()
    return list(_reg["errors"])


def env_names(name: str) -> list[str]:
    return list((get(name) or {}).get("env", []))


def has_key(name: str) -> bool:
    entry = get(name)
    return bool(entry) and all(os.environ.get(v, "").strip() for v in entry["env"])


def is_on(name: str) -> bool:
    entry = get(name)
    return bool(entry) and entry["enabled"] and has_key(name)


def personal_ok(name: str) -> bool:
    entry = get(name)
    return bool(entry) and entry["personal_ok"]


def timeout_s(name: str, default: float) -> float:
    entry = get(name)
    return entry["timeout_ms"] / 1000 if entry else default


def soft_timeout_s(name: str, default: float) -> float:
    entry = get(name)
    return entry["soft_timeout_ms"] / 1000 if entry and entry.get("soft_timeout_ms") else default


def unit_of(name: str) -> str:
    entry = get(name)
    return entry["allowance"]["unit"] if entry else "calls"


def cost(name: str, op: str, default: float = 1.0) -> float:
    entry = get(name)
    return (entry or {}).get("costs", {}).get(op, default)


def per_model(name: str) -> bool:
    entry = get(name)
    return bool(entry) and entry["allowance"]["per_model"]


def startup_line() -> str:
    """Which providers are on, names only (never a key or a hint of one)."""
    on, off = [], []
    for e in entries():
        if is_on(e["name"]):
            on.append(e["name"])
        else:
            off.append(f"{e['name']} ({'disabled' if not e['enabled'] else 'no key'})")
    line = f"[providers] enabled: {', '.join(on) or 'none'}"
    if off:
        line += f" · off: {', '.join(off)}"
    if _reg["errors"]:
        line += f" · {len(_reg['errors'])} problem(s) in {os.path.basename(REGISTRY_FILE)}"
    return line


# ---------- usage ledger ----------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
    op TEXT NOT NULL DEFAULT '', units REAL NOT NULL DEFAULT 0, unit TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER, status TEXT NOT NULL DEFAULT 'ok', detail TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS calls_by_provider ON calls (provider, ts);
CREATE TABLE IF NOT EXISTS cooldowns (
    provider TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', until REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (provider, model));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
_db = {"path": None}


@contextlib.contextmanager
def _db_con():
    """One connection per use (the server's threads, the CLI and the test suite all write to the ledger), committed
    when the block ends. The schema and the one-time import run the first time a path is used."""
    path = DB_FILE
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    try:
        if _db["path"] != path:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)       # owner-only, like Javid's other records of what Mahadi does
            with con:
                con.executescript(_SCHEMA)
            _import_once(con)
            _db["path"] = path
        with con:
            yield con
    finally:
        con.close()


def _jsonl(path: str):
    try:
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def _import_once(con) -> None:
    """The usage logged before the ledger existed (tavily_usage.jsonl, llm_usage.jsonl), so this month's credits and
    today's requests start out right."""
    if con.execute("SELECT 1 FROM meta WHERE key = 'imported'").fetchone():
        return
    rows = []
    for e in _jsonl(IMPORT_FILES.get("tavily") or ""):
        if isinstance(e.get("ts"), (int, float)):
            rows.append((e["ts"], "tavily", "", e.get("kind") or "search", float(e.get("credits") or 0), "credits",
                         None, "cached" if e.get("cached") else "ok", "imported"))
    since = time.time() - 35 * 86400
    for e in _jsonl(IMPORT_FILES.get("llm") or ""):
        if not isinstance(e.get("ts"), (int, float)) or e["ts"] < since:
            continue
        model = str(e.get("model") or "")
        prefix, sep, _ = model.partition(":")
        provider = prefix if sep and prefix in ("gemini", "deepseek", "ifm", "openrouter", "cloudflare") else "groq"
        units, unit = ((float(e.get("total") or 0), "tokens") if provider == "groq" else (1.0, "requests"))
        rows.append((e["ts"], provider, model, "chat", units, unit, None, "ok", "imported"))
    with con:
        con.executemany("INSERT INTO calls (ts, provider, model, op, units, unit, latency_ms, status, detail) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('imported', ?)", (str(round(time.time())),))
    if rows:
        print(f"[providers] ledger started from {len(rows)} earlier logged calls")


def record(provider: str, units: float = 0.0, *, model: str = "", op: str = "", unit: str | None = None,
           latency_ms: float | None = None, status: str = "ok", detail: str = "", ts: float | None = None) -> None:
    """One call in the ledger. `detail` is a short, non-personal note (an error's kind, never a query or a key). A
    failed write is printed and dropped: accounting must never cost Mahadi his answer."""
    try:
        with _db_con() as con:
            con.execute("INSERT INTO calls (ts, provider, model, op, units, unit, latency_ms, status, detail) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (ts or time.time(), provider, model or "", op or "", float(units or 0), unit or unit_of(provider),
                         None if latency_ms is None else int(latency_ms), status, (detail or "")[:200]))
    except sqlite3.Error as e:
        print(f"[providers] ledger write failed: {e}")


def used(provider: str, since: float | None = None, model: str | None = None, unit: str | None = None) -> float:
    """Units used (in `unit`, the provider's own by default) since `since`; one model's only if `model` is given."""
    q, args = "SELECT COALESCE(SUM(units), 0) FROM calls WHERE provider = ? AND unit = ?", [provider, unit or unit_of(provider)]
    if since is not None:
        q, args = q + " AND ts >= ?", args + [since]
    if model:
        q, args = q + " AND model = ?", args + [model]
    try:
        with _db_con() as con:
            return float(con.execute(q, args).fetchone()[0])
    except sqlite3.Error:
        return 0.0


def calls(provider: str, since: float | None = None, model: str | None = None, ops: tuple = (),
          statuses: tuple = (), exclude: tuple = ("refused",)) -> list[dict]:
    q, args = "SELECT ts, model, op, units, unit, latency_ms, status, detail FROM calls WHERE provider = ?", [provider]
    if since is not None:
        q, args = q + " AND ts >= ?", args + [since]
    if model:
        q, args = q + " AND model = ?", args + [model]
    if ops:
        q, args = q + f" AND op IN ({','.join('?' * len(ops))})", args + list(ops)
    if statuses:
        q, args = q + f" AND status IN ({','.join('?' * len(statuses))})", args + list(statuses)
    if exclude:
        q, args = q + f" AND status NOT IN ({','.join('?' * len(exclude))})", args + list(exclude)
    try:
        with _db_con() as con:
            rows = con.execute(q + " ORDER BY ts", args).fetchall()
    except sqlite3.Error:
        return []
    keys = ("ts", "model", "op", "units", "unit", "latency_ms", "status", "detail")
    return [dict(zip(keys, r)) for r in rows]


def last_error(provider: str, model: str | None = None) -> dict | None:
    q, args = (f"SELECT ts, model, op, status, detail FROM calls WHERE provider = ? AND status IN "
               f"({','.join('?' * len(FAILED))})", [provider, *FAILED])
    if model:
        q, args = q + " AND model = ?", args + [model]
    try:
        with _db_con() as con:
            row = con.execute(q + " ORDER BY ts DESC LIMIT 1", args).fetchone()
    except sqlite3.Error:
        return None
    return dict(zip(("at", "model", "op", "kind", "message"), row)) if row else None


def last_used(provider: str) -> float | None:
    try:
        with _db_con() as con:
            row = con.execute("SELECT MAX(ts) FROM calls WHERE provider = ? AND status != 'refused'", (provider,)).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


# ---------- reset windows ----------
def _midnight(now: float, tz) -> datetime:
    return datetime.fromtimestamp(now, tz).replace(hour=0, minute=0, second=0, microsecond=0)


def window_start(reset: str, now: float | None = None) -> float | None:
    """Start of the window an allowance is counted over (None: the provider reports it itself)."""
    now = now or time.time()
    if reset == "daily_utc":
        return _midnight(now, timezone.utc).timestamp()
    if reset == "daily_pacific":
        return _midnight(now, PACIFIC).timestamp()
    if reset == "monthly_utc":
        return _midnight(now, timezone.utc).replace(day=1).timestamp()
    if reset == "rolling_24h":
        return now - 86400
    return None


def next_reset(reset: str, now: float | None = None) -> float | None:
    """When the allowance refills (None: continuously, or the provider says)."""
    now = now or time.time()
    if reset in ("daily_utc", "daily_pacific"):
        start = _midnight(now, timezone.utc if reset == "daily_utc" else PACIFIC)
        return (start + timedelta(days=1)).timestamp()      # wall-clock arithmetic: still midnight across DST
    if reset == "monthly_utc":
        first = _midnight(now, timezone.utc).replace(day=1)
        return (first.replace(year=first.year + 1, month=1) if first.month == 12
                else first.replace(month=first.month + 1)).timestamp()
    return None


# ---------- what's used of each allowance ----------
_reporters: dict = {}


def set_reporter(name: str, fn) -> None:
    """For an allowance `from: headers` or `from: api`: fn(model) gives {"used", "amount", "unit", "next_reset"?,
    "parts"?} from what the provider reported, or None when it hasn't reported yet."""
    _reporters[name] = fn


def usage(name: str, model: str = "", now: float | None = None) -> dict:
    entry = get(name)
    now = now or time.time()
    a = entry["allowance"]
    out = {"amount": a["amount"], "unit": a["unit"], "per": a["per"], "verified": a["verified"], "from": a["from"],
           "per_model": a["per_model"], "used": None, "share": None, "window_start": None, "next_reset": None}
    if a["from"]:
        report = _reporters.get(name, lambda m: None)(model)
        if report:
            out.update({k: report[k] for k in ("used", "amount", "unit", "next_reset", "parts", "per") if k in report})
    else:
        start = window_start(entry["reset"], now)
        out.update(window_start=start, next_reset=next_reset(entry["reset"], now),
                   used=used(name, start, model if a["per_model"] and model else None, a["unit"]))
    if out["amount"] and out["used"] is not None:
        out["share"] = out["used"] / out["amount"]
    return out


def share_state(share: float | None) -> str:
    """The guard's verdict on a share of an allowance: ready, reserve (>= 80 %) or off (>= 95 %)."""
    if share is None:
        return "ready"
    return "off" if share >= OFF_AT else "reserve" if share >= RESERVE_AT else "ready"


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if n >= 100 or float(n).is_integer() else f"{n:,.2f}".rstrip("0").rstrip(".")


def state(name: str, model: str = "", cost: float = 0.0, now: float | None = None) -> dict:
    """{"state", "why", "until", "usage"}: ready, reserve (used only when the others fail), off (allowance used up or
    no free tier; "until" = its reset), cooldown ("until" = retry-after), nokey, disabled, unknown. With `cost`, also
    off when this call would go past the free allowance."""
    entry = get(name)
    now = now or time.time()
    if not entry:
        return {"state": "unknown", "why": "not in config/providers.yaml", "until": None, "usage": None}
    if not entry["enabled"]:
        return {"state": "disabled", "why": "turned off in config/providers.yaml", "until": None, "usage": None}
    missing = [v for v in entry["env"] if not os.environ.get(v, "").strip()]
    if missing:
        return {"state": "nokey", "why": f"no {' / '.join(missing)} in .env", "until": None, "usage": None}
    until, reason = cooldown(name, model, now)
    if until:
        return {"state": "cooldown", "why": reason or "resting after an error", "until": until, "usage": None}
    rate = entry["rate_limit"]
    if rate:
        recent = calls(name, now - 60)
        if len(recent) + 1 > rate["amount"]:
            return {"state": "cooldown", "why": f"{_fmt(rate['amount'])} {rate['unit']} a minute",
                    "until": recent[-int(rate["amount"])]["ts"] + 60 if len(recent) >= rate["amount"] else now + 60,
                    "usage": None}
    u = usage(name, model, now)
    if u["amount"] == 0:
        return {"state": "off", "why": "no free tier", "until": None, "usage": u}
    if u["amount"] and u["used"] is not None:
        verdict = share_state(u["share"])
        if verdict == "off":
            return {"state": "off", "why": f"{u['share']:.0%} of its free {u['unit']} used", "until": u["next_reset"],
                    "usage": u}
        if cost and u["used"] + cost > u["amount"]:
            return {"state": "off", "why": "this call would go past its free allowance", "until": u["next_reset"],
                    "usage": u}
        if verdict == "reserve":
            return {"state": "reserve", "why": f"{u['share']:.0%} of its free {u['unit']} used", "until": None,
                    "usage": u}
    return {"state": "ready", "why": "", "until": None, "usage": u}


def allow(name: str, model: str = "", cost: float = 0.0) -> tuple[bool, dict]:
    """May this call be made now? (yes for ready and reserve: a caller with no other provider uses a reserve one)."""
    st = state(name, model, cost)
    return st["state"] in ("ready", "reserve"), st


def ordered(kind: str) -> list[str]:
    """Providers of a kind that may be called now, by priority, reserve ones last."""
    states = [(e["name"], state(e["name"])["state"]) for e in entries(kind)]
    return [n for n, s in states if s == "ready"] + [n for n, s in states if s == "reserve"]


# ---------- cooldowns ----------
def cool(name: str, seconds: float, reason: str = "", model: str = "") -> float:
    """Keep the provider (or one model of it) out of use for `seconds`. A longer cooldown already running stays."""
    until = time.time() + max(1.0, float(seconds))
    try:
        with _db_con() as con:
            con.execute("INSERT INTO cooldowns (provider, model, until, reason) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT (provider, model) DO UPDATE SET reason = excluded.reason, "
                        "until = MAX(until, excluded.until)", (name, model or "", until, (reason or "")[:120]))
    except sqlite3.Error as e:
        print(f"[providers] cooldown write failed: {e}")
    return until


def cooldown(name: str, model: str = "", now: float | None = None) -> tuple[float | None, str]:
    """(until, reason) of the cooldown that applies to this provider or model now, or (None, "")."""
    now = now or time.time()
    try:
        with _db_con() as con:
            row = con.execute("SELECT until, reason FROM cooldowns WHERE provider = ? AND model IN ('', ?) AND until > ? "
                              "ORDER BY until DESC LIMIT 1", (name, model or "", now)).fetchone()
    except sqlite3.Error:
        return None, ""
    return (row[0], row[1]) if row else (None, "")


def clear_cooldown(name: str, model: str | None = None) -> None:
    """After a success (or a new key): the provider-wide cooldown and this model's, or all of them."""
    try:
        with _db_con() as con:
            if model is None:
                con.execute("DELETE FROM cooldowns WHERE provider = ?", (name,))
            else:
                con.execute("DELETE FROM cooldowns WHERE provider = ? AND model IN ('', ?)", (name, model))
    except sqlite3.Error:
        pass


# ---------- errors: kind and how long to wait ----------
_QUOTA_WORDS = re.compile(r"quota|exceed|usage limit|plan limit|insufficient|balance|out of credits|limit reached|"
                          r"resource.?exhausted|spend limit", re.I)
_WAIT_TEXT = re.compile(r"(?:try again in|retry in|retry after|retryDelay['\"]?:\s*['\"]?)\s*"
                        r"(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", re.I)


def _status(err) -> int | None:
    code = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def explicit_wait(err) -> float | None:
    """Seconds the error itself says to wait: a retry-after header (seconds or a date), or text like Groq's "try again
    in 7m11s" or Google's "retry in 6h21m37s". None when it doesn't say."""
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    header = headers.get("retry-after") if hasattr(headers, "get") else None
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            try:
                return max(1.0, parsedate_to_datetime(header).timestamp() - time.time())
            except (TypeError, ValueError, IndexError):
                pass
    m = _WAIT_TEXT.search(str(err))
    if m and any(m.groups()):
        h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
        return max(1.0, h * 3600 + mins * 60 + secs)
    return None


def error_kind(err) -> str:
    """rate_limited (429), payment (402), quota (a quota or plan-limit message), auth (401/403), missing (404),
    unavailable (5xx, network, timeout) or error."""
    code, name, text = _status(err), type(err).__name__, str(err)
    if code == 429 or name in ("RateLimitError", "UsageLimitExceededError"):
        return "rate_limited"
    if code == 402:
        return "payment"
    if name == "ForbiddenError" or code in (401, 403) or name in ("AuthenticationError", "PermissionDeniedError",
                                                                    "InvalidAPIKeyError"):
        return "quota" if _QUOTA_WORDS.search(text) else "auth"      # Tavily's plan limits (432/433) are "forbidden"
    if code == 404 or name == "NotFoundError":
        return "missing"
    if (code and code >= 500) or name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "TimeoutError",
                                          "ConnectionError", "ConnectTimeout", "ReadTimeout", "ConnectError"):
        return "unavailable"
    if _QUOTA_WORDS.search(text) and re.search(r"quota|limit|credits|balance", text, re.I):
        return "quota"
    return "error"


def cooldown_seconds(name: str, err, now: float | None = None) -> float | None:
    """How long a provider (or model) rests after this error: a 429 until its retry-after (60 s if it doesn't say);
    a 402 or a quota error until its retry-after, else until the allowance resets (an hour if that isn't known); a
    rejected key or a missing model an hour; an outage 30 s. None: not a reason to rest."""
    now = now or time.time()
    kind, wait = error_kind(err), explicit_wait(err)
    if kind == "rate_limited":
        return wait or 60.0
    if kind in ("quota", "payment"):
        entry = get(name)
        reset_at = next_reset(entry["reset"], now) if entry else None
        return wait or (reset_at - now if reset_at else 3600.0)
    if kind in ("auth", "missing"):
        return 3600.0
    if kind == "unavailable":
        return 30.0
    return None


# ---------- ElevenLabs: the balance from its API ----------
_balance = {"at": 0.0, "data": None, "error": None, "busy": False}


def _get_json(url: str, headers: dict, timeout: float) -> dict:
    """GET a provider's JSON API (regression.py swaps this for recorded fixtures)."""
    import httpx
    resp = httpx.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def refresh_balance(wait: bool = False) -> None:
    """Ask ElevenLabs for the character balance again (in the background unless `wait`)."""
    if not is_on("elevenlabs") or _balance["busy"]:
        return
    _balance["busy"] = True
    if wait:
        _fetch_balance()
    else:
        threading.Thread(target=_fetch_balance, name="elevenlabs-balance", daemon=True).start()


def _fetch_balance() -> None:
    key = os.environ.get(env_names("elevenlabs")[0], "").strip()
    t0 = time.time()
    try:
        data = _get_json("https://api.elevenlabs.io/v1/user/subscription", {"xi-api-key": key},
                         timeout_s("elevenlabs", 10.0))
        reset_at = data.get("next_character_count_reset_unix")
        _balance.update(at=time.time(), error=None, data={
            "used": float(data["character_count"]), "amount": float(data["character_limit"]),
            "next_reset": float(reset_at) if reset_at else None, "tier": str(data.get("tier") or "")})
        record("elevenlabs", 0, op="balance", latency_ms=(time.time() - t0) * 1000, status="ok")
    except Exception as e:
        kind = error_kind(e) if not isinstance(e, (KeyError, TypeError, ValueError)) else "error"
        _balance.update(at=time.time(), error={"kind": kind, "at": time.time()})
        record("elevenlabs", 0, op="balance", latency_ms=(time.time() - t0) * 1000, status=kind, detail=type(e).__name__)
        if (rest := cooldown_seconds("elevenlabs", e)) and kind != "error":
            cool("elevenlabs", rest, kind)
    finally:
        _balance["busy"] = False


def _elevenlabs_report(model: str = "") -> dict | None:
    if is_on("elevenlabs") and time.time() - _balance["at"] > BALANCE_SEC and not cooldown("elevenlabs")[0]:
        refresh_balance()
    data = _balance["data"]
    return {**{k: data[k] for k in ("used", "amount", "next_reset")}, "unit": "characters"} if data else None


set_reporter("elevenlabs", _elevenlabs_report)


# ---------- the Providers panel ----------
def status(now: float | None = None) -> dict:
    """Every provider: on or off and why, used / allowance in its own unit, when that resets, its last error."""
    now = now or time.time()
    rows = []
    for e in entries():
        name = e["name"]
        st = state(name, now=now)
        u = st["usage"] or (usage(name, now=now) if e["enabled"] and has_key(name) else None)
        a = e["allowance"]
        rows.append({
            "name": name, "label": e["label"], "kind": e["kind"], "uses": USES.get(name, "not used by Javid yet"),
            "env": e["env"], "has_key": has_key(name), "enabled": e["enabled"], "on": is_on(name),
            "priority": e["priority"], "personal_ok": e["personal_ok"], "timeout_ms": e["timeout_ms"],
            "state": st["state"], "why": st["why"], "until": st["until"],
            "allowance": {"amount": (u or {}).get("amount", a["amount"]), "unit": (u or {}).get("unit", a["unit"]),
                          "per": a["per"], "verified": a["verified"], "from": a["from"], "per_model": a["per_model"]},
            "used": (u or {}).get("used"), "share": (u or {}).get("share"), "parts": (u or {}).get("parts"),
            "reset": e["reset"], "next_reset": (u or {}).get("next_reset") or next_reset(e["reset"], now),
            "window_start": (u or {}).get("window_start"), "rate_limit": e["rate_limit"], "costs": e["costs"],
            "calls": len(calls(name, (u or {}).get("window_start"))) if (u or {}).get("window_start") else None,
            "last_used": last_used(name), "last_error": last_error(name),
        })
    return {"providers": rows, "errors": errors(), "reserve_at": RESERVE_AT, "off_at": OFF_AT, "now": now,
            "registry": os.path.relpath(REGISTRY_FILE, os.path.dirname(os.path.abspath(__file__)))}
