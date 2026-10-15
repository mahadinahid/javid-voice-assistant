"""
Memory v2: two markdown files Javid reads but never writes on his own, plus a search index over past chats.

    memory/USER.md     facts about Mahadi
    memory/MEMORY.md   lessons: how Mahadi likes things done

One "- " line per entry. Both files go into the system message as a fixed block (block()), so the
prompt prefix only changes when an entry does. The model can only *propose* an entry
(propose_memory): it is saved when Mahadi approves it on the card in the page, or says yes on the very
next message (checked in code, like delete_event's confirmation). Secrets (passwords, keys, ID and
passport numbers) are refused on every path, including edits in the sidebar.

Past chats: search_history(query) runs an SQLite FTS5 query over chat_log.jsonl, indexed incrementally.
"""
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime

MEMORY_DIR = "memory"
FILES = {"user": "USER.md", "memory": "MEMORY.md"}
TITLES = {"user": "About Mahadi", "memory": "How Mahadi likes things done"}
HEADERS = {"user": "# About Mahadi\n\nFacts Javid may use. One per line; edit freely.\n\n",
           "memory": "# How Mahadi likes things done\n\nLessons and preferences. One per line; edit freely.\n\n"}
LEGACY_FILE = "memory.json"           # v1 facts, copied into USER.md once
HISTORY_FILE = "chat_log.jsonl"
INDEX_FILE = "chat_index.db"
MAX_ENTRY_CHARS = 300


# ---------- secrets are never stored ----------
_SECRET_WORDS = re.compile(
    r"pass\s*word|passcode|pass\s*phrase|\bpin\b|\bpins\b|\botp\b|api[\s_-]*key|secret|token|credential|"
    r"private key|seed phrase|recovery (code|phrase)|passport|national id|\bnid\b|\bssn\b|social security|"
    r"driver'?s licen[cs]e|licen[cs]e number|\bid (card|number|no)\b|account number|card number|\bcvv\b|\biban\b|"
    r"পাসওয়ার্ড|পিন|পাসপোর্ট|জাতীয় পরিচয়|এনআইডি", re.I)
_SECRET_SHAPES = re.compile(
    r"(?:gsk_|AIza|sk-|tvly-|ya29\.|ghp_|xox[bp]-)[A-Za-z0-9_\-.]{6,}"     # provider key shapes
    r"|\b[A-Z]{1,2}\d{6,9}\b"                                          # passport-style numbers
    r"|(?:\d[\s-]?){8,}"                                               # long digit runs: ID/card/account numbers
    r"|\b[A-Za-z0-9+/_\-]{32,}\b")                                     # long opaque strings


_PHONE_WORDS = re.compile(r"phone|mobile|cell|whats\s*app|call|ফোন|মোবাইল|নম্বর", re.I)


def secret_reason(text: str) -> str | None:
    """Why this text must not be stored, or None if it's fine."""
    if _SECRET_WORDS.search(text):
        return "it looks like a password, key or ID document detail"
    for m in _SECRET_SHAPES.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # A phone number is fine to keep ("Mom's mobile is 01712 345678"); ID/card/account numbers are not.
        if _PHONE_WORDS.search(text) and 10 <= len(digits) <= 13 and m.group(0)[0].isdigit():
            continue
        return "it contains something shaped like a key or an ID/card number"
    return None


# ---------- the two files ----------
def _path(file: str) -> str:
    if file not in FILES:
        raise ValueError(f"Unknown memory file {file!r}")
    return os.path.join(MEMORY_DIR, FILES[file])


def _migrate() -> None:
    """Copy v1 facts (memory.json) into USER.md the first time; memory.json is left as it was."""
    if os.path.exists(_path("user")) or not os.path.exists(LEGACY_FILE):
        return
    try:
        with open(LEGACY_FILE) as f:
            facts = json.load(f).get("facts", [])
    except (OSError, ValueError):
        return
    write("user", [f for f in facts if not secret_reason(f)])


def read(file: str) -> list[str]:
    _migrate()
    try:
        with open(_path(file), encoding="utf-8") as f:
            return [line[2:].strip() for line in f if line.startswith("- ") and line[2:].strip()]
    except OSError:
        return []


def write(file: str, entries: list[str]) -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(_path(file), "w", encoding="utf-8") as f:
        f.write(HEADERS[file] + "".join(f"- {e}\n" for e in entries))


def _clean(text: str) -> str:
    text = " ".join((text or "").replace("\n", " ").split()).lstrip("-• ").strip()
    if not text:
        raise ValueError("Nothing to save.")
    if len(text) > MAX_ENTRY_CHARS:
        raise ValueError(f"Too long (keep it under {MAX_ENTRY_CHARS} characters).")
    if (why := secret_reason(text)):
        raise ValueError(f"Not saved: {why}. Javid never stores passwords, keys or ID numbers.")
    return text


def add(file: str, text: str) -> str:
    text = _clean(text)
    entries = read(file)
    if text.lower() not in (e.lower() for e in entries):
        write(file, entries + [text])
    return text


def edit(file: str, index: int, text: str) -> bool:
    entries = read(file)
    if not 0 <= index < len(entries):
        return False
    entries[index] = _clean(text)
    write(file, entries)
    return True


def delete(file: str, index: int) -> bool:
    entries = read(file)
    if not 0 <= index < len(entries):
        return False
    del entries[index]
    write(file, entries)
    return True


def all_entries() -> dict:
    return {f: read(f) for f in FILES}


def block() -> str:
    """The fixed memory block that follows the system prompt."""
    parts = []
    for f in FILES:
        entries = read(f)
        if entries:
            parts.append(f"{TITLES[f]} ({FILES[f]}):\n" + "\n".join(f"- {e}" for e in entries))
    body = "\n\n".join(parts) if parts else "Nothing saved yet."
    return ("[Memory] " + body + "\nYou can't write memory yourself: to keep something, call propose_memory and "
            "Mahadi approves it. Never propose passwords, keys or ID numbers.")


# ---------- proposals: saved only after approval ----------
_pending: dict[str, dict] = {}
_turn = 0


def new_turn() -> None:
    global _turn
    _turn += 1


def propose(text: str, file: str = "user") -> tuple[str, dict | None]:
    """(tool result, proposal or None). Nothing is written here."""
    file = file if file in FILES else "user"
    try:
        text = _clean(text)
    except ValueError as e:
        return f"{e} Tell Mahadi it wasn't saved.", None
    if text.lower() in (e.lower() for e in read(file)):
        return f"Already saved in {FILES[file]}: {text}", None
    p = {"id": uuid.uuid4().hex[:10], "text": text, "file": file, "turn": _turn, "at": time.time()}
    _pending[p["id"]] = p
    return (f"Proposed for {FILES[file]}, NOT saved yet: \"{text}\". Ask Mahadi whether to save it "
            "(he can say yes, or use the card on screen). Don't say it's saved."), p


def pending() -> list[dict]:
    return sorted(_pending.values(), key=lambda p: p["at"])


def approve(pid: str, text: str | None = None) -> str:
    p = _pending.get(pid)
    if not p:
        raise KeyError(pid)
    saved = add(p["file"], text if text is not None else p["text"])   # re-checked for secrets
    del _pending[pid]
    return saved


def reject(pid: str) -> bool:
    return _pending.pop(pid, None) is not None


_YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|please do|do it|save it|go ahead|correct|right|"
                  r"হ্যাঁ|হ্যা|জি|ঠিক আছে|রাখো|সেভ কর)(?=[\s,.!?।]|$)[\w\s,.!'ঀ-৿]{0,30}$", re.I)
_NO = re.compile(r"^\W*(no|nope|nah|don'?t|do not|cancel|never ?mind|forget it|না|থাক|রেখো না)(?=[\s,.!?।]|$)", re.I)   # not \b: it fails after a Bengali vowel sign


def answer_from_reply(user_text: str) -> tuple[str, dict] | None:
    """A plain yes/no on the message right after a proposal decides it, in code (the model can't save).
    Returns ("approved"|"rejected", proposal) or None."""
    last = [p for p in _pending.values() if p["turn"] == _turn - 1]
    if not last:
        return None
    p = last[-1]
    text = user_text.strip()
    if _NO.match(text):
        reject(p["id"])
        return "rejected", p
    if _YES.match(text):
        try:
            p = {**p, "text": approve(p["id"])}
        except ValueError:
            reject(p["id"])
            return "rejected", p
        return "approved", p
    return None


# ---------- past-chat search (SQLite FTS5) ----------
INDEX_SCHEMA = "2"     # 2: + session column. An older index is dropped and rebuilt from chat_log.


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(INDEX_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    row = con.execute("SELECT v FROM meta WHERE k = 'schema'").fetchone()
    if not row or row[0] != INDEX_SCHEMA:
        con.execute("DROP TABLE IF EXISTS chats")
        con.execute("DELETE FROM meta")
        con.execute("INSERT INTO meta (k, v) VALUES ('schema', ?)", (INDEX_SCHEMA,))
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chats USING fts5(text, role UNINDEXED, ts UNINDEXED, "
                "session UNINDEXED, tokenize='unicode61 remove_diacritics 2')")
    con.commit()
    return con


def sync_index() -> int:
    """Index chat_log lines added since the last sync (the log only grows). Returns lines added."""
    con = _db()
    row = con.execute("SELECT v FROM meta WHERE k = 'lines'").fetchone()
    done = int(row[0]) if row else 0
    if not os.path.exists(HISTORY_FILE):
        con.close()
        return 0
    added, n = 0, 0
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if n <= done:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("role") in ("user", "assistant") and e.get("text"):
                con.execute("INSERT INTO chats (text, role, ts, session) VALUES (?, ?, ?, ?)",
                            (e["text"], e["role"], e.get("ts", ""), e.get("session", "")))
                added += 1
    if n < done:            # the log shrank (a chat was deleted, or tests replaced it): rebuild
        con.execute("DELETE FROM chats")
        con.execute("DELETE FROM meta WHERE k = 'lines'")
        con.commit()
        con.close()
        return sync_index()
    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('lines', ?)", (str(n),))
    con.commit()
    con.close()
    return added


_STOP = {"what", "did", "i", "say", "said", "tell", "told", "you", "me", "about", "the", "a", "an", "we", "talk",
         "talked", "last", "week", "time", "when", "my", "to", "of", "in", "on", "is", "was", "do", "does", "that",
         "it", "and", "or", "remind", "earlier", "before", "ago", "mention", "mentioned", "ask", "asked", "for"}


_WORD = re.compile(r"[\w\u0980-\u09ff]+")


def search_sessions(query: str, limit: int = 200) -> list[str]:
    """Chat ids whose messages contain every word typed in the Chats search box (prefix match)."""
    sync_index()
    words = [w for w in _WORD.findall((query or "").lower()) if len(w) > 1]
    if not words:
        return []
    con = _db()
    rows = con.execute("SELECT DISTINCT session FROM chats WHERE chats MATCH ? LIMIT ?",
                       (" AND ".join(f'"{w}"*' for w in words[:8]), limit)).fetchall()
    con.close()
    return [r[0] for r in rows if r[0]]


def search_history(query: str, limit: int = 5) -> str:
    sync_index()
    words = [w for w in _WORD.findall((query or "").lower()) if w not in _STOP and len(w) > 1]
    if not words:
        return "Give a word or two to search past chats for."
    match = " OR ".join(f'"{w}"*' for w in words[:8])
    con = _db()
    rows = con.execute("SELECT text, role, ts FROM chats WHERE chats MATCH ? ORDER BY bm25(chats) LIMIT ?",
                       (match, limit)).fetchall()
    con.close()
    if not rows:
        return f"Nothing in past chats matches '{query}'. Say you couldn't find it."
    lines = [f"Past chats matching '{query}' (best first):"]
    for text, role, ts in rows:
        when = datetime.fromisoformat(ts).strftime("%a %b %d, %I:%M %p") if ts else "unknown date"
        who = "Mahadi" if role == "user" else "Javid"
        lines.append(f"- {when}, {who}: {' '.join(text.split())[:200]}")
    lines.append("Answer from these; say the date. If none is about it, say you couldn't find it.")
    return "\n".join(lines)
