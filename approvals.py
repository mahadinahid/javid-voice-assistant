"""
Write actions that wait for approval. On Telegram (telegram_bot.py) every one of javid's WRITE_TOOLS waits for a
button: think_stream(gate=...) hands each call to propose(), which works out exactly what it would do - the task
line, the calendar event (found by title words as always; then only its id is kept), the day and time resolved to a
date and a clock time, the memory text - describes it in words and keeps it for APPROVE_SEC. Only take() + run() with
that action's id does it, once: a second press or a late one does nothing. The model is told the action is waiting,
never that it's done.

On the web page and in the CLI (2026-11-14, SECURITY.md item 2) only the DESTRUCTIVE tools wait: javid's own gate
(_web_gate) proposes them with typed_yes=True, the page shows an Approve card (POST /approvals/{id}/approve), and a
plain yes typed or said push-to-talk on the very next message approves too (answer(), decided in code like a memory
proposal); a hands-free yes never does (the page can hear the TV). The CLI asks y/N after the reply.
"""
import re
import secrets
import time

import calendar_tools
import drive_tools
import javid
import memory_store

APPROVE_SEC = 120
DESTRUCTIVE = ("delete_task", "delete_event", "move_event")    # what waits on the web page and in the CLI
_actions: dict[str, dict] = {}      # id -> {"id", "tool", "at", "turn", "typed_yes", "text", plus what run() needs}
_turn = 0                           # bumped once per user message (javid.think_stream): a typed yes answers the last turn's card


def new_turn() -> None:
    global _turn
    _turn += 1


def propose(tool: str, args: dict, refuse_memory: str | None = None, typed_yes: bool = False) -> tuple[str, dict | None]:
    """(what the model is told, the pending action, or None when there is nothing to approve).
    refuse_memory: why a memory proposal is refused on this turn (the channel decides; None = allowed).
    typed_yes: a plain yes on the next message may approve it (the web page, the CLI), not only a button (Telegram)."""
    plan = _plan(tool, dict(args), refuse_memory)
    if isinstance(plan, str):
        return plan, None
    action = {"id": secrets.token_urlsafe(9), "tool": tool, "at": time.time(), "turn": _turn, "typed_yes": typed_yes, **plan}
    _actions[action["id"]] = action
    if typed_yes:
        return (f"NOT done yet: it waits for Mahadi's approval of \"{action['text']}\" within 2 minutes: the Approve card "
                "on screen, or a plain yes as his next message. Tell him that in a few words; don't say it's done."), action
    return (f"NOT done yet: it waits for Mahadi to tap Approve on \"{action['text']}\" within 2 minutes (a typed yes "
            "doesn't count). Tell him to tap Approve; don't say it's done."), action


def pending() -> list[dict]:
    """The web page's cards still waiting (id, text, tool, seconds left), oldest first."""
    now = time.time()
    return [{"id": a["id"], "text": a["text"], "tool": a["tool"], "expires_in": max(0, int(APPROVE_SEC - (now - a["at"])))}
            for a in sorted(_actions.values(), key=lambda a: a["at"]) if a["typed_yes"] and now - a["at"] <= APPROVE_SEC]


def made_this_turn() -> list[dict]:
    """The actions this turn proposed (the CLI asks y/N for each after the reply)."""
    return [a for a in _actions.values() if a["turn"] == _turn and a["typed_yes"]]


_YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok(ay)?|please do|do it|go ahead|confirm(ed)?|approved?|"
                  r"(delete|move|remove) it|হ্যাঁ|হ্যা|জি|ঠিক আছে|মুছে দাও|করে দাও)(?=[\s,.!?।]|$)[\w\s,.!'ঀ-৿]{0,30}$", re.I)
_NO = re.compile(r"^\W*(no|nope|nah|don'?t|do not|cancel|never ?mind|forget it|stop|না|থাক|বাদ দাও)(?=[\s,.!?।]|$)", re.I)


def waiting() -> dict | None:
    """The web card a plain yes/no on this message would answer: the newest one made on the previous message."""
    last = [a for a in _actions.values() if a["typed_yes"] and a["turn"] == _turn - 1]
    return last[-1] if last else None


def answer(user_text: str, yes_ok: bool = True) -> tuple[str, dict, str] | None:
    """A plain yes/no right after a web card decides it in code (no model request): ("done", action, the outcome),
    ("rejected", ...), ("expired", ...), or with yes_ok False (hands-free) ("tap", ...): the card is asked about again
    and stays answerable by the next message. None when the message isn't a yes/no, or no card waits."""
    action = waiting()
    if not action:
        return None
    text = user_text.strip()
    if _NO.match(text):
        drop(action["id"])
        return "rejected", action, "Nothing was done."
    if not _YES.match(text):
        return None
    if not yes_ok:
        action["turn"] = _turn
        return "tap", action, ""
    taken, note = take(action["id"])
    if not taken:
        return "expired", action, note
    return "done", action, run(taken)


def _plan(tool: str, args: dict, refuse_memory: str | None):
    """What the action will do, bound to its exact target, or a sentence for the model (nothing to approve)."""
    if tool == "add_task":
        task = " ".join(str(args.get("task") or "").split())
        return {"text": f"Add task: {task}", "args": {"task": task}} if task else "No task given."
    if tool in ("complete_task", "delete_task"):
        lines = javid._read_lines()
        if not lines:
            return "No tasks yet."
        i = javid._find_task(lines, args.get("number"), args.get("text"))
        if isinstance(i, str):
            return i
        if tool == "complete_task" and lines[i].startswith("✓"):
            return f"Task #{i + 1} is already marked complete."
        verb = "Delete" if tool == "delete_task" else "Mark done"
        return {"text": f"{verb} task: {javid._task_parts(lines[i])[2]}", "line": lines[i]}
    if tool == "add_event":
        plan = calendar_tools.describe_add(**args)
        return plan if isinstance(plan, str) else {"text": plan["text"], "args": plan["args"]}
    if tool in ("move_event", "delete_event"):
        ev = calendar_tools.lookup_event(args.get("title", ""), args.get("day", ""), args.get("time", ""))
        if isinstance(ev, str):
            return ev
        if tool == "delete_event":
            return {"text": f"Delete {calendar_tools.describe_event(ev)}", "event": ev["id"]}
        move = calendar_tools.describe_move(ev, args.get("new_day", ""), args.get("new_time", ""))
        if isinstance(move, str):
            return move
        return {"text": move["text"], "event": ev["id"], "args": {"new_day": move["new_day"], "new_time": move["new_time"]}}
    if tool == "propose_memory":
        text = " ".join(str(args.get("text") or "").split())
        if refuse_memory:
            return refuse_memory
        if (why := memory_store.secret_reason(text)):
            return f"NOT saved: {why}. Tell Mahadi it wasn't saved."
        if not text:
            return "Nothing to save."
        file = args.get("file") if args.get("file") in memory_store.FILES else "user"
        return {"text": f"Save to memory: {text}", "args": {"text": text, "file": file}}
    if tool == "save_report":          # Google Drive: bound to the document as it is now
        report = args.get("report") or {}
        if not report.get("screen"):
            return "There's no report to save."
        doc = drive_tools.document(report)
        if (why := drive_tools.problem(doc)):
            return drive_tools.say(why)
        return {"text": f"Save to Google Drive: {doc['name']}", "report": report}
    return f"{tool} can't wait for approval."


def take(aid: str) -> tuple[dict | None, str]:
    """The action to run now, or (None, why not). Single use: once taken it's gone."""
    action = _actions.pop(aid, None)
    if not action:
        return None, "Expired or already handled. Nothing was done."
    if time.time() - action["at"] > APPROVE_SEC:
        return None, "Expired: nothing was done. Ask again if you still want it."
    return action, ""


def drop(aid: str) -> bool:
    return _actions.pop(aid, None) is not None


def clear() -> None:
    _actions.clear()


def sweep() -> list[str]:
    """Drop expired actions; their ids (the channel marks their buttons expired)."""
    gone = [aid for aid, a in _actions.items() if time.time() - a["at"] > APPROVE_SEC]
    for aid in gone:
        del _actions[aid]
    return gone


def run(action: dict) -> str:
    """Do exactly the approved action. The outcome in a sentence. Call under javid.hold_turn."""
    tool = action["tool"]
    if tool == "add_task":
        result = javid.add_task(action["args"]["task"])
    elif tool in ("complete_task", "delete_task"):
        lines = javid._read_lines()
        if action["line"] not in lines:        # bound to the line shown, not to a number that may have shifted
            return "That task has changed since, so nothing was done."
        done = javid.delete_task if tool == "delete_task" else javid.complete_task
        result = done(number=lines.index(action["line"]) + 1)
    elif tool == "add_event":
        result = calendar_tools.add_event(**action["args"])
    elif tool == "move_event":
        result = calendar_tools.move_event_id(action["event"], **action["args"])
    elif tool == "delete_event":
        result = calendar_tools.delete_event_id(action["event"])
    elif tool == "propose_memory":
        result, proposal = memory_store.propose(**action["args"])
        if proposal:
            try:
                result = f"Saved to memory: {memory_store.approve(proposal['id'])}"   # secrets checked again here
            except ValueError as e:
                memory_store.reject(proposal["id"])
                result = f"Not saved: {e}"
    elif tool == "save_report":
        done = drive_tools.save(action["report"])
        result = drive_tools.say(done["said"]) + (f" {done['link']}" if done.get("link") else "")
        return result
    else:
        return "Nothing was done."
    return result.split(". ")[0].rstrip(".") + "."     # the tool's first sentence: what happened
