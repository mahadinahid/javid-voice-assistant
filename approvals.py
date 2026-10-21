"""
Write actions that wait for a button. On a channel whose writes need one (Telegram, telegram_bot.py), javid's
WRITE_TOOLS don't run when the model calls them: think_stream(gate=...) hands each call to propose(), which works
out exactly what it would do - the task line, the calendar event (found by title words as always; then only its
id is kept), the day and time resolved to a date and a clock time, the memory text - describes it in words and
keeps it for APPROVE_SEC. Only take() + run() with that action's id does it, once: a typed "yes", a second press
or a late one does nothing. The model is told the action is waiting, never that it's done.
"""
import secrets
import time

import calendar_tools
import javid
import memory_store

APPROVE_SEC = 120
_actions: dict[str, dict] = {}      # id -> {"id", "tool", "at", "text", plus what run() needs}


def propose(tool: str, args: dict, refuse_memory: str | None = None) -> tuple[str, dict | None]:
    """(what the model is told, the pending action, or None when there is nothing to approve).
    refuse_memory: why a memory proposal is refused on this turn (the channel decides; None = allowed)."""
    plan = _plan(tool, dict(args), refuse_memory)
    if isinstance(plan, str):
        return plan, None
    action = {"id": secrets.token_urlsafe(9), "tool": tool, "at": time.time(), **plan}
    _actions[action["id"]] = action
    return (f"NOT done yet: it waits for Mahadi to tap Approve on \"{action['text']}\" within 2 minutes (a typed yes "
            "doesn't count). Tell him to tap Approve; don't say it's done."), action


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
    else:
        return "Nothing was done."
    return result.split(". ")[0].rstrip(".") + "."     # the tool's first sentence: what happened
