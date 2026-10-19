"""
Agents: specialists with their own LLM loop, for jobs too big for one voice turn. Each is
agents/<name>/AGENT.md - frontmatter like a skill's (name, description, allowed-tools, and metadata:
max-rounds, spoken-seconds, limits, triggers, required) and a body that is the agent's system prompt.

Javid hands a job over with delegate(agent, task); a request matching a required agent's triggers is handed
over in code (think_stream), with the last two turns of the chat, so no request is spent choosing. The agent
starts fresh - its prompt with the memory block, then the [Context] note (language, date, the next 14 days)
and the task; no chat history - with only its own tools, per-tool call limits and its own round budget. Tools
listed under `prefetch` are called in code first (with the task's day) and their results come with the task:
gpt-oss-20b calls one tool per round, so the planner ran out of rounds before it had looked at the tasks.
Its tool calls stream to the page as progress notes.
It answers like a skill, SPOKEN: / SCREEN:. The spoken part becomes Javid's reply as it is (a handoff, see
think_stream: no second request to reword it) and the screen part goes under the reply. An answer built on
web results gets the usual figure checks: one rewrite inside the agent, then screen lines that still carry an
unconfirmed figure are dropped (finalize drops such sentences from the spoken part).
"""
import json
import os
import re
import uuid
from datetime import date, timedelta

import calendar_tools
import skills
import tracing
import web_tools

AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents")
DEFAULT_ROUNDS = 4
DEFAULT_SPOKEN_SECONDS = 40

_agents: dict[str, dict] = {}
_cfg: dict = {}     # set by javid.py: complete, tools, tool_map, context, memory, api_errors
_turn: dict = {}


def load() -> dict[str, dict]:
    _agents.clear()
    if os.path.isdir(AGENTS_DIR):
        for name in sorted(os.listdir(AGENTS_DIR)):
            path = os.path.join(AGENTS_DIR, name, "AGENT.md")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    agent = skills._parse(f.read())      # the same frontmatter layout as a skill
                if agent and agent["name"] == name:
                    _agents[name] = agent
    return _agents


def configure(**kw) -> None:
    _cfg.update(kw)


def all_agents() -> list[dict]:
    return [{"name": a["name"], "description": a["description"], "tools": a["tools"]} for a in _agents.values()]


def trigger_pattern() -> str:
    return "|".join(f"(?:{a['metadata']['triggers']})" for a in _agents.values() if a["metadata"].get("triggers"))


def required_agent(text: str) -> str | None:
    """The first required agent whose triggers match the request: that turn must go through delegate."""
    for a in _agents.values():
        t = a["metadata"].get("triggers")
        if a["metadata"].get("required") == "true" and t and re.search(t, text, re.I):
            return a["name"]
    return None


def agent_tools(name: str) -> list[str]:
    return list(_agents[name]["tools"]) if name in _agents else []


def followup_tools(name: str) -> list[str]:
    """Tools the turn after this agent's answer gets: its own plus metadata `follow-up-tools` (the planner
    asks whether to add its blocks, so a "yes" needs add_event)."""
    if name not in _agents:
        return []
    return agent_tools(name) + (_agents[name]["metadata"].get("follow-up-tools") or "").split()


# ---------- per turn ----------
def new_turn() -> None:
    _turn.clear()
    _turn.update(used=set(), handoff=None)


new_turn()


def tools_used() -> set[str]:
    """Tools the agents called this turn (they count for the turn's required tools)."""
    return set(_turn["used"])


def take_handoff() -> dict | None:
    h, _turn["handoff"] = _turn["handoff"], None
    return h


def _int(meta: dict, key: str, default: int) -> int:
    try:
        return int(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _limits(meta: dict) -> dict[str, int]:
    out = {}
    for part in (meta.get("limits") or "").split():
        name, _, n = part.partition("=")
        if n.isdigit():
            out[name] = int(n)
    return out


_UNCONFIRMED = ("isn't in any source", "couldn't confirm")


_DAY_WORD = re.compile(r"\b(today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                       r"\d{4}-\d{2}-\d{2})\b|(আজ|আগামীকাল|কাল|পরশু)", re.I)


def _task_day(task: str) -> str:
    """The day a task is about, as the calendar tools take it ('tomorrow', 'friday', 'কাল'); default today."""
    m = _DAY_WORD.search(task)
    if not m:
        return "today"
    word = (m.group(1) or m.group(2)).lower()
    return "today" if word == "tonight" else word


def _day_label(day: str) -> str:
    try:
        d = calendar_tools._parse_day(day)
    except ValueError:
        return day
    rel = {0: " (today)", 1: " (tomorrow)"}.get((d - date.today()).days, "")
    return f"{d:%A}, {d:%B} {d.day}{rel}"


_ITEM = re.compile(r"^- .*?(?:AM|PM|all day)\s*: (.+)$", re.M | re.I)    # "- 1:00 PM to 2:30 PM: Lunch with Sam"


def _missing(meta: dict, fetched: dict, day: str, spoken: str, screen: str) -> list[str]:
    """What an answer leaves out that metadata `must-show` requires: every item of those prefetched results
    (the day's calendar events), and the day itself. The planner's plan once left out both of tomorrow's
    events and was headed with today's date."""
    if not meta.get("must-show"):
        return []
    text = (spoken + "\n" + screen).lower()
    out = []
    for tool in meta["must-show"].split():
        for title in _ITEM.findall(fetched.get(tool, "")):
            words = re.findall(r"[a-z0-9']+", title.lower())[:3]
            if words and not all(w in text for w in words):
                out.append(title.strip())
    weekday = _day_label(day).split(",")[0]
    if weekday.lower() not in screen.lower():
        out.append(f"the day itself ({_day_label(day)}) at the top")
    return out


def _drop_unconfirmed(screen: str) -> str:
    return "\n".join(line for line in screen.splitlines()
                     if not any(m in p for p in web_tools.check_reply(line) for m in _UNCONFIRMED))


def _run_tool(agent: dict, call: dict, counts: dict, limits: dict):
    """One of the agent's tool calls (a generator: yields its progress note, returns the result)."""
    name = call["name"]
    if name not in agent["tools"]:
        return f"{name} isn't one of your tools ({', '.join(agent['tools'])})."
    if counts.get(name, 0) >= limits.get(name, 99):
        return f"You've used {name} {counts[name]} times, the limit. Write the answer now with what you have."
    try:
        args = json.loads(call["arguments"] or "{}")
    except ValueError:
        return f"Those {name} arguments weren't valid JSON; call it again."
    counts[name] = counts.get(name, 0) + 1
    _turn["used"].add(name)
    yield ("tool", name)
    span = tracing.tool_start(name)
    try:
        result = _cfg["tool_map"][name](args)
    except TypeError as e:
        result = f"Wrong arguments for {name}: {e}. Use only the parameters in its schema."
    except _cfg["api_errors"]:
        raise
    except Exception as e:
        result = f"{name} failed ({type(e).__name__})."
    finally:
        tracing.tool_end(span)
    if not isinstance(result, str):     # another agent: agents don't delegate
        return "That isn't available inside an agent."
    print(f"[agent] {agent['name']} {name}({args}) → {result[:80]}")
    return result


def delegate(agent: str = "", task: str = ""):
    """Run an agent. A generator: yields ("agent", name) and ("tool", name) progress events and returns the
    tool result for Javid's own conversation; the answer itself is left for take_handoff()."""
    a = _agents.get((agent or "").strip().lower())
    if not a:
        return f"No agent called '{agent}'. Agents: {', '.join(_agents) or 'none'}. Answer without one."
    task = " ".join((task or "").split()) or "(no task given)"
    yield ("agent", a["name"])
    meta = a["metadata"]
    limits, counts = _limits(meta), {}
    tools = [t for t in _cfg["tools"] if t["function"]["name"] in a["tools"]]
    msgs = [{"role": "system", "content": a["body"] + "\n\n" + _cfg["memory"]()},
            {"role": "user", "content": _cfg["context"]() + f"Mahadi's request: {task}"}]
    rounds = _int(meta, "max-rounds", DEFAULT_ROUNDS)
    text = ""
    strayed = False
    day, fetched = _task_day(task), {}
    try:
        if meta.get("prefetch"):
            facts = [f"The day to plan: {_day_label(day)}."]
            for name in meta["prefetch"].split():
                schema = next((t for t in tools if t["function"]["name"] == name), None)
                if not schema:
                    continue
                args = {"day": day} if "day" in schema["function"]["parameters"].get("properties", {}) else {}
                result = yield from _run_tool(a, {"name": name, "arguments": json.dumps(args)}, counts, limits)
                fetched[name] = result
                facts.append(f"{name}: {result}")
            msgs[1]["content"] += ("\n\nWhat your tools returned for this (current; use it, and call a tool only "
                                   "for something not here):\n\n" + "\n\n".join(facts))
        for round_no in range(rounds):
            tool_calls = {}
            # Last round: no tools, so the agent writes its answer with what it has.
            try:
                text = "".join(_cfg["complete"](tool_calls, msgs, [] if round_no == rounds - 1 else tools))
            except _cfg["api_errors"] as e:
                if "not in request" not in str(e) or strayed:
                    raise
                strayed = True      # it reached for one of Javid's tools it doesn't have: say which it has
                print(f"[agent] {a['name']} called a tool it doesn't have: {str(e)[:120]}")
                msgs.append({"role": "user", "content": f"[Note] Your only tools are {', '.join(a['tools'])}. "
                                                        "Use those, or write your answer now."})
                continue
            if not tool_calls:
                if text.strip():
                    break
                continue                    # an empty reply: ask again (it costs a round)
            calls = [tool_calls[i] for i in sorted(tool_calls)]
            for c in calls:
                c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:12]}"
            msgs.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]},
                 **({"extra_content": c["extra"]} if c.get("extra") else {})} for c in calls]})
            for c in calls:
                result = yield from _run_tool(a, c, counts, limits)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        spoken, screen = skills.split_reply(text)
        if not screen and text.strip():     # it skipped the screen part (the research report, once): ask for it
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "[Note] Your answer has no SCREEN part. Write it now: start with "
                                                 "'SCREEN:' and follow the format exactly."}]
            extra = "".join(_cfg["complete"]({}, msgs, []))
            screen = skills.split_reply(extra)[1] or re.sub(r"^\W*screen\W*:?\s*", "", extra.strip(), flags=re.I)
        if (missing := _missing(meta, fetched, day, spoken, screen)):
            print(f"[agent] {a['name']} left out: {missing}")
            msgs += [{"role": "assistant", "content": f"SPOKEN: {spoken}\nSCREEN:\n{screen}"},
                     {"role": "user", "content": "[Check, not from Mahadi] Your plan leaves out: " + "; ".join(missing)
                                                 + ". Rewrite the whole answer with them in it, at their times, in the "
                                                   "same format."}]
            retry = "".join(_cfg["complete"]({}, msgs, []))
            if retry.strip():
                spoken, screen = skills.split_reply(retry)
        if web_tools.used():
            problems = web_tools.check_reply(spoken + "\n" + screen)
            if problems:
                print(f"[agent] {a['name']} answer rejected: {problems}")
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": web_tools.correction(problems, in_skill=True)}]
                retry = "".join(_cfg["complete"]({}, msgs, []))
                if retry.strip():
                    spoken, screen = skills.split_reply(retry)
            screen = _drop_unconfirmed(screen)
    except _cfg["api_errors"]:
        raise                               # think_stream's API-failure reply (and regression's "inconclusive")
    except Exception as e:
        print(f"[agent] {a['name']} failed: {e!r}")
        return (f"The {a['name']} agent stopped ({type(e).__name__}). Tell Mahadi it couldn't finish, and "
                "answer briefly yourself if you can.")
    spoken = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", spoken)     # "[3]" citation marks are for the screen, not the voice
    spoken = skills.trim_spoken(spoken, _int(meta, "spoken-seconds", DEFAULT_SPOKEN_SECONDS))
    if not spoken:
        return f"The {a['name']} agent came back empty. Tell Mahadi it couldn't finish."
    _turn["handoff"] = {"agent": a["name"], "spoken": spoken, "screen": screen}
    return f"The {a['name']} agent answered Mahadi (spoken, and in full on screen): {spoken}"


def _schema() -> dict:
    listed = "; ".join(f"{a['name']}: {a['description'].split('. ')[0].rstrip('.')}" for a in _agents.values())
    return {"type": "function", "function": {
        "name": "delegate",
        "description": "Hand a big job to a specialist agent: it works on it with its own tools and answers Mahadi "
                       f"itself (spoken summary, full version on screen). Agents - {listed}.",
        "parameters": {"type": "object", "properties": {
            "agent": {"type": "string", "enum": list(_agents)},
            "task": {"type": "string", "description": "His request, plus what it needs from the chat (it can't see it)."}},
            "required": ["agent", "task"]}}}


load()
TOOLS = [_schema()] if _agents else []
TOOL_MAP = {"delegate": lambda args: delegate(**args)}
