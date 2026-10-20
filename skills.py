"""
Skills: step-by-step playbooks in skills/<name>/SKILL.md (agentskills.io layout).

    ---
    name: morning-brief
    description: One line saying when to use it.
    allowed-tools: morning_briefing news_headlines     # tools the skill unlocks for the rest of the turn
    metadata:
      spoken-seconds: "45"                           # spoken part's budget, enforced in code
      triggers: "\\bbrief\\b|headlines"               # words that send use_skill with a request
    ---
    Body: steps, tools to use, output format.

Only names and descriptions go in the (static) system prompt; the body is loaded when the model calls
use_skill(name). A skill's reply may have two parts, "SPOKEN: ..." and "SCREEN: ...": the spoken part is
read aloud (cut to its time budget here), the screen part is only shown.
"""
import os
import re

SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
WORDS_PER_SECOND = 2.3      # Andrew speaks ~140 words a minute
DEFAULT_SPOKEN_SECONDS = 30

_skills: dict[str, dict] = {}
_turn: dict = {}


def _parse(text: str) -> dict | None:
    m = re.match(r"---\s*\n(.*?)\n---\s*\n?(.*)", text, re.S)
    if not m:
        return None
    meta, current = {}, None
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"').strip("'")
        if line.startswith((" ", "\t")) and current is not None:
            meta[current][key.strip()] = value     # raw: triggers are regexes ("\\bbrief\\b")
        elif value:
            meta[key.strip()], current = value, None
        else:
            meta[key.strip()], current = {}, key.strip()
    if not meta.get("name") or not meta.get("description"):
        return None
    return {"name": meta["name"], "description": meta["description"],
            "tools": (meta.get("allowed-tools") or "").split(), "metadata": meta.get("metadata") or {},
            "body": m.group(2).strip()}


def load() -> dict[str, dict]:
    _skills.clear()
    if os.path.isdir(SKILLS_DIR):
        for name in sorted(os.listdir(SKILLS_DIR)):
            path = os.path.join(SKILLS_DIR, name, "SKILL.md")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    skill = _parse(f.read())
                if skill and skill["name"] == name:     # agentskills: name must match its folder
                    _skills[name] = skill
    return _skills


def all_skills() -> list[dict]:
    return [{"name": s["name"], "description": s["description"]} for s in _skills.values()]


def prompt_index() -> str:
    if not _skills:
        return ""
    listed = " ".join(f"{s['name']}: {s['description'].rstrip('.')}." for s in _skills.values())
    return (" Skills (playbooks): when a request fits one, call use_skill with its name first and follow "
            f"what it returns. {listed}")


def trigger_pattern(required_only: bool = False) -> str:
    """Every skill's triggers as one regex; required_only: just the skills marked `required: "true"`,
    whose requests must go through use_skill (gpt-oss-120b answered "yuan rate in taka" from four web
    searches instead of the exchange-rates skill, since answered in code by rates.py)."""
    return "|".join(f"(?:{s['metadata']['triggers']})" for s in _skills.values() if s["metadata"].get("triggers")
                    and (not required_only or s["metadata"].get("required") == "true"))


# ---------- per turn ----------
def new_turn() -> None:
    _turn.clear()


def active() -> dict | None:
    return _turn.get("skill")


def use_skill(name: str) -> str:
    skill = _skills.get((name or "").strip())
    if not skill:
        return f"No skill called '{name}'. Skills: {', '.join(_skills) or 'none'}. Answer without one."
    _turn["skill"] = skill
    tools = f" Tools now available: {', '.join(skill['tools'])}." if skill["tools"] else ""
    return f"Skill {skill['name']} loaded.{tools}\n\n{skill['body']}"


def spoken_seconds(skill: dict) -> int:
    try:
        return int(skill["metadata"].get("spoken-seconds", DEFAULT_SPOKEN_SECONDS))
    except ValueError:
        return DEFAULT_SPOKEN_SECONDS


_SPOKEN = re.compile(r"^\W*spoken\W*:?\s*", re.I | re.M)
_SCREEN = re.compile(r"^\W*screen\W*:?\s*$|^\W*screen\W*:\s*", re.I | re.M)


def split_reply(text: str) -> tuple[str, str]:
    """(spoken, screen) from a skill reply. Without the markers, all of it is spoken (and trimmed)."""
    m = _SCREEN.search(text)
    spoken, screen = (text[:m.start()], text[m.end():]) if m else (text, "")
    spoken = _SPOKEN.sub("", spoken, count=1)
    return " ".join(spoken.split()), screen.strip()


_INSTRUCTION = re.compile(r"^\s*(Use these|Answer |Only state|If the snippets|Tell Mahadi|Say |Sources give)", re.I)


def screen_from_results(results: list[str]) -> str:
    """The full version when the model left out its SCREEN part: what the skill's tools returned,
    minus the lines that were instructions to the model."""
    lines = [l.rstrip() for r in results for l in r.splitlines()]
    return "\n".join(l for l in lines if l.strip() and not _INSTRUCTION.match(l)).strip()


_BOUNDARY = re.compile(r"(?<=[.!?।])[\"')\]]*\s+")


def trim_spoken(text: str, seconds: int) -> str:
    """Whole sentences that fit in `seconds` of speech (the first one always stays)."""
    budget = int(seconds * WORDS_PER_SECOND)
    out, words = [], 0
    for sentence in _BOUNDARY.split(text.strip()):
        n = len(sentence.split())
        if out and words + n > budget:
            break
        out.append(sentence)
        words += n
    return " ".join(out)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "use_skill",
        "description": "Load a skill's step-by-step instructions (and unlock its tools) before doing the task.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "The skill's name."}},
                       "required": ["name"]},
    },
}]
TOOL_MAP = {"use_skill": lambda args: use_skill(**args)}

load()
