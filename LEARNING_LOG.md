# Javid Learning Log

## Day 1 — Voice Loop v0 (2026-10-11)

### What I built
A push-to-talk voice assistant called Javid:
- Microphone input recorded between two Enter presses (sounddevice)
- Speech transcribed to text via Groq Whisper (whisper-large-v3-turbo)
- Text sent to Groq LLM (openai/gpt-oss-20b) with full conversation history
- Reply spoken aloud via macOS `say` with Aman voice
- Session memory: the full `messages` list is sent every turn so Javid remembers the conversation

### What broke and how we fixed it
1. **Word "goodbye" in first transcription** — mic picked up background audio from the smoke test, which included "Goodbye". The exit trigger fired immediately. Fix: run again and speak cleanly.
2. **BadRequestError on internet questions** — `openai/gpt-oss-20b` tried to call a built-in `browser.run` tool, but no tools were registered. Fix: wrapped `client.chat.completions.create()` in a try/except BadRequestError that returns a graceful fallback message. Tool calling comes in Day 2.
3. **API key pasted in chat twice** — pasted the key into the chat window instead of the .env file. Both keys were rotated immediately on Groq console. Fix: open .env in TextEdit and type there, never here.

### Quiz answers
1. **Q: What does Javid lose if you remove `messages.append({"role": "assistant", "content": reply})`?**
   A: He loses memory of his own previous replies. The LLM still runs, but it never sees what it said before, so context breaks across turns.

2. **Q: Why use `io.BytesIO()` instead of writing to disk?**
   A: Speed and cleanliness — RAM is faster than disk, and there's no file to clean up. `transcribe()` only needs the bytes, not a saved file.

3. **Q: What happens if `SAMPLE_RATE` is changed from 16000 to 8000?**
   A: Audio quality drops (only captures up to 4 kHz, telephone quality), and Whisper accuracy suffers because it's trained on 16 kHz audio.

### Tomorrow's goal — Day 2: Tool Calling
Add a `add_task` tool so saying "add a task: buy groceries" writes an entry to `tasks.txt`. This introduces Groq's tool-calling API and shows how the LLM can trigger real actions, not just answer questions.

---

## Day 2 — Tool Calling (2026-10-11)

### What I built
Added tool calling to Javid (now v1):
- `TOOLS` list describes `add_task` to the LLM in JSON schema format
- `add_task()` function writes a timestamped entry to `tasks.txt`
- `think()` updated to handle the two-step roundtrip: first API call returns a tool call request, code executes the function, second API call turns the result into a spoken reply
- The LLM never touches the file system directly — it only requests actions; Python executes them

### Key insight
The LLM is not stateful and cannot DO anything on its own. Tool calling is the bridge: the model says "call function X with argument Y", your code runs it, and the result is sent back so the model can reply naturally. You stay in control of what actually happens.

### What broke and how we fixed it
Nothing broke — smoke test passed first try. The Day 1 BadRequestError (model trying to call `browser.run`) is now less likely since we provide valid tools the model can call instead.

### Tomorrow's goal — Day 3: list_tasks + Web Search
Add a `list_tasks` tool so Javid can read back your task list. Then add a web search tool using a search API so Javid can answer questions about current events.

---

## Day 3 — Multiple Tools + TOOL_MAP Dispatcher (2026-10-11)

### What I built
Upgraded Javid to v2 with three tools and a clean dispatcher:
- `list_tasks()` — reads tasks.txt and returns all entries so Javid can read your list back aloud
- `web_search(query)` — searches DuckDuckGo via the `ddgs` package (no extra API key needed)
- `TOOL_MAP` dictionary — maps tool names to functions, replacing the hardcoded if/else. Adding a new tool now only requires one entry in TOOL_MAP and one schema in TOOLS
- Added a `[tool]` print line so you can see which tool fired and what it returned during a session

### Key insight
An agent's power comes from its tools, not from the LLM itself. The LLM decides *which* tool to call and *when* — your Python code decides *what actually happens*. The TOOL_MAP pattern makes this easy to extend.

### What broke and how we fixed it
1. **`duckduckgo_search` package renamed** — the package was renamed to `ddgs` in a recent release. Fix: `pip install ddgs`, update import to `from ddgs import DDGS`, removed old package.

### Tomorrow's goal — Day 4
Add persistent memory across sessions (save key facts to a JSON file so Javid remembers things between restarts), and improve the system prompt with Mahadi's preferences.

---

## Day 4 — Persistent Memory (2026-10-11)

### What I built
Upgraded Javid to v3 with long-term memory across restarts:
- `memory.json` stores a list of facts as plain JSON
- `remember(fact)` tool — LLM calls this to save something important about Mahadi
- `recall()` tool — LLM calls this to read back stored facts mid-conversation
- `load_memory()` + `build_system_prompt()` — on every startup, facts from memory.json are injected into the system prompt so Javid already knows them before the first word is spoken

### Three layers of memory (key mental model)
| Layer | Where | Survives restart? |
|---|---|---|
| In-context | `messages` list in RAM | No |
| Persistent | `memory.json` on disk | Yes |
| Model weights | Baked into the LLM | Always (but can't change) |

### What broke and how we fixed it
Nothing broke. Smoke test confirmed Javid answered "What is my favourite programming language?" correctly on a fresh start, with no prompting — proof the memory injection works.

### Tomorrow's goal — Day 5
Add a `delete_task` tool and a `clear_memory` tool for housekeeping. Then explore making Javid proactive: after adding a task, check if it conflicts with anything in the list.

---

## Day 5 — Task Management + Proactive Hints (2026-10-11)

### What I built
Upgraded Javid to v4 with full task management and prompt-driven proactivity:
- `list_tasks` upgraded to return numbered output (1. 2. 3. …) so tasks can be referenced by number
- `delete_task(number)` — removes task N, shifts remaining tasks up
- `complete_task(number)` — marks task N with ✓ prefix in place
- `clear_memory()` — resets memory.json to empty facts list
- `_read_lines()` / `_write_lines()` — shared helpers so all mutating tools use one consistent read/write pattern
- Proactive duplicate detection: one sentence added to the system prompt tells Javid to call list_tasks after adding a task and warn about similar entries — no extra code needed

### Key insight: prompt engineering vs. code
Making Javid proactive took one sentence in the system prompt, not a new function. This is the core prompt engineering lesson: behavior that feels like intelligence is often just a well-worded instruction. Code handles the *mechanism*; the prompt shapes the *judgment*.

### What broke and how we fixed it
Nothing broke. Smoke test confirmed: complete, delete, and clear all mutate the file correctly. Javid flagged a duplicate task ("Prepare Day 3 notes" appeared twice) without being asked — proof the proactive instruction works.

### Tomorrow's goal — Day 6
Add time and date awareness (Javid knows what day/time it is), a morning briefing tool (reads tasks + weather summary when you say "good morning"), and personality tuning.

---

## Day 6 — Context Awareness + Personality (2026-10-11)

### What I built
Upgraded Javid to v5 with time awareness and a distinct voice:
- `get_datetime()` — returns current day, date, and time; called whenever Javid needs to know when it is
- `morning_briefing()` — greets appropriately (morning/afternoon/evening), states the time, counts pending vs completed tasks, lists pending ones for a spoken overview
- Personality rewrite — system prompt now describes Javid as "sharp, reliable, calm and confident like a friend, not a corporate bot." Test 3 shows the difference immediately
- Tool instruction refactored — system prompt now lists all tools in one tight sentence each, easier to read and extend

### Key insight: static vs. dynamic context
- Static context (who Mahadi is, his preferences) → injected into system prompt at startup from memory.json. Costs zero tokens at runtime.
- Dynamic context (what time is it, what tasks are pending) → tools the LLM calls on demand. Only fetched when actually needed.
- Never inject dynamic data at startup — it goes stale immediately. Use a tool instead.

### What broke and how we fixed it
Nothing broke. All three smoke tests passed first try.

### Tomorrow's goal — Day 7
Final day: polish and wrap-up. Add a startup greeting (Javid speaks when he starts, not just when asked), improve error messages, review all 7 days of code for anything to clean up, and do a final commit tagging v1.0.
