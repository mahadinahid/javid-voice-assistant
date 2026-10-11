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
