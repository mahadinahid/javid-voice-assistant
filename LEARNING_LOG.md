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
