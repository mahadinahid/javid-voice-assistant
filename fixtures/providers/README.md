# Provider fixtures

Responses `regression.py` replays so its default run never calls a real API. Each file says where it comes from:

- `groq_headers.json`: **recorded** 2026-10-25 from one real request to `openai/gpt-oss-20b` (only the
  `x-ratelimit-*` headers and the token counts were kept).
- `groq_429.json`, `gemini_429.json`: the error texts as Javid has logged them (org id replaced).
- `deepseek_402.json`, `tavily_432.json`, `elevenlabs_subscription.json`: the providers' documented formats, not
  recorded (no key, or the error never happened here).
- `openrouter_429_daily.json`, `openrouter_429_upstream.json`: OpenRouter's documented 429s (the account's free
  daily limit, whose reset time `RESET_MS` the test fills in; a free model busy upstream). Not recorded: no key yet.

ElevenLabs' text-to-speech is faked in code (`FakeElevenLabs` in `regression.py`: the documented subscription
response above, fake MP3 bytes for speech): no key yet, so nothing is recorded.

When a provider is set up, record its real response with `python regression.py --live` and replace the documented
one. Never put a key, a request id or personal data in a fixture.
