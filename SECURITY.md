# Security

Javid is a single-user assistant on Mahadi's Mac: a local web server (`server.py`, 127.0.0.1:8000), a terminal
loop (`javid.py`) and a Telegram bot (`telegram_bot.py`). It holds API keys, a Google Calendar token, tasks,
memory and every chat. This file lists what it defends against, how, and what is still open.

Last reviewed: 2026-10-21 (news, watches and the self-check added 2026-10-24; providers and personal data
2026-10-25; colour themes, Groq cleared for personal data, and the status of items 3 and 4 updated 2026-11-02).

## Threats

1. **A stranger on Telegram** finds the bot and tries to use it, pair with it, or flood it.
2. **A lost or stolen phone** with the paired Telegram account still logged in.
3. **Prompt injection**: text Javid didn't write (a forwarded message, a web page, a link's content, a calendar
   invite) tells the model to delete, change or remember something.
4. **The model acting on its own**: calling a write tool on a misunderstanding, or claiming a typed "yes" was an
   approval.
5. **Leaking secrets or ids**: API keys, the bot token, the OAuth token or account ids ending up in git, logs or
   replies.
6. **Other websites and programs** reaching the local server (cross-site requests, DNS rebinding, other local
   accounts).
7. **Third parties seeing personal data**: model providers, search, speech services.
8. **Vulnerable dependencies**.

## Telegram (done 2026-10-21)

| Threat | Protection (in code) | Test (`regression.py --offline -k telegram`) |
|---|---|---|
| Stranger | Only ids in `TELEGRAM_ALLOWED_IDS` (`.env`) reach Javid. Anyone else gets **no reply at all**, not even to a wrong pairing code or a button press. Each attempt is logged to `data/telegram_denied.jsonl` (mode 600) as account id, time and kind, never the content, capped at 20 lines a minute. | unknown account gets silence; only id and time logged |
| Pairing brute force | A one-time 6-digit code, accepted only while the allowlist is empty. A new code after 5 wrong tries, and pairing closes after 20 until restart. The code comes from the Mac's own page (QR) or the server log. | pairing by code fills the allowlist |
| Bot token | Only in `.env` (gitignored, written owner-only from the moment it exists). Never logged: error text is cleaned (`_safe`), and the `httpx`/`httpcore` loggers are pinned to WARNING because they would log token-bearing URLs. Verified 2026-10-21 that it is not in any git object (reachable or not) or any log file. The bot is @Javidinstractor_bot, not @mahadi_crm_bot. | set up from the page (`.env` mode 600) |
| Open port | Long polling only: nothing listens for Telegram, and a webhook, if one was ever set, is deleted at start. (No webhook was set on 2026-10-21.) | start removes a webhook |
| Writes | Adding, completing or deleting a task, adding, moving or deleting an event, and saving to memory never run when the model calls them. Each becomes an action bound to its exact target (the task's line, the event's id, the date resolved now), shown as "Approve this?" with ✅ Approve / ✖ Reject. It runs once, only from that message's button, within 2 minutes. A typed "yes", a late, repeated or foreign press, or a stranger's press does nothing. The gate sits inside `TOOL_MAP`, so agents' calls go through it too. | saving to memory waits for Approve; expired Approve / typed yes / Reject; deletes exactly the task it showed, once |
| Forwarded messages | Never a turn: the text is kept in the chat tagged as someone else's words, with a canned reply. No model request is made, so nothing in it can act. Forwarded voice notes and files aren't opened. | a forwarded "delete all events" does nothing |
| Files and links | Files aren't opened, and only a voice note recorded in Telegram is transcribed (not an audio file). A message with a link gets no web tools and no research agent, so Javid never opens or searches it. | a message with a link gets no web tools |
| Untrusted text to memory | A memory save is refused when the message has a link, the text holds one, or a forwarded message is among the turns the model can see (or in this turn's search results). | forwarded…never reaches memory; link isn't saved |
| Flood | At most 20 messages and presses a minute, with one "slow down" notice a minute. | 20 messages a minute |
| Lost phone | `/lock` (it works even past the rate limit) turns remote control off: no turns, no button actions, no reminder pushes, and pending approvals are dropped. It persists across restarts. Only the Mac's `/telegram` page can unlock it (`POST /telegram/lock`, same-origin only). | /lock stops everything until the Mac unlocks it |
| Ids and keys in replies | Every message, edit, button answer and voice note passes `_scrub`. It hides configured keys and key shapes, the bot token, the allowed and chat ids, the bot's id, passport-style numbers and 9+ digit runs outside links. A phone number right after "phone/mobile/call" stays. The `/telegram` status page shows no ids either. | replies never carry keys or ID numbers |

## Secrets (verified 2026-10-21)

- `.env` (Groq, Gemini, Tavily and IFM keys, the bot token, the Telegram allowlist), `token.json` (Google OAuth
  access and refresh token), `credentials.json` (OAuth client secret) and `telegram.json` are gitignored and mode
  600. gitleaks found nothing in the 47 commits, and no secret file was ever committed.
- Keys pasted in the Models panel are verified with the provider, written to `.env` owner-only and never sent back.
  (The panel still shows each key's last 4 characters: see open items.)
- The Google scope is `calendar.events`, not full `calendar`.

## News, watches and the self-check (added 2026-10-24)

- **Google News** (`gnews.py`): Javid fetches news.google.com's RSS for the topics asked (and, for a news watch, its
  subject). Google sees the topic words, nothing else: no account, key or personal data goes with them.
- **Watches** (`watches.py`) only fetch prices (Binance, Yahoo Finance), headlines (Google News) and the forecast
  (Open-Meteo) on a timer, and their only action is an alert that rings like a timer. A watch is made only from a
  message Mahadi sent himself (the page, the CLI, or the allowed Telegram account; a forwarded Telegram message is
  never a turn), in code, not by the model; a news subject with a link is refused. Cancelling on the page goes
  through `_local_page_only`; `GET /watches` has no check, like the other read routes (open item 1). Setting or
  cancelling a watch from Telegram doesn't need Approve, like a timer (open item 9).
- **Self-check** (`selfcheck.py`): the last 300 clips' outcomes and first 120 characters, edge-tts and Whisper
  failures are kept in memory only (never written to disk) and shown only when Mahadi asks what's wrong; the
  last ignored clip's words are on that answer's screen. Its Groq probe lists models (no message content).

## Providers and personal data (added 2026-10-25)

- **Registry** (`config/providers.yaml`): every outside service with a key or an allowance. A provider is called only
  when it is enabled there and every key it names is in `.env`; the start-up line and the Providers panel show
  variable names, never values (the model keys' last 4 characters in the key controls are open item 7).
- **Personal data only where Mahadi allows it**: `personal_ok` is false for a provider until he sets it after
  reading its data policy. He set it for Groq (2026-10-26, committed 2026-11-02); it is false for every other
  provider. A provider without it gets each request without the memory block, the chat history, the
  older-chat summary, the carried calendar facts and earlier drafts, and isn't offered the calendar, task, memory or
  past-chat tools. A turn that needs those (by its words, required tools or agent, a personal tool's result, or a model
  reaching for a personal tool) goes only to providers with `personal_ok`; with none, nothing is sent and Javid says
  so. The privacy tags never leave Javid (`_messages_for` strips them). Agents follow the same rule.
- **Search queries**: a query carrying words from a task, a memory line or a calendar or task result isn't sent to a
  search engine without `personal_ok` (the search router checks each engine: Tavily, Exa, Brave), nor to Cohere for
  reranking, nor to the Google News fallback (Mahadi's own words in that message are allowed). Firecrawl only ever
  gets a result's URL. The router's cache (in memory) and the ledger never keep a query.
- **Spending**: a provider is held back at 80 % of its free allowance and off at 95 %; a call whose cost would pass the
  allowance isn't made; a 429, a 402 or a quota error rests it until its retry-after, across restarts.
  - **OpenRouter** (2026-10-25) is free models only, enforced in code: a model id not ending in `:free` is refused
    before any request (`javid._create`, `PaidModelRefused`), can't be saved in the chain, listed, pinged or smoke-
    tested, and request fields that would route to another model are never sent. A response that reports any cost
    rests OpenRouter for 30 days. Belt and braces on the account side: give the OpenRouter key a credit limit of $0.
    Its free endpoints may log prompts or train on them (OpenRouter's data policy), one more reason `personal_ok`
    stays false there.
  - **Cloudflare Workers AI** is counted in neurons from each reply's tokens at the model's published price; a model
    without a price is never called, and a request whose estimate could pass the free 10,000 a day isn't sent (on
    the Workers Paid plan, more would be billed).
- **The backup voice** (ElevenLabs, 2026-10-26) gets the text of a sentence only after every edge-tts try failed, and
  never a reply to a turn that needed calendar, tasks, memory or past chats while it has no `personal_ok`. Its key goes
  only in the `xi-api-key` header. Clips of Javid's fixed phrases are kept in `data/tts_cache/` (mode 600): no
  reply text is stored. edge-tts itself still gets every spoken reply (Microsoft), as before.
- **Google Drive** (2026-10-26): Javid's own OAuth client with `drive.file` only (it sees only the files it made), its
  own token `token_drive.json` (mode 600, gitignored), used only if its client id is Javid's and its scope exactly
  `drive.file` (never another app's token, such as Hermes's). Saving is decided in code, never by the model, and asks
  first unless Mahadi said "save"; on Telegram it waits for Approve. Only a report built from a reply is uploaded,
  never a file: not the memory files, not chat history, and not a document holding a key, a sign-in secret or a
  memory line. A saved brief can hold calendar and task lines (that's what a brief is), in Mahadi's own Drive.
- **Which models may change data** (2026-10-25): only a model with a pass in `config/model_tests.json` (tracked, written
  by `regression.py --certify` from live tool tests: no unsafe call, tool-call accuracy ≥ 80 %) gets tools. Any other
  model is chat-only: no tools in its requests, never a turn that needs one, and a write tool refuses to run for it
  even if a call reaches it. A tool the round didn't offer never runs, whatever the provider passes through (though
  the first such call in a turn has the round asked again with every tool: see open item 3), and malformed tool
  arguments are refused instead of crashing the turn.
- **The usage ledger** (`data/providers.db`, mode 600) holds provider, model, units, latency, status and an error's
  class name per call: no query, no message text, no key.
- Tests: `regression.py -k privacy` (stripped requests, refusals, search queries, a model reaching for the calendar,
  OpenRouter and Cloudflare), `-k chain:` (OpenRouter's `:free` rule, a billed call, its 429s; Cloudflare's neurons),
  `-k gate:` (chat-only models, the second locks, calls outside the offered tools, malformed arguments),
  `-k providers` (keys, windows, guard, cooldowns, the panel's JSON without keys), `-k "tests: the default run"`
  (no network in the default run).

## Colour themes (added 2026-10-26)

- A theme file in `themes/`, and every request from the theme editor, holds only a name, `#RRGGBB` colours and two
  numbers, checked by `themes.clean` before anything reaches the page: no CSS text, URL or script from a theme ever
  does, and the name is shown as text. File names (the theme's id) are lower-case letters, digits and dashes, so no
  request reaches outside `themes/`; a file over 16 KB isn't read.
- Making, changing and deleting a theme (`POST /themes`, `POST /themes/preview`, `PUT` and `DELETE /themes/{id}`) go
  through `_local_page_only`. `GET /themes` has no check, like the other read routes (open item 1); it holds only
  theme names and colours.
- The page keeps the chosen theme's colours in `localStorage`, and the head script applies only `--name` variables
  whose values hold none of `;{}<>\`.
- Tests: `regression.py -k themes:` (light, malformed or injected colours and path tricks refused; only the page
  writes; the presets and 400 random themes keep text at 4.5:1).

## Open (found in the 2026-10-21 audit, not fixed yet)

1. **Local web server**: most routes have no Host/Origin check and there is no API token. Any website can POST a
   multipart audio clip to `/talk` (a "simple" request, with no CORS preflight) and Javid will act on it. A DNS-rebinding
   page can read `/history`, `/memory`, `/calendar` and `/tasks`. Planned: a random token in `.env` required on
   every route, plus Host/Origin checks in one middleware.
2. **Write confirmation on the web page and CLI**: only Telegram has the Approve gate. On the page, `delete_task`
   runs at once, and `delete_event`'s "confirm" is set by the model, not by Mahadi's words. In hands-free mode a
   delete should need a tap.
3. **Prompt injection on the web path**: web results go to the model as raw text. Since 2026-10-25 a call to a tool
   the round didn't offer never runs, but the first one in a turn has the round asked again with every tool offered
   (that is how a follow-up such as "yes" reaches the tool it needs), so it doesn't stop an injected instruction from
   reaching a write tool on the web page; on Telegram the Approve button still catches it. Planned: wrap web, page and
   calendar text as untrusted data, and refuse writes and memory after web content in a turn.
4. **Personal data to providers**: the router is in place (2026-10-25, above), and Mahadi cleared Groq: calendar,
   task, memory and past-chat requests go to Groq's models and to no other provider. When Groq's models can't take
   one (resting, or at their allowance), Javid says so rather than send it elsewhere. The data terms of Gemini,
   OpenRouter, Cloudflare and IFM K2 are still his to read before clearing any of them. Not routed, so not covered:
   Whisper (voice clips to Groq), edge-tts (spoken replies to Microsoft), Telegram. The search check is best effort and
   doesn't look at chat history. A model that reaches for a personal tool it wasn't offered makes the turn personal,
   so the retry goes only to providers with `personal_ok`.
5. **Dependencies**: `certifi` 2026.7.22 has PYSEC-2026-4320 (fixed in 2026.10.19). The 51 packages pulled in
   by the direct ones aren't pinned, and the unused `ddgs` is installed.
6. **Logs**: `javid._redact` doesn't know Tavily, Telegram or Google refresh-token shapes, and a few error prints
   bypass it. `data/traces.jsonl` (the start of each message) is never pruned (plan: 14 days).
7. **Local files**: chats, memory, tasks and traces are world-readable (644 in a 755 folder), and this Mac has a
   second account. The Models panel sends each key's last 4 characters.
8. **No `.claude/settings.json` deny rules** for `.env` and the token files.
9. **Telegram, still open**: setting or cancelling a timer doesn't need Approve. Bot chats aren't end-to-end
   encrypted, so Telegram sees them. The rate limit resets on restart (the lock doesn't). Any program running as
   Mahadi can edit `.env`, the allowlist included.
10. **Outside services**: reply text goes to Microsoft's unofficial edge-tts endpoint (the voice is locked; see
    CLAUDE.md), voice clips to Groq's Whisper, and search queries (written by the model) to Tavily (checked for
    personal data since 2026-10-25, see above).
