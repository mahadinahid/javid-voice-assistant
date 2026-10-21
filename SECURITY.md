# Security

Javid is a single-user assistant on Mahadi's Mac: a local web server (`server.py`, 127.0.0.1:8000), a terminal
loop (`javid.py`) and a Telegram bot (`telegram_bot.py`). It holds API keys, a Google Calendar token, tasks,
memory and every chat. This file lists what it defends against, how, and what is still open.

Last reviewed: 2026-10-21.

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

## Open (found in the 2026-10-21 audit, not fixed yet)

1. **Local web server**: most routes have no Host/Origin check and there is no API token. Any website can POST a
   multipart audio clip to `/talk` (a "simple" request, with no CORS preflight) and Javid will act on it. A DNS-rebinding
   page can read `/history`, `/memory`, `/calendar` and `/tasks`. Planned: a random token in `.env` required on
   every route, plus Host/Origin checks in one middleware.
2. **Write confirmation on the web page and CLI**: only Telegram has the Approve gate. On the page, `delete_task`
   runs at once, and `delete_event`'s "confirm" is set by the model, not by Mahadi's words. In hands-free mode a
   delete should need a tap.
3. **Prompt injection on the web path**: web results go to the model as raw text, and the dispatcher runs any
   tool the model names, even one it wasn't offered. Planned: wrap web, page and calendar text as untrusted data,
   refuse writes and memory after web content in a turn, and refuse tools that weren't offered.
4. **Gemini (free tier)** gets memory, chat history and calendar results when it answers. Planned: strip or skip
   personal data in the router. IFM K2's data terms haven't been checked.
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
    CLAUDE.md), voice clips to Groq's Whisper, and search queries (written by the model) to Tavily.
