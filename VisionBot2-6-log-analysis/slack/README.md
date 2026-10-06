# SupportIQ Slack Bot — Setup Guide

## Overview
The Slack bot is a Bolt + Socket Mode bot that connects to the SupportIQ FastAPI server and answers support queries directly in Slack threads. It does not touch Qdrant directly — it calls the `/query` endpoint on the running server.

---

## Prerequisites
- Python 3.9+
- The SupportIQ FastAPI server must be running before starting the bot
- A Slack app with Socket Mode enabled (see Slack App Setup below)

---

## Installation

1. **Clone the repo** and set up the virtual environment:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Add the following to your `.env` file:**
   ```
   SLACK_BOT_TOKEN=xoxb-...        # Bot OAuth token
   SLACK_APP_TOKEN=xapp-...        # Socket Mode app-level token
   API_SECRET_TOKEN=...            # Must match the FastAPI server's token
   ```

---

## Running

**Step 1 — Start the FastAPI server** (in one terminal):
```bash
.venv/bin/uvicorn api.server:app --port 8000
```
Wait for it to fully load (~25 seconds, loads 88k vectors from Qdrant).

**Step 2 — Start the Slack bot** (in a separate terminal):
```bash
.venv/bin/python -m slack.bot
```

Both processes must be running at the same time.

---

## Slack App Setup

Go to https://api.slack.com/apps and create a new app (or use an existing one).

### Enable Socket Mode
- Go to **Socket Mode** → Enable Socket Mode
- Generate an **App-Level Token** with `connections:write` scope → this is your `SLACK_APP_TOKEN` (xapp-...)

### OAuth & Permissions
Under **OAuth & Permissions → Bot Token Scopes**, add:
- `chat:write`
- `im:history`
- `im:read`
- `channels:history` (if using in public channels)

Install the app to your workspace → copy the **Bot User OAuth Token** → this is your `SLACK_BOT_TOKEN` (xoxb-...)

### Event Subscriptions
Under **Event Subscriptions → Subscribe to bot events**, add:
- `message.im` (for DMs)
- `app_mention` (for @mentions in channels)

### App Home
Under **App Home**, enable the **Messages Tab** and check **Allow users to send Slash commands and messages from the messages tab**.

---

## How to Use

### In a channel (team use)
@mention the bot with your question:
```
@SupportIQ SIP 403 on outbound calls from US to India
```
The bot replies in a thread. All follow-ups in the same thread maintain conversation context.

### In a DM
Send a message directly to the bot — no @mention needed.

### Pasting a Zendesk ticket URL
Paste a Zendesk ticket URL in your message and the bot will fetch the live ticket and provide a diagnosis:
```
@SupportIQ https://plivo.zendesk.com/agent/tickets/441532 what's the issue here?
```
Follow-up questions in the same thread will retain the ticket context for 1 hour.

---

## Key Behaviours

- **Multi-turn memory**: Conversation history is maintained per thread. Survives server restarts.
- **Ticket context persistence**: Once a Zendesk URL is pasted, ticket context is retained for follow-up questions in the same thread (1-hour TTL). Also persists across new threads started by the same user.
- **Intent-aware responses**: The bot formats responses differently based on query type — troubleshooting queries get a full 5-section guide, find-similar gets a precedent list, conversational replies are brief.
- **Feedback**: After each answer, the bot posts a 1–5 rating prompt. Feedback is logged to `logs/slack_feedback_logs.jsonl`.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Bot not responding | Check that both the server and bot processes are running |
| `422 Unprocessable Entity` | Check that `API_SECRET_TOKEN` in `.env` matches the server |
| Bot replies with old answers | Clear the answer cache: `truncate -s 0 logs/answer_cache.jsonl` |
| Server loads stale code | Never use `--reload` with uvicorn — Qdrant's file lock blocks reloads silently. Always restart manually. |
| Bot loses ticket context in new thread | Expected on first message — context restores from user-level fallback after first query |
