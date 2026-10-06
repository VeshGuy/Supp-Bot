"""
Slack bot for SupportIQ — Bolt + Socket Mode (no public URL needed).

Calls the FastAPI /query endpoint, so it runs alongside the server
without touching Qdrant directly (no file-lock conflict).

Usage:
    .venv/bin/python -m slack.bot

Required in .env:
    SLACK_APP_TOKEN=xapp-...   (Socket Mode app-level token)
    SLACK_BOT_TOKEN=xoxb-...   (Bot OAuth token)
    API_SECRET_TOKEN=...        (same token the FastAPI server uses)
    PAI_API_KEY=...             (internal voice call debugger)
    DOBBY_API_KEY=...           (internal log investigation agent)
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

from api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8000"

_ZENDESK_URL_RE = re.compile(r'https?://[a-zA-Z0-9-]+\.zendesk\.com/agent/tickets/(\d+)')

app = AsyncApp(token=settings.slack_bot_token)

# Per-session chat history kept in bot memory.
# Survives server restarts — sent as prior_context on every request so
# LangGraph can rebuild thread state even if MemorySaver was wiped.
# Key: session_id → list of {role, content} dicts (max 20 turns).
_session_history: dict[str, list[dict]] = defaultdict(list)
_MAX_HISTORY_TURNS = 20  # 10 exchanges

# Tracks the most recent session_id per user_id.
# When an agent starts a new @mention thread (new session_id, empty history),
# we look up their last session's history as fallback — last 4 turns only.
_user_last_session: dict[str, str] = {}  # user_id → last session_id

# Active ticket tracking — set when an agent pastes a Zendesk URL.
# Keyed by BOTH session_id AND user_id so ticket context survives across threads.
_active_ticket: dict[str, tuple] = {}  # session_id or "user-{uid}" → (ticket_id, set_at)
_ACTIVE_TICKET_TTL = 3600

# ── Feedback ──────────────────────────────────────────────────────────────────

SLACK_FEEDBACK_LOG = Path("logs/slack_feedback_logs.jsonl")

# Stores pending feedback context keyed by message ts (feedback_id).
_pending_feedback: dict[str, dict] = {}
_FEEDBACK_TTL = 86400  # 24 hours


def _store_pending_feedback(
    feedback_id: str, question: str, session_id: str, thread_ts: str, channel_id: str
) -> None:
    now = time.time()
    stale = [k for k, v in _pending_feedback.items() if now - v["stored_at"] > _FEEDBACK_TTL]
    for k in stale:
        del _pending_feedback[k]
    _pending_feedback[feedback_id] = {
        "question": question,
        "session_id": session_id,
        "thread_ts": thread_ts,
        "channel_id": channel_id,
        "stored_at": now,
    }


def _log_slack_feedback(
    user_id: str, user_name: str, feedback_id: str, rating: int, comment: str = None
) -> None:
    data = _pending_feedback.get(feedback_id, {})
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "user_name": user_name,
        "channel_id": data.get("channel_id"),
        "thread_ts": data.get("thread_ts"),
        "session_id": data.get("session_id"),
        "question": data.get("question"),
        "rating": rating,
        "comment": comment,
        "source": "slack",
    }
    SLACK_FEEDBACK_LOG.parent.mkdir(exist_ok=True)
    with open(SLACK_FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Slack feedback logged: user={user_name} rating={rating}")


async def _call_feedback_api(session_id: str, question: str, rating: int, comment: str = None) -> None:
    payload = {
        "session_id": session_id,
        "question": question,
        "rating": rating,
        "comment": comment,
        "ticket_ids": [],
    }
    headers = {"Authorization": f"Bearer {settings.api_secret_token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(f"{API_BASE}/feedback", json=payload, headers=headers)
        except Exception as e:
            logger.warning(f"Feedback API call failed: {e}")


def _build_answer_blocks(reply: str, feedback_id: str) -> list:
    """Build Slack blocks: answer section + 👍/👎 feedback row."""
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": reply[:3000]},
        },
        {
            "type": "actions",
            "block_id": f"fb_{feedback_id[-20:]}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👍 Helpful"},
                    "action_id": "sfeedback_good",
                    "value": feedback_id,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👎 Not helpful"},
                    "action_id": "sfeedback_bad",
                    "value": feedback_id,
                },
            ],
        },
    ]


# ── Markdown → Slack mrkdwn conversion ───────────────────────────────────────

def _to_slack_md(text: str) -> str:
    """Convert GitHub-flavoured markdown to Slack mrkdwn."""
    # Headers: # / ## / ### → *bold*
    text = re.sub(r'^#{1,3}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    # Bold: **text** → *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    return text


# ── Log analysis (PAI / Dobby) ────────────────────────────────────────────────
#
# Two entry points:
#   A — UUID(s) appear directly in the engineer's question → skip RAG, go straight to analysis
#   B — UUID(s) appear only in the injected ticket thread → RAG first, then offer button
#
# PAI  : voice call debugger (Claude Opus 4.6 + CDR + PCAP, ~5 min, ~$0.75/call UUID)
# Dobby: agentic log investigator (messaging / api-messaging logs, 90–120s)
#
# CRITICAL: PAI and Dobby must NEVER be named in any Slack-visible message.
# All user-facing text uses only "log analysis" / "investigating logs".

PAI_API_URL = "https://pai-voice.eks-bqbtjv.prod.plivops.com/api/agent/stream"
DOBBY_API_URL = "https://dobby.plivops.com/api/v1/analyze/stream"
UUID_CACHE_FILE = Path("logs/uuid_analysis_cache.jsonl")
UUID_CACHE_TTL = 86400  # 24 hours

# Standard UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
_UUID_RE = re.compile(
    r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
    re.IGNORECASE,
)

# Plivo Auth IDs: exactly 20 uppercase alphanumeric chars.
# Must contain at least one non-hex letter (G–Z) to distinguish from UUID hex fragments.
_AUTH_ID_RE = re.compile(r'\b([A-Z0-9]{20})\b')

# Keywords indicating that checking logs is a primary recommended step
_LOG_CHECK_RE = re.compile(
    r'\b('
    r'check\s+(?:the\s+)?(?:cdr|logs?|uuid|call)'
    r'|investigate\s+(?:the\s+)?(?:call|uuid|logs?)'
    r'|look\s+up\s+(?:the\s+)?(?:logs?|cdr)'
    r'|fetch\s+(?:the\s+)?(?:cdr|logs?)'
    r'|review\s+(?:the\s+)?(?:cdr|logs?)'
    r'|pull\s+(?:the\s+)?(?:cdr|logs?)'
    r')\b',
    re.IGNORECASE,
)

# PAI vs Dobby routing: count voice vs SMS signals in context
_VOICE_CONTEXT_RE = re.compile(
    r'\b(call|sip|voice|audio|webrtc|rtp|pcap|trunk|zentrunk|dtmf|media|'
    r'one.?way\s+audio|call.?uuid|calluuid|hangup|dial|mos|jitter)\b',
    re.IGNORECASE,
)
_SMS_CONTEXT_RE = re.compile(
    r'\b(message|sms|mms|messaging|delivery|delivered|10dlc|sender.?id|'
    r'message.?uuid|msg.?uuid|short.?code|template|opt.?out)\b',
    re.IGNORECASE,
)

# E.164-style phone numbers in agent messages (e.g. +14151234567, +919876543210)
_PHONE_NUMBER_RE = re.compile(r'\+\d{7,15}\b')

# For contradiction detection between RAG answer and log result
_SIP_CODE_RE = re.compile(r'\bSIP\s+(\d{3})\b', re.IGNORECASE)
_HANGUP_RE = re.compile(
    r'\b(NORMAL_CLEARING|NO_ROUTE_DESTINATION|USER_BUSY|CALL_REJECTED|'
    r'UNALLOCATED_NUMBER|NORMAL_TEMPORARY_FAILURE|RECOVERY_ON_TIMER_EXPIRE|'
    r'INCOMPATIBLE_DESTINATION)\b',
    re.IGNORECASE,
)

# Fix 1: "investigate them" — no UUID in message but investigation intent + pronoun reference.
# Triggers a scan of session history for UUIDs.
_INVESTIGATE_PRONOUN_RE = re.compile(
    r'\b(investigate|debug|analyse|analyze|look\s+into|dig\s+into)\b'
    r'.{0,40}'
    r'\b(them|these|those|it|the\s+uuids?|the\s+calls?|the\s+messages?)\b',
    re.IGNORECASE,
)

# Fix 2: Dobby follow-up — user says "yes" after Dobby offers another pass.
_FOLLOWUP_YES_RE = re.compile(
    r'^\s*(yeah|yes|sure|ok|okay|go\s+ahead|do\s+it|do\s+(?:one\s+more|another)\s+pass|'
    r'(?:one\s+more|another)\s+pass|continue|proceed)\s*[.!]?\s*$',
    re.IGNORECASE,
)
# Pattern in Dobby result indicating it offered to do more work
_DOBBY_OFFER_RE = re.compile(
    r'if\s+you\s+want[,\s]+(i\s+can|we\s+can).{10,200}',
    re.IGNORECASE | re.DOTALL,
)

# In-memory UUID result cache — loaded from file at startup
_uuid_cache: dict = {}

# Per-session auth_id from log results (ground truth).
# Value: (auth_id, source) where source ∈ {log_result, customer, internal}
_session_auth_id: dict = {}

# Pending investigation contexts — keyed by action_value, set when offer button is posted.
# Engineer clicks "Investigate" → we look up this dict to know what to run.
_pending_investigations: dict = {}
_INVESTIGATION_TTL = 3600  # 1 hour

# Fix 2: Per-session pending Dobby follow-up offer.
# Set when a Dobby result ends with "If you want, I can do another pass..."
# Cleared after engineer responds or after 10 minutes.
_pending_dobby_followup: dict = {}  # session_id → {uuid, offer_text, stored_at}
_DOBBY_FOLLOWUP_TTL = 600  # 10 minutes


# ── UUID cache ────────────────────────────────────────────────────────────────

def _load_uuid_cache() -> None:
    """Load UUID result cache from file at startup. Skips expired entries."""
    if not UUID_CACHE_FILE.exists():
        return
    now = time.time()
    loaded = 0
    with open(UUID_CACHE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                uuid = entry.get("uuid", "").lower()
                ts_str = entry.get("ts", "")
                if not uuid or not ts_str:
                    continue
                entry_time = datetime.fromisoformat(ts_str).timestamp()
                if now - entry_time < UUID_CACHE_TTL:
                    _uuid_cache[uuid] = entry
                    loaded += 1
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    if loaded:
        logger.info(f"UUID analysis cache loaded: {loaded} entries")


def _get_cached_analysis(uuid: str) -> dict:
    """Return cached analysis for this UUID if still valid, else None."""
    entry = _uuid_cache.get(uuid.lower())
    if not entry:
        return None
    try:
        entry_time = datetime.fromisoformat(entry["ts"]).timestamp()
        if time.time() - entry_time < UUID_CACHE_TTL:
            return entry
    except (ValueError, KeyError):
        pass
    _uuid_cache.pop(uuid.lower(), None)
    return None


def _save_to_uuid_cache(uuid: str, result_text: str, tool: str, auth_id: str = None) -> None:
    """Persist UUID analysis result to memory and file."""
    entry = {
        "uuid": uuid.lower(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "result_text": result_text,
        "tool": tool,
        "auth_id": auth_id,
    }
    _uuid_cache[uuid.lower()] = entry
    UUID_CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(UUID_CACHE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Auth ID extraction ────────────────────────────────────────────────────────

def _extract_auth_id(text: str) -> str:
    """
    Extract a Plivo Auth ID from text.
    Auth IDs are 20 uppercase alphanumeric chars that contain at least one
    non-hex letter (G–Z), distinguishing them from UUID hex fragments.
    Returns the first match or None.
    """
    for match in _AUTH_ID_RE.finditer(text):
        candidate = match.group(1)
        if re.search(r'[G-Z]', candidate):  # non-hex char confirms it's an auth_id
            return candidate
    return None


def _extract_auth_id_with_source(injected_context: str) -> tuple:
    """
    Extract auth_id from a Zendesk ticket thread, noting comment visibility.
    The zendesk_fetcher formats comments as [Public | timestamp] or [Internal | timestamp].
    Returns (auth_id, source) where source ∈ {customer, internal}.
    Priority: public (customer) > internal note.
    """
    if not injected_context:
        return None, None

    # Split on comment markers and check visibility of each block
    # Pattern: [Public | timestamp] or [Internal | timestamp]
    blocks = re.split(r'\[(Public|Internal)\s*\|[^\]]+\]', injected_context)
    # re.split with groups includes the capture groups: [pre, vis1, block1, vis2, block2, ...]
    visibility_markers = re.findall(r'\[(Public|Internal)\s*\|', injected_context)

    customer_auth_id = None
    internal_auth_id = None

    for i, vis in enumerate(visibility_markers):
        # re.split with capturing group → [pre, marker0, text0, marker1, text1, ...]
        # Text after marker i is at index 2*(i+1).
        block_idx = 2 * (i + 1)
        if block_idx < len(blocks):
            auth = _extract_auth_id(blocks[block_idx])
            if auth:
                if vis.strip().startswith("Public") and not customer_auth_id:
                    customer_auth_id = auth
                elif vis.strip().startswith("Internal") and not internal_auth_id:
                    internal_auth_id = auth

    if customer_auth_id:
        return customer_auth_id, "customer"
    if internal_auth_id:
        return internal_auth_id, "internal"
    return None, None


# ── PAI runner ────────────────────────────────────────────────────────────────

async def _run_pai_analysis(uuid: str, session_id: str, channel: str, thread_ts: str) -> str:
    """
    Stream a voice call UUID debug request through PAI.
    Posts intermediate narration messages to the Slack thread in real-time.
    Returns the final analysis text or None on failure.
    """
    prompt = f"debug this calluuid - {uuid}"
    headers = {
        "X-API-Key": settings.pai_api_key,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    result_text = None
    current_event = None

    try:
        async with httpx.AsyncClient(timeout=360.0) as client:
            async with client.stream(
                "POST", PAI_API_URL, json={"prompt": prompt}, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if current_event == "assistant":
                            # Stream narration text to Slack in real-time.
                            # type:thinking and type:tool_use are internal — skip them.
                            if isinstance(data, dict) and data.get("type") == "text":
                                narration = (data.get("text") or "").strip()
                                if narration:
                                    short = narration[:500] + ("…" if len(narration) > 500 else "")
                                    try:
                                        await app.client.chat_postMessage(
                                            channel=channel,
                                            text=f"_🔍 {short}_",
                                            thread_ts=thread_ts,
                                        )
                                    except Exception as narr_err:
                                        logger.warning(f"PAI narration post failed (continuing stream): {narr_err}")

                        elif current_event == "result":
                            if isinstance(data, dict):
                                result_text = data.get("text", "")

                    elif not line:
                        current_event = None

        return result_text or None

    except Exception as e:
        logger.error(f"PAI analysis error for UUID {uuid[:8]}...: {e}")
        return None


# ── Dobby runner ──────────────────────────────────────────────────────────────

async def _run_dobby_analysis(query: str, channel: str, thread_ts: str) -> str:
    """
    Send a natural language query to Dobby (agentic log investigator).
    Returns the formatted analysis text or None on failure.
    """
    headers = {
        "X-API-Key": settings.dobby_api_key,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }

    result_data = None
    current_event = None
    events_seen = []

    logger.info(f"Dobby query: {query[:200]}")

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", DOBBY_API_URL, json={"message": query}, headers=headers
            ) as response:
                logger.info(f"Dobby HTTP status: {response.status_code}")
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("event: "):
                        current_event = line[7:].strip()
                        if current_event not in events_seen:
                            events_seen.append(current_event)
                    elif line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if current_event == "result":
                            result_data = data
                            logger.info(f"Dobby result keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                    elif not line:
                        current_event = None

        logger.info(f"Dobby stream finished. Events seen: {events_seen}. result_data={'yes' if result_data else 'None'}")

        if result_data is None:
            logger.warning(f"Dobby returned no result event. Events seen: {events_seen}")
            return None

        # Prefer the pre-formatted markdown; fall back to reconstructing from data dict
        formatted = result_data.get("formatted", "")
        if not formatted:
            inner = result_data.get("data", {})
            lines = []
            tldr = inner.get("tldr", "")
            rc = inner.get("root_cause", {})
            suggestions = inner.get("suggestions", [])
            if tldr:
                lines.append(f"**Summary:** {tldr}")
            if rc:
                lines.append(f"**Root cause:** {rc.get('summary', '')}")
                sev = rc.get("severity", "")
                comp = rc.get("affected_component", "")
                if sev or comp:
                    lines.append(f"**Severity:** {sev}  |  **Component:** {comp}")
            if suggestions:
                lines.append("\n**Suggested actions:**")
                for s in suggestions[:3]:
                    lines.append(f"- {s}")
            formatted = "\n".join(lines)

        return formatted or None

    except Exception as e:
        logger.error(f"Dobby analysis error: {e}")
        return None


# ── Contradiction detection ───────────────────────────────────────────────────

async def _check_contradiction(result_text: str, rag_action: str, channel: str, thread_ts: str) -> None:
    """
    Compare SIP codes and hangup causes between log result and RAG suggestion.
    If they differ on something specific, warn the engineer.
    """
    log_sip = set(_SIP_CODE_RE.findall(result_text))
    rag_sip = set(_SIP_CODE_RE.findall(rag_action))
    log_hangup = {m.upper() for m in _HANGUP_RE.findall(result_text)}
    rag_hangup = {m.upper() for m in _HANGUP_RE.findall(rag_action)}

    contradiction = (
        (log_sip and rag_sip and log_sip.isdisjoint(rag_sip))
        or (log_hangup and rag_hangup and log_hangup.isdisjoint(rag_hangup))
    )
    if contradiction:
        await app.client.chat_postMessage(
            channel=channel,
            text="⚠️ Log analysis finding differs from initial diagnosis — the log analysis should take priority.",
            thread_ts=thread_ts,
        )


# ── Log analysis orchestrator ─────────────────────────────────────────────────

def _split_at_paragraphs(text: str, limit: int = 2800) -> list:
    """
    Split text at paragraph boundaries (double newlines) so Slack messages
    never break mid-sentence. Paragraphs longer than limit are hard-split as
    a last resort.
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_parts = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n separator
        if current_len + para_len > limit and current_parts:
            chunks.append('\n\n'.join(current_parts))
            current_parts = []
            current_len = 0
        # Paragraph itself exceeds limit — hard split as last resort
        if para_len > limit:
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts = []
                current_len = 0
            for i in range(0, len(para), limit):
                chunks.append(para[i:i + limit])
        else:
            current_parts.append(para)
            current_len += para_len
    if current_parts:
        chunks.append('\n\n'.join(current_parts))
    return [c for c in chunks if c.strip()]


async def _post_log_result(
    uuid: str,
    result_text: str,
    session_id: str,
    channel: str,
    thread_ts: str,
    rag_suggested_action: str,
    from_cache: bool,
) -> None:
    """Format and post a single UUID analysis result to Slack."""
    cache_note = " _(cached)_" if from_cache else ""

    # Extract a one-line root cause for the header
    rc_match = re.search(r'(?:root\s+cause[:\s*]+)(.+?)(?:\n|$)', result_text, re.IGNORECASE)
    root_cause_line = rc_match.group(1).strip() if rc_match else ""

    header = f"🔬 *Log Analysis — `{uuid}`*{cache_note}"
    if root_cause_line:
        header += f"\n{root_cause_line}"

    await app.client.chat_postMessage(
        channel=channel,
        text=header,
        thread_ts=thread_ts,
    )

    # Fix 3: Split at paragraph boundaries so chunks never break mid-sentence.
    slack_result = _to_slack_md(result_text)
    chunks = _split_at_paragraphs(slack_result, limit=2800)
    for chunk in chunks[:5]:
        await app.client.chat_postMessage(
            channel=channel,
            text=chunk,
            thread_ts=thread_ts,
        )

    # Inject result into session history so follow-up questions are grounded in log findings
    history = _session_history[session_id]
    history.append({
        "role": "assistant",
        "content": f"[Log analysis for UUID {uuid}]\n{result_text[:2000]}",
    })
    if len(history) > _MAX_HISTORY_TURNS:
        _session_history[session_id] = history[-_MAX_HISTORY_TURNS:]

    # Fix 2: If Dobby offered to do another pass, store it so engineer can say "yeah do it"
    offer_match = _DOBBY_OFFER_RE.search(result_text)
    if offer_match:
        offer_text = offer_match.group(0).strip()
        _pending_dobby_followup[session_id] = {
            "uuid": uuid,
            "offer_text": offer_text,
            "stored_at": time.time(),
        }
        logger.info(f"Stored Dobby follow-up offer for session {session_id[:16]}")

    # Contradiction check against the RAG answer's suggested_action
    if rag_suggested_action:
        await _check_contradiction(result_text, rag_suggested_action, channel, thread_ts)


async def _run_log_analysis(
    uuids: list,
    context_text: str,
    session_id: str,
    channel: str,
    thread_ts: str,
    question: str,
    rag_suggested_action: str = "",
) -> None:
    """
    Orchestrate log analysis for one or more UUIDs.
    Routes each UUID to PAI (voice) or Dobby (messaging) based on context signals.
    Checks cache before calling the external tool.
    """
    # Score context to determine routing (higher score = more signals for that product)
    voice_score = len(_VOICE_CONTEXT_RE.findall(context_text))
    sms_score = len(_SMS_CONTEXT_RE.findall(context_text))

    for uuid in uuids[:3]:  # cap at 3 UUIDs per investigation
        uuid_lower = uuid.lower()

        # Cache hit → return immediately, no external call
        cached = _get_cached_analysis(uuid_lower)
        if cached:
            logger.info(f"UUID cache hit: {uuid_lower[:8]}... ({cached.get('tool')})")
            await _post_log_result(
                uuid=uuid,
                result_text=cached["result_text"],
                session_id=session_id,
                channel=channel,
                thread_ts=thread_ts,
                rag_suggested_action=rag_suggested_action,
                from_cache=True,
            )
            # Also cross-reference auth_id if we have a session-level one
            cached_auth = cached.get("auth_id")
            if cached_auth and session_id not in _session_auth_id:
                _session_auth_id[session_id] = (cached_auth, "log_result")
            continue

        # Route: default to PAI (voice) unless SMS context clearly dominates.
        # PAI also requires a valid key — skip it entirely if not configured.
        pai_available = bool(settings.pai_api_key)
        use_pai = pai_available and (voice_score > sms_score)  # tie → Dobby (safer fallback)

        logger.info(
            f"UUID {uuid[:8]}… routing: voice_score={voice_score} sms_score={sms_score} "
            f"pai_available={pai_available} use_pai={use_pai}"
        )

        result_text = None
        tool = None

        if use_pai:
            result_text = await _run_pai_analysis(uuid, session_id, channel, thread_ts)
            tool = "pai"
            if not result_text:
                logger.info(f"PAI returned no result for {uuid[:8]}… — falling back to Dobby")

        if not result_text:
            # Dobby: either SMS dominates, PAI key missing, or PAI returned nothing
            dobby_query = f"Investigate this UUID: {uuid}\n\nContext: {context_text[:800]}"
            result_text = await _run_dobby_analysis(dobby_query, channel, thread_ts)
            tool = "dobby"

        if not result_text:
            await app.client.chat_postMessage(
                channel=channel,
                text=f"⚠️ Log analysis returned no results for `{uuid}`. The logs may not be available.",
                thread_ts=thread_ts,
            )
            continue

        # Extract auth_id from result (ground truth — overrides ticket notes)
        result_auth = _extract_auth_id(result_text)
        if result_auth:
            existing = _session_auth_id.get(session_id)
            if existing and existing[1] == "internal" and existing[0] != result_auth:
                # Internal note had a different auth_id → flag it
                await app.client.chat_postMessage(
                    channel=channel,
                    text=(
                        f"⚠️ Auth ID mismatch: internal note shows `{existing[0]}` but log analysis "
                        f"found `{result_auth}`. The log analysis auth ID is more reliable."
                    ),
                    thread_ts=thread_ts,
                )
            _session_auth_id[session_id] = (result_auth, "log_result")

        _save_to_uuid_cache(uuid_lower, result_text, tool, result_auth)
        await _post_log_result(
            uuid=uuid,
            result_text=result_text,
            session_id=session_id,
            channel=channel,
            thread_ts=thread_ts,
            rag_suggested_action=rag_suggested_action,
            from_cache=False,
        )


# ── Investigation offer ───────────────────────────────────────────────────────

async def _maybe_offer_log_analysis(
    uuid_details: list,
    session_id: str,
    channel: str,
    thread_ts: str,
    question: str,
    context_text: str,
    rag_suggested_action: str,
) -> None:
    """
    Post an investigation offer after a RAG answer when UUIDs were found in the
    live ticket thread and log investigation is a primary recommended step.

    uuid_details: [{uuid, source}] already in reverse-chrono order (most recent first).
    source is "customer" (public comment) or "internal" (internal note).
    """
    # Deduplicate preserving order, normalise to lowercase
    seen_set: set = set()
    deduped: list = []
    for item in uuid_details:
        u = item["uuid"].lower() if isinstance(item, dict) else item.lower()
        src = item.get("source", "unknown") if isinstance(item, dict) else "unknown"
        if u not in seen_set:
            seen_set.add(u)
            deduped.append({"uuid": u, "source": src})

    if not deduped:
        return

    action_value = "inv_" + hashlib.md5(
        (thread_ts + "".join(d["uuid"] for d in deduped[:3])).encode()
    ).hexdigest()[:16]

    # Clean up expired pending investigations
    now = time.time()
    stale = [k for k, v in _pending_investigations.items()
             if now - v.get("stored_at", 0) > _INVESTIGATION_TTL]
    for k in stale:
        del _pending_investigations[k]

    _pending_investigations[action_value] = {
        "uuid_details": deduped,
        "uuids": [d["uuid"] for d in deduped],  # plain strings for _run_log_analysis
        "session_id": session_id,
        "channel": channel,
        "thread_ts": thread_ts,
        "question": question,
        "context_text": context_text,
        "rag_suggested_action": rag_suggested_action,
        "stored_at": now,
    }

    # Build UUID list with source labels (most recent first)
    _SOURCE_EMOJI = {"customer": "👤", "internal": "📝"}
    _SOURCE_LABEL = {"customer": "customer comment", "internal": "internal note"}
    uuid_lines = []
    for item in deduped[:5]:
        emoji = _SOURCE_EMOJI.get(item["source"], "•")
        label = _SOURCE_LABEL.get(item["source"], item["source"])
        uuid_lines.append(f"{emoji} `{item['uuid']}` _({label})_")

    list_text = "\n".join(uuid_lines)
    header_text = f"🔍 Found UUIDs in this ticket (most recent first):\n{list_text}"
    vpn_note = "⚠️ Make sure you're connected to the *AWS VPN* before starting log analysis."
    question_text = "Investigate all, or select a specific UUID:"

    # Buttons: Investigate All + individual UUID buttons (up to 3) + Skip
    buttons = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": "Investigate All"},
            "action_id": "investigate_uuids",
            "value": action_value,
            "style": "primary",
        }
    ]
    for idx, item in enumerate(deduped[:3]):
        short_uuid = item["uuid"][:8] + "…"
        src_short = "👤" if item["source"] == "customer" else "📝"
        buttons.append({
            "type": "button",
            "text": {"type": "plain_text", "text": f"{src_short} {short_uuid}"},
            "action_id": f"investigate_one_uuid_{idx}",
            "value": f"{action_value}::{idx}",
        })
    buttons.append({
        "type": "button",
        "text": {"type": "plain_text", "text": "Skip"},
        "action_id": "skip_investigation",
        "value": action_value,
    })

    full_text = f"{header_text}\n\n{vpn_note}\n{question_text}"
    await app.client.chat_postMessage(
        channel=channel,
        text=full_text,
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": header_text},
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": vpn_note}],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": question_text},
            },
            {
                "type": "actions",
                "elements": buttons,
            },
        ],
        thread_ts=thread_ts,
    )


# ── API call ──────────────────────────────────────────────────────────────────

async def _query_api(question: str, session_id: str, user_id: str = None) -> dict:
    """
    Call the /query SSE endpoint and collect the full response.
    Sends prior_context from local history so multi-turn works even after server restarts.
    Updates local history with the new Q&A turn after a successful response.
    """
    current_history = list(_session_history[session_id])
    if not current_history and user_id and user_id in _user_last_session:
        last_sid = _user_last_session[user_id]
        prior = list(_session_history[last_sid])[-4:]
    else:
        prior = current_history[-6:]

    if user_id:
        _user_last_session[user_id] = session_id

    url_match = _ZENDESK_URL_RE.search(question)
    if url_match:
        tid = url_match.group(1)
        _active_ticket[session_id] = (tid, time.time())
        if user_id:
            _active_ticket[f"user-{user_id}"] = (tid, time.time())

    active_ticket_id = None
    now = time.time()
    for key in ([session_id] + ([f"user-{user_id}"] if user_id else [])):
        entry = _active_ticket.get(key)
        if not entry:
            continue
        ticket_id_stored, set_at = entry
        if now - set_at < _ACTIVE_TICKET_TTL:
            active_ticket_id = ticket_id_stored
            break
        else:
            del _active_ticket[key]
            logger.info(f"Active ticket expired for key {key[:20]}")

    payload = {
        "question": question,
        "session_id": session_id,
        "prior_context": prior,
        "metadata_filters": {},
        "active_ticket_id": active_ticket_id,
    }
    headers = {
        "Authorization": f"Bearer {settings.api_secret_token}",
        "Accept": "text/event-stream",
    }

    tokens = []
    final_data = {}

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{API_BASE}/query", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "token":
                    tokens.append(data.get("content", ""))
                elif data.get("type") == "done":
                    final_data = data
                elif data.get("type") == "error":
                    raise RuntimeError(data.get("content", "Server error"))

    answer_text = "".join(tokens)

    history = _session_history[session_id]
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer_text})
    if len(history) > _MAX_HISTORY_TURNS:
        _session_history[session_id] = history[-_MAX_HISTORY_TURNS:]

    return {
        "answer": answer_text,
        "citations": final_data.get("citations") or [],
        "suggested_action": final_data.get("suggested_action", ""),
        "confidence_score": final_data.get("confidence_score", 0.0),
        "related_tickets": final_data.get("related_tickets") or [],
        "awaiting_clarification": final_data.get("awaiting_clarification", False),
        "intent": final_data.get("intent", ""),
        # UUIDs extracted from the live ticket thread by the server
        "context_uuids": final_data.get("context_uuids") or [],
        # Classifier decided the agent wants log investigation on specific UUIDs
        "wants_log_investigation": final_data.get("wants_log_investigation", False),
    }


# ── Message formatting ────────────────────────────────────────────────────────

def _format_citations(citations: list, confidence: float) -> str:
    """Build the references block for ticket and doc citations."""
    if not citations:
        return ""
    conf_pct = int(confidence * 100)
    conf_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
    lines = [f"\n{conf_emoji} *References* (confidence: {conf_pct}%)"]
    for c in citations[:5]:
        source = c.get("source", "ticket")
        if source == "ticket":
            tid = c.get("ticket_id") or ""
            subject = (c.get("subject") or "")[:60]
            url = c.get("zendesk_url") or ""
            csat = c.get("csat_score")
            csat_str = f" · CSAT {csat}" if csat else ""
            if url and tid:
                lines.append(f"• <{url}|#{tid}>{csat_str} — {subject}")
            elif tid:
                lines.append(f"• #{tid}{csat_str} — {subject}")
        else:
            page_title = (c.get("page_title") or c.get("subject") or "")[:60]
            section = (c.get("section_title") or "")[:40]
            url = c.get("url") or ""
            label = page_title + (f" › {section}" if section else "")
            src_label = source.capitalize()
            if url and label:
                lines.append(f"• <{url}|{src_label}: {label}>")
            elif label:
                lines.append(f"• {src_label}: {label}")
    return "\n".join(lines)


def _format_response(result: dict) -> str:
    """
    Format API result as Slack mrkdwn, varying by intent:
      conversational  — plain answer only, no confidence/citations block
      find-similar    — answer + citations, no suggested action pill
      troubleshoot /
      summarize       — full format: answer + suggested action + citations + reasoning
      explain /
      product_question— answer + citations (no suggested action pill)
    """
    intent = result.get("intent", "")
    answer = _to_slack_md(result.get("answer", "")).strip()
    citations = result.get("citations") or []
    confidence = result.get("confidence_score", 0.0)
    suggested = (result.get("suggested_action") or "").strip()
    reasoning = (result.get("reasoning") or "").strip()

    # ── Conversational: clean 2-4 sentence reply, no metadata noise ──────────
    if intent == "conversational":
        text = answer or "Got it."
        if len(text) > 2900:
            text = text[:2900] + "\n_…(truncated)_"
        return text

    parts = []
    if answer:
        parts.append(answer)

    # ── Suggested action — only for troubleshoot / summarize ─────────────────
    if suggested and intent in ("troubleshoot", "summarize", "other", ""):
        parts.append(f"\n💡 *Suggested action:* {suggested}")

    # ── Citations block ───────────────────────────────────────────────────────
    citations_block = _format_citations(citations, confidence)
    if citations_block:
        parts.append(citations_block)

    # ── Reasoning footer — skip for conversational + find-similar ────────────
    if reasoning and intent not in ("conversational", "find-similar"):
        parts.append(f"\n_How I answered: {reasoning}_")

    text = "\n".join(parts)
    if len(text) > 2900:
        text = text[:2900] + "\n_…(truncated)_"

    return text or "I couldn't find a relevant answer. Could you rephrase your question?"


# ── Core handler ──────────────────────────────────────────────────────────────

async def _handle_question(
    question: str,
    session_id: str,
    channel: str,
    thread_ts: str,
    user_id: str = None,
) -> None:
    """Look up the question and post the answer in the thread."""
    if not question.strip():
        await app.client.chat_postMessage(
            channel=channel,
            text="Hi! Ask me a question about Plivo products.",
            thread_ts=thread_ts,
        )
        return

    # ── Entry Point A: UUID(s) directly in the question ───────────────────────
    # Engineer is explicitly asking to debug a UUID → skip RAG, go straight to log analysis.
    # Examples: "debug this calluuid abc-123...", "can you investigate 9080f4aa-..."
    question_uuids = _UUID_RE.findall(question)
    if question_uuids:
        await app.client.chat_postMessage(
            channel=channel,
            text="⚠️ Ensure you're connected to the *AWS VPN* before viewing log results.",
            blocks=[{"type": "context", "elements": [
                {"type": "mrkdwn", "text": "⚠️ Ensure you're connected to the *AWS VPN* before viewing log results."}
            ]}],
            thread_ts=thread_ts,
        )
        thinking = await app.client.chat_postMessage(
            channel=channel,
            text="🔍 Investigating logs…",
            thread_ts=thread_ts,
        )
        # Seed session history with the question so follow-ups have context
        _session_history[session_id].append({"role": "user", "content": question})
        if user_id:
            _user_last_session[user_id] = session_id

        # Also scan active ticket context for routing signals
        context_text = question
        active_entry = _active_ticket.get(session_id) or _active_ticket.get(f"user-{user_id}" if user_id else "")
        # (active ticket text isn't available here — just use question for routing)

        # Seed session auth_id from ticket context if available (for cross-reference later)
        for key in ([session_id] + ([f"user-{user_id}"] if user_id else [])):
            tid_entry = _active_ticket.get(key)
            if tid_entry and time.time() - tid_entry[1] < _ACTIVE_TICKET_TTL:
                break  # ticket_id found but we don't have the fetched context here — OK

        try:
            deduped_uuids = list(dict.fromkeys(u.lower() for u in question_uuids))
            await _run_log_analysis(
                uuids=deduped_uuids,
                context_text=context_text,
                session_id=session_id,
                channel=channel,
                thread_ts=thread_ts,
                question=question,
            )
        except Exception as e:
            logger.error(f"Log analysis error: {e}", exc_info=True)
            await app.client.chat_postMessage(
                channel=channel,
                text=f"⚠️ Log analysis failed: {e}",
                thread_ts=thread_ts,
            )
        finally:
            # Remove the "Investigating logs…" placeholder
            try:
                await app.client.chat_delete(channel=thinking["channel"], ts=thinking["ts"])
            except Exception:
                pass
        return

    # Entry Point A.2 removed — "investigate this/them/uuids" is now handled by the LLM
    # classifier (wants_log_investigation field). After RAG runs, if wants_log_investigation=True,
    # log analysis fires automatically. See "Normal RAG flow" section below.

    # ── Entry Point A.3: Dobby follow-up — engineer said "yeah do it" after an offer ──
    # Dobby sometimes ends results with "If you want, I can do another pass..."
    # If the engineer replies with yes/do it, trigger a follow-up Dobby call.
    if _FOLLOWUP_YES_RE.match(question):
        followup = _pending_dobby_followup.get(session_id)
        if followup and time.time() - followup.get("stored_at", 0) < _DOBBY_FOLLOWUP_TTL:
            del _pending_dobby_followup[session_id]
            uuid = followup["uuid"]
            offer_text = followup["offer_text"]
            thinking = await app.client.chat_postMessage(
                channel=channel,
                text="🔍 Running deeper analysis…",
                thread_ts=thread_ts,
            )
            _session_history[session_id].append({"role": "user", "content": question})
            try:
                followup_query = (
                    f"For message/call UUID {uuid}, please continue: {offer_text}\n\n"
                    f"This is a follow-up to a previous investigation of the same UUID."
                )
                # Use the same routing as the original (context will be SMS-heavy for Dobby)
                history_text = ' '.join(
                    m.get('content', '') for m in list(_session_history[session_id])[-8:]
                )
                voice_score = len(_VOICE_CONTEXT_RE.findall(history_text))
                sms_score = len(_SMS_CONTEXT_RE.findall(history_text))
                if voice_score >= sms_score:
                    result_text = await _run_pai_analysis(uuid, session_id, channel, thread_ts)
                else:
                    result_text = await _run_dobby_analysis(followup_query, channel, thread_ts)
                if result_text:
                    await _post_log_result(
                        uuid=uuid,
                        result_text=result_text,
                        session_id=session_id,
                        channel=channel,
                        thread_ts=thread_ts,
                        rag_suggested_action="",
                        from_cache=False,
                    )
                else:
                    await app.client.chat_postMessage(
                        channel=channel,
                        text="⚠️ No additional findings from the deeper analysis.",
                        thread_ts=thread_ts,
                    )
            except Exception as e:
                logger.error(f"Dobby follow-up error: {e}", exc_info=True)
                await app.client.chat_postMessage(
                    channel=channel,
                    text=f"⚠️ Follow-up analysis failed: {e}",
                    thread_ts=thread_ts,
                )
            finally:
                try:
                    await app.client.chat_delete(channel=thinking["channel"], ts=thinking["ts"])
                except Exception:
                    pass
            return

    # ── Normal RAG flow ───────────────────────────────────────────────────────
    thinking = await app.client.chat_postMessage(
        channel=channel,
        text="On it…",
        thread_ts=thread_ts,
    )

    try:
        result = await _query_api(question, session_id, user_id=user_id)

        awaiting = result.get("awaiting_clarification", False)
        intent = result.get("intent", "")

        if awaiting:
            reply = result["answer"]
            show_feedback = False
        else:
            reply = _format_response(result)
            show_feedback = intent != "conversational"

        if show_feedback:
            feedback_id = thinking["ts"]
            _store_pending_feedback(feedback_id, question, session_id, thread_ts, channel)
            await app.client.chat_update(
                channel=thinking["channel"],
                ts=thinking["ts"],
                text=reply,
                blocks=_build_answer_blocks(reply, feedback_id),
            )
        else:
            await app.client.chat_update(
                channel=thinking["channel"],
                ts=thinking["ts"],
                text=reply,
            )

        # ── Entry Point A.2 (LLM-based): agent wants log investigation on prior UUIDs ──
        # The classifier set wants_log_investigation=True — meaning the question was something
        # like "investigate this", "check the CDRs", "investigate uuids" — the LLM understood
        # the intent without needing brittle regex.
        # Scan session history for UUIDs and run log analysis automatically (no button needed).
        if not awaiting and result.get("wants_log_investigation"):
            history_text = ' '.join(
                m.get('content', '') for m in list(_session_history[session_id])[-8:]
            )
            history_uuids = _UUID_RE.findall(history_text)
            if history_uuids:
                deduped_uuids = list(dict.fromkeys(u.lower() for u in history_uuids))
                await app.client.chat_postMessage(
                    channel=channel,
                    text="⚠️ Ensure you're connected to the *AWS VPN* before viewing log results.",
                    blocks=[{"type": "context", "elements": [
                        {"type": "mrkdwn", "text": "⚠️ Ensure you're connected to the *AWS VPN* before viewing log results."}
                    ]}],
                    thread_ts=thread_ts,
                )
                try:
                    await _run_log_analysis(
                        uuids=deduped_uuids,
                        context_text=history_text + " " + question,
                        session_id=session_id,
                        channel=channel,
                        thread_ts=thread_ts,
                        question=question,
                    )
                except Exception as e:
                    logger.error(f"Log analysis error: {e}", exc_info=True)
                    await app.client.chat_postMessage(
                        channel=channel,
                        text=f"⚠️ Log analysis failed: {e}",
                        thread_ts=thread_ts,
                    )
            else:
                # No UUID — check if a phone number was mentioned (e.g. "search logs where FROM is +14151234567")
                phone_matches = _PHONE_NUMBER_RE.findall(question + " " + history_text)
                if phone_matches:
                    phone_number = phone_matches[0]
                    await app.client.chat_postMessage(
                        channel=channel,
                        text="⚠️ Ensure you're connected to the *AWS VPN* before viewing log results.",
                        blocks=[{"type": "context", "elements": [
                            {"type": "mrkdwn", "text": "⚠️ Ensure you're connected to the *AWS VPN* before viewing log results."}
                        ]}],
                        thread_ts=thread_ts,
                    )
                    dobby_query = (
                        f"Search message/call logs where the source number (FROM) is {phone_number}. "
                        f"Find recent UUIDs and investigate delivery/call failures. "
                        f"Context: {question}"
                    )
                    try:
                        await _run_dobby_analysis(dobby_query, channel, thread_ts)
                    except Exception as e:
                        logger.error(f"Dobby phone number analysis error: {e}", exc_info=True)
                        await app.client.chat_postMessage(
                            channel=channel,
                            text=f"⚠️ Log analysis failed: {e}",
                            thread_ts=thread_ts,
                        )
                else:
                    # No UUID or phone number — ask the agent for an identifier
                    await app.client.chat_postMessage(
                        channel=channel,
                        text="To verify this with actual call/message logs, share a call UUID, message UUID, or source phone number and I'll investigate.",
                        thread_ts=thread_ts,
                    )

        # ── Entry Point B: UUIDs in ticket context → offer investigation button ──────
        # context_uuids from the injected live ticket thread. Only offer the button for
        # troubleshooting intents — not for summarize/explain/conversational/find-similar.
        # Skip if wants_log_investigation already fired (avoid double investigation).
        if (not awaiting
                and intent == "troubleshoot"
                and not result.get("wants_log_investigation")):
            context_uuids = result.get("context_uuids") or []
            if context_uuids:
                suggested = result.get("suggested_action", "")
                answer_text = result.get("answer", "")
                await _maybe_offer_log_analysis(
                    uuid_details=context_uuids,
                    session_id=session_id,
                    channel=channel,
                    thread_ts=thread_ts,
                    question=question,
                    context_text=answer_text + " " + suggested,
                    rag_suggested_action=suggested,
                )

    except httpx.ConnectError:
        await app.client.chat_update(
            channel=thinking["channel"],
            ts=thinking["ts"],
            text="⚠️ Can't reach the SupportIQ server. Make sure it's running on port 8000.",
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        await app.client.chat_update(
            channel=thinking["channel"],
            ts=thinking["ts"],
            text=f"⚠️ Something went wrong: {e}",
        )


# ── Event handlers ────────────────────────────────────────────────────────────

@app.event("app_mention")
async def handle_mention(event, say):  # noqa: F811
    """Handle @mentions in channels/groups."""
    text = event.get("text", "")
    question = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

    thread_ts = event.get("thread_ts") or event["ts"]
    session_id = f"slack-{thread_ts}"
    user_id = event.get("user")

    await _handle_question(question, session_id, event["channel"], thread_ts, user_id=user_id)


@app.event("message")
async def handle_dm(event, say):  # noqa: F811
    """Handle direct messages to the bot."""
    if event.get("bot_id") or event.get("subtype"):
        return
    if event.get("channel_type") != "im":
        return

    question = event.get("text", "").strip()
    if not question:
        return

    session_id = f"slack-dm-{event['channel']}"
    thread_ts = event.get("thread_ts") or event["ts"]
    user_id = event.get("user")

    await _handle_question(question, session_id, event["channel"], thread_ts, user_id=user_id)


# ── Investigation action handlers ─────────────────────────────────────────────

@app.action("investigate_uuids")
async def handle_investigate_uuids(ack, body, client):
    await ack()
    action_value = body["actions"][0]["value"]
    channel = body["channel"]["id"]
    msg_ts = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts") or msg_ts

    # Replace offer block with in-progress indicator
    await client.chat_update(
        channel=channel,
        ts=msg_ts,
        text="🔍 Investigating logs…",
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": "🔍 Investigating logs…"}},
        ],
    )

    pending = _pending_investigations.get(action_value)
    if not pending:
        await app.client.chat_postMessage(
            channel=channel,
            text="⚠️ Investigation request expired. Please ask again.",
            thread_ts=thread_ts,
        )
        return

    # Run analysis in background — don't block the ack response
    asyncio.create_task(_run_log_analysis(
        uuids=pending.get("uuids") or [],
        context_text=pending.get("context_text", ""),
        session_id=pending["session_id"],
        channel=pending["channel"],
        thread_ts=pending["thread_ts"],
        question=pending["question"],
        rag_suggested_action=pending.get("rag_suggested_action", ""),
    ))


@app.action(re.compile(r"investigate_one_uuid(_\d+)?"))
async def handle_investigate_one_uuid(ack, body, client):
    await ack()
    raw_value = body["actions"][0]["value"]
    channel = body["channel"]["id"]
    msg_ts = body["message"]["ts"]
    thread_ts = body["message"].get("thread_ts") or msg_ts

    try:
        action_value, idx_str = raw_value.rsplit("::", 1)
        idx = int(idx_str)
    except (ValueError, AttributeError):
        await app.client.chat_postMessage(
            channel=channel, text="⚠️ Couldn't parse investigation request.", thread_ts=thread_ts
        )
        return

    pending = _pending_investigations.get(action_value)
    if not pending:
        await app.client.chat_postMessage(
            channel=channel, text="⚠️ Investigation request expired. Please ask again.", thread_ts=thread_ts
        )
        return

    uuids = pending.get("uuids", [])
    if idx >= len(uuids):
        await app.client.chat_postMessage(
            channel=channel, text="⚠️ UUID index out of range.", thread_ts=thread_ts
        )
        return

    selected_uuid = uuids[idx]
    short = selected_uuid[:8] + "…"

    await client.chat_update(
        channel=channel,
        ts=msg_ts,
        text=f"🔍 Investigating `{short}`…",
        blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": f"🔍 Investigating `{short}`…"}}],
    )

    asyncio.create_task(_run_log_analysis(
        uuids=[selected_uuid],
        context_text=pending.get("context_text", ""),
        session_id=pending["session_id"],
        channel=pending["channel"],
        thread_ts=pending["thread_ts"],
        question=pending["question"],
        rag_suggested_action=pending.get("rag_suggested_action", ""),
    ))


@app.action("skip_investigation")
async def handle_skip_investigation(ack, body, client):
    await ack()
    await client.chat_update(
        channel=body["channel"]["id"],
        ts=body["message"]["ts"],
        text="Investigation skipped.",
        blocks=[
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "_Investigation skipped._"}]},
        ],
    )


# ── Feedback action handlers ──────────────────────────────────────────────────

@app.action("sfeedback_good")
async def handle_feedback_good(ack, body, client):
    await ack()
    feedback_id = body["actions"][0]["value"]
    user_id = body["user"]["id"]
    user_name = body["user"].get("username") or body["user"].get("name") or user_id

    _log_slack_feedback(user_id, user_name, feedback_id, rating=8)

    data = _pending_feedback.get(feedback_id, {})
    await _call_feedback_api(
        session_id=data.get("session_id", ""),
        question=data.get("question", ""),
        rating=8,
    )

    await client.chat_update(
        channel=body["channel"]["id"],
        ts=body["message"]["ts"],
        text=body["message"].get("text", ""),
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": body["message"].get("text", "")[:3000]}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "✅ Thanks for the feedback!"}]},
        ],
    )


@app.action("sfeedback_bad")
async def handle_feedback_bad(ack, body, client):
    await ack()
    feedback_id = body["actions"][0]["value"]

    await client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "slack_feedback_modal",
            "private_metadata": json.dumps({
                "feedback_id": feedback_id,
                "channel": body["channel"]["id"],
                "ts": body["message"]["ts"],
                "text": body["message"].get("text", "")[:3000],
            }),
            "title": {"type": "plain_text", "text": "Feedback"},
            "submit": {"type": "plain_text", "text": "Submit"},
            "close": {"type": "plain_text", "text": "Cancel"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "comment_block",
                    "optional": True,
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "comment_input",
                        "multiline": True,
                        "placeholder": {"type": "plain_text", "text": "What could be improved? (optional)"},
                    },
                    "label": {"type": "plain_text", "text": "What was wrong?"},
                }
            ],
        },
    )


@app.view("slack_feedback_modal")
async def handle_feedback_modal(ack, body, client):
    await ack()
    meta = json.loads(body["view"]["private_metadata"])
    feedback_id = meta["feedback_id"]
    channel = meta["channel"]
    ts = meta["ts"]
    original_text = meta["text"]

    user_id = body["user"]["id"]
    user_name = body["user"].get("username") or body["user"].get("name") or user_id
    comment = body["view"]["state"]["values"]["comment_block"]["comment_input"].get("value") or None

    _log_slack_feedback(user_id, user_name, feedback_id, rating=2, comment=comment)

    data = _pending_feedback.get(feedback_id, {})
    await _call_feedback_api(
        session_id=data.get("session_id", ""),
        question=data.get("question", ""),
        rating=2,
        comment=comment,
    )

    await client.chat_update(
        channel=channel,
        ts=ts,
        text=original_text,
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": original_text[:3000]}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "✅ Thanks for the feedback!"}]},
        ],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    if not settings.slack_app_token:
        raise RuntimeError("SLACK_APP_TOKEN is not set in .env")
    if not settings.slack_bot_token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set in .env")

    _load_uuid_cache()

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)
    logger.info("Starting SupportIQ Slack bot (Socket Mode)…")
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
