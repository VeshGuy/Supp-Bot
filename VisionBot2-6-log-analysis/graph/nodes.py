"""
LangGraph node functions. Each function handles one reasoning step.
All nodes receive the full AgentState and return a partial state update.
"""

import asyncio
import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from api.cost_tracker import estimate_tokens, make_entry
from graph.state import AgentState
from retrieval.answer_cache import get_cached, set_cached
from retrieval.hyde import generate_hyde
from retrieval.hybrid_retriever import retrieve
from retrieval.reranker import rerank
from retrieval.zendesk_fetcher import fetch_ticket_by_id

_ZENDESK_URL_RE = re.compile(r'https?://[a-zA-Z0-9-]+\.zendesk\.com/agent/tickets/(\d+)')

# Bare ticket ID patterns — no full URL needed.
# Matches: "ticket #441532", "ticket 441532", "ZD441532", "ZD #441532", "#441532"
_BARE_TICKET_ID_RE = re.compile(
    r'\bticket\s*#?\s*(\d{5,7})\b'       # "ticket 441532", "ticket #441532"
    r'|\bZD\s*#?\s*(\d{5,7})\b'          # "ZD441532", "ZD #441532"
    r'|(?:^|(?<=\s))#(\d{5,7})\b',       # "#441532" — bare hash at start or after space
    re.IGNORECASE | re.MULTILINE,
)

# Signals that indicate the query contains exact technical identifiers.
# For these queries HyDE is bypassed — the raw question is embedded directly.
# Reason: HyDE hallucinates a generic resolution for exact error strings/UUIDs,
# pulling in wrong tickets. BM25 sparse vectors do exact string matching better.
_TECHNICAL_SIGNAL_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'  # UUID / call UUID
    r'|\bSIP\s*[3-5]\d{2}\b'                                             # SIP response codes
    r'|\bHTTP\s*[3-5]\d{2}\b'                                            # HTTP status codes
    r'|Traceback \(most recent|Exception in thread|at com\.|at org\.'    # Stack traces
    r'|EHOSTUNREACH|ECONNREFUSED|ECONNRESET|ETIMEDOUT|ENOTFOUND'         # POSIX network errors
    r'|HangupCauseCode|NORMAL_CLEARING|NO_ROUTE_DESTINATION'             # SIP/telephony hangup causes
    r'|USER_BUSY|NO_ANSWER|CALL_REJECTED|UNALLOCATED_NUMBER'             # More SIP causes
    r'|websocket\.send|websocket\.receive|ASGI message'                  # ASGI/WebSocket errors
    r'|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',                        # IP addresses
    re.IGNORECASE,
)

# Ambiguous follow-up signals — question has no standalone topic, relies on previous response.
# When matched, classify_intent will prepend the last bot response as explicit topic context
# so the classifier knows what "it / this / that / example" refers to.
_FOLLOWUP_RE = re.compile(
    r'\bgive\s+(?:me\s+)?(?:an?\s+)?example\b'                  # "give me an example"
    r'|\belaborate\b'                                            # "elaborate", "can you elaborate"
    r'|\btell\s+me\s+more\b'                                     # "tell me more"
    r'|\bwhat\s+does\s+(?:that|this|it)\s+mean\b'               # "what does that mean"
    r'|\bhow\s+does\s+(?:that|this|it)\s+work\b'                # "how does that work"
    r'|\bcan\s+you\s+(?:clarify|simplify|rephrase|elaborate)\b'  # "can you clarify/elaborate"
    r'|\bexplain\s+(?:it|this|that)\b'                          # "explain it/this/that"
    r'|\bexplain\s+(?:with|using)\s+(?:an?\s+)?example\b',      # "explain with/using an example"
    re.IGNORECASE,
)

# Ticket-metadata follow-up signals.
# Priority 3 in fetch_ticket_context_node only fires if the question matches this —
# prevents general product questions from accidentally picking up old ticket URLs
# in chat history from unrelated prior conversations.
_TICKET_FOLLOWUP_RE = re.compile(
    r'\bassigned\s+to\b'                          # "who is assigned to it"
    r'|\bassignee\b'                               # "who is the assignee"
    r'|\bwho\s+(?:is|has|handles?)\s+(?:it|this|the\s+ticket)\b'  # "who has this ticket"
    r'|\bwho\s+is\s+the\s+(?:requester|customer|agent)\b'
    r'|\bwhat(?:\'?s|\s+is)\s+(?:the\s+)?(?:status|priority|subject)\b'  # "what's the status"
    r'|\bwho\s+(?:raised|submitted|created|opened)\s+(?:it|this)\b'
    r'|\bwhen\s+was\s+(?:it|this|the\s+ticket)\s+(?:created|updated|opened|raised)\b',
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)

_RETRIEVAL_DEBUG_FILE = Path("logs/retrieval_debug.jsonl")

# Server-side ticket context cache — avoids re-fetching the same ticket on every turn.
# Key: ticket_id → {"context": str, "fetched_at": float}
# TTL: 10 minutes (tickets can be updated mid-conversation)
_ticket_context_cache: dict = {}
_TICKET_CACHE_TTL = 600


def _log_retrieval_debug(state: AgentState) -> None:
    """
    Write a per-query retrieval debug record so we can inspect exactly what
    GPT-4o worked with: HyDE text, top-20 retrieved chunks, top-5/8 reranked,
    confidence score, and which chunks made it to the answer.
    """
    def _slim(chunks: list[dict], score_key: str = "rrf_score") -> list[dict]:
        return [
            {
                "ticket_id": c.get("ticket_id"),
                "subject": c.get("subject") or c.get("page_title") or c.get("title") or c.get("summary") or c.get("nav_path", ""),
                "source": c.get("source", "ticket"),
                "chunk_type": c.get("chunk_type", ""),
                score_key: round(float(c.get(score_key, 0) or 0), 4),
            }
            for c in (chunks or [])
        ]

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": state.get("session_id", ""),
        "question": state.get("question", ""),
        "query_type": state.get("query_type", ""),
        "hyde_text": state.get("hyde_text", ""),
        "retrieved_top20": _slim(state.get("retrieved_chunks", []), "rrf_score"),
        "reranked_top_n": _slim(state.get("reranked_chunks", []), "rerank_score"),
        "confidence_score": round(float(state.get("confidence_score", 0) or 0), 3),
        "confidence_factors": state.get("confidence_factors", {}),
    }
    try:
        with open(_RETRIEVAL_DEBUG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write retrieval debug log: {e}")


# ── Prompts ──────────────────────────────────────────────────────────────────

CLASSIFY_HYDE_PROMPT = """You are a classifier and retrieval assistant for SupportIQ — an internal tool for Plivo support agents answering questions about Plivo products (Voice API, SMS/Messaging, Phone Numbers, SIP Trunking).

Return ONLY a JSON object:
{
  "is_voice_related": true|false,
  "query_type": "ticket_search|product_question|conversational",
  "intent": "troubleshoot|explain|find-similar|summarize|conversational|other",
  "is_ambiguous": true|false,
  "needs_clarification": true|false,
  "clarification_question": "question to ask if is_ambiguous OR needs_clarification, else null",
  "pre_diagnosis": {
    "error_code": "SIP/HTTP error code or SMS error code explicitly mentioned, e.g. SIP 403, HTTP 500, or null",
    "failure_point": "outbound_call_setup|inbound_call_setup|media_audio|registration|webhook|dtmf|recording|call_transfer|ivr|message_delivery|message_routing|sender_id|template_approval|opt_out|dlt_registration|number_provisioning|null",
    "direction": "inbound|outbound|both|null",
    "sdk": "browser_sdk|android_sdk|ios_sdk|rest_api|sip_trunk|sms_api|null",
    "onset_pattern": "sudden|gradual|specific_carrier|specific_customer|always_failing|intermittent|null",
    "scope": "all_calls|specific_carrier|specific_customer|intermittent|null",
    "product_area": "voice_api|zentrunk|webrtc|sip_trunking|sms|messaging|phone_numbers|null"
  },
  "hyde_text": "hypothetical document text for embedding (see instructions below)",
  "wants_log_investigation": true|false
}

IMPORTANT — follow-up questions: If "[Follow-up context — the agent is continuing from this previous bot response:]" appears at the top of the message, the agent is asking a follow-up about THAT previous topic. Use the previous response to determine is_voice_related, query_type, intent, and hyde_text. Do NOT re-derive the topic from the follow-up question text alone or from older history messages below it.

STEP 0 — conversational (check THIS FIRST, before all other steps):
Set query_type="conversational" and intent="conversational" if the message falls into EITHER of these two categories:

Category A — meta-comment about a PREVIOUS assistant response (not a new technical question):
- Refers to the bot's previous response: "you said", "you gave", "you told", "your answer", "your response", "you were wrong", "you made a mistake"
- Asks if bot was correct: "so you gave the wrong resolution right?", "did you get that right?", "was that correct?", "that's wrong right?"
- User provides a correction: "actually...", "my senior said...", "the correct answer is...", "that's not right because..."
- Simple acknowledgment: "ok thanks", "got it", "makes sense", "that helped"
- Confirming bot was right: "yes that's correct", "exactly", "good answer"
- Challenging or questioning citations/evidence: "these tickets aren't related", "these results don't match", "none of these are relevant", "this ticket has nothing to do with the issue", "the tickets you gave are wrong"
- Asking the bot to refine/choose from its own previous answer: "pin it down to one issue", "narrow it down", "which one is most likely", "what's the most probable cause", "just give me the top one", "which of these is it"
  These are meta-instructions — the agent wants the bot to choose from what it already said, NOT trigger new retrieval.

Category B — simple data-lookup about a LIVE TICKET (ONLY when "[Live ticket submitted for analysis]" appears at the top of this message):
- Asking who is assigned / handling the ticket: "who is assigned", "who has this ticket", "who is handling it", "who is the assignee"
- Asking ticket status: "what's the status", "is it open", "is it closed", "is it pending"
- Asking ticket priority: "what's the priority", "is it urgent", "is it high priority"
- Asking about the requester or customer: "who is the customer", "who raised this", "who is the requester", "who submitted"
- Asking simple ticket metadata: "when was it created", "what are the tags", "what is the subject", "who updated it"
These are direct field lookups — the answer is already in the ticket context, no retrieval needed.

CRITICAL — do NOT classify these as Category B conversational:
- "what has happened in this ticket", "what happened so far", "summarize this ticket", "tell me what's going on"
- "what is the issue", "what is the problem", "what is the customer reporting"
- "analyze this ticket", "give me a summary", "what's the history"
- Any question asking for a narrative, summary, or analysis of the ticket thread
- "check the carrier contact sheet", "carrier support email", "escalation contact for X", "support email of carrier"
- "check confluence", "check the docs", "look it up in", any request to look up an external document or resource
These require retrieval from indexed docs — classify them as query_type="product_question", is_voice_related=true so they go through the RAG pipeline.

When conversational: set is_voice_related=true, query_type="conversational", intent="conversational", is_ambiguous=false, needs_clarification=false, hyde_text="".
Do NOT generate HyDE text for conversational queries.

STEP 1 — is_voice_related (field name is historical — it now means "is Plivo product related"):
- true if the query is about ANY Plivo product or API, including:
  Voice: calls, SIP, WebRTC, PSTN, call quality, audio, DTMF, call routing, IVR, caller ID, call recordings, voice SDK, outbound/inbound calls, call failures, Zentrunk, CPS limits, voice capacity
  SMS/Messaging: SMS delivery, MMS, message routing, message templates, sender IDs, short codes, long codes, toll-free SMS, 10DLC, DLT registration, UCC registration, opt-out handling, delivery failures, message errors, messaging API
  Phone Numbers: number provisioning, number porting, number search, number capabilities, inbound routing, geographic numbers, toll-free numbers
  General: Plivo account, API authentication, webhooks, Plivo console, PHLO, any Plivo API question
  Carrier support resources: carrier contact sheets, carrier escalation emails, carrier support contacts, carrier contact details used when investigating or resolving Plivo SMS/voice delivery issues (e.g. "Jio support email", "AT&T escalation contact", "carrier contact sheet")
- false ONLY for queries that have zero connection to any Plivo product: pure general IT questions, greetings, general small talk, questions about competitors.
- WHEN IN DOUBT: set is_voice_related=true, needs_clarification=true, and ask a short question. Never reject something that might be Plivo-related — ask first.
If false: set query_type="ticket_search", intent="other", is_ambiguous=false, needs_clarification=false, hyde_text=<the original question>.

STEP 2 — query_type (if voice_related):
- "product_question": user wants to understand a feature/concept.
  Examples: "What is Play API?", "How does DTMF work?", "What are Zentrunk rate limits?"
- "ticket_search": user has a problem or wants past cases.
  Examples: "Play API returning 404", "SIP 403 Forbidden", "WebRTC drops after 30s", "how do I resolve this ticket", "what's the next step", "what should I do next"
If unsure, mark is_ambiguous=true and ask one clarifying question.

STEP 3 — intent (if voice_related):
- "troubleshoot": user wants to resolve a problem or needs next steps. Examples: "calls dropping", "SIP 403 error", "audio choppy", "how do I resolve this", "what is the next action", "what should we do next", "how to fix this", "can you help resolve this ticket", "what's the next step for this ticket"
- "explain": user wants to understand a feature or concept. Examples: "how does DTMF work?", "what is WebRTC?"
- "find-similar": user wants to know if this scenario has been seen before, or wants past ticket examples. Examples: "have there been cases like this?", "any similar tickets?", "has this happened before?", "are there tickets where X occurred?", "any precedents for this?"
- "summarize": user explicitly wants a summary or timeline of what happened. Examples: "summarize this ticket", "what has happened so far", "give me a summary", "tell me what happened", "what's the history of this ticket"

CRITICAL — intent distinction when a live ticket is present:
- "summarize this ticket" / "what happened here" / "give me a summary" → intent=summarize
- "how do I resolve this" / "what's the next action" / "what should we do" / "help me close this" → intent=troubleshoot, query_type=ticket_search
- When in doubt between summarize and troubleshoot, prefer troubleshoot.

IMPORTANT: "have there been cases like these?" or "any tickets for this?" with a topic in context = find-similar. Never mark these as needs_clarification — the topic is already known from context or the question itself.

STEP 4 — is_ambiguous:
Set true ONLY if the query has two or more completely different possible interpretations.
Example: "issue with calls" — could mean setup, quality, billing, routing.
Ask the single most useful clarifying question.

STEP 5 — needs_clarification (voice_related=true, NOT ambiguous):
IMPORTANT: Default is needs_clarification=false. Only set true as a last resort.

Set true ONLY when ALL of these are true simultaneously:
  (a) the query is ticket_search (NOT product_question — product questions never need clarification)
  (b) the symptom is so vague that retrieval would return random results
  (c) there is literally no specific signal — no feature name, no direction, no behaviour description

GOOD examples of when to clarify:
- "my calls don't work" → needs_clarification=true (zero signal)
- "I have an issue" → needs_clarification=true (no feature, no symptom)

BAD examples (DO NOT clarify — retrieve and answer):
- "audio is bad on calls" → needs_clarification=false (symptom is clear, retrieve on it)
- "recording has voice overlap in part of the call but not the full recording" → needs_clarification=false (specific symptom, retrieve)
- "audio one-way on WebRTC" → needs_clarification=false (has SDK + direction)
- "SIP 403 on outbound calls" → needs_clarification=false (has error + direction)
- "WebRTC drops after 30s" → needs_clarification=false (has SDK + timing)
- "How does DTMF work?" → needs_clarification=false (product question)
- "tell me about call recordings" → needs_clarification=false (feature name is enough to retrieve)
- "help me with this ticket" (with ticket context above) → needs_clarification=false (ticket IS the context)
- "have there been cases like this?" → needs_clarification=false, intent=find-similar (topic comes from context)
- "any similar tickets?" → needs_clarification=false, intent=find-similar
- "any tickets raised for this?" → needs_clarification=false, intent=find-similar
- "has this happened before?" → needs_clarification=false, intent=find-similar

If needs_clarification=true: ask for the single most valuable missing piece.
  e.g. "What error code or behaviour are you seeing?" / "Which SDK — Browser, Android, or iOS?"
Do NOT set needs_clarification=true if the query contains ANY specific technical signal.

STEP 6 — pre_diagnosis:
For ticket_search queries: extract diagnostic signals from the question. Only include what is explicitly stated or strongly implied — use null for anything not mentioned.
For product_question, ambiguous, or non-voice queries: set all pre_diagnosis fields to null.
- error_code: any SIP (3xx/4xx/5xx) or HTTP error code explicitly mentioned
- failure_point: the specific stage in the call flow where failure occurs
- direction: whether the issue affects inbound calls, outbound, or both
- sdk: which Plivo SDK or interface is involved (browser_sdk, android_sdk, ios_sdk, rest_api, sip_trunk)
- onset_pattern: how/when the issue started
- scope: how broadly the issue affects calls
- product_area: the Plivo product most closely involved

STEP 7 — hyde_text:
Write a short hypothetical document (2-4 sentences) that would perfectly answer this query.
- For ticket_search: write as if you are the resolution note in a Zendesk ticket. Use the pre_diagnosis signals you extracted to make this Plivo-specific — incorporate the exact failure_point, SDK, product_area, and error_code into the resolution note so the embedding targets the right ticket cluster. Do NOT write generic telecom text.
  Voice example — "WebRTC browser SDK audio cuts out after 30s": write "WebRTC Browser SDK call dropped due to ICE connection timeout after 30 seconds. Root cause was TURN server credentials expiring mid-call. Resolved by refreshing ICE credentials before the 30s window."
  SMS example — "SMS not delivered to US numbers": write "SMS messages to US long codes failing delivery. Root cause was 10DLC campaign registration not completed. Resolved by registering the campaign via Plivo console and waiting for carrier approval."
- For product_question: write as if you are a Plivo docs page explaining the feature.
- Keep it dense with technical terms relevant to the query (Plivo API names, error codes, carrier names).
If is_ambiguous=true or needs_clarification=true or is_voice_related=false, set hyde_text to the original question verbatim.

STEP 8 — wants_log_investigation:
Set true when the agent clearly wants to investigate actual call or message logs for specific identifiers — NOT a general troubleshooting question.

Set true for:
- A UUID appears in the question AND the agent is asking to investigate/debug/check it
- The agent refers to previously mentioned UUIDs with investigation intent, even without naming them explicitly:
  "investigate this", "investigate uuids", "investigate these calls", "check the logs", "check the CDRs", "debug these", "look into this", "what do the logs show", "can you check what happened"
- The agent says "investigate them", "investigate it", "run the investigation", "analyze the logs"
- The agent wants to search logs by phone number: "search for UUIDs in logs where FROM is X", "search logs by number", "find calls from this number", "check logs for number +1234567890", "investigate calls to/from this number", "find messages sent from +X", "look up logs for this number"

Set false for:
- General troubleshooting questions ("my calls are dropping") — no specific identifier to look up
- Product questions, find-similar, summarize, explain
- Conversational follow-ups that don't involve log investigation
- Any question where the agent is asking for guidance, not log data

When in doubt: if the agent seems to want an investigation of a specific call/message identifier rather than general advice, set true."""

DOCS_ANSWER_PROMPT = """You are SupportIQ — a senior Plivo engineer with deep knowledge of Voice API, SMS/Messaging API, Phone Numbers, and SIP Trunking. When an agent asks how something works, give a direct, complete explanation.

Rules:
1. If a LIVE TICKET is present at the top of the context, use its thread to understand what the customer is asking about. Do NOT treat the ticket's own resolution as authoritative — use it for context only and answer from your Plivo API knowledge.
2. Explain directly — the agent wants to understand how something works, not a doc summary.
3. Structure your answer using markdown: **What it is** → **How it works** → **Key constraints**. Use `\n\n` between sections. Use bullet points for lists. Use backticks for parameter names and code values.
4. If the question asks for an implementation example, include a short code snippet.
5. Cite docs by title: "Per the Confluence doc 'Voice API Overview'..." or "Per the Plivo docs..."
6. Do NOT reference support tickets for product explanation questions.
7. If the docs don't cover something, say so — do not hallucinate.
8. End with one actionable next step. If you reference a doc URL in suggested_action, use ONLY URLs from the "Available doc URLs" list supplied in the user message — never invent or guess URLs.

Return ONLY a JSON object. The "answer" field MUST use the exact three-section structure below — never a single paragraph:
{
  "reasoning": "1 sentence: which docs were relevant and why",
  "answer": "**What it is**\\n[one sentence definition]\\n\\n**How it works**\\n- [key point]\\n- [key point]\\n- [key point]\\n\\n**Key constraints / gotchas**\\n- [constraint or limit]\\n- [common mistake]",
  "suggested_action": "one actionable next step (if linking to docs, use only URLs from the Available doc URLs list)",
  "citations": [
    {
      "ticket_id": null,
      "subject": "doc title",
      "excerpt": "most relevant sentence from the doc",
      "resolution_type": null
    }
  ]
}"""

FIND_SIMILAR_PROMPT = """You are SupportIQ — an internal tool for Plivo support agents covering Voice API, SMS/Messaging, Phone Numbers, and SIP Trunking. The agent wants to find past tickets or precedents that match a scenario — not a troubleshooting guide.

Rules:
1. DO NOT invent a problem. The agent is asking "have you seen this before?" — not "I have this problem, fix it."
2. For each source, give: ticket ID or doc title, a 1–2 sentence summary of what happened, and what the resolution was.
3. If multiple tickets match, list all of them — most relevant first.
4. End with a 1-sentence takeaway: the common pattern across matching cases.
5. Do NOT generate Diagnosis / Possible causes / Primary fix sections. That is the wrong format for this query.

Return ONLY a JSON object:
{
  "reasoning": "1 sentence: what scenario was searched for and what pattern was found",
  "answer": "Yes/No — [direct answer to whether this scenario has been seen]\\n\\n[For each matching source: **Ticket #XXXXX** or **doc title** — [1-2 sentence summary of what happened and how it resolved.]]\\n\\n**Pattern:** [1-sentence takeaway across all cases]",
  "suggested_action": "one sentence — the most useful next step given the precedents found, or empty string if not applicable",
  "citations": [
    {
      "ticket_id": "... or null",
      "subject": "...",
      "excerpt": "most relevant sentence showing the match",
      "resolution_type": "..."
    }
  ]
}"""

ANSWER_PROMPT = """You are SupportIQ — a senior Plivo engineer who has resolved hundreds of support cases across Voice API, SMS/Messaging, Phone Numbers, and SIP Trunking. When an agent brings you a problem, give them a complete resolution guide — not just an investigation starting point.

Rules:
1. If a LIVE TICKET is present at the top of the context, use it to understand what the CUSTOMER is experiencing — the ticket thread tells you the problem. Do NOT treat the ticket's own resolution notes as the definitive answer; previous agents may have given incomplete or incorrect advice. Cross-reference the ticket's symptoms with your broader Plivo API knowledge and the retrieved evidence to produce the correct resolution.
2. Start from symptoms: identify exactly what the customer is experiencing before naming causes.
3. Identify ALL plausible root causes ranked by probability. Use ticket evidence to support rankings.
4. Only cite a source if its root cause genuinely matches the described symptoms.
5. Do NOT introduce products, features, or tools (e.g. PHLO) unless the customer mentioned them or multiple sources strongly point to them.
6. Cite ticket sources as "Ticket #36286 showed the same pattern..." — evidence supporting your ranking.
7. Cite doc sources as "Per the Confluence SOP on X..." or "Jira VT-123 confirms..."
8. The suggested_action is the single most important first step. One sentence, one Ryuk path or tool. If you include a doc URL, use ONLY URLs from the "Available doc URLs" list — never invent or guess URLs.
9. If context is insufficient, state what information would help — do not hallucinate.
10. Temporal awareness: if an older source contradicts a newer one on the same behaviour, rely on the newer source and note that platform behaviour has changed. If ALL citations are older than 1 year, add a brief caveat at the end of the answer: "Note: These findings are from [year]. Carrier policies and platform behaviour may have changed — verify before applying."
11. Ryuk navigation — always use exact paths:

    VOICE:
    - Account state: Account Management → Accounts → [search email] → Account Settings tab.
      Key fields: Enabled, Voice Enabled, Zentrunk Enabled, CPS Allowed, Zentrunk CPS Allowed, Voice Carrier Type, Plivo Carrier Fallback, Risk Status, Account Notes.
    - Call records (Voice API): Voice → Fetch CDRs (by call UUID or date range)
    - Call visual trace: Voice → Call Timeline (by call UUID — full SIP/media event sequence)
    - PCAP analysis: Voice → PCAP Analyser v2 (load PCAP for call UUID — SIP responses, RTP, carrier leg failures)
    - CPS limits: Voice → Voice CPS utilisation
    - Voice geo permissions: Voice → Voice Geo Permissions (by username — check if destination country is enabled)
    - Zentrunk trunks: Zentrunk → Trunks (credentials, IP whitelist, registration status)
    - Zentrunk call records: Zentrunk → Fetch CDRs
    - Zentrunk routing: Zentrunk → Routing (routing profile assigned to account)

    SMS / MESSAGING:
    - SMS message records: Messaging → Fetch MDRs (search by Auth ID + From/To number or Message UUID + date range).
      Check Status column (queued/sent/delivered/undelivered/failed) and error codes on each record.
    - Decode an error code: Messaging → Error Code Mapping (enter Plivo error code → shows carrier error code, description, recommended action).
      For all standard codes: Messaging → Error Code Mapping → Plivo Master List.
    - Account SMS config (scroll 3 of Account Settings): Account Management → Accounts → [account] → Account Settings.
      Key SMS fields: SMS Carrier Profile Group, SMS Routing Strategy, SMS MPS Allowed (messages-per-second limit).
    - SMS blocked at operator level: Account Management → Accounts → [account] → Account Settings → Messaging Actions.
      Check if "Disable Messages at Operator Level" is active. Also check "SMS Cost Check" — if a daily spend threshold was hit, outbound SMS is blocked.
    - Country geo permissions for SMS: Messaging → SMS Geo Permissions (search by Auth ID).
      Check if destination country is Enabled. If Hard Disabled=Yes, escalate — cannot be unlocked at support level.
    - Carrier routing: Messaging → Outbound SMS Routing (Outgoing Carriers) — shows which carrier handles this account for each country/message type, priority, and active/inactive status.
    - Powerpack: Messaging → Powerpack — check if account uses a number pool, pool size, sticky sender status.
    - US 10DLC: Messaging → 10DLC & Tollfree Management → Campaigns tab — check campaign status, failure reasons, per-carrier MNO status columns (AT&T, T-Mobile, Verizon separately).
    - US Tollfree SMS: Messaging → 10DLC & Tollfree Management → Tollfree Verifications — check verification status.
    - India DLT templates: Messaging → Template Management (search by Auth ID or DLT Template ID) — check Status (Approved/Pending/Rejected/In Review).
    - Sender ID registration: Messaging → Sender ID Registration (search by Auth ID or Sender ID) — check status, Carrier Mapping view, History tab for audit trail.

    NUMBERS / FINANCE:
    - Number issues: Numbers → Number Management or Number History
    - Finance / limit charges: Finance → Recurring Charge

MANDATORY ANSWER FORMAT — five sections, separated by blank lines. Never collapse into a single paragraph.

**Diagnosis**
[One sentence — the most likely root cause based on the symptoms.]

**Possible causes**
1. [Most likely cause — why]
2. [Second most likely — why]
3. [Third if applicable — why]

**Evidence from past tickets**
[What each cited ticket showed — "Ticket #XXXXX showed [specific finding]." One line per ticket.]

**Primary fix**
Only include steps you can ground in specific evidence or Ryuk paths from Rule 11. Each step must have a concrete action + specific location + what to look for or what value indicates the problem. 1-2 precise steps beat 4 vague ones. Never write a step that amounts to "retest", "wait", "confirm the issue exists", or "check general settings" — omit it instead.

**If unresolved — fallback actions**
Only include if you have a genuinely different diagnostic path to try. If the primary fix covers it, leave this section brief or with just the escalation line. Escalation must name the specific team and what to prepare (logs, UUIDs, error codes, Ryuk findings).

Return ONLY a JSON object:
{
  "reasoning": "1-2 sentences: what are the symptoms, most likely root cause and why",
  "answer": "**Diagnosis**\\n[root cause]\\n\\n**Possible causes**\\n1. [primary]\\n2. [secondary]\\n3. [third if clearly distinct]\\n\\n**Evidence from past tickets**\\n[ticket evidence]\\n\\n**Primary fix**\\n1. [specific action + Ryuk path + what value to look for]\\n2. [next concrete step if warranted by evidence]\\n\\n**If unresolved — fallback actions**\\n[different diagnostic path if applicable, then escalation line with specific team + what to prepare]",
  "suggested_action": "REQUIRED. Copy step 1 from Primary fix as a single sentence. Must NOT be empty.",
  "citations": [
    {
      "ticket_id": "... or null for non-ticket sources",
      "subject": "...",
      "excerpt": "most relevant sentence",
      "resolution_type": "..."
    }
  ]
}"""

SUMMARIZE_PROMPT = """You are SupportIQ — an internal assistant for Plivo support agents.

The agent has shared a live ticket and wants a plain summary of what has happened — not a diagnosis or resolution guide.

Write a clear, chronological summary of the ticket thread. Cover:
1. What the customer reported (the original issue)
2. What has been investigated or tried so far (by either side)
3. Key findings or information shared (call UUIDs, error codes, logs, screenshots mentioned)
4. Current status — is it resolved, pending customer response, pending internal action?
5. What is still unresolved or unclear (if anything)

Keep it factual and concise. Do not diagnose root causes, do not suggest fixes, do not add sections like "Possible causes" or "Primary fix". Just tell the agent what happened.

Return ONLY a JSON object:
{
  "reasoning": "1 sentence: what this ticket is about",
  "answer": "Plain chronological summary covering the 5 points above. Use bullet points or short paragraphs. No diagnosis sections.",
  "suggested_action": "",
  "citations": []
}"""

CONVERSATIONAL_PROMPT = """You are SupportIQ — an internal assistant for Plivo support agents.

You are handling one of two cases:

CASE A — Meta-comment about a previous response:
The agent is making a correction, confirmation, or meta-question about your previous answer. Respond naturally and concisely.
- If they are pointing out you made a mistake: acknowledge it clearly, state the correct information.
- If they ask "were you right/wrong?": answer directly (yes/no), then give the correct answer in one sentence.
- If they provided a correction (e.g. "my senior said..."): confirm the correction is right and briefly why.
- If they are confirming your response was correct: acknowledge simply.
- If they say the cited tickets/results are wrong or not relevant ("these tickets aren't related", "none of these match"): acknowledge it — say the coverage for this topic may be limited and suggest what the agent should do next (check Zendesk directly, escalate, etc.). Do NOT say "I don't see that in the ticket."
- If they ask you to narrow down or pick one from your previous answer ("pin it down", "which one is most likely", "narrow it down"): choose the single most likely cause from your previous response and explain why in 2 sentences.
- Keep the response to 2–4 sentences maximum.

CASE B — Simple data-lookup about a live ticket:
If ticket information is shown above, the agent is asking about ticket metadata. Answer directly and only from the ticket fields shown — do not guess or add context.
- "who is assigned?" → "Ticket #XXXXX is assigned to [Assignee name]."
- "what's the status?" → "The ticket status is [status]."
- "who is the requester/customer?" → "The requester is [requester name]."
- "what's the priority?" → "The priority is [priority]."
Keep the response to 1–2 sentences. If the information is not in the ticket, say "I don't see that in the ticket — please check Zendesk directly."

Do NOT produce a 5-section resolution guide. Do NOT cite tickets. Do NOT call Plivo tools or APIs."""


# ── Node: conversational_reply ────────────────────────────────────────────────

async def conversational_reply_node(state: AgentState, client: AsyncOpenAI, model: str) -> dict:
    """
    Handles conversational follow-ups: corrections, confirmations, meta-questions
    about the bot's own previous responses. Skips all retrieval — chat history only.
    """
    t0 = time.monotonic()
    history = list(state.get("chat_history") or [])

    messages = [{"role": "system", "content": CONVERSATIONAL_PROMPT}]
    # Include last 8 turns of history so the bot has context of what it said before
    messages.extend(history[-8:])

    # If a live ticket was injected (either from current message or from chat history),
    # prepend it to the user message so the bot can answer metadata questions directly.
    user_content = state["question"]
    injected = state.get("injected_context")
    if injected:
        user_content = f"[Live ticket context:]\n{injected}\n\n---\nAgent's question: {user_content}"
    messages.append({"role": "user", "content": user_content})

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            seed=42,
            max_tokens=300,
        )
        answer_text = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Conversational reply failed: {e}")
        answer_text = "Got it — thanks for the correction."

    updated_history = list(history)
    updated_history.append({"role": "user", "content": state["question"]})
    updated_history.append({"role": "assistant", "content": answer_text})

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Conversational reply: {elapsed}ms")
    return {
        "answer": answer_text,
        "citations": [],
        "suggested_action": "",
        "related_tickets": [],
        "confidence_score": 1.0,
        "confidence_factors": {},
        "reasoning": "Conversational follow-up — no retrieval.",
        "chat_history": updated_history,
        "timings": {**(state.get("timings") or {}), "answer_ms": elapsed},
    }


# ── Node: fetch_ticket_context ───────────────────────────────────────────────

async def fetch_ticket_context_node(state: AgentState, settings) -> dict:
    """
    If the user's message contains a Zendesk ticket URL OR a bare ticket ID,
    fetch that ticket live and inject its full thread as context.
    URLs are removed from the question; bare IDs are kept (natural sentence structure).

    Priority 3: if the current message has no URL/ID, scan the last 8 chat history
    entries (≈4 turns) for a recent ticket URL — so follow-up questions like
    "who is assigned to it?" still have ticket context without the agent re-pasting the URL.
    """
    question = state["question"]
    ticket_id = None
    cleaned_question = question
    strip_id_from_question = False

    # Priority 1: full Zendesk URL in current message
    url_only = False
    url_match = _ZENDESK_URL_RE.search(question)
    if url_match:
        ticket_id = url_match.group(1)
        cleaned = _ZENDESK_URL_RE.sub("", question).strip()
        if not cleaned:
            # Agent pasted URL with no question — ask what they want
            url_only = True
            cleaned_question = f"Ticket #{ticket_id} submitted for analysis."
        else:
            cleaned_question = cleaned
        strip_id_from_question = True
    else:
        # Priority 2: bare ticket ID in current message
        bare_match = _BARE_TICKET_ID_RE.search(question)
        if bare_match:
            ticket_id = next(g for g in bare_match.groups() if g is not None)
        else:
            # Priority 3: use active_ticket_id tracked by the Slack bot.
            # When an agent pastes a ticket URL in a thread, the Slack bot stores
            # that ticket ID and sends it on every subsequent turn. No history
            # scanning needed — the session knows which ticket is active.
            ticket_id = state.get("active_ticket_id") or None

    if not ticket_id:
        return {"injected_context": None}

    # Check server-side cache before calling Zendesk
    cached = _ticket_context_cache.get(ticket_id)
    if cached and (time.time() - cached["fetched_at"]) < _TICKET_CACHE_TTL:
        logger.info(f"Ticket #{ticket_id} served from cache ({len(cached['context'])} chars)")
        result = {"injected_context": cached["context"], "live_ticket_id": ticket_id}
        if strip_id_from_question:
            result["question"] = cleaned_question
        if url_only:
            result["url_only_ticket"] = True
            result["clarification_question"] = f"Got ticket *#{ticket_id}*! What would you like me to do?\n• *Summarize* — what happened, who said what, current status\n• *Diagnose* — root cause analysis with suggested next steps"
        return result

    try:
        context = await fetch_ticket_by_id(
            ticket_id=ticket_id,
            subdomain=settings.zendesk_subdomain,
            email=settings.zendesk_email,
            api_key=settings.zendesk_api_key,
        )
        logger.info(f"Fetched live ticket #{ticket_id} from Zendesk ({len(context)} chars)")
        _ticket_context_cache[ticket_id] = {"context": context, "fetched_at": time.time()}
        result = {"injected_context": context, "live_ticket_id": ticket_id}
        if strip_id_from_question:
            result["question"] = cleaned_question
        if url_only:
            result["url_only_ticket"] = True
            result["clarification_question"] = f"Got ticket *#{ticket_id}*! What would you like me to do?\n• *Summarize* — what happened, who said what, current status\n• *Diagnose* — root cause analysis with suggested next steps"
        return result
    except Exception as e:
        logger.error(f"Failed to fetch ticket #{ticket_id}: {e}")
        return {
            "injected_context": None,
            "error": f"⚠️ Couldn't load ticket #{ticket_id} from Zendesk. Please check the URL is correct and try again.",
        }


# ── Node: classify_intent ────────────────────────────────────────────────────

async def classify_intent(state: AgentState, client: AsyncOpenAI, model: str) -> dict:
    """Classify intent, query_type, and generate HyDE text — all in one LLM call."""
    t0 = time.monotonic()
    history_text = ""
    if state.get("chat_history"):
        last = state["chat_history"][-6:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in last)

    user_msg = state["question"]
    if history_text:
        user_msg = f"Conversation so far:\n{history_text}\n\nNew question: {state['question']}"

    # Follow-up grounding: if the question is ambiguous ("can you explain with an example",
    # "give me an example", "elaborate", etc.) it has no standalone topic signal.
    # Explicitly prepend the last bot response so the classifier knows what "it" refers to,
    # rather than letting older history messages (e.g. unrelated Voice API questions) bias
    # is_voice_related and HyDE generation.
    if _FOLLOWUP_RE.search(state["question"]):
        last_assistant = next(
            (m["content"] for m in reversed(state.get("chat_history") or [])
             if m.get("role") == "assistant"),
            None,
        )
        if last_assistant:
            user_msg = (
                f"[Follow-up context — the agent is continuing from this previous bot response:]\n"
                f"{last_assistant[:400]}\n\n"
                f"{user_msg}"
            )

    # If a live ticket was injected, prepend a summary so HyDE is grounded in the actual problem
    injected = state.get("injected_context")
    if injected:
        # Give the LLM the first ~800 chars of the ticket thread for classification + HyDE
        snippet = injected[:800]
        user_msg = f"[Live ticket submitted for analysis]\n{snippet}\n\n---\nAgent's question: {user_msg}"

    classify_cost_entry = None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_HYDE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if response.usage:
            classify_cost_entry = make_entry(
                "classify+hyde", model,
                response.usage.prompt_tokens, response.usage.completion_tokens,
            )
    except Exception as e:
        logger.error(f"Classify+HyDE failed: {e}")
        result = {
            "is_voice_related": True,
            "query_type": "ticket_search",
            "intent": "troubleshoot",
            "is_ambiguous": False,
            "clarification_question": None,
            "hyde_text": state["question"],
        }

    is_voice = result.get("is_voice_related", True)
    query_type = result.get("query_type", "ticket_search")
    hyde_text = result.get("hyde_text") or state["question"]
    pre_dx = result.get("pre_diagnosis") or {}

    # HyDE bypass: if query contains exact technical signals (UUIDs, SIP/HTTP error codes,
    # stack traces, IP addresses), embed the raw question instead of the hallucinated
    # hypothetical. BM25 sparse vectors handle exact string matching better than a
    # generic LLM-generated resolution document.
    hyde_bypassed = False
    if _TECHNICAL_SIGNAL_RE.search(state["question"]):
        hyde_text = state["question"]
        hyde_bypassed = True

    costs = list(state.get("cost_entries") or [])
    if classify_cost_entry:
        costs.append(classify_cost_entry)

    is_ambiguous = result.get("is_ambiguous", False)
    needs_clarification = result.get("needs_clarification", False)

    # If a live ticket was injected, the ticket IS the context — never ask for clarification.
    # The agent pasted a URL specifically to get an answer; asking follow-up questions is wrong.
    if state.get("injected_context"):
        is_ambiguous = False
        needs_clarification = False
        # Keep query_type as classified — live ticket queries benefit from balanced lanes (5+5),
        # not the docs-heavy allocation (2+8). The agent needs similar past tickets as evidence,
        # not feature docs. Prompt selection already uses injected_context to pick ANSWER_PROMPT.

    wants_log_investigation = result.get("wants_log_investigation", False)

    # Map pre_diagnosis signals to Qdrant metadata filters.
    # Product-area filters prevent cross-product ticket contamination (e.g. SMS tickets
    # appearing in voice queries because both mention "spam" or "delivery failure").
    # Don't override filters the frontend already set (e.g. from UI filter chips).
    metadata_filters = state.get("metadata_filters") or {}
    if not metadata_filters and query_type == "ticket_search":
        product_area = pre_dx.get("product_area")
        failure_point = pre_dx.get("failure_point")
        if product_area == "zentrunk":
            metadata_filters = {"product": "sip_trunking"}
        elif product_area in ("voice_api", "webrtc"):
            metadata_filters = {"product": "voice_api"}
        elif product_area in ("sms", "messaging"):
            metadata_filters = {"product": "sms"}
        elif product_area == "phone_numbers":
            metadata_filters = {"product": "phone_numbers"}
        elif failure_point == "dtmf":
            metadata_filters = {"issue_type": "dtmf"}

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(
        f"Classify+HyDE: voice={is_voice} | type={query_type} | intent={result.get('intent')} "
        f"| ambiguous={is_ambiguous} | needs_clarification={needs_clarification} "
        f"| hyde_bypass={hyde_bypassed} | pre_dx={pre_dx} | {elapsed}ms"
    )
    return {
        "is_voice_related": is_voice,
        "query_type": query_type,
        "intent": result.get("intent", "troubleshoot"),
        "is_ambiguous": is_ambiguous,
        "needs_clarification": needs_clarification,
        "clarification_question": result.get("clarification_question"),
        "awaiting_clarification": is_ambiguous or needs_clarification,
        "hyde_text": hyde_text,
        "pre_diagnosis": pre_dx if pre_dx else None,
        "metadata_filters": metadata_filters,
        "wants_log_investigation": wants_log_investigation,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "classify_ms": elapsed},
    }



# ── Node: generate_hyde ──────────────────────────────────────────────────────

async def generate_hyde_node(
    state: AgentState,
    client: AsyncOpenAI,
    embedder,
    model: str,
) -> dict:
    """Embed the hyde_text produced by classify_intent. Pure embed — no LLM call."""
    t0 = time.monotonic()
    # hyde_text already set by classify_intent (merged call)
    hyde_text = state.get("hyde_text") or state["question"]
    dense, sparse = await embedder.embed_query_async(hyde_text)

    embed_entry = make_entry("embed_hyde", embedder.dense_model, estimate_tokens(hyde_text))
    costs = list(state.get("cost_entries") or [])
    costs.append(embed_entry)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Embed: {elapsed}ms")
    return {
        "hyde_dense": dense,
        "hyde_sparse": sparse,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "hyde_ms": elapsed},
    }


# ── Node: retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, qdrant: QdrantClient, top_k: int) -> dict:
    """Guaranteed-lane hybrid retrieval using embedded HyDE text."""
    t0 = time.monotonic()
    chunks = retrieve(
        client=qdrant,
        dense_vector=state["hyde_dense"],
        sparse_vector=state["hyde_sparse"],
        top_k=top_k,
        filters=state.get("metadata_filters") or None,
        query_type=state.get("query_type", "ticket_search"),
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Retrieve: {len(chunks)} chunks in {elapsed}ms")
    return {
        "retrieved_chunks": chunks,
        "timings": {**(state.get("timings") or {}), "retrieve_ms": elapsed},
    }


# ── Node: rerank ─────────────────────────────────────────────────────────────

def rerank_node(state: AgentState, top_n: int) -> dict:
    """Cross-encoder reranking of retrieved chunks."""
    t0 = time.monotonic()
    # Use hyde_text as rerank query instead of the raw question.
    # Raw questions often include conversational preamble ("this is sarvesh, I got a ticket...")
    # which confuses the cross-encoder and produces universally negative scores.
    # hyde_text is a clean, Plivo-specific problem description that scores much better.
    rerank_query = state.get("hyde_text") or state["question"]

    # Exclude the live ticket being discussed — it's already in injected_context and
    # citing it as "Evidence from past tickets" is confusing/redundant.
    live_id = state.get("live_ticket_id")
    candidates = state["retrieved_chunks"]
    if live_id:
        candidates = [c for c in candidates if str(c.get("ticket_id", "")) != str(live_id)]

    reranked = rerank(
        query=rerank_query,
        chunks=candidates,
        top_n=top_n,
    )
    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Rerank: {len(reranked)} chunks in {elapsed}ms")
    return {
        "reranked_chunks": reranked,
        "timings": {**(state.get("timings") or {}), "rerank_ms": elapsed},
    }


# ── Node: compress ───────────────────────────────────────────────────────────

async def compress_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
    qdrant: QdrantClient = None,
) -> dict:
    """
    For ticket_search: parent-child expansion — fetches all sibling chunks (problem +
    investigation + resolution) for each winning ticket and combines into one block.
    GPT-4o sees the full diagnostic thread, not just the chunk that won the reranker.

    For product_question: pass-through — LLM compression removed. GPT-4o-mini was
    stripping relevant doc context; GPT-4o has 128k context and doesn't need it.
    """
    t0 = time.monotonic()

    if state.get("query_type") == "ticket_search":
        # Parent-child expansion: for each winning ticket, fetch ALL its sibling chunks
        # (problem + investigation + resolution) and combine into one context block.
        # This ensures GPT-4o sees the full diagnostic thread, not just the one chunk
        # that won the reranker lottery.
        _CHUNK_ORDER = {"problem": 0, "investigation": 1, "resolution": 2}
        winning_chunks = state.get("reranked_chunks", [])[:4]
        expanded = []
        for c in winning_chunks:
            tid = c.get("ticket_id")
            if not tid or not qdrant:
                c = dict(c)
                c["compressed_text"] = c.get("text", "")
                expanded.append(c)
                continue

            siblings = _fetch_all_ticket_chunks(qdrant, str(tid))
            siblings.sort(key=lambda x: _CHUNK_ORDER.get(x.get("chunk_type", ""), 99))

            parts = []
            for s in siblings:
                label = s.get("chunk_type", "context").upper()
                parts.append(f"[{label}]\n{s.get('text', '')}")

            c = dict(c)
            c["compressed_text"] = "\n\n".join(parts) if parts else c.get("text", "")
            expanded.append(c)

        elapsed = int((time.monotonic() - t0) * 1000)
        logger.info(f"Compress: parent-child expansion {len(winning_chunks)} tickets → {len(expanded)} expanded in {elapsed}ms")
        return {
            "compressed_chunks": expanded,
            "timings": {**(state.get("timings") or {}), "compress_ms": elapsed},
        }

    # product_question: pass through reranked chunks without LLM compression.
    # GPT-4o-mini compression was stripping relevant doc context; GPT-4o has 128k
    # context and doesn't need it. Raw chunks give the model the full picture.
    input_chunks = state["reranked_chunks"][:4]
    passed_through = []
    for c in input_chunks:
        c = dict(c)
        c["compressed_text"] = c.get("text", "")
        passed_through.append(c)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Compress: pass-through (no LLM) {len(passed_through)} doc chunks in {elapsed}ms")
    return {
        "compressed_chunks": passed_through,
        "timings": {**(state.get("timings") or {}), "compress_ms": elapsed},
    }


# ── Node: check_confidence ───────────────────────────────────────────────────

_DOC_SOURCE_QUALITY = {"confluence": 1.0, "docs": 0.9, "slack": 0.7, "jira": 0.6}


def check_confidence_node(state: AgentState, threshold: float) -> dict:
    """
    Compute confidence score from multiple signals.
    Uses different formula for product_question vs ticket_search because
    ticket-specific signals (resolution_type, csat, recency) are meaningless for docs.
    """
    chunks = state.get("reranked_chunks", [])
    if not chunks:
        return {
            "confidence_score": 0.0,
            "confidence_factors": {"error": "No chunks retrieved"},
        }

    # Factor 1: Top reranker score normalised 0-1 via sigmoid.
    # bge-reranker-v2-m3 returns raw logits (any range); sigmoid maps them cleanly to 0-1.
    top_score = chunks[0].get("rerank_score", 0)
    retrieval_sim = 1 / (1 + math.exp(-top_score))

    if state.get("query_type") == "product_question":
        # ── Docs path ──────────────────────────────────────────────────────────
        # Ticket signals (resolution_type, csat, recency) don't exist on doc chunks.
        # Use retrieval relevance (60%) + source quality (40%).
        src_scores = [_DOC_SOURCE_QUALITY.get(c.get("source", ""), 0.5) for c in chunks]
        avg_src = sum(src_scores) / len(src_scores)

        score = round(min(max(retrieval_sim * 0.60 + avg_src * 0.40, 0.0), 1.0), 3)
        factors = {
            "retrieval_similarity": round(retrieval_sim, 3),
            "source_quality": f"avg {round(avg_src, 2)} "
                              f"({'Confluence/Docs' if avg_src > 0.8 else 'Jira/Slack'})",
        }

    else:
        # ── Ticket path ────────────────────────────────────────────────────────
        # Factor 2: Resolution consistency (how many top chunks share same resolution_type)
        res_types = [c.get("resolution_type") for c in chunks if c.get("resolution_type")]
        if res_types:
            most_common = max(set(res_types), key=res_types.count)
            consistency = res_types.count(most_common) / len(res_types)
        else:
            consistency = 0.5

        # Factor 3: Recency (average age, penalise old tickets)
        ages = []
        for c in chunks:
            created = c.get("created_at")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    ages.append((datetime.now(timezone.utc) - dt).days)
                except Exception:
                    pass
        recency = 1.0 - min(sum(ages) / max(len(ages), 1) / 365, 1.0) if ages else 0.7

        # Factor 4: CSAT
        _CSAT_MAP = {"good": 5, "bad": 1, "offered": 3, "unoffered": 3}
        csats_raw = [c.get("csat_score") for c in chunks if c.get("csat_score") is not None]
        csats = [_CSAT_MAP.get(str(v).lower(), 3) for v in csats_raw]
        csat_norm = (sum(csats) / len(csats)) / 5.0 if csats else 0.7

        score = round(min(max(
            retrieval_sim * 0.40
            + consistency * 0.25
            + recency * 0.15
            + csat_norm * 0.20,
            0.0), 1.0), 3)

        factors = {
            "retrieval_similarity": round(retrieval_sim, 3),
            "resolution_consistency": f"{'High' if consistency > 0.7 else 'Medium' if consistency > 0.4 else 'Low'}",
            "recency": f"{'< 90 days' if recency > 0.75 else '< 1 year' if recency > 0.5 else '> 1 year'}",
            "source_quality": f"avg CSAT {round(sum(csats)/len(csats), 1) if csats else 'N/A'}",
        }

    logger.info(f"Confidence [{state.get('query_type','ticket_search')}]: {score} | {factors}")
    return {
        "confidence_score": score,
        "confidence_factors": factors,
        "timings": {**(state.get("timings") or {}), "confidence_ms": 0},
    }


# ── Node: generate_answer ────────────────────────────────────────────────────

async def generate_answer_node(
    state: AgentState,
    client: AsyncOpenAI,
    model: str,
    qdrant: QdrantClient,
) -> dict:
    """CoT answer generation using compressed chunks as context."""
    t0 = time.monotonic()

    # Log full retrieval pipeline state for debugging answer quality
    _log_retrieval_debug(state)

    # Cache check — skip retrieval + LLM for repeated questions.
    # Don't cache when a live ticket is injected (ticket-specific context must be fresh).
    if not state.get("injected_context"):
        cached = get_cached(state["question"], state.get("query_type", ""))
        if cached:
            history = list(state.get("chat_history") or [])
            history.append({"role": "user", "content": state["question"]})
            history.append({"role": "assistant", "content": cached.get("answer", "")})
            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(f"Answer generation: cache hit in {elapsed}ms")
            return {**cached, "chat_history": history,
                    "timings": {**(state.get("timings") or {}), "answer_ms": elapsed}}

    chunks = state.get("compressed_chunks") or state.get("reranked_chunks", [])

    DOC_SOURCES = {"docs", "confluence", "jira", "slack", "ryuk", "support"}

    # Build context — live ticket first (if present), then RAG chunks
    context_parts = []
    injected = state.get("injected_context")
    if injected:
        context_parts.append(
            f"=== LIVE TICKET (submitted by agent for analysis) ===\n{injected}\n==="
        )

    for i, c in enumerate(chunks, 1):
        text = c.get("compressed_text") or c.get("text", "")
        source = c.get("source", "")
        if source in DOC_SOURCES:
            if source == "confluence":
                label = f"Confluence — {c.get('title', 'N/A')}"
            elif source == "jira":
                label = f"Jira {c.get('project', '')} — {c.get('summary', c.get('title', 'N/A'))}"
            elif source == "slack":
                label = f"Slack #{c.get('channel', 'N/A')}"
            elif source == "ryuk":
                label = f"Ryuk — {c.get('nav_path', c.get('section', 'N/A'))}"
            elif source == "support":
                label = f"Plivo Support — {c.get('page_title', 'N/A')} ({c.get('category', '')})"
            else:
                label = f"Plivo Docs — {c.get('page_title', 'N/A')} > {c.get('section_title', '')}"
            context_parts.append(f"[{label}]\n{text}")
        else:
            root_cause = c.get("root_cause") or c.get("resolution_summary") or ""
            root_cause_line = f"Root cause: {root_cause}\n" if root_cause else ""
            context_parts.append(
                f"[Ticket #{c.get('ticket_id', 'N/A')} — {c.get('subject', 'N/A')}]\n"
                f"Type: {c.get('chunk_type')} | Product: {c.get('product')} | "
                f"Region: {c.get('region')} | CSAT: {c.get('csat_score', 'N/A')} | "
                f"Solved: {c.get('solved_at') or c.get('created_at', 'N/A')}\n"
                f"{root_cause_line}"
                f"{text}"
            )
    context = "\n\n---\n\n".join(context_parts)

    # Collect real URLs from retrieved doc chunks to ground suggested_action
    # Prevents the LLM from hallucinating Plivo docs URLs — it must use only these
    URL_SOURCES = {"docs", "confluence"}
    doc_urls = []
    for c in chunks:
        url = c.get("url", "")
        if c.get("source") in URL_SOURCES and url and url not in doc_urls:
            doc_urls.append(url)

    doc_urls_section = ""
    if doc_urls:
        url_list = "\n".join(f"- {u}" for u in doc_urls)
        doc_urls_section = f"\n\nAvailable doc URLs (use ONLY these if referencing docs in suggested_action):\n{url_list}"

    # Pick prompt based on intent + query_type.
    # Priority order:
    # 1. Live ticket + summarize intent → SUMMARIZE_PROMPT (plain timeline, no diagnosis)
    # 2. Live ticket + other intent → ANSWER_PROMPT (full resolution guide)
    # 3. find-similar intent → FIND_SIMILAR_PROMPT (list precedents, no troubleshooting guide)
    # 4. product_question → DOCS_ANSWER_PROMPT (explain how something works)
    # 5. everything else → ANSWER_PROMPT (5-section resolution guide)
    if state.get("injected_context") and state.get("intent") == "summarize":
        prompt = SUMMARIZE_PROMPT
    elif state.get("injected_context"):
        prompt = ANSWER_PROMPT
    elif state.get("intent") == "find-similar":
        prompt = FIND_SIMILAR_PROMPT
    elif state.get("query_type") == "product_question":
        prompt = DOCS_ANSWER_PROMPT
    else:
        prompt = ANSWER_PROMPT

    # Include recent chat history for multi-turn awareness
    messages = [{"role": "system", "content": prompt}]
    for turn in (state.get("chat_history") or [])[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    # Prepend pre_diagnosis signals so GPT-4o can verify root cause match per ticket
    pre_dx = state.get("pre_diagnosis")
    pre_dx_section = ""
    if pre_dx and state.get("query_type") == "ticket_search":
        signals = {k: v for k, v in pre_dx.items() if v}
        if signals:
            sig_lines = "\n".join(f"  {k}: {v}" for k, v in signals.items())
            pre_dx_section = (
                f"\n\n[Pre-diagnosis signals extracted from this question]\n{sig_lines}"
                f"\nUse these to verify whether each ticket's root cause actually matches "
                f"the current issue before citing it."
            )

    # When a live ticket is injected, ANSWER_PROMPT (5-section) is always used regardless
    # of query_type. The format_reminder must match — use ticket_search format.
    effective_qt = "ticket_search" if state.get("injected_context") else state.get("query_type", "")

    format_reminder = ""
    if state.get("intent") == "find-similar":
        format_reminder = (
            "\n\nFORMAT REMINDER: The agent is asking for past precedents, NOT a troubleshooting guide. "
            "DO NOT generate Diagnosis / Possible causes / Primary fix sections. "
            "Your answer MUST follow this format:\n"
            "[Yes/No — direct answer to whether this scenario has been seen]\n\n"
            "**[Ticket #XXXXX or Doc Title]** — [1-2 sentences: what happened and how it resolved]\n"
            "**[Ticket #XXXXX or Doc Title]** — [1-2 sentences: what happened and how it resolved]\n\n"
            "**Pattern:** [1-sentence takeaway across all cases]"
        )
    elif effective_qt == "ticket_search":
        format_reminder = (
            "\n\nFORMAT REMINDER: Five sections, bold headers, blank lines between them:\n"
            "**Diagnosis** — one sentence root cause.\n\n"
            "**Possible causes** — ranked list, only causes you can justify.\n\n"
            "**Evidence from past tickets** — one line per ticket, what it specifically showed.\n\n"
            "**Primary fix** — QUALITY RULE: each step must have (a) a specific action, (b) exact Ryuk path from Rule 11 or specific API/tool, (c) what value or finding confirms the problem. "
            "Write only steps you can ground in the evidence. 1-2 sharp steps > 4 vague ones. "
            "NEVER write steps like 'Retest the issue', 'Verify the problem persists', 'Check general settings', 'Contact carrier to confirm' without a specific contact path — these are not actionable, omit them.\n\n"
            "**If unresolved — fallback actions** — only if there is a genuinely different path to try. "
            "Always end with one escalation line: who specifically (e.g. Messaging Operations Team, Voice Platform Team) + what to prepare (error codes, UUIDs, Ryuk CDR screenshot, MDR export)."
        )
    elif effective_qt == "product_question":
        format_reminder = (
            "\n\nFORMAT REMINDER: Your answer MUST use exactly these three sections with bold headers and blank lines between them. Never return a single paragraph:\n"
            "**What it is**\n[one sentence definition]\n\n"
            "**How it works**\n- [bullet point]\n- [bullet point]\n\n"
            "**Key constraints / gotchas**\n- [bullet point]\n- [bullet point]"
        )

    messages.append({
        "role": "user",
        "content": f"Question: {state['question']}\n\nContext (tickets, docs, Confluence, Jira, Slack):\n{context}{doc_urls_section}{pre_dx_section}{format_reminder}",
    })

    answer_cost_entry = None
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        if response.usage:
            answer_cost_entry = make_entry(
                "answer", model,
                response.usage.prompt_tokens, response.usage.completion_tokens,
            )
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        result = {
            "reasoning": "Could not generate answer",
            "answer": "I encountered an error generating an answer. Please try rephrasing your question.",
            "suggested_action": None,
            "citations": [],
        }

    # Fetch related tickets from the same cluster
    related = _fetch_related(qdrant, chunks)

    # Append this turn to chat_history for multi-turn context
    answer_text = result.get("answer", "")
    history = list(state.get("chat_history") or [])
    history.append({"role": "user", "content": state["question"]})
    history.append({"role": "assistant", "content": answer_text})

    # Enrich LLM citations with metadata from the actual chunks
    llm_citations = result.get("citations", [])
    chunk_map = {str(c.get("ticket_id")): c for c in chunks if c.get("ticket_id")}

    enriched_citations = []
    for cit in llm_citations:
        tid = cit.get("ticket_id")
        if not tid or str(tid).lower() in ("none", "null", ""):
            continue  # Doc citations — handled by the supplementary loop below
        chunk = chunk_map.get(str(tid), {})
        enriched_citations.append({
            "source": "ticket",
            "ticket_id": tid,
            "subject": cit.get("subject"),
            "excerpt": cit.get("excerpt", ""),
            "resolution_type": cit.get("resolution_type") or chunk.get("resolution_type"),
            "csat_score": chunk.get("csat_score"),
            "zendesk_url": chunk.get("zendesk_url"),
        })

    # Append non-ticket chunks that appeared in context as supplementary citations
    DOC_SOURCES = {"docs", "confluence", "jira", "slack", "ryuk", "support"}
    for c in chunks:
        if c.get("source") in DOC_SOURCES:
            enriched_citations.append({
                "source": c.get("source"),
                "page_title": c.get("page_title") or c.get("title") or c.get("nav_path") or c.get("summary", ""),
                "section_title": c.get("section_title") or c.get("section") or c.get("channel") or c.get("project", ""),
                "url": c.get("url", ""),
                "excerpt": (c.get("compressed_text") or c.get("text", ""))[:200],
            })

    costs = list(state.get("cost_entries") or [])
    if answer_cost_entry:
        costs.append(answer_cost_entry)

    elapsed = int((time.monotonic() - t0) * 1000)
    logger.info(f"Answer generation: {elapsed}ms")

    response_payload = {
        "answer": answer_text,
        "reasoning": result.get("reasoning", ""),
        "citations": enriched_citations,
        "suggested_action": result.get("suggested_action", ""),
        "related_tickets": related,
        "confidence_score": state.get("confidence_score", 0.0),
        "confidence_factors": state.get("confidence_factors", {}),
    }

    # Store in cache for future identical questions (skip for live-ticket queries)
    if not state.get("injected_context"):
        set_cached(state["question"], state.get("query_type", ""), response_payload)

    return {
        **response_payload,
        "chat_history": history,
        "cost_entries": costs,
        "timings": {**(state.get("timings") or {}), "answer_ms": elapsed},
    }


def _fetch_all_ticket_chunks(qdrant: QdrantClient, ticket_id: str) -> list[dict]:
    """
    Fetch all sibling chunks (problem + investigation + resolution) for a ticket.
    Searches both Qdrant collections so parent-child context is always complete.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    all_chunks = []
    for collection in ("ticket_problems", "ticket_resolutions"):
        try:
            results = qdrant.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="ticket_id", match=MatchValue(value=ticket_id))]
                ),
                limit=5,
                with_vectors=False,
            )
            for point in results[0]:
                all_chunks.append(point.payload)
        except Exception as e:
            logger.warning(f"Failed to fetch {collection} chunks for ticket {ticket_id}: {e}")
    return all_chunks


def _fetch_related(qdrant: QdrantClient, chunks: list[dict]) -> list[dict]:
    """Fetch other tickets in the same cluster as the top result."""
    if not chunks:
        return []

    cluster_id = chunks[0].get("cluster_id")
    if not cluster_id:
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = qdrant.scroll(
            collection_name="ticket_resolutions",
            scroll_filter=Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
            ),
            limit=5,
        )
        related = []
        top_ticket_id = chunks[0].get("ticket_id")
        for point in results[0]:
            if point.payload.get("ticket_id") != top_ticket_id:
                related.append({
                    "ticket_id": point.payload.get("ticket_id"),
                    "subject": point.payload.get("subject"),
                    "resolution_type": point.payload.get("resolution_type"),
                })
        return related[:4]
    except Exception as e:
        logger.warning(f"Related ticket fetch failed: {e}")
        return []
