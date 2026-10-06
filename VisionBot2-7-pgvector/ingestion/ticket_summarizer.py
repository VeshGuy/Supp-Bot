"""
Use GPT-4o to extract structured fields from each ticket's conversation thread.
Results are cached to summaries_cache.jsonl to avoid re-processing on incremental runs.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

CACHE_FILE = Path("summaries_cache.jsonl")
BATCH_SIZE = 50  # Concurrent GPT-4o-mini calls


def get_cache_file(year: int = None) -> Path:
    """Return year-specific cache file path for parallel summarization."""
    if year:
        return Path(f"summaries_cache_{year}.jsonl")
    return CACHE_FILE


def load_all_caches() -> dict[str, dict]:
    """Merge all summaries_cache*.jsonl files into one dict keyed by ticket_id."""
    cache = {}
    cache_files = list(Path(".").glob("summaries_cache*.jsonl"))
    for path in sorted(cache_files):
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        cache[str(entry["ticket_id"])] = entry
    logger.info(f"Loaded {len(cache)} cached summaries from {len(cache_files)} cache file(s)")
    return cache

_SCHEMA = """{
  "problem_description": "...",
  "root_cause": "...",
  "resolution_summary": "...",
  "suggested_action": "...",
  "diagnostic_steps": ["step 1 the agent took", "step 2", "..."],
  "issue_type": "one of: dtmf, sms, voice_call, sip, api, authentication, latency, number_porting, compliance_activation, number_config, other",
  "product": "one of: voice_api, messaging_api, sip_trunking, phone_numbers, plivo_cx, other",
  "region": "one of: US, EMEA, APAC, LATAM, unknown",
  "resolution_type": "one of: carrier_escalation, config_fix, code_fix, user_error, known_issue, other"
}"""

SYSTEM_PROMPT_SIMPLE = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- Be concise. One sentence per text field is sufficient for simple tickets.
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""

SYSTEM_PROMPT_MEDIUM = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- This ticket has moderate complexity. Use 2-3 sentences for problem_description,
  root_cause, and resolution_summary to capture key details.
- diagnostic_steps: list every distinct investigative action the agent took (up to 8 steps).
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""

SYSTEM_PROMPT_COMPLEX = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- This ticket is COMPLEX (many comments, carrier escalations, or multi-step investigation).
  Write thorough summaries — do NOT compress important diagnostic details.
- problem_description: capture exactly what the customer reported, including error messages,
  timestamps, affected numbers, and any patterns.
- root_cause: explain the full chain of causation, not just the surface symptom.
- resolution_summary: describe every step taken to resolve, including all carrier interactions,
  configuration changes, escalations, and final outcome.
- diagnostic_steps: list EVERY distinct action taken with no limit on steps.
- suggested_action: concrete playbook for the next agent who sees this exact issue.
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""

# Keep the original name as an alias so existing callers that reference
# SYSTEM_PROMPT directly don't break.
SYSTEM_PROMPT = SYSTEM_PROMPT_SIMPLE


def _classify_complexity(ticket: dict) -> str:
    """Return 'simple', 'medium', or 'complex' based on thread length."""
    comments = ticket.get("comments") or []
    n = len(comments)
    total_chars = sum(len((c.get("plain_body") or "")) for c in comments)
    if n <= 3 and total_chars < 1500:
        return "simple"
    elif n <= 8 and total_chars < 5000:
        return "medium"
    else:
        return "complex"


_PROMPT_MAP = {
    "simple": SYSTEM_PROMPT_SIMPLE,
    "medium": SYSTEM_PROMPT_MEDIUM,
    "complex": SYSTEM_PROMPT_COMPLEX,
}


def _build_ticket_text(ticket: dict) -> str:
    """
    Build the full conversation text.
    Correctly attributes each comment using author_id vs requester_id.
    Falls back to position-based attribution (index 0 = customer) when IDs are unavailable.
    Internal notes (public=False) are labelled as such.
    """
    parts = [f"Subject: {ticket.get('subject', 'N/A')}"]
    requester_id = ticket.get("requester_id")

    for i, comment in enumerate(ticket.get("comments") or []):
        body = (comment.get("plain_body") or comment.get("body") or "").strip()
        if not body:
            continue

        is_public = comment.get("public", True)
        author_id = comment.get("author_id")

        if author_id and requester_id and author_id == requester_id:
            role = "Customer"
        elif i == 0:
            role = "Customer"  # first comment is always the requester
        elif not is_public:
            role = "Agent (internal note)"
        else:
            role = "Agent"

        ts = (comment.get("created_at") or "")[:10]
        label = f"[{role} | {ts}]" if ts else f"[{role}]"
        parts.append(f"\n{label}\n{body}")

    return "\n".join(parts)


def _load_cache(cache_file: Path = None) -> dict[str, dict]:
    """Load existing summaries cache keyed by ticket_id."""
    path = cache_file or CACHE_FILE
    cache = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[str(entry["ticket_id"])] = entry
    logger.info(f"Loaded {len(cache)} cached summaries from {path.name}")
    return cache


def _append_cache(entry: dict, cache_file: Path = None):
    """Append a single summary to cache file."""
    path = cache_file or CACHE_FILE
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


class TicketSummarizer:
    def __init__(self, api_key: str, model: str = "gpt-4o", year: int = None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cache_file = get_cache_file(year)
        self.cache = _load_cache(self.cache_file)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def _summarize_one(self, ticket: dict) -> dict:
        """Summarize a single ticket using the appropriate prompt for its complexity."""
        ticket_id = str(ticket["id"])

        if ticket_id in self.cache:
            return self.cache[ticket_id]

        complexity = _classify_complexity(ticket)
        system_prompt = _PROMPT_MAP[complexity]
        text = _build_ticket_text(ticket)

        summary = None
        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this ticket:\n\n{text}"},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                summary = json.loads(response.choices[0].message.content)
                if response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"Summarization failed for ticket {ticket_id}: {e}")
                    break

        if summary is None:
            summary = {
                "problem_description": ticket.get("subject"),
                "root_cause": None,
                "resolution_summary": None,
                "suggested_action": None,
                "diagnostic_steps": [],
                "issue_type": "other",
                "product": "other",
                "region": "unknown",
                "resolution_type": "other",
            }

        entry = {"ticket_id": ticket_id, **summary}
        _append_cache(entry, self.cache_file)
        self.cache[ticket_id] = entry
        return entry

    async def summarize_batch(self, tickets: list[dict]) -> list[dict]:
        """Summarize a list of tickets with concurrency control."""
        results = []
        semaphore = asyncio.Semaphore(BATCH_SIZE)

        async def bounded(ticket):
            async with semaphore:
                return await self._summarize_one(ticket)

        tasks = [bounded(t) for t in tickets]
        results = await asyncio.gather(*tasks)
        return list(results)

    def summarize_all(self, tickets: list[dict]) -> list[dict]:
        """Synchronous entry point."""
        return asyncio.run(self.summarize_batch(tickets))
