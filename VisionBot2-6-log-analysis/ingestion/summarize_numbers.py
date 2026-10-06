"""
Adaptive summarizer for numbers_tickets.json using GPT-4.1-mini.

Key difference from summarize_sms/sip: summary depth scales with ticket complexity.
  - Simple  (≤3 comments, <1500 chars): one sentence per field
  - Medium  (4-8 comments, <5000 chars): 2-3 sentences per field
  - Complex (>8 comments OR >5000 chars): full narrative per field, all diagnostic steps

This matters for phone number tickets because:
  - Porting tickets: 10-20+ comments with carrier back-and-forth, LOA, FOC dates,
    rejection reasons — crushing to one sentence loses all diagnostic value.
  - Compliance tickets: regulatory rejection reasons, required documents, resubmission
    history — all essential context for agents handling similar tickets.

Run filter_tickets.py --product numbers first.

Usage:
  .venv/bin/python -m ingestion.summarize_numbers
  .venv/bin/python -m ingestion.summarize_numbers --limit 100   # test run
  .venv/bin/python -m ingestion.summarize_numbers --model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
INPUT_FILE = INGESTION_DIR / "numbers_tickets.json"
CACHE_FILE = INGESTION_DIR / "summaries_cache_numbers.jsonl"

BATCH_SIZE = 30  # conservative — phone tickets are longer


# ── Prompt templates ──────────────────────────────────────────────────────────
# Three tiers based on ticket complexity. All return the same JSON schema so
# indexing is uniform. The difference is how much detail the LLM is asked to capture.

_SCHEMA = """{
  "problem_description": "...",
  "root_cause": "...",
  "resolution_summary": "...",
  "suggested_action": "...",
  "diagnostic_steps": ["step 1", "step 2", "..."],
  "issue_type": "one of: number_porting, compliance_activation, number_purchase, number_release, cnam_caller_id, toll_free, short_code, number_config, other",
  "product": "one of: phone_numbers, messaging_api, voice_api, other",
  "region": "one of: US, EMEA, APAC, LATAM, unknown",
  "resolution_type": "one of: carrier_escalation, compliance_approval, config_fix, user_error, number_released, known_issue, other",
  "country": "ISO country code of the number in question, or null",
  "carrier": "carrier name involved if mentioned, or null",
  "porting_rejection_reason": "specific rejection reason if this is a porting ticket, or null"
}"""

SYSTEM_PROMPT_SIMPLE = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- Be concise. One sentence per text field is enough for simple tickets.
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""

SYSTEM_PROMPT_MEDIUM = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- This ticket has moderate complexity. Use 2-3 sentences for problem_description,
  root_cause, and resolution_summary to capture the key details.
- diagnostic_steps: list every distinct investigative action the agent took (up to 8 steps).
- Capture the country, carrier, and any porting rejection reasons explicitly.
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""

SYSTEM_PROMPT_COMPLEX = f"""You are a technical support analyst. Given a Zendesk ticket conversation,
extract structured information and return ONLY a JSON object with these exact fields:

{_SCHEMA}

Rules:
- This ticket is COMPLEX (many comments, carrier escalation, or regulatory back-and-forth).
  Write thorough summaries — do NOT compress important diagnostic details.
- problem_description: capture exactly what the customer reported including any error messages,
  carrier names, number formats, and timeline.
- root_cause: explain the full chain of causation, not just the surface symptom.
- resolution_summary: describe every step taken to resolve, including carrier interactions,
  documents submitted, approvals obtained, and final outcome.
- diagnostic_steps: list EVERY distinct action taken — carrier escalations, document requests,
  Ryuk checks, LOA submissions, FOC date negotiations, etc. No limit on steps.
- suggested_action: concrete playbook for the next agent who sees this exact issue.
- porting_rejection_reason: if this is a porting ticket, quote the exact rejection reason
  given by the carrier if mentioned.
- If a field cannot be determined, use null.
- Return ONLY the JSON object, no markdown, no explanation."""


# ── Complexity classifier ─────────────────────────────────────────────────────

def _classify_complexity(ticket: dict) -> str:
    """Return 'simple', 'medium', or 'complex' based on ticket thread length."""
    comments = ticket.get("comments") or []
    n_comments = len(comments)
    total_chars = sum(len((c.get("plain_body") or "")) for c in comments)

    if n_comments <= 3 and total_chars < 1500:
        return "simple"
    elif n_comments <= 8 and total_chars < 5000:
        return "medium"
    else:
        return "complex"


_PROMPT_MAP = {
    "simple": SYSTEM_PROMPT_SIMPLE,
    "medium": SYSTEM_PROMPT_MEDIUM,
    "complex": SYSTEM_PROMPT_COMPLEX,
}


# ── Ticket text builder ───────────────────────────────────────────────────────

def _build_ticket_text(ticket: dict) -> str:
    """
    Build the conversation thread text.
    Correctly attributes each comment: the first public comment is the customer;
    subsequent comments alternate. Public = visible to customer. Internal = agent note.
    Uses author_id vs requester_id to distinguish customer from agent messages.
    """
    parts = [f"Subject: {ticket.get('subject', 'N/A')}"]
    requester_id = ticket.get("requester_id")

    for comment in ticket.get("comments") or []:
        body = (comment.get("plain_body") or comment.get("body") or "").strip()
        if not body:
            continue

        is_public = comment.get("public", True)
        author_id = comment.get("author_id")

        if author_id and requester_id and author_id == requester_id:
            role = "Customer"
        elif not is_public:
            role = "Agent (internal note)"
        else:
            role = "Agent"

        ts = (comment.get("created_at") or "")[:10]  # YYYY-MM-DD only
        label = f"[{role} | {ts}]" if ts else f"[{role}]"
        parts.append(f"\n{label}\n{body}")

    return "\n".join(parts)


# ── Core summarizer ───────────────────────────────────────────────────────────

async def _summarize_one(
    ticket: dict,
    client: AsyncOpenAI,
    model: str,
    cache: dict,
    cache_file: Path,
) -> dict:
    ticket_id = str(ticket["id"])

    if ticket_id in cache:
        return cache[ticket_id]

    complexity = _classify_complexity(ticket)
    system_prompt = _PROMPT_MAP[complexity]
    text = _build_ticket_text(ticket)

    summary = None
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this ticket:\n\n{text}"},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            summary = json.loads(response.choices[0].message.content)
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1})")
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
            "product": "phone_numbers",
            "region": "unknown",
            "resolution_type": "other",
            "country": None,
            "carrier": None,
            "porting_rejection_reason": None,
        }

    entry = {"ticket_id": ticket_id, "complexity": complexity, **summary}
    with open(cache_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
    cache[ticket_id] = entry
    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max tickets to summarize this run (resume-safe)")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini",
                        help="Model to use (default: gpt-4.1-mini)")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        logger.error(
            f"{INPUT_FILE.name} not found. "
            "Run: .venv/bin/python -m ingestion.filter_tickets --product numbers"
        )
        return

    logger.info(f"Loading tickets from {INPUT_FILE.name}...")
    with open(INPUT_FILE) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} phone number tickets")

    # Load cache (resume support)
    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[str(entry["ticket_id"])] = entry
        logger.info(f"Resuming — {len(cache):,} already summarized")

    pending = [t for t in tickets if str(t["id"]) not in cache]
    if not pending:
        logger.info("All tickets already summarized. Nothing to do.")
        return

    if args.limit:
        pending = pending[:args.limit]
        logger.info(f"--limit {args.limit}: processing {len(pending):,} tickets this run")

    # Log complexity breakdown for the pending batch
    from collections import Counter
    complexity_counts = Counter(_classify_complexity(t) for t in pending)
    logger.info(
        f"Complexity breakdown: "
        f"simple={complexity_counts['simple']}, "
        f"medium={complexity_counts['medium']}, "
        f"complex={complexity_counts['complex']}"
    )

    model = args.model
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    completed = 0

    async def bounded(ticket):
        nonlocal completed
        async with semaphore:
            result = await _summarize_one(ticket, client, model, cache, CACHE_FILE)
            completed += 1
            if completed % 100 == 0:
                pct = completed / len(pending) * 100
                logger.info(f"Progress: {completed:,}/{len(pending):,} ({pct:.1f}%)")
            return result

    logger.info(f"Summarizing {len(pending):,} tickets with {model} ({BATCH_SIZE} concurrent)...")
    await asyncio.gather(*[bounded(t) for t in pending])
    logger.info(f"Done. {len(pending):,} tickets summarized → {CACHE_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
