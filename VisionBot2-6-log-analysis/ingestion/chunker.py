"""
Conversation-aware chunker.
Produces 3 chunks per ticket: problem, investigation, resolution.
Resolution chunks get a boost flag used at retrieval time.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _comment_text(comment: dict) -> str:
    return (comment.get("plain_body") or comment.get("body") or "").strip()


def _is_agent_comment(comment: dict, ticket: dict) -> bool:
    """Heuristic: first comment is always customer; rest alternate or are by agent."""
    return comment.get("author_id") != ticket.get("requester_id")


def chunk_ticket(ticket: dict, summary: dict, zendesk_subdomain: str = "") -> list[dict]:
    """
    Split a ticket into 3 semantic chunks.
    Each chunk carries all metadata needed for retrieval + display.
    """
    ticket_id = str(ticket["id"])
    comments = ticket.get("comments", [])

    base_metadata = {
        "ticket_id": ticket_id,
        "subject": ticket.get("subject", ""),
        "product": summary.get("product") or "other",
        "issue_type": summary.get("issue_type") or "other",
        "region": summary.get("region") or "unknown",
        "resolution_type": summary.get("resolution_type") or "other",
        "csat_score": ticket.get("satisfaction_rating", {}).get("score") if ticket.get("satisfaction_rating") else None,
        "created_at": ticket.get("created_at"),
        "solved_at": ticket.get("updated_at"),
        "zendesk_url": f"https://{zendesk_subdomain}.zendesk.com/agent/tickets/{ticket_id}" if zendesk_subdomain else f"tickets/{ticket_id}",
        "cluster_id": None,  # Filled in by indexer after clustering
    }

    chunks = []

    # ── chunk_0: PROBLEM ────────────────────────────────────────────────────
    problem_parts = [f"Subject: {ticket.get('subject', '')}"]
    if summary.get("problem_description"):
        problem_parts.append(f"Problem: {summary['problem_description']}")

    # Add first 2 customer comments
    customer_comments = [c for c in comments if not _is_agent_comment(c, ticket)]
    for c in customer_comments[:2]:
        text = _comment_text(c)
        if text:
            problem_parts.append(text[:600])

    chunks.append({
        **base_metadata,
        "chunk_id": f"{ticket_id}_0",
        "chunk_type": "problem",
        "text": "\n\n".join(problem_parts),
        "retrieval_boost": 1.0,
    })

    # ── chunk_1: INVESTIGATION ───────────────────────────────────────────────
    investigation_parts = []
    if summary.get("diagnostic_steps"):
        steps = summary["diagnostic_steps"]
        investigation_parts.append("Diagnostic steps:\n" + "\n".join(f"- {s}" for s in steps))

    # Add middle comments (agent exchanges)
    middle_comments = comments[1:-1] if len(comments) > 2 else []
    for c in middle_comments[:4]:
        text = _comment_text(c)
        if text:
            investigation_parts.append(text[:400])

    if investigation_parts:
        chunks.append({
            **base_metadata,
            "chunk_id": f"{ticket_id}_1",
            "chunk_type": "investigation",
            "text": "\n\n".join(investigation_parts),
            "retrieval_boost": 1.0,
        })

    # ── chunk_2: RESOLUTION ──────────────────────────────────────────────────
    resolution_parts = []
    if summary.get("root_cause"):
        resolution_parts.append(f"Root cause: {summary['root_cause']}")
    if summary.get("resolution_summary"):
        resolution_parts.append(f"Resolution: {summary['resolution_summary']}")
    if summary.get("suggested_action"):
        resolution_parts.append(f"Suggested action: {summary['suggested_action']}")

    # Add last comment (usually the resolution/closure note)
    if comments:
        last = _comment_text(comments[-1])
        if last:
            resolution_parts.append(last[:600])

    chunks.append({
        **base_metadata,
        "chunk_id": f"{ticket_id}_2",
        "chunk_type": "resolution",
        "text": "\n\n".join(resolution_parts),
        "retrieval_boost": 1.3,  # Resolution chunks ranked higher at retrieval
    })

    return [c for c in chunks if c["text"].strip()]


def chunk_tickets(tickets: list[dict], summaries: dict[str, dict], zendesk_subdomain: str = "") -> list[dict]:
    """Chunk all tickets. summaries keyed by ticket_id string."""
    all_chunks = []
    for ticket in tickets:
        tid = str(ticket["id"])
        summary = summaries.get(tid, {})
        chunks = chunk_ticket(ticket, summary, zendesk_subdomain=zendesk_subdomain)
        all_chunks.extend(chunks)
    logger.info(f"Created {len(all_chunks)} chunks from {len(tickets)} tickets")
    return all_chunks
