"""
OpenAI API cost tracking.

Two log files:
  cost_logs.jsonl          — per-query breakdown (written after answer is streamed)
  pipeline_cost_logs.jsonl — per ingestion run breakdown (written at end of script)

One running total:
  monthly_cost_summary.json — cumulative query + pipeline costs per calendar month

All file writes happen AFTER the main operation (zero latency impact on queries).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── OpenAI pricing per token (March 2026) ─────────────────────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":                 {"input": 2.50  / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":            {"input": 0.15  / 1_000_000, "output": 0.60  / 1_000_000},
    "text-embedding-3-large": {"input": 0.13  / 1_000_000, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02  / 1_000_000, "output": 0.0},
}

QUERY_COST_LOG    = Path("logs/cost_logs.jsonl")
PIPELINE_COST_LOG = Path("logs/pipeline_cost_logs.jsonl")
MONTHLY_SUMMARY   = Path("logs/monthly_cost_summary.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Return USD cost for a single API call."""
    p = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    return round(input_tokens * p["input"] + output_tokens * p["output"], 8)


def make_entry(step: str, model: str, input_tokens: int, output_tokens: int = 0) -> dict:
    """Build a cost entry dict for one LLM/embedding call."""
    return {
        "step": step,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": compute_cost(model, input_tokens, output_tokens),
    }


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate for embedding calls where the API doesn't return usage.
    Rule of thumb: ~1 token per 4 characters (OpenAI BPE average for English).
    """
    return max(1, len(text) // 4)


def _update_monthly_summary(cost_usd: float, category: str):
    """Add cost to the monthly running total. category = 'query' | 'pipeline'."""
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    summary: dict = {}

    if MONTHLY_SUMMARY.exists():
        try:
            with open(MONTHLY_SUMMARY) as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    if month not in summary:
        summary[month] = {
            "query_cost_usd": 0.0,
            "pipeline_cost_usd": 0.0,
            "total_usd": 0.0,
            "query_count": 0,
            "last_updated": "",
        }

    key = f"{category}_cost_usd"
    summary[month][key] = round(summary[month].get(key, 0.0) + cost_usd, 6)
    if category == "query":
        summary[month]["query_count"] += 1
    summary[month]["total_usd"] = round(
        summary[month]["query_cost_usd"] + summary[month]["pipeline_cost_usd"], 6
    )
    summary[month]["last_updated"] = datetime.now(timezone.utc).isoformat()

    with open(MONTHLY_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)


# ── Query cost logger (called from server.py after streaming) ─────────────────

def log_query_cost(
    session_id: str,
    question: str,
    query_type: str,
    cost_entries: list[dict],
):
    """
    Write per-query cost breakdown to cost_logs.jsonl.
    Must be called AFTER the SSE stream is complete — not during retrieval.
    """
    if not cost_entries:
        return

    total = round(sum(e["cost_usd"] for e in cost_entries), 8)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question[:120],
        "query_type": query_type,
        "breakdown": cost_entries,
        "total_usd": total,
    }

    with open(QUERY_COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    _update_monthly_summary(total, "query")

    steps = " + ".join(
        f"{e['step']}(${e['cost_usd']:.6f})" for e in cost_entries
    )
    logger.info(f"[cost] query total=${total:.6f} | {steps}")


# ── Pipeline cost logger (called from ingestion scripts) ──────────────────────

def log_pipeline_cost(
    script: str,
    reason: str,
    cost_entries: list[dict],
):
    """
    Write pipeline run cost to pipeline_cost_logs.jsonl.
    Call at the end of each ingestion script after all processing is done.

    Args:
        script: Script name e.g. 'summarize_voice.py'
        reason: Brief note on why this run happened e.g. 'Initial voice ticket summarization'
        cost_entries: List of cost entry dicts from make_entry()
    """
    if not cost_entries:
        return

    total = round(sum(e["cost_usd"] for e in cost_entries), 4)

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "reason": reason,
        "breakdown": cost_entries,
        "total_usd": total,
        "summary": _pipeline_summary(cost_entries, total),
    }

    with open(PIPELINE_COST_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    _update_monthly_summary(total, "pipeline")
    logger.info(f"[cost] pipeline={script} total=${total:.4f} | reason: {reason}")


def _pipeline_summary(entries: list[dict], total: float) -> str:
    """Generate a human-readable explanation of pipeline cost breakdown."""
    lines = []
    for e in entries:
        pct = (e["cost_usd"] / total * 100) if total > 0 else 0
        lines.append(
            f"{e['step']}: {e['input_tokens']:,} tokens @ {e['model']} "
            f"= ${e['cost_usd']:.4f} ({pct:.0f}% of total)"
        )
    lines.append(f"Total: ${total:.4f}")
    return " | ".join(lines)
