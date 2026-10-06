"""
Main ingestion pipeline.

Usage:
  python -m ingestion.run_pipeline --since 2025-01-01   # Fetch from a specific date
  python -m ingestion.run_pipeline --since 2021-01-01   # Another date range
  python -m ingestion.run_pipeline --incremental        # Re-index last 24h resolved tickets
  python -m ingestion.run_pipeline --full               # Full index from beginning

Each --since run writes to its own checkpoint file:
  raw_tickets_2025-01-01.jsonl, raw_tickets_2021-01-01.jsonl, etc.

All checkpoint files are merged automatically before summarization.
Safe to kill + resume — checkpoint saves progress as tickets are fetched.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm
from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from api.cost_tracker import estimate_tokens, log_pipeline_cost, make_entry
from ingestion.zendesk_extractor import ZendeskExtractor
from ingestion.ticket_summarizer import TicketSummarizer, load_all_caches
from ingestion.chunker import chunk_tickets
from ingestion.embedder import Embedder
from ingestion.indexer import init_collections, upsert_chunks, assign_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent


def _checkpoint_file(since_date: str = None) -> Path:
    """Return the checkpoint file path for a given date range."""
    if since_date:
        return INGESTION_DIR / f"raw_tickets_{since_date}.jsonl"
    return INGESTION_DIR / "raw_tickets_full.jsonl"


def _load_checkpoint(path: Path) -> list[dict]:
    """Load previously fetched tickets from a checkpoint file."""
    if not path.exists():
        return []
    tickets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(json.loads(line))
    logger.info(f"Loaded {len(tickets)} tickets from {path.name}")
    return tickets


def _load_all_checkpoints() -> list[dict]:
    """Merge all raw_tickets_*.jsonl files, deduplicating by ticket ID."""
    all_tickets = {}
    checkpoint_files = sorted(INGESTION_DIR.glob("raw_tickets_*.jsonl"))
    for path in checkpoint_files:
        for t in _load_checkpoint(path):
            all_tickets[str(t["id"])] = t
    logger.info(f"Merged {len(all_tickets)} unique tickets from {len(checkpoint_files)} checkpoint file(s): {[p.name for p in checkpoint_files]}")
    return list(all_tickets.values())


def _append_checkpoint(path: Path, ticket: dict):
    """Append a single ticket to a checkpoint file."""
    with open(path, "a") as f:
        f.write(json.dumps(ticket) + "\n")


def _parse_since(since_str: str) -> str:
    """Parse --since YYYY-MM-DD into ISO8601 string."""
    dt = datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.isoformat()


async def run(incremental: bool = False, since_date: str = None, fetch_only: bool = False, stop_after_summarize: bool = False, summarize_year: int = None):
    mode = "incremental" if incremental else (f"since {since_date}" if since_date else "full")
    pipeline_cost_entries: list[dict] = []
    logger.info(f"Starting ingestion pipeline — mode: {mode}")

    extractor = ZendeskExtractor(
        subdomain=settings.zendesk_subdomain,
        email=settings.zendesk_email,
        api_key=settings.zendesk_api_key,
    )

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    checkpoint_path = _checkpoint_file(since_date)
    existing = _load_checkpoint(checkpoint_path)

    if existing and not incremental:
        logger.info(f"Step 1: Resuming from checkpoint ({checkpoint_path.name}) — {len(existing)} tickets already fetched")
        # Still merge all checkpoints for summarization
        raw_tickets = _load_all_checkpoints()
    else:
        if incremental:
            updated_after = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        elif since_date:
            updated_after = _parse_since(since_date)
        else:
            updated_after = None

        # Clear only this run's checkpoint for a fresh fetch
        if not incremental and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Cleared {checkpoint_path.name} for fresh fetch")

        logger.info(f"Step 1: Fetching tickets from Zendesk (since {updated_after or 'beginning'})...")
        BATCH = 20
        batch = []
        total = 0
        for ticket in tqdm(extractor.fetch_solved_tickets(updated_after=updated_after), desc="Fetching"):
            batch.append(ticket)
            if len(batch) >= BATCH:
                enriched_batch = await extractor.fetch_tickets_with_comments_batch(batch)
                for enriched in enriched_batch:
                    _append_checkpoint(checkpoint_path, enriched)
                total += len(enriched_batch)
                batch = []
        if batch:
            enriched_batch = await extractor.fetch_tickets_with_comments_batch(batch)
            for enriched in enriched_batch:
                _append_checkpoint(checkpoint_path, enriched)
            total += len(enriched_batch)
        logger.info(f"Fetched {total} tickets with comments")

        # Merge all checkpoint files for downstream steps
        raw_tickets = _load_all_checkpoints()

    logger.info(f"Total tickets across all checkpoints: {len(raw_tickets)}")

    if not raw_tickets:
        logger.info("No tickets to process — pipeline complete")
        return

    if fetch_only:
        logger.info("--fetch-only flag set — stopping after fetch. Run without --fetch-only to continue.")
        return

    summarizer = TicketSummarizer(api_key=settings.openai_api_key, model=settings.llm_mini_model, year=summarize_year)

    # ── Step 2: Summarize ────────────────────────────────────────────────────
    if summarize_year:
        tickets_to_summarize = [t for t in raw_tickets if t.get("updated_at", "")[:4] == str(summarize_year)]
        logger.info(f"Step 2: Summarizing {len(tickets_to_summarize)} tickets for year {summarize_year}...")
    else:
        tickets_to_summarize = raw_tickets
        logger.info("Step 2: Summarizing tickets with GPT-4o-mini...")
    summaries_list = await summarizer.summarize_batch(tickets_to_summarize)
    summaries = {str(s["ticket_id"]): s for s in summaries_list}
    logger.info(f"Summarized {len(summaries)} tickets")

    if summarizer.total_input_tokens > 0:
        pipeline_cost_entries.append(make_entry(
            "summarize", settings.llm_mini_model,
            summarizer.total_input_tokens, summarizer.total_output_tokens,
        ))

    if stop_after_summarize or summarize_year:
        if pipeline_cost_entries:
            log_pipeline_cost(
                script="run_pipeline.py",
                reason=f"Summarization only — {mode} ({len(tickets_to_summarize)} tickets)",
                cost_entries=pipeline_cost_entries,
            )
        logger.info("Stopping after summarization. Run without --summarize-year / --stop-after-summarize to continue.")
        return

    # Merge all year-specific cache files for downstream steps
    summaries = load_all_caches()

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    logger.info("Step 3: Chunking conversations...")
    chunks = chunk_tickets(raw_tickets, summaries, zendesk_subdomain=settings.zendesk_subdomain)
    logger.info(f"Created {len(chunks)} chunks")

    # ── Step 4: Embed ────────────────────────────────────────────────────────
    logger.info("Step 4: Generating embeddings...")
    embed_token_estimate = sum(estimate_tokens(c.get("text", "")) for c in chunks)
    pipeline_cost_entries.append(make_entry("embed_chunks", settings.embedding_model, embed_token_estimate))
    embedded_chunks = await embedder.embed_chunks(chunks)

    # ── Step 5: Upsert to Qdrant ─────────────────────────────────────────────
    logger.info("Step 5: Upserting to Qdrant...")
    upsert_chunks(qdrant, embedded_chunks)

    # ── Step 6: Cluster ──────────────────────────────────────────────────────
    logger.info("Step 6: Assigning clusters...")
    assign_clusters(qdrant, embedded_chunks)

    if pipeline_cost_entries:
        log_pipeline_cost(
            script="run_pipeline.py",
            reason=f"Full ingestion — {mode} ({len(raw_tickets)} tickets, {len(chunks)} chunks)",
            cost_entries=pipeline_cost_entries,
        )

    extractor.close()
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    incremental = "--incremental" in sys.argv
    fetch_only = "--fetch-only" in sys.argv
    stop_after_summarize = "--stop-after-summarize" in sys.argv
    since_date = None
    summarize_year = None
    if "--since" in sys.argv:
        idx = sys.argv.index("--since")
        since_date = sys.argv[idx + 1]
    if "--summarize-year" in sys.argv:
        idx = sys.argv.index("--summarize-year")
        summarize_year = int(sys.argv[idx + 1])
    asyncio.run(run(incremental=incremental, since_date=since_date, fetch_only=fetch_only, stop_after_summarize=stop_after_summarize, summarize_year=summarize_year))
