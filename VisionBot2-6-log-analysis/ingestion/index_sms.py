"""
Chunk → Embed → Index pipeline for SMS/Messaging tickets.

Reads:
  ingestion/sms_tickets.json            (filtered SMS tickets)
  ingestion/summaries_cache_sms.jsonl   (GPT-4o summaries)

Writes to Qdrant (local file mode):
  ticket_problems    — problem + investigation chunks
  ticket_resolutions — resolution chunks + cluster assignments

Safe to re-run — Qdrant upsert is idempotent (point IDs are deterministic UUIDs).

Usage:
  # Stop uvicorn first, then:
  .venv/bin/python -m ingestion.index_sms
  # Restart uvicorn after.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.chunker import chunk_tickets
from ingestion.embedder import Embedder
from ingestion.indexer import init_collections, upsert_chunks, assign_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
TICKETS_FILE = INGESTION_DIR / "sms_tickets.json"
SUMMARIES_FILE = INGESTION_DIR / "summaries_cache_sms.jsonl"


def load_summaries() -> dict:
    summaries = {}
    with open(SUMMARIES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                summaries[str(entry["ticket_id"])] = entry
    logger.info(f"Loaded {len(summaries):,} summaries from {SUMMARIES_FILE.name}")
    return summaries


async def main():
    if not TICKETS_FILE.exists():
        logger.error(f"{TICKETS_FILE.name} not found. Run: .venv/bin/python -m ingestion.filter_tickets --product sms")
        return
    if not SUMMARIES_FILE.exists():
        logger.error(f"{SUMMARIES_FILE.name} not found. Run: .venv/bin/python -m ingestion.summarize_sms")
        return

    logger.info(f"Loading tickets from {TICKETS_FILE.name}...")
    with open(TICKETS_FILE) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} SMS tickets")

    summaries = load_summaries()

    # Only index tickets that have been summarized
    summarized_ids = set(summaries.keys())
    tickets = [t for t in tickets if str(t["id"]) in summarized_ids]
    logger.info(f"{len(tickets):,} tickets have summaries — indexing these")

    logger.info("Chunking tickets (problem / investigation / resolution)...")
    chunks = chunk_tickets(tickets, summaries, zendesk_subdomain=settings.zendesk_subdomain)
    logger.info(f"Created {len(chunks):,} chunks from {len(tickets):,} tickets")

    logger.info(f"Embedding with {settings.embedding_model} (dense) + BM25 (sparse)...")
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    embedded_chunks = await embedder.embed_chunks(chunks)
    logger.info(f"Embedded {len(embedded_chunks):,} chunks")

    logger.info("Upserting to Qdrant...")
    qdrant = QdrantClient(path=settings.qdrant_path)
    init_collections(qdrant, dim=settings.embedding_dim)
    upsert_chunks(qdrant, embedded_chunks)
    logger.info("Upsert complete")

    logger.info("Assigning root-cause clusters (cosine > 0.85)...")
    assign_clusters(qdrant, embedded_chunks)

    problem_chunks = [c for c in chunks if c["chunk_type"] in ("problem", "investigation")]
    resolution_chunks = [c for c in chunks if c["chunk_type"] == "resolution"]
    logger.info(
        f"Done. {len(problem_chunks):,} problem/investigation chunks → ticket_problems | "
        f"{len(resolution_chunks):,} resolution chunks → ticket_resolutions"
    )


if __name__ == "__main__":
    asyncio.run(main())
