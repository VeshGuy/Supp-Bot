"""
Chunk → Embed → Index pipeline for voice API tickets.

Reads:
  ingestion/voice_api_tickets.json        (15,342 raw tickets)
  ingestion/summaries_cache_voice.jsonl   (GPT-4o summaries)

Writes to Qdrant (local file mode):
  ticket_problems    — problem + investigation chunks
  ticket_resolutions — resolution chunks + cluster assignments

Safe to re-run — Qdrant upsert is idempotent (point IDs are deterministic UUIDs).

Usage:
  .venv/bin/python -m ingestion.index_voice
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
TICKETS_FILE = INGESTION_DIR / "voice_api_tickets.json"
SUMMARIES_FILE = INGESTION_DIR / "summaries_cache_voice.jsonl"


def load_summaries() -> dict[str, dict]:
    """Load voice summaries keyed by ticket_id string."""
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
    # ── Load ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading tickets from {TICKETS_FILE.name}...")
    with open(TICKETS_FILE) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} voice API tickets")

    summaries = load_summaries()

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    logger.info("Step 3: Chunking tickets (problem / investigation / resolution)...")
    chunks = chunk_tickets(tickets, summaries, zendesk_subdomain=settings.zendesk_subdomain)
    logger.info(f"Created {len(chunks):,} chunks from {len(tickets):,} tickets")

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    logger.info(f"Step 4: Embedding with {settings.embedding_model} (dense) + BM25 (sparse)...")
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    embedded_chunks = await embedder.embed_chunks(chunks)
    logger.info(f"Embedded {len(embedded_chunks):,} chunks")

    # ── Step 5: Upsert to Qdrant ──────────────────────────────────────────────
    logger.info("Step 5: Initialising Qdrant collections and upserting...")
    qdrant = QdrantClient(path=settings.qdrant_path)
    init_collections(qdrant, dim=settings.embedding_dim)
    upsert_chunks(qdrant, embedded_chunks)
    logger.info("Upsert complete")

    # ── Step 6: Cluster ───────────────────────────────────────────────────────
    logger.info("Step 6: Assigning root-cause clusters (cosine > 0.85)...")
    assign_clusters(qdrant, embedded_chunks)
    logger.info("Pipeline complete!")

    # ── Summary ───────────────────────────────────────────────────────────────
    problem_chunks = [c for c in chunks if c["chunk_type"] in ("problem", "investigation")]
    resolution_chunks = [c for c in chunks if c["chunk_type"] == "resolution"]
    logger.info(
        f"Indexed {len(problem_chunks):,} problem/investigation chunks → ticket_problems | "
        f"{len(resolution_chunks):,} resolution chunks → ticket_resolutions"
    )


if __name__ == "__main__":
    asyncio.run(main())
