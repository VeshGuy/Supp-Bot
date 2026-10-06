"""
Standalone summarizer for voice_api_tickets.json using GPT-4o.

Reads ingestion/voice_api_tickets.json, summarizes with GPT-4o (full model for
better root_cause / resolution_summary quality), writes to summaries_cache_voice.jsonl.

Safe to kill + resume — already-processed tickets are skipped via cache.

Usage:
  .venv/bin/python -m ingestion.summarize_voice
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.ticket_summarizer import TicketSummarizer, BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
INPUT_FILE = INGESTION_DIR / "voice_api_tickets.json"
CACHE_FILE = INGESTION_DIR / "summaries_cache_voice.jsonl"


def _load_existing_cache() -> set[str]:
    """Return set of already-summarized ticket IDs."""
    done = set()
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done.add(str(entry["ticket_id"]))
    return done


async def main():
    logger.info(f"Loading tickets from {INPUT_FILE.name}...")
    with open(INPUT_FILE) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} voice API tickets")

    already_done = _load_existing_cache()
    if already_done:
        logger.info(f"Resuming — {len(already_done):,} already summarized, {len(tickets) - len(already_done):,} remaining")

    pending = [t for t in tickets if str(t["id"]) not in already_done]
    if not pending:
        logger.info("All tickets already summarized. Nothing to do.")
        return

    # Use GPT-4o (llm_model) — better root_cause/resolution_summary extraction
    summarizer = TicketSummarizer(
        api_key=settings.openai_api_key,
        model=settings.llm_model,  # gpt-4o
    )
    # Override cache file to voice-specific one
    summarizer.cache_file = CACHE_FILE
    summarizer.cache = {str(e["ticket_id"]): e for e in [
        json.loads(l) for l in open(CACHE_FILE).readlines() if l.strip()
    ]} if CACHE_FILE.exists() else {}

    logger.info(f"Summarizing {len(pending):,} tickets with {settings.llm_model} ({BATCH_SIZE} concurrent)...")

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    completed = 0

    async def bounded(ticket):
        nonlocal completed
        async with semaphore:
            result = await summarizer._summarize_one(ticket)
            completed += 1
            if completed % 100 == 0:
                logger.info(f"Progress: {completed:,}/{len(pending):,} ({completed/len(pending)*100:.1f}%)")
            return result

    tasks = [bounded(t) for t in pending]
    results = await asyncio.gather(*tasks)

    logger.info(f"Done. {len(results):,} tickets summarized → {CACHE_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
