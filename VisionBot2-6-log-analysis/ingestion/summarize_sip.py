"""
Standalone summarizer for sip_tickets.json using GPT-4o.

Run filter_tickets.py --product sip first to generate sip_tickets.json.

Safe to kill + resume — already-processed tickets are skipped via cache.

Usage:
  .venv/bin/python -m ingestion.summarize_sip
  .venv/bin/python -m ingestion.summarize_sip --limit 35   # test run
  .venv/bin/python -m ingestion.summarize_sip --model gpt-4.1-mini
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.ticket_summarizer import TicketSummarizer, BATCH_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
INPUT_FILE = INGESTION_DIR / "sip_tickets.json"
CACHE_FILE = INGESTION_DIR / "summaries_cache_sip.jsonl"


def _load_existing_cache() -> set[str]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max tickets to summarize this run (for test runs)")
    parser.add_argument("--model", type=str, default=None, help="Override model (e.g. gpt-4.1-mini)")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        logger.error(f"{INPUT_FILE.name} not found. Run: .venv/bin/python -m ingestion.filter_tickets --product sip")
        return

    logger.info(f"Loading tickets from {INPUT_FILE.name}...")
    with open(INPUT_FILE) as f:
        tickets = json.load(f)
    logger.info(f"Loaded {len(tickets):,} SIP Trunking tickets")

    already_done = _load_existing_cache()
    if already_done:
        logger.info(f"Resuming — {len(already_done):,} already summarized, {len(tickets) - len(already_done):,} remaining")

    pending = [t for t in tickets if str(t["id"]) not in already_done]
    if not pending:
        logger.info("All tickets already summarized. Nothing to do.")
        return

    if args.limit:
        pending = pending[:args.limit]
        logger.info(f"--limit {args.limit}: processing {len(pending):,} tickets this run")

    model = args.model or settings.llm_model
    summarizer = TicketSummarizer(
        api_key=settings.openai_api_key,
        model=model,
    )
    summarizer.cache_file = CACHE_FILE
    summarizer.cache = {str(e["ticket_id"]): e for e in [
        json.loads(l) for l in open(CACHE_FILE).readlines() if l.strip()
    ]} if CACHE_FILE.exists() else {}

    logger.info(f"Summarizing {len(pending):,} tickets with {model} ({BATCH_SIZE} concurrent)...")

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    completed = 0

    async def bounded(ticket):
        nonlocal completed
        async with semaphore:
            result = await summarizer._summarize_one(ticket)
            completed += 1
            if completed % 10 == 0:
                logger.info(f"Progress: {completed:,}/{len(pending):,} ({completed/len(pending)*100:.1f}%)")
            return result

    tasks = [bounded(t) for t in pending]
    results = await asyncio.gather(*tasks)

    logger.info(f"Done. {len(results):,} tickets summarized → {CACHE_FILE.name}")


if __name__ == "__main__":
    asyncio.run(main())
