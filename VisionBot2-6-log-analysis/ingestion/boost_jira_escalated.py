"""
One-off script: boost retrieval_boost for jira_escalated tickets in Qdrant.

For every ticket in ticket_problems + ticket_resolutions that has the
jira_escalated tag in the raw files:
  - retrieval_boost: 1.0/1.3 → 1.5
  - jira_project: "VPT"|"VT"|"SUP"|"ZEN"|"CSDK"|null (extracted from comments)

Idempotent — safe to re-run. Only updates points whose ticket_id is in the
jira_escalated set.

Usage:
  # Stop uvicorn first (Qdrant single-process lock), then:
  .venv/bin/python -m ingestion.boost_jira_escalated
  # Restart uvicorn after.
"""

import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
RAW_FILES = [
    INGESTION_DIR / "raw_tickets_2022-01-01.jsonl",
    INGESTION_DIR / "raw_tickets_2025-01-01.jsonl",
]
QDRANT_PATH = "./qdrant_data"
COLLECTIONS = ["ticket_problems", "ticket_resolutions"]
BOOSTED_VALUE = 1.5

JIRA_RE = re.compile(r"\b(VT|VPT|ZEN|SUP|SP|CAL|CSDK)-\d+\b", re.IGNORECASE)

# chunk_id suffixes: 0=problem, 1=investigation, 2=resolution
CHUNK_SUFFIXES = ["_0", "_1", "_2"]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def build_jira_escalated_map() -> Dict[str, Optional[str]]:
    """
    Scan raw ticket files and return {ticket_id: jira_project} for all
    jira_escalated tickets. jira_project is the first Jira project found
    in comments (VPT/VT/SUP/etc.), or None if not found.
    """
    logger.info("Scanning raw ticket files for jira_escalated tickets...")
    escalated: dict[str, str | None] = {}

    for fpath in RAW_FILES:
        if not fpath.exists():
            logger.warning(f"File not found, skipping: {fpath}")
            continue
        logger.info(f"Reading {fpath.name}...")
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tags = [str(tg).lower() for tg in (t.get("tags") or [])]
                if "jira_escalated" not in tags:
                    continue
                ticket_id = str(t["id"])
                # Extract Jira project from comments
                jira_project = None
                for comment in t.get("comments", []):
                    body = comment.get("plain_body") or ""
                    match = JIRA_RE.search(body)
                    if match:
                        jira_project = match.group(1).upper()
                        break
                escalated[ticket_id] = jira_project

    logger.info(
        f"Found {len(escalated):,} jira_escalated tickets "
        f"({sum(1 for v in escalated.values() if v):,} with Jira project refs)"
    )
    return escalated


def update_qdrant(escalated: Dict[str, Optional[str]]) -> None:
    client = QdrantClient(path=QDRANT_PATH)

    for collection in COLLECTIONS:
        logger.info(f"Processing collection: {collection}")

        # Build point_id → (ticket_id, jira_project) map for all escalated tickets
        updates: dict[str, dict] = {}
        for ticket_id, jira_project in escalated.items():
            for suffix in CHUNK_SUFFIXES:
                chunk_id = ticket_id + suffix
                pid = _point_id(chunk_id)
                updates[pid] = {
                    "retrieval_boost": BOOSTED_VALUE,
                    "jira_escalated": True,
                    "jira_project": jira_project,
                }

        # Batch update in chunks of 500
        point_ids = list(updates.keys())
        batch_size = 500
        updated = 0

        for i in range(0, len(point_ids), batch_size):
            batch = point_ids[i : i + batch_size]

            # Fetch existing points to confirm they exist in this collection
            # (not all ticket_ids will have chunks in both collections)
            try:
                existing = client.retrieve(
                    collection_name=collection,
                    ids=batch,
                    with_payload=False,
                )
            except Exception as e:
                logger.error(f"Retrieve failed on batch {i}: {e}")
                continue

            if not existing:
                continue

            existing_ids = [str(p.id) for p in existing]

            # Group by payload value to minimise set_payload calls
            # (most will have the same jira_project — batch those together)
            payload_groups: dict[str, list[str]] = {}
            for pid in existing_ids:
                payload = updates[pid]
                key = f"{payload['retrieval_boost']}|{payload['jira_project']}"
                payload_groups.setdefault(key, []).append(pid)

            for key, pids in payload_groups.items():
                boost, proj = key.split("|", 1)
                client.set_payload(
                    collection_name=collection,
                    payload={
                        "retrieval_boost": float(boost),
                        "jira_escalated": True,
                        "jira_project": proj if proj != "None" else None,
                    },
                    points=pids,
                )
                updated += len(pids)

        logger.info(f"  Updated {updated:,} points in {collection}")


def main() -> None:
    # Safety check — confirm Qdrant data dir exists
    if not Path(QDRANT_PATH).exists():
        logger.error(f"Qdrant path not found: {QDRANT_PATH}. Are you in the project root?")
        sys.exit(1)

    escalated = build_jira_escalated_map()
    if not escalated:
        logger.error("No jira_escalated tickets found — check raw file paths.")
        sys.exit(1)

    update_qdrant(escalated)
    logger.info("Done.")


if __name__ == "__main__":
    main()
