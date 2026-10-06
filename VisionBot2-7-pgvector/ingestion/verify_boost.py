"""
Quick verification for boost_jira_escalated.py.

Checks:
1. How many points in ticket_problems + ticket_resolutions have retrieval_boost=1.5
2. Samples 5 boosted points and prints their ticket_id, subject, jira_project
3. Samples 5 non-boosted points to confirm they're untouched

Usage:
  .venv/bin/python -m ingestion.verify_boost
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QDRANT_PATH = "./qdrant_data"
COLLECTIONS = ["ticket_problems", "ticket_resolutions"]


def verify(collection: str) -> None:
    client = QdrantClient(path=QDRANT_PATH)

    # Count boosted points
    boosted = client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[FieldCondition(key="jira_escalated", match=MatchValue(value=True))]
        ),
        exact=True,
    )
    total = client.count(collection_name=collection, exact=True)
    logger.info(f"{collection}: {boosted.count:,} boosted / {total.count:,} total ({100*boosted.count//total.count}%)")

    # Sample 5 boosted points
    boosted_sample, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="jira_escalated", match=MatchValue(value=True))]
        ),
        limit=5,
        with_payload=True,
    )
    print(f"\n  Sample boosted points ({collection}):")
    for p in boosted_sample:
        pl = p.payload or {}
        print(f"    ticket_id={pl.get('ticket_id')} | boost={pl.get('retrieval_boost')} | "
              f"jira_project={pl.get('jira_project')} | subject={pl.get('subject', '')[:60]}")

    # Sample 5 non-boosted points to confirm untouched
    non_boosted_sample, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must_not=[FieldCondition(key="jira_escalated", match=MatchValue(value=True))]
        ),
        limit=5,
        with_payload=True,
    )
    print(f"\n  Sample non-boosted points ({collection}):")
    for p in non_boosted_sample:
        pl = p.payload or {}
        print(f"    ticket_id={pl.get('ticket_id')} | boost={pl.get('retrieval_boost')} | "
              f"jira_escalated={pl.get('jira_escalated')} | subject={pl.get('subject', '')[:60]}")
    print()


def main() -> None:
    for collection in COLLECTIONS:
        verify(collection)


if __name__ == "__main__":
    main()
