"""
Chunk → Embed → Index pipeline for Jira issues.

Reads ingestion/jira_checkpoint.jsonl (22,513 resolved issues).
Chunks each issue into up to 2 chunks:
  - chunk_0: summary + description (the problem/context)
  - chunk_1: comments thread (the discussion/resolution)  ← only if comments exist

Upserts into the existing 'support_docs' Qdrant collection alongside Plivo docs.
Safe to re-run — Qdrant upsert is idempotent (deterministic UUIDs).

Usage:
  .venv/bin/python -m ingestion.index_jira
"""

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue,
    PointStruct, SparseVector, SparseVectorParams, VectorParams,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INGESTION_DIR = Path(__file__).parent
CHECKPOINT_FILE = INGESTION_DIR / "jira_checkpoint.jsonl"
COLLECTION = "support_docs"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_issues() -> list[dict]:
    issues = []
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    logger.info(f"Loaded {len(issues):,} Jira issues from {CHECKPOINT_FILE.name}")
    return issues


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_issue(issue: dict) -> list[dict]:
    """
    Split a Jira issue into up to 2 chunks.
    chunk_0: summary + description  (problem context)
    chunk_1: comment thread          (discussion + resolution)
    """
    base = {
        "source": "jira",
        "issue_id": issue["id"],
        "issue_key": issue["key"],
        "project": issue["project"],
        "issue_type": issue["issue_type"],
        "status": issue["status"],
        "resolution": issue.get("resolution", ""),
        "labels": issue.get("labels", []),
        "created": issue.get("created", ""),
        "updated": issue.get("updated", ""),
        "resolutiondate": issue.get("resolutiondate", ""),
        "url": issue.get("url", ""),
        "summary": issue["summary"],
    }

    chunks = []

    # chunk_0: summary + description
    desc = issue.get("description", "").strip()
    desc_text = f"{issue['summary']}\n\n{desc}" if desc else issue["summary"]
    if desc_text.strip():
        chunks.append({
            **base,
            "chunk_id": f"jira_{issue['id']}_0",
            "chunk_type": "description",
            "text": desc_text[:4000],  # Cap at ~1k tokens
        })

    # chunk_1: comments (concatenated, most recent last)
    comments = issue.get("comments", [])
    if comments:
        comment_parts = []
        for c in comments:
            author = c.get("author", "")
            text = c.get("text", "").strip()
            if text:
                comment_parts.append(f"[{author}]: {text}")
        comment_text = "\n\n".join(comment_parts)
        if comment_text.strip():
            chunks.append({
                **base,
                "chunk_id": f"jira_{issue['id']}_1",
                "chunk_type": "comments",
                "text": comment_text[:6000],  # Cap at ~1.5k tokens
            })

    return chunks


def chunk_all(issues: list[dict]) -> list[dict]:
    all_chunks = []
    for issue in issues:
        all_chunks.extend(chunk_issue(issue))
    logger.info(f"Created {len(all_chunks):,} chunks from {len(issues):,} issues")
    return all_chunks


# ── Qdrant ────────────────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient, dim: int):
    """Create support_docs collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        logger.info(f"Collection '{COLLECTION}' already exists — upserting into it")
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    for field in ("source", "project", "issue_type", "chunk_type"):
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema="keyword",
        )
    logger.info(f"Created collection '{COLLECTION}'")


def upsert_chunks(client: QdrantClient, chunks: list[dict], batch_size: int = 50):
    points = []
    for chunk in chunks:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        payload = {k: v for k, v in chunk.items() if k not in ("dense_vector", "sparse_vector")}
        points.append(PointStruct(
            id=point_id,
            vector={
                "dense": chunk["dense_vector"],
                "sparse": SparseVector(
                    indices=chunk["sparse_vector"]["indices"],
                    values=chunk["sparse_vector"]["values"],
                ),
            },
            payload=payload,
        ))

    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        if (i // batch_size) % 20 == 0:
            logger.info(f"Upserted {min(i + batch_size, total):,}/{total:,} to '{COLLECTION}'")
    logger.info(f"Upsert complete — {total:,} points in '{COLLECTION}'")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    issues = load_issues()

    logger.info("Chunking issues...")
    chunks = chunk_all(issues)

    logger.info(f"Embedding {len(chunks):,} chunks with {settings.embedding_model} + BM25...")
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    embedded = await embedder.embed_chunks(chunks)
    logger.info(f"Embedding complete")

    logger.info("Upserting to Qdrant...")
    qdrant = QdrantClient(path=settings.qdrant_path)
    ensure_collection(qdrant, dim=settings.embedding_dim)
    upsert_chunks(qdrant, embedded)

    desc_chunks = sum(1 for c in chunks if c["chunk_type"] == "description")
    comment_chunks = sum(1 for c in chunks if c["chunk_type"] == "comments")
    logger.info(
        f"Done. {desc_chunks:,} description chunks + {comment_chunks:,} comment chunks "
        f"→ '{COLLECTION}'"
    )


if __name__ == "__main__":
    asyncio.run(main())
