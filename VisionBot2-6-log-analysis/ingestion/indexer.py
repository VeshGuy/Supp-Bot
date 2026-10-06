"""
Upsert embedded chunks to Qdrant (two collections) and assign cluster IDs
by grouping tickets with similar root causes.
"""

import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    UpdateStatus,
    VectorParams,
    VectorsConfig,
)

logger = logging.getLogger(__name__)

UPSERT_BATCH = 50
CLUSTER_THRESHOLD = 0.85  # cosine similarity threshold for root cause clustering


def init_collections(client: QdrantClient, dim: int = 1536):
    """Create Qdrant collections if they don't exist."""
    for collection_name in ("ticket_problems", "ticket_resolutions"):
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            logger.info(f"Collection '{collection_name}' already exists — skipping creation")
            continue

        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        # Create payload indexes for metadata filtering
        for field in ("product", "issue_type", "region", "cluster_id", "chunk_type"):
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword",
            )

        logger.info(f"Created collection '{collection_name}'")


def _chunk_to_point(chunk: dict) -> PointStruct:
    """Convert an embedded chunk dict to a Qdrant PointStruct."""
    payload = {k: v for k, v in chunk.items()
               if k not in ("dense_vector", "sparse_vector")}

    return PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
        vector={
            "dense": chunk["dense_vector"],
            "sparse": SparseVector(
                indices=chunk["sparse_vector"]["indices"],
                values=chunk["sparse_vector"]["values"],
            ),
        },
        payload=payload,
    )


def upsert_chunks(client: QdrantClient, chunks: list[dict]):
    """Upsert chunks to the correct collection based on chunk_type."""
    problem_chunks = [c for c in chunks if c["chunk_type"] in ("problem", "investigation")]
    resolution_chunks = [c for c in chunks if c["chunk_type"] == "resolution"]

    for collection, chunk_list in [
        ("ticket_problems", problem_chunks),
        ("ticket_resolutions", resolution_chunks),
    ]:
        for i in range(0, len(chunk_list), UPSERT_BATCH):
            batch = chunk_list[i : i + UPSERT_BATCH]
            points = [_chunk_to_point(c) for c in batch]
            client.upsert(collection_name=collection, points=points)
            logger.info(f"Upserted {min(i + UPSERT_BATCH, len(chunk_list))}/{len(chunk_list)} to '{collection}'")


def assign_clusters(client: QdrantClient, chunks: list[dict]):
    """
    Group resolution chunks by root cause similarity.
    Tickets with cosine similarity > CLUSTER_THRESHOLD share a cluster_id.
    Updates Qdrant payloads in place.
    """
    resolution_chunks = [c for c in chunks if c["chunk_type"] == "resolution"]
    if not resolution_chunks:
        return

    vectors = np.array([c["dense_vector"] for c in resolution_chunks])
    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms

    cluster_map: dict[int, str] = {}  # idx → cluster_id

    for i in range(len(resolution_chunks)):
        if i in cluster_map:
            continue
        cluster_id = str(uuid.uuid4())[:8]
        cluster_map[i] = cluster_id

        # Find all similar chunks
        similarities = vectors_norm @ vectors_norm[i]
        for j in range(i + 1, len(resolution_chunks)):
            if j not in cluster_map and similarities[j] >= CLUSTER_THRESHOLD:
                cluster_map[j] = cluster_id

    # Group points by cluster_id and batch-update Qdrant payloads
    # (one set_payload call per cluster instead of one per chunk)
    cluster_to_point_ids: dict[str, list[str]] = {}
    for idx, chunk in enumerate(resolution_chunks):
        cid = cluster_map.get(idx)
        chunk["cluster_id"] = cid
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        cluster_to_point_ids.setdefault(cid, []).append(point_id)

    for cid, point_ids in cluster_to_point_ids.items():
        client.set_payload(
            collection_name="ticket_resolutions",
            payload={"cluster_id": cid},
            points=point_ids,
        )

    n_clusters = len(set(cluster_map.values()))
    logger.info(f"Assigned {n_clusters} clusters across {len(resolution_chunks)} resolution chunks")
