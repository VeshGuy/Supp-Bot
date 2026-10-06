"""
Index Ryuk screenshots into the ryuk_docs Qdrant collection.

Ryuk gets its own dedicated collection (not support_docs) so it can occupy a
guaranteed Lane C in retrieval — always fetching 1 Ryuk result regardless of how
many tickets or product docs score higher. The cross-encoder reranker then decides
if the Ryuk navigation page is actually relevant to the query.

Reads ingestion/ryuk_captions.jsonl (written by reading all screenshots),
embeds each caption using text-embedding-3-large, and upserts to ryuk_docs.

Run once after adding new screenshots / captions:
    cd /Users/sarvesh.patil/Desktop/vision-resolve
    .venv/bin/python -m ingestion.index_ryuk

Cost estimate: 104 chunks × ~150 tokens avg ≈ 15,600 tokens ≈ $0.01 (text-embedding-3-large)
Time estimate: < 30 seconds
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CAPTIONS_FILE = Path(__file__).parent / "ryuk_captions.jsonl"
COLLECTION = "ryuk_docs"
BATCH_SIZE = 50


def load_captions() -> list[dict]:
    """Load all Ryuk captions from JSONL."""
    captions = []
    with open(CAPTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line))
    logger.info(f"Loaded {len(captions)} Ryuk captions from {CAPTIONS_FILE.name}")
    return captions


def build_embed_text(caption: dict) -> str:
    """
    Build the text to embed for a Ryuk caption.
    Combines nav_path + caption + keywords so keyword searches and
    semantic queries both hit relevant chunks.
    """
    keywords_str = ", ".join(caption.get("keywords", []))
    return (
        f"Navigation: {caption['nav_path']}\n\n"
        f"{caption['caption']}\n\n"
        f"Keywords: {keywords_str}"
    )


async def main():
    captions = load_captions()

    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    qdrant = QdrantClient(path=settings.qdrant_path)

    # Create ryuk_docs collection if it doesn't exist; drop+recreate for clean re-index
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        qdrant.delete_collection(COLLECTION)
        logger.info(f"Dropped existing '{COLLECTION}' for clean re-index")

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
    )
    # Payload indexes for filtering by section (useful for future targeted Ryuk searches)
    for field in ("section", "source"):
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema="keyword",
        )
    logger.info(f"Created collection '{COLLECTION}' with section + source indexes")

    # Build embed texts
    chunks = []
    for cap in captions:
        chunks.append({
            "text": build_embed_text(cap),
            "image_path": cap["image_path"],
            "nav_path": cap["nav_path"],
            "section": cap["section"],
            "caption": cap["caption"],
            "keywords": cap.get("keywords", []),
            "source": "ryuk",
        })

    total = len(chunks)
    upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        embedded = await embedder.embed_chunks([{"text": c["text"]} for c in batch])

        points = []
        for chunk, emb in zip(batch, embedded):
            dense = emb["dense_vector"]
            sparse = emb["sparse_vector"]
            # Stable point ID from image_path
            point_id = abs(hash(f"ryuk_{chunk['image_path']}")) % (2**31)
            points.append(PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "source": "ryuk",
                    "section": chunk["section"],
                    "nav_path": chunk["nav_path"],
                    "image_path": chunk["image_path"],
                    "caption": chunk["caption"],
                    "keywords": chunk["keywords"],
                    "text": chunk["text"],
                },
            ))

        qdrant.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)
        logger.info(f"Upserted {upserted}/{total} Ryuk chunks")

    logger.info(f"Done. {upserted} Ryuk navigation chunks indexed into '{COLLECTION}'")


if __name__ == "__main__":
    asyncio.run(main())
