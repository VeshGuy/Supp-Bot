"""
Parse Confluence PSA space export (CSV format) and index all pages into Qdrant support_docs.
Previously voice-only filtered — now indexes all Plivo products (Voice, SMS, Numbers, Billing, etc).
"""

import asyncio
import csv
import logging
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, SparseIndexParams

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).parent / "Confluence-export-plivo-team.atlassian.net-PSA"
COLLECTION = "support_docs"


class ConfluenceTextExtractor(HTMLParser):
    """Strip Confluence storage format XML/HTML tags to plain text."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_tags = {"ac:parameter", "ri:attachment"}
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip += 1
        if tag in ("tr", "li"):
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")
        if tag in ("h1", "h2", "h3", "h4", "p"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            text = data.strip()
            if text:
                self.parts.append(text)

    def get_text(self):
        text = " ".join(self.parts)
        # Collapse whitespace
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def extract_text(html: str) -> str:
    parser = ConfluenceTextExtractor()
    parser.feed(html)
    return parser.get_text()


def load_export():
    """Load content + bodycontent CSVs from the export directory."""
    with open(EXPORT_DIR / "content.csv.gz") as f:
        pages = {
            r["contentid"]: r
            for r in csv.DictReader(f)
            if r.get("contenttype") == "PAGE" and r.get("content_status") == "current"
        }

    with open(EXPORT_DIR / "bodycontent.csv.gz") as f:
        bodies = {r["contentid"]: r["body"] for r in csv.DictReader(f)}

    return pages, bodies



def chunk_page(page: dict, text: str) -> list[dict]:
    """Split page into chunks of ~1000 words."""
    title = page.get("title", "")
    words = text.split()
    chunks = []
    size = 1000

    for i, start in enumerate(range(0, len(words), size)):
        chunk_text = " ".join(words[start:start + size])
        if not chunk_text.strip():
            continue
        chunks.append({
            "source": "confluence",
            "title": title,
            "chunk_index": i,
            "text": f"{title}\n\n{chunk_text}",
            "content_id": page["contentid"],
            "last_modified": page.get("lastmoddate", ""),
        })

    return chunks


async def main():
    logger.info("Loading Confluence export...")
    pages, bodies = load_export()
    logger.info(f"Loaded {len(pages)} current pages")

    # Filter + extract text
    chunks = []
    skipped = 0
    for cid, page in pages.items():
        body = bodies.get(cid, "")
        if not body:
            skipped += 1
            continue
        text = extract_text(body)
        if not text or len(text) < 100:
            skipped += 1
            continue
        chunks.extend(chunk_page(page, text))

    logger.info(f"Total chunks: {len(chunks)} (skipped {skipped} empty/short pages)")

    # Init embedder + Qdrant
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    qdrant = QdrantClient(path=settings.qdrant_path)

    # Ensure collection exists
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
        )
        logger.info(f"Created collection: {COLLECTION}")

    # Embed + upsert in batches
    batch_size = 50
    total = len(chunks)
    upserted = 0

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        embedded = await embedder.embed_chunks([{"text": t} for t in texts])

        points = []
        for j, (chunk, emb) in enumerate(zip(batch, embedded)):
            dense, sparse = emb["dense_vector"], emb["sparse_vector"]
            point_id = abs(hash(f"confluence_{chunk['content_id']}_{chunk['chunk_index']}")) % (2**31)
            points.append(PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": sparse},
                payload={k: v for k, v in chunk.items() if k != "text"} | {"text": chunk["text"][:2000]},
            ))

        qdrant.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)
        logger.info(f"Upserted {upserted}/{total} chunks")

    logger.info(f"Done. {upserted} Confluence chunks indexed into '{COLLECTION}'")


if __name__ == "__main__":
    asyncio.run(main())
