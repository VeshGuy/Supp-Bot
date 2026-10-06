"""
Fetch → Chunk → Embed → Index pipeline for Plivo Support Help Center.

Source: https://support.plivo.com/hc/en-us (Zendesk Help Center)
Uses the public Zendesk Help Center REST API — no auth required.
Indexes all articles across all categories into the existing 'support_docs' collection.

Payload schema (source="support"):
  source        : "support"
  article_id    : str  (Zendesk article ID)
  chunk_id      : str  (support_{article_id}_{chunk_index})
  page_title    : str  (article title)
  section       : str  (section name, e.g. "SDK")
  category      : str  (category name, e.g. "Voice API")
  url           : str  (full article URL)
  updated_at    : str  (ISO date)
  text          : str  (clean plain text)

Safe to re-run — UUIDs are deterministic from chunk_id.
Deletes existing source="support" points before upserting (clean re-index).

Usage:
  # Stop uvicorn first (Qdrant single-process lock), then:
  .venv/bin/python -m ingestion.index_support_docs
  # Restart uvicorn after
"""

import asyncio
import html
import json
import logging
import re
import sys
import uuid
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchValue,
    PointStruct, SparseVector,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://support.plivo.com/api/v2/help_center/en-us"
COLLECTION = "support_docs"
SOURCE = "support"
MAX_CHUNK_CHARS = 4000   # ~1k tokens — split articles longer than this
CHUNK_OVERLAP_CHARS = 200


# ── HTML → plain text ─────────────────────────────────────────────────────────

def _html_to_text(raw: str) -> str:
    """Strip HTML tags and decode entities to get clean plain text."""
    # Remove <style> and <script> blocks entirely
    raw = re.sub(r'<(style|script)[^>]*>.*?</\1>', ' ', raw, flags=re.DOTALL | re.IGNORECASE)
    # Replace block elements with newlines
    raw = re.sub(r'<(br|p|div|li|h[1-6]|tr)[^>]*>', '\n', raw, flags=re.IGNORECASE)
    # Strip all remaining tags
    raw = re.sub(r'<[^>]+>', ' ', raw)
    # Decode HTML entities (&amp; &lt; etc.)
    raw = html.unescape(raw)
    # Collapse whitespace
    raw = re.sub(r'[ \t]+', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Split long article text into chunks of MAX_CHUNK_CHARS.
    Splits at paragraph boundaries where possible.
    Short articles return a single chunk.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    paragraphs = text.split('\n\n')
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            chunks.append('\n\n'.join(current))
            # Overlap: carry last paragraph into next chunk
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2  # +2 for '\n\n'

    if current:
        chunks.append('\n\n'.join(current))

    return chunks if chunks else [text[:MAX_CHUNK_CHARS]]


# ── Zendesk Help Center API ───────────────────────────────────────────────────

async def _paginate(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Paginate through a Zendesk Help Center list endpoint."""
    results = []
    next_url = f"{url}?per_page=100"
    while next_url:
        resp = await client.get(next_url)
        resp.raise_for_status()
        data = resp.json()
        # The key is the last segment of the URL path (articles, sections, categories)
        key = url.rstrip('/').split('/')[-1]
        results.extend(data.get(key, []))
        next_url = data.get("next_page")
        if next_url:
            logger.debug(f"Fetching next page: {next_url}")
    return results


async def fetch_all() -> tuple[dict, dict, list[dict]]:
    """
    Returns:
      category_map: {category_id: category_name}
      section_map:  {section_id: {name, category_name}}
      articles:     list of raw article dicts
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        logger.info("Fetching categories...")
        categories = await _paginate(client, f"{BASE_URL}/categories")
        category_map = {c["id"]: c["name"] for c in categories}
        logger.info(f"  {len(category_map)} categories")

        logger.info("Fetching sections...")
        sections = await _paginate(client, f"{BASE_URL}/sections")
        section_map = {
            s["id"]: {
                "name": s["name"],
                "category_name": category_map.get(s["category_id"], ""),
            }
            for s in sections
        }
        logger.info(f"  {len(section_map)} sections")

        logger.info("Fetching articles...")
        articles = await _paginate(client, f"{BASE_URL}/articles")
        # Filter to English articles only
        articles = [a for a in articles if a.get("locale", "").startswith("en")]
        logger.info(f"  {len(articles)} articles (English)")

    return category_map, section_map, articles


# ── Chunking pipeline ─────────────────────────────────────────────────────────

def build_chunks(articles: list[dict], section_map: dict) -> list[dict]:
    chunks = []
    skipped = 0

    for article in articles:
        body_html = article.get("body") or ""
        text = _html_to_text(body_html).strip()
        if not text:
            skipped += 1
            continue

        section_id = article.get("section_id")
        section_info = section_map.get(section_id, {})
        section_name = section_info.get("name", "")
        category_name = section_info.get("category_name", "")

        article_id = str(article["id"])
        title = article.get("title", "").strip()
        url = article.get("html_url", "")
        updated_at = article.get("updated_at", "")

        text_chunks = _chunk_text(text)
        for i, chunk_text in enumerate(text_chunks):
            # Prepend title to every chunk so embedding captures topic
            full_text = f"{title}\n\n{chunk_text}" if i == 0 else f"[{title} — continued]\n\n{chunk_text}"
            chunks.append({
                "source": SOURCE,
                "article_id": article_id,
                "chunk_id": f"support_{article_id}_{i}",
                "page_title": title,
                "section": section_name,
                "category": category_name,
                "url": url,
                "updated_at": updated_at,
                "text": full_text,
            })

    logger.info(f"Built {len(chunks):,} chunks from {len(articles) - skipped:,} articles "
                f"({skipped} skipped — empty body)")
    return chunks


# ── Qdrant ────────────────────────────────────────────────────────────────────

def delete_existing(client: QdrantClient):
    """Delete all existing source='support' points before re-indexing."""
    result = client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=SOURCE))]
        ),
    )
    logger.info(f"Deleted existing source='{SOURCE}' points (result: {result})")


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
        if (i // batch_size) % 10 == 0:
            logger.info(f"  Upserted {min(i + batch_size, total):,}/{total:,}")
    logger.info(f"Upsert complete — {total:,} points added to '{COLLECTION}'")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=== Plivo Support Docs Indexer ===")

    # 1. Fetch from Zendesk Help Center API
    category_map, section_map, articles = await fetch_all()

    # 2. Build chunks
    chunks = build_chunks(articles, section_map)
    if not chunks:
        logger.error("No chunks built — aborting")
        return

    # 3. Embed (dense + BM25 sparse)
    logger.info(f"Embedding {len(chunks):,} chunks...")
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    embedded = await embedder.embed_chunks(chunks)
    logger.info("Embedding complete")

    # 4. Upsert to Qdrant
    logger.info("Connecting to Qdrant...")
    qdrant = QdrantClient(path=settings.qdrant_path)

    logger.info(f"Deleting existing source='{SOURCE}' points...")
    delete_existing(qdrant)

    logger.info("Upserting new chunks...")
    upsert_chunks(qdrant, embedded)

    # 5. Summary by category
    from collections import Counter
    cat_counts = Counter(c["category"] for c in chunks)
    logger.info("Chunks by category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count:,}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
