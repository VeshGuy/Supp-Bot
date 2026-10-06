"""
Plivo docs scraper.

Fetches all /docs/* pages from the Plivo sitemap, extracts content,
chunks by heading sections, embeds, and upserts to the support_docs
Qdrant collection.

Usage:
  python -m ingestion.docs_scraper

Checkpoint: ingestion/docs_checkpoint.jsonl — safe to kill and resume.
"""

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.plivo.com"
DOCS_PREFIX = "/docs/"
CHECKPOINT_FILE = Path(__file__).parent / "docs_checkpoint.jsonl"
COLLECTION_NAME = "support_docs"
REQUEST_DELAY = 0.4  # seconds between requests — polite crawling

# Seed URLs — BFS spiders outward from these
SEED_URLS = [
    # Home
    "https://www.plivo.com/docs/home",
    # Voice
    "https://www.plivo.com/docs/voice/quickstart/quickstart",
    "https://www.plivo.com/docs/voice/concepts/overview",
    "https://www.plivo.com/docs/voice/xml/overview",
    "https://www.plivo.com/docs/voice/api/overview",
    # Messaging / SMS
    "https://www.plivo.com/docs/messaging/concepts/overview",
    "https://www.plivo.com/docs/messaging/api/overview",
    "https://www.plivo.com/docs/messaging/concepts/10dlc/overview",
    "https://www.plivo.com/docs/messaging/concepts/sender-id/overview",
    "https://www.plivo.com/docs/messaging/concepts/toll-free/overview",
    "https://www.plivo.com/docs/messaging/concepts/short-code/overview",
    "https://www.plivo.com/docs/faq/messaging/messaging-api",
    # SIP Trunking
    "https://www.plivo.com/docs/sip-trunking",
    "https://www.plivo.com/docs/sip-trunking/api/overview",
    # Phone Numbers
    "https://www.plivo.com/docs/numbers/phone-numbers",
    "https://www.plivo.com/docs/numbers/api/overview",
    # Account
    "https://www.plivo.com/docs/account/overview",
    "https://www.plivo.com/docs/account/api/overview",
    # Lookup
    "https://www.plivo.com/docs/lookup/overview",
    # AI Agent
    "https://www.plivo.com/docs/aiagent/getstarted/overview",
]


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _load_checkpoint() -> dict[str, dict]:
    """Load already-scraped pages keyed by URL."""
    cache = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cache[entry["url"]] = entry
    logger.info(f"Loaded {len(cache)} scraped pages from checkpoint")
    return cache


def _append_checkpoint(entry: dict):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── BFS Crawler ───────────────────────────────────────────────────────────────

def discover_docs_urls(client: httpx.Client, already_scraped: set[str]) -> list[str]:
    """
    BFS from seed URLs. Follows all /docs/ links found in each page's content area.
    Returns all discovered URLs in crawl order, skipping already-scraped ones.
    """
    visited: set[str] = set()
    queue: list[str] = []

    for seed in SEED_URLS:
        full = seed if seed.startswith("http") else BASE_URL + seed
        # Normalise — strip fragment
        full = full.split("#")[0].rstrip("/")
        if full not in visited:
            visited.add(full)
            queue.append(full)

    all_urls: list[str] = []

    while queue:
        url = queue.pop(0)
        all_urls.append(url)

        if url in already_scraped:
            continue  # Still return in list so caller knows about it, skip re-fetch

        try:
            resp = client.get(url, timeout=20)
            if resp.status_code != 200:
                time.sleep(REQUEST_DELAY)
                continue
        except Exception as e:
            logger.warning(f"Crawl fetch failed {url}: {e}")
            time.sleep(REQUEST_DELAY)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.find(id="content-area") or soup.find(id="content") or soup.find("main")
        if not content:
            time.sleep(REQUEST_DELAY)
            continue

        for a in content.find_all("a", href=True):
            href = a["href"].split("#")[0].rstrip("/")
            if not href.startswith("/docs/"):
                continue
            full_link = BASE_URL + href
            if full_link not in visited:
                visited.add(full_link)
                queue.append(full_link)

        time.sleep(REQUEST_DELAY)

    logger.info(f"BFS complete — discovered {len(all_urls)} docs URLs")
    return all_urls


# ── Scraping ──────────────────────────────────────────────────────────────────

def _extract_breadcrumb(soup: BeautifulSoup) -> str:
    """Extract breadcrumb trail from page for context."""
    breadcrumb_el = soup.find(attrs={"aria-label": "breadcrumb"})
    if breadcrumb_el:
        parts = [s.strip() for s in breadcrumb_el.get_text(" > ", strip=True).split(">") if s.strip()]
        return " > ".join(parts)
    return ""


def _clean_text(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def scrape_page(url: str, client: httpx.Client) -> Optional[dict]:
    """Fetch a docs page and return structured content."""
    try:
        resp = client.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract main content area
    content_el = soup.find(id="content-area") or soup.find("article") or soup.find("main")
    if not content_el:
        return None

    # Page title
    title_el = content_el.find("h1") or soup.find("h1")
    page_title = title_el.get_text(strip=True) if title_el else urlparse(url).path.split("/")[-1]

    breadcrumb = _extract_breadcrumb(soup)

    return {
        "url": url,
        "page_title": page_title,
        "breadcrumb": breadcrumb,
        "content_el": content_el,  # BeautifulSoup element — chunked below
    }


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_page(page: dict) -> list[dict]:
    """
    Split a page into chunks at H2/H3 boundaries.
    Each chunk = one section with its heading as context.
    """
    content_el = page["content_el"]
    url = page["url"]
    page_title = page["page_title"]
    breadcrumb = page["breadcrumb"]

    chunks = []
    current_heading = page_title
    current_parts = []

    for el in content_el.children:
        if not hasattr(el, "name"):
            continue

        if el.name in ("h2", "h3"):
            # Save previous section
            text = _clean_text("\n".join(current_parts))
            if text and len(text) > 50:
                chunks.append(_make_chunk(url, page_title, breadcrumb, current_heading, text, len(chunks)))
            current_heading = el.get_text(strip=True)
            current_parts = []
        else:
            text = el.get_text(separator="\n", strip=True)
            if text:
                current_parts.append(text)

    # Last section
    text = _clean_text("\n".join(current_parts))
    if text and len(text) > 50:
        chunks.append(_make_chunk(url, page_title, breadcrumb, current_heading, text, len(chunks)))

    # If no sections found, treat whole page as one chunk
    if not chunks:
        full_text = _clean_text(content_el.get_text(separator="\n", strip=True))
        if full_text and len(full_text) > 50:
            chunks.append(_make_chunk(url, page_title, breadcrumb, page_title, full_text, 0))

    return chunks


def _make_chunk(url: str, page_title: str, breadcrumb: str, section_title: str, text: str, idx: int) -> dict:
    url_slug = urlparse(url).path.strip("/").replace("/", "_")
    return {
        "chunk_id": f"docs_{url_slug}_{idx}",
        "source": "docs",
        "url": url,
        "page_title": page_title,
        "section_title": section_title,
        "breadcrumb": breadcrumb,
        # Text prepends breadcrumb + section for richer embedding signal
        "text": f"{breadcrumb + ' > ' if breadcrumb else ''}{section_title}\n\n{text}",
        "retrieval_boost": 1.0,
    }


# ── Qdrant ────────────────────────────────────────────────────────────────────

def init_docs_collection(client: QdrantClient, dim: int = 1536):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    for field in ("source", "page_title", "url"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema="keyword",
        )
    logger.info(f"Created collection '{COLLECTION_NAME}'")


def upsert_docs(client: QdrantClient, chunks: list[dict], batch_size: int = 50):
    import uuid
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

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Upserted {min(i + batch_size, len(points))}/{len(points)} docs chunks")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run():
    logger.info("Starting Plivo docs scrape")

    http_client = httpx.Client(
        headers={"User-Agent": "SupportIQ-DocsScraper/1.0"},
        follow_redirects=True,
    )
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    qdrant = QdrantClient(path=settings.qdrant_path)

    init_docs_collection(qdrant, dim=settings.embedding_dim)

    # ── Step 1: Discover URLs via BFS ─────────────────────────────────────────
    checkpoint = _load_checkpoint()
    already_scraped = set(checkpoint.keys())

    logger.info("Discovering docs URLs via BFS crawl from seed URLs...")
    all_urls = discover_docs_urls(http_client, already_scraped)

    todo_urls = [u for u in all_urls if u not in already_scraped]
    logger.info(f"Total docs URLs: {len(all_urls)} | Already scraped: {len(already_scraped)} | To scrape: {len(todo_urls)}")

    # ── Step 2: Scrape + chunk ────────────────────────────────────────────────
    all_chunks = []

    # Load already-scraped chunks from checkpoint
    for entry in checkpoint.values():
        all_chunks.extend(entry.get("chunks", []))

    for i, url in enumerate(todo_urls):
        page = scrape_page(url, http_client)
        if not page:
            logger.warning(f"Skipping {url} — no content")
            time.sleep(REQUEST_DELAY)
            continue

        chunks = chunk_page(page)
        entry = {
            "url": url,
            "page_title": page["page_title"],
            "chunks": chunks,
        }
        _append_checkpoint(entry)
        all_chunks.extend(chunks)

        if (i + 1) % 20 == 0:
            logger.info(f"Scraped {i + 1}/{len(todo_urls)} pages — {len(all_chunks)} chunks total")

        time.sleep(REQUEST_DELAY)

    logger.info(f"Scraping complete — {len(all_chunks)} total chunks from {len(all_urls)} pages")

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    logger.info("Embedding docs chunks...")
    embedded_chunks = await embedder.embed_chunks(all_chunks)

    # ── Step 4: Upsert ────────────────────────────────────────────────────────
    logger.info("Upserting to Qdrant support_docs collection...")
    upsert_docs(qdrant, embedded_chunks)

    http_client.close()
    logger.info("Docs ingestion complete!")


if __name__ == "__main__":
    asyncio.run(run())
