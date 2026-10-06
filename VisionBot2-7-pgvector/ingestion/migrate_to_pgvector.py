"""
Migrate all vectors from Qdrant local file mode → pgvector (Postgres).

Collections migrated:
  ticket_problems    → ticket_chunks (chunk_type: problem/investigation)
  ticket_resolutions → ticket_chunks (chunk_type: resolution)
  support_docs       → doc_chunks
  ryuk_docs          → ryuk_chunks

Run:
  .venv/bin/python -m ingestion.migrate_to_pgvector

Idempotent: uses INSERT ... ON CONFLICT DO NOTHING so safe to re-run.
Reads POSTGRES_URL from .env (falls back to local default).
Qdrant must NOT have uvicorn running (file lock). Stop server first.
"""

import logging
import os
import sys
import uuid
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from qdrant_client import QdrantClient
from tqdm import tqdm

# ── Bootstrap project path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

QDRANT_PATH = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_data"))
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    f"postgresql://{os.getenv('USER', 'postgres')}@localhost:5432/supportiq",
)

BATCH_SIZE = 200   # rows per INSERT batch
SCROLL_SIZE = 200  # Qdrant scroll page size


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID from chunk_id — same formula as the indexers."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _clean(s) -> str:
    """Strip NUL bytes and return a plain str. PostgreSQL rejects 0x00 in text fields."""
    if s is None:
        return ""
    return str(s).replace("\x00", "")


def _get_dense(point):
    """Extract dense vector from a Qdrant ScoredPoint or Record."""
    vecs = getattr(point, "vector", None)
    if vecs is None:
        return None
    if isinstance(vecs, dict):
        return vecs.get("dense") or vecs.get("text-dense")
    if isinstance(vecs, list):
        return vecs
    return None


def _scroll_all(client: QdrantClient, collection: str) -> list:
    """Scroll entire collection, returning all points with vectors."""
    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    return points


def _insert_batch(cur, sql: str, rows: list):
    cur.executemany(sql, rows)


# ── Migration: ticket collections ─────────────────────────────────────────────

def migrate_tickets(qdrant: QdrantClient, cur):
    ticket_sql = """
        INSERT INTO ticket_chunks (
            id, ticket_id, chunk_id, chunk_type, subject, text,
            product, issue_type, region, resolution_type, csat_score,
            created_at, solved_at, zendesk_url, cluster_id,
            retrieval_boost, jira_escalated, jira_project,
            embedding, ts_text
        ) VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s::halfvec, to_tsvector('english', %s)
        )
        ON CONFLICT (id) DO NOTHING
    """

    for collection in ("ticket_resolutions", "ticket_problems"):
        logger.info(f"Scrolling {collection}…")
        try:
            points = _scroll_all(qdrant, collection)
        except Exception as e:
            logger.error(f"Failed to scroll {collection}: {e}")
            continue

        logger.info(f"  {len(points)} points — inserting into ticket_chunks…")
        rows = []
        skipped = 0

        for point in tqdm(points, desc=collection, unit="chunks"):
            p = point.payload or {}
            vec = _get_dense(point)
            if not vec:
                skipped += 1
                continue

            chunk_id = _clean(p.get("chunk_id", ""))
            row_id = _chunk_id_to_uuid(chunk_id) if chunk_id else str(uuid.uuid4())
            text = _clean(p.get("text", ""))

            rows.append((
                row_id,
                _clean(p.get("ticket_id", "")),
                chunk_id,
                _clean(p.get("chunk_type", "")),
                _clean(p.get("subject")),
                text,
                _clean(p.get("product")),
                _clean(p.get("issue_type")),
                _clean(p.get("region")),
                _clean(p.get("resolution_type")),
                _clean(p.get("csat_score")),
                _clean(p.get("created_at")),
                _clean(p.get("solved_at")),
                _clean(p.get("zendesk_url")),
                _clean(p.get("cluster_id")),
                float(p.get("retrieval_boost", 1.0)),
                bool(p.get("jira_escalated", False)),
                _clean(p.get("jira_project")),
                vec,
                text or " ",  # tsvector requires non-empty input
            ))

            if len(rows) >= BATCH_SIZE:
                _insert_batch(cur, ticket_sql, rows)
                rows = []

        if rows:
            _insert_batch(cur, ticket_sql, rows)

        if skipped:
            logger.warning(f"  {skipped} points skipped (no dense vector)")
        logger.info(f"  Done: {collection}")


# ── Migration: support_docs ───────────────────────────────────────────────────

def migrate_docs(qdrant: QdrantClient, cur):
    doc_sql = """
        INSERT INTO doc_chunks (
            id, source, text,
            issue_key, issue_id, project, chunk_type, summary,
            page_title, section_title, content_id, url,
            channel, channel_id, created_at,
            embedding, ts_text
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s::halfvec, to_tsvector('english', %s)
        )
        ON CONFLICT (id) DO NOTHING
    """

    logger.info("Scrolling support_docs…")
    try:
        points = _scroll_all(qdrant, "support_docs")
    except Exception as e:
        logger.error(f"Failed to scroll support_docs: {e}")
        return

    logger.info(f"  {len(points)} points — inserting into doc_chunks…")
    rows = []
    skipped = 0

    for point in tqdm(points, desc="support_docs", unit="chunks"):
        p = point.payload or {}
        vec = _get_dense(point)
        if not vec:
            skipped += 1
            continue

        # Deterministic ID: use existing Qdrant point ID if it's a UUID string, else generate
        try:
            row_id = str(uuid.UUID(str(point.id)))
        except (ValueError, AttributeError):
            chunk_id = p.get("chunk_id") or p.get("issue_key") or str(point.id)
            row_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))

        text = _clean(p.get("text", ""))
        rows.append((
            row_id,
            _clean(p.get("source", "")),
            text,
            _clean(p.get("issue_key")),
            _clean(p.get("issue_id")),
            _clean(p.get("project")),
            _clean(p.get("chunk_type")),
            _clean(p.get("summary")),
            _clean(p.get("page_title") or p.get("title")),
            _clean(p.get("section_title")),
            _clean(p.get("content_id")),
            _clean(p.get("url")),
            _clean(p.get("channel")),
            _clean(p.get("channel_id")),
            _clean(p.get("created_at") or p.get("created") or p.get("last_modified")),
            vec,
            text or " ",  # tsvector requires non-empty input
        ))

        if len(rows) >= BATCH_SIZE:
            _insert_batch(cur, doc_sql, rows)
            rows = []

    if rows:
        _insert_batch(cur, doc_sql, rows)

    if skipped:
        logger.warning(f"  {skipped} points skipped (no dense vector)")
    logger.info("  Done: support_docs")


# ── Migration: ryuk_docs ──────────────────────────────────────────────────────

def migrate_ryuk(qdrant: QdrantClient, cur):
    ryuk_sql = """
        INSERT INTO ryuk_chunks (id, nav_path, text, embedding, ts_text)
        VALUES (%s, %s, %s, %s::halfvec, to_tsvector('english', %s))
        ON CONFLICT (id) DO NOTHING
    """

    logger.info("Scrolling ryuk_docs…")
    try:
        points = _scroll_all(qdrant, "ryuk_docs")
    except Exception as e:
        logger.error(f"Failed to scroll ryuk_docs: {e}")
        return

    logger.info(f"  {len(points)} points — inserting into ryuk_chunks…")
    rows = []
    skipped = 0

    for point in tqdm(points, desc="ryuk_docs", unit="chunks"):
        p = point.payload or {}
        vec = _get_dense(point)
        if not vec:
            skipped += 1
            continue

        try:
            row_id = str(uuid.UUID(str(point.id)))
        except (ValueError, AttributeError):
            row_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point.id)))

        text = _clean(p.get("text", ""))
        rows.append((
            row_id,
            _clean(p.get("nav_path")),
            text,
            vec,
            text or " ",
        ))

    if rows:
        _insert_batch(cur, ryuk_sql, rows)

    if skipped:
        logger.warning(f"  {skipped} points skipped (no dense vector)")
    logger.info("  Done: ryuk_docs")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info(f"Connecting to Qdrant at {QDRANT_PATH}…")
    qdrant = QdrantClient(path=QDRANT_PATH)

    collections = [c.name for c in qdrant.get_collections().collections]
    logger.info(f"Qdrant collections: {collections}")

    logger.info(f"Connecting to Postgres at {POSTGRES_URL}…")
    conn = psycopg2.connect(POSTGRES_URL)
    register_vector(conn)
    cur = conn.cursor()

    try:
        migrate_tickets(qdrant, cur)
        conn.commit()
        logger.info("ticket_chunks committed")

        migrate_docs(qdrant, cur)
        conn.commit()
        logger.info("doc_chunks committed")

        migrate_ryuk(qdrant, cur)
        conn.commit()
        logger.info("ryuk_chunks committed")

        # Final counts
        for table in ("ticket_chunks", "doc_chunks", "ryuk_chunks"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            logger.info(f"  {table}: {count:,} rows")

    except Exception as e:
        conn.rollback()
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
