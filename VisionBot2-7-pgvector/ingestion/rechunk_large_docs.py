"""
Re-chunk oversized doc_chunks in pgvector.

Supports source='docs' (4000-char limit) and source='jira' (2000-char limit).
Chunks with text > limit are split at paragraph/line boundaries and re-embedded.
The original row is deleted; sub-chunks are inserted with new deterministic IDs.

Run:
    .venv/bin/python -m ingestion.rechunk_large_docs

Safe to re-run: uses ON CONFLICT DO NOTHING and deletes only rows it processes.
"""

import asyncio
import hashlib
import logging
import os
import sys
import uuid
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ingestion.embedder import Embedder
from api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Per-source size limits
_SOURCE_MAX_CHARS = {
    "docs": 4000,    # Plivo scraper docs — heading-split sections
    "jira": 2000,    # Jira descriptions/comments — dense technical content
}
OVERLAP_CHARS = 200
DRY_RUN = False  # set True to see what would change without writing


def _split_lines(lines: list[str], sep: str) -> list[str]:
    """
    Pack lines into chunks of MAX_CHARS, joined by sep.
    Carries the last line of each chunk into the next for overlap.
    """
    parts = []
    current = []
    current_len = 0
    for line in lines:
        if not line.strip():
            continue
        if current_len + len(line) > MAX_CHARS and current:
            parts.append(sep.join(current))
            carry = current[-1]
            current = [carry, line]
            current_len = len(carry) + len(line)
        else:
            current.append(line)
            current_len += len(line)
    if current:
        parts.append(sep.join(current))
    return parts


def _split_text(text: str) -> list[str]:
    """
    Split text at paragraph boundaries (\\n\\n), falling back to single-newline
    boundaries, and finally to hard character splits for very long lines.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    # Try double-newline paragraphs first
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    # If any single paragraph still exceeds MAX_CHARS, break it further at single newlines
    expanded = []
    for para in paragraphs:
        if len(para) > MAX_CHARS:
            sub_lines = [l for l in para.split('\n') if l.strip()]
            # Pack sub_lines; if a single line still exceeds MAX_CHARS, hard-split it
            for sl in sub_lines:
                if len(sl) > MAX_CHARS:
                    for i in range(0, len(sl), MAX_CHARS - OVERLAP_CHARS):
                        expanded.append(sl[i:i + MAX_CHARS])
                else:
                    expanded.append(sl)
        else:
            expanded.append(para)

    # Now pack all expanded segments into MAX_CHARS chunks
    parts = _split_lines(expanded, '\n')
    return parts if parts else [text[:MAX_CHARS]]


def _sub_chunk_id(original_id: str, idx: int) -> str:
    """Deterministic UUID for a sub-chunk derived from the original chunk ID."""
    key = f"{original_id}::sub::{idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


async def _rechunk_source(source: str, max_chars: int, embedder, cur, conn, insert_sql: str):
    """Re-chunk all oversized rows for a given source."""
    cur.execute("""
        SELECT id, source, text, page_title, section_title, url, channel, channel_id,
               issue_key, issue_id, project, chunk_type, summary, content_id, created_at
        FROM doc_chunks
        WHERE source = %s AND length(text) > %s
        ORDER BY length(text) DESC
    """, (source, max_chars))
    rows = cur.fetchall()
    logger.info(f"[{source}] Found {len(rows)} oversized chunks (>{max_chars} chars)")

    if not rows:
        return 0, 0

    total_deleted = 0
    total_inserted = 0

    for row in rows:
        (row_id, src, text, page_title, section_title, url, channel, channel_id,
         issue_key, issue_id, project, chunk_type, summary, content_id, created_at) = row

        # Use per-source MAX_CHARS for splitting
        sub_texts = _split_text_with_max(text, max_chars)

        logger.info(
            f"  {str(row_id)[:8]}… '{(page_title or summary or '')[:50]}' "
            f"{len(text):,} chars → {len(sub_texts)} sub-chunks"
        )

        if DRY_RUN:
            continue

        chunks_to_embed = [{"text": t} for t in sub_texts]
        embedded = await embedder.embed_chunks(chunks_to_embed)

        cur.execute("DELETE FROM doc_chunks WHERE id = %s", (str(row_id),))
        total_deleted += 1

        for idx, (sub_text, emb) in enumerate(zip(sub_texts, embedded)):
            sub_id = _sub_chunk_id(str(row_id), idx)
            vec = '[' + ','.join(str(x) for x in emb["dense_vector"]) + ']'
            cur.execute(insert_sql, (
                sub_id, src, sub_text,
                issue_key, issue_id, project, chunk_type, summary,
                page_title, section_title, content_id, url,
                channel, channel_id, str(created_at) if created_at else None,
                vec, sub_text or " ",
            ))
            total_inserted += 1

        conn.commit()

    logger.info(f"[{source}] Deleted {total_deleted}, inserted {total_inserted} sub-chunks")
    return total_deleted, total_inserted


def _split_text_with_max(text: str, max_chars: int) -> list[str]:
    """Same logic as _split_text but parametrised on max_chars."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    expanded = []
    for para in paragraphs:
        if len(para) > max_chars:
            for line in para.split('\n'):
                if not line.strip():
                    continue
                if len(line) > max_chars:
                    for i in range(0, len(line), max_chars - OVERLAP_CHARS):
                        expanded.append(line[i:i + max_chars])
                else:
                    expanded.append(line)
        else:
            expanded.append(para)

    parts = []
    current = []
    current_len = 0
    for seg in expanded:
        # Account for joining newlines: joined length = sum(lens) + n_segments - 1
        # So projected joined length = current_len + len(current) [newlines so far] + len(seg)
        if current_len + len(current) + len(seg) > max_chars and current:
            parts.append('\n'.join(current))
            carry = current[-1]
            # Only carry if carry + sep + seg fits within max_chars — otherwise start fresh
            if len(carry) + 1 + len(seg) <= max_chars:
                current = [carry, seg]
                current_len = len(carry) + len(seg)
            else:
                current = [seg]
                current_len = len(seg)
        else:
            current.append(seg)
            current_len += len(seg)
    if current:
        parts.append('\n'.join(current))
    return parts if parts else [text[:max_chars]]


async def main():
    POSTGRES_URL = os.getenv("POSTGRES_URL",
        f"postgresql://{os.getenv('USER', 'postgres')}@localhost:5432/supportiq")

    conn = psycopg2.connect(POSTGRES_URL)
    register_vector(conn)
    cur = conn.cursor()

    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)

    insert_sql = """
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

    grand_deleted = 0
    grand_inserted = 0

    for source, max_chars in _SOURCE_MAX_CHARS.items():
        d, i = await _rechunk_source(source, max_chars, embedder, cur, conn, insert_sql)
        grand_deleted += d
        grand_inserted += i

    logger.info(
        f"\nTotal: deleted {grand_deleted}, inserted {grand_inserted} sub-chunks "
        f"(avg {grand_inserted / max(grand_deleted, 1):.1f} per original)."
    )
    cur.close()
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
