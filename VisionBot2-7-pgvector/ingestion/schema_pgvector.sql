-- SupportIQ pgvector schema
-- Replaces 4 Qdrant collections: ticket_problems, ticket_resolutions, support_docs, ryuk_docs
--
-- Embedding type: halfvec(3072) — half-precision float (16-bit vs 32-bit).
--   pgvector's vector type caps HNSW/IVFFlat at 2000 dims; halfvec supports up to 4000 dims.
--   Precision loss is negligible for cosine similarity with text embeddings.
--
-- Sparse BM25 replaced by: tsvector full-text search (exact term matching, similar benefit).
-- Hybrid score = RRF(cosine_rank, fts_rank) computed in Python, same as before.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Ticket chunks (ticket_problems + ticket_resolutions merged) ────────────────
-- Merging both Qdrant collections into one table simplifies parent-child expansion
-- (compress node can fetch all chunk_types for a ticket_id in one query).
CREATE TABLE IF NOT EXISTS ticket_chunks (
    id              UUID    PRIMARY KEY,
    ticket_id       TEXT    NOT NULL,
    chunk_id        TEXT    NOT NULL,
    chunk_type      TEXT    NOT NULL,   -- 'problem'|'investigation'|'resolution'
    subject         TEXT,
    text            TEXT    NOT NULL,
    product         TEXT,
    issue_type      TEXT,
    region          TEXT,
    resolution_type TEXT,
    csat_score      TEXT,
    created_at      TEXT,
    solved_at       TEXT,
    zendesk_url     TEXT,
    cluster_id      TEXT,
    retrieval_boost FLOAT   DEFAULT 1.0,
    jira_escalated  BOOLEAN DEFAULT FALSE,
    jira_project    TEXT,
    embedding       HALFVEC(3072) NOT NULL,
    ts_text         TSVECTOR              -- populated on insert via to_tsvector('english', text)
);

CREATE INDEX IF NOT EXISTS ticket_chunks_embedding_idx
    ON ticket_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ticket_chunks_ts_idx      ON ticket_chunks USING GIN (ts_text);
CREATE INDEX IF NOT EXISTS ticket_chunks_ticket_id   ON ticket_chunks (ticket_id);
CREATE INDEX IF NOT EXISTS ticket_chunks_chunk_type  ON ticket_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS ticket_chunks_created_at  ON ticket_chunks (created_at);

-- ── Doc chunks (support_docs: jira, confluence, slack, docs, support) ──────────
CREATE TABLE IF NOT EXISTS doc_chunks (
    id            UUID PRIMARY KEY,
    source        TEXT NOT NULL,        -- 'jira'|'confluence'|'slack'|'docs'|'support'
    text          TEXT NOT NULL,
    -- jira fields
    issue_key     TEXT,
    issue_id      TEXT,
    project       TEXT,
    chunk_type    TEXT,
    summary       TEXT,
    -- confluence / docs / support fields
    page_title    TEXT,
    section_title TEXT,
    content_id    TEXT,
    url           TEXT,
    -- slack fields
    channel       TEXT,
    channel_id    TEXT,
    -- common
    created_at    TEXT,
    embedding     HALFVEC(3072) NOT NULL,
    ts_text       TSVECTOR
);

CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS doc_chunks_ts_idx ON doc_chunks USING GIN (ts_text);
CREATE INDEX IF NOT EXISTS doc_chunks_source ON doc_chunks (source);

-- ── Ryuk chunks ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ryuk_chunks (
    id        UUID PRIMARY KEY,
    nav_path  TEXT,
    text      TEXT NOT NULL,
    embedding HALFVEC(3072) NOT NULL,
    ts_text   TSVECTOR
);

CREATE INDEX IF NOT EXISTS ryuk_chunks_embedding_idx
    ON ryuk_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ryuk_chunks_ts_idx ON ryuk_chunks USING GIN (ts_text);
