"""
Hybrid retrieval: dense cosine similarity + full-text search (tsvector), guaranteed lanes.
pgvector replacement for hybrid_retriever.py (Qdrant).

Lane A (tickets): ticket_chunks (resolution) + ticket_chunks (problem/investigation) → RRF → top N
Lane B (docs):    doc_chunks → score × source_multiplier → top N
Lane C (ryuk):    ryuk_chunks → always top-1 (reranker decides final relevance)

Dense:   ORDER BY embedding <=> query_vec::halfvec (cosine distance → similarity = 1 - distance)
FTS:     WHERE ts_text @@ plainto_tsquery('english', query_text) — replaces BM25 sparse
RRF:     same Python implementation as Qdrant version (k=60, weighted per lane)

All lanes always searched in parallel via ThreadPoolExecutor.
query_type shifts ticket/docs slot allocation, not source exclusion.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

RRF_K = 60
RESOLUTION_WEIGHT = 0.65
PROBLEM_WEIGHT = 0.35

_SOURCE_MULTIPLIERS = {
    "confluence": 2.0,
    "support": 1.8,
    "slack": 1.5,
    "docs": 1.5,
}
_JIRA_HIGH_VALUE_PROJECTS = {"VT", "VPT"}

_LANE_SLOTS = {
    "ticket_search":    (5, 5),
    "product_question": (2, 8),
}
_DEFAULT_SLOTS = (5, 5)

_TICKET_MIN_DATE = "2024-01-01"


# ── Connection pool ───────────────────────────────────────────────────────────

def create_pool(postgres_url: str, minconn: int = 2, maxconn: int = 10) -> psycopg2.pool.ThreadedConnectionPool:
    """Create a thread-safe connection pool. Call once at startup."""
    pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, postgres_url)
    # Warm one connection to verify connectivity
    conn = pool.getconn()
    register_vector(conn)
    pool.putconn(conn)
    return pool


def _get_conn(pool: psycopg2.pool.ThreadedConnectionPool):
    """Borrow a connection from pool and register pgvector type."""
    conn = pool.getconn()
    register_vector(conn)
    return conn


# ── Lane helpers (identical logic to Qdrant version) ─────────────────────────

def _recency_boost(created_at: str) -> float:
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days < 180:
            return 1.5
        elif age_days < 365:
            return 1.3
        elif age_days < 730:
            return 1.1
        else:
            return 0.85
    except Exception:
        return 1.0


def _source_multiplier(hit: dict) -> float:
    source = hit.get("source", "")
    if source in _SOURCE_MULTIPLIERS:
        return _SOURCE_MULTIPLIERS[source]
    if source == "jira":
        project = str(hit.get("project", "")).upper()
        return 1.0 if project in _JIRA_HIGH_VALUE_PROJECTS else 0.5
    return 1.0


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _rrf_merge(dense_hits: list[dict], fts_hits: list[dict], id_field: str) -> list[dict]:
    """
    RRF merge of dense and FTS result lists. Deduplicates by id_field.
    Returns merged list sorted by combined RRF score descending.
    """
    scores: dict = {}
    payloads: dict = {}

    for rank, hit in enumerate(dense_hits):
        key = hit[id_field]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        payloads[key] = hit

    for rank, hit in enumerate(fts_hits):
        key = hit[id_field]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        if key not in payloads:
            payloads[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in ranked:
        hit = dict(payloads[key])
        hit["similarity"] = round(score, 6)
        results.append(hit)
    return results


def _merge_ticket_lane(
    resolution_results: list[dict],
    problem_results: list[dict],
    top_k: int,
) -> list[dict]:
    """RRF merge within the ticket lane. Deduped by ticket_id. Drops pre-2024 tickets."""
    scores: dict = {}
    payloads: dict = {}

    resolution_results = [h for h in resolution_results if h.get("created_at", "") >= _TICKET_MIN_DATE]
    problem_results    = [h for h in problem_results    if h.get("created_at", "") >= _TICKET_MIN_DATE]

    for rank, hit in enumerate(resolution_results):
        key = hit["ticket_id"]
        recency = _recency_boost(hit.get("created_at", ""))
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank) * RESOLUTION_WEIGHT * recency
        payloads[key] = hit

    for rank, hit in enumerate(problem_results):
        key = hit["ticket_id"]
        recency = _recency_boost(hit.get("created_at", ""))
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank) * PROBLEM_WEIGHT * recency
        if key not in payloads:
            payloads[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for key, score in ranked:
        hit = dict(payloads[key])
        hit["rrf_score"] = round(score, 4)
        hit["lane"] = "ticket"
        results.append(hit)
    return results


def _rank_docs_lane(docs_results: list[dict], top_k: int) -> list[dict]:
    """Score docs by similarity × source_multiplier × recency_boost. Returns top_k."""
    for hit in docs_results:
        recency = _recency_boost(hit.get("created_at", "")) if hit.get("created_at") else 1.0
        hit["rrf_score"] = round(hit.get("similarity", 0) * _source_multiplier(hit) * recency, 4)
        hit["lane"] = "docs"
    docs_results.sort(key=lambda h: h["rrf_score"], reverse=True)
    return docs_results[:top_k]


def _ryuk_lane(ryuk_results: list[dict], top_k: int = 1) -> list[dict]:
    """Lane C: always include the top Ryuk result. Reranker decides final relevance."""
    for hit in ryuk_results:
        hit["lane"] = "ryuk"
    ryuk_results.sort(key=lambda h: h.get("similarity", 0), reverse=True)
    return ryuk_results[:top_k]


# ── SQL helpers ───────────────────────────────────────────────────────────────

_TICKET_COLS = """
    ticket_id, chunk_id, chunk_type, subject, text,
    product, issue_type, region, resolution_type, csat_score,
    created_at, solved_at, zendesk_url, cluster_id,
    retrieval_boost, jira_escalated, jira_project
"""

_DOC_COLS = """
    source, text, issue_key, issue_id, project, chunk_type, summary,
    page_title, section_title, content_id, url, channel, channel_id, created_at
"""

_RYUK_COLS = "nav_path, text"


def _row_to_ticket_dict(row: tuple, cols: list[str]) -> dict:
    d = dict(zip(cols, row))
    # Normalise bool to Python bool (psycopg2 returns native bool, but guard anyway)
    d["jira_escalated"] = bool(d.get("jira_escalated", False))
    return d


def _row_to_doc_dict(row: tuple, cols: list[str]) -> dict:
    return dict(zip(cols, row))


def _row_to_ryuk_dict(row: tuple, cols: list[str]) -> dict:
    return dict(zip(cols, row))


def _build_filter_clause(filters: Optional[dict], table_alias: str = "") -> tuple[str, list]:
    """
    Build a parameterised SQL WHERE fragment from metadata_filters dict.
    Returns (sql_fragment, params_list).  sql_fragment starts with ' AND ...' or is empty.
    Only fields that exist on ticket_chunks are applied (product, issue_type, region).
    """
    if not filters:
        return "", []

    _ALLOWED_TICKET_FIELDS = {"product", "issue_type", "region"}
    clauses = []
    params = []
    prefix = f"{table_alias}." if table_alias else ""
    for k, v in filters.items():
        if k in _ALLOWED_TICKET_FIELDS:
            clauses.append(f"{prefix}{k} = %s")
            params.append(v)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


# ── Per-source search functions ───────────────────────────────────────────────

def _search_tickets(
    pool: psycopg2.pool.ThreadedConnectionPool,
    chunk_type_filter: str,   # 'resolution' or 'problem'
    dense_vec: list,
    query_text: str,
    per_k: int,
    filters: Optional[dict],
) -> list[dict]:
    """
    Hybrid search on ticket_chunks for one chunk_type group.
    Returns RRF-merged dicts with 'similarity' field.
    """
    filter_sql, filter_params = _build_filter_clause(filters)

    # chunk_type_filter is 'resolution' or 'problem' (problem includes investigation)
    if chunk_type_filter == "resolution":
        type_sql = "chunk_type = 'resolution'"
    else:
        type_sql = "chunk_type IN ('problem', 'investigation')"

    date_sql = f"created_at >= '{_TICKET_MIN_DATE}'"

    base_where = f"WHERE {type_sql} AND {date_sql}{filter_sql}"

    dense_sql = f"""
        SELECT {_TICKET_COLS},
               1 - (embedding <=> %s::halfvec) AS similarity
        FROM ticket_chunks
        {base_where}
        ORDER BY embedding <=> %s::halfvec
        LIMIT %s
    """
    fts_sql = f"""
        SELECT {_TICKET_COLS},
               ts_rank(ts_text, plainto_tsquery('english', %s)) AS similarity
        FROM ticket_chunks
        {base_where}
          AND ts_text @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT %s
    """

    col_names = [c.strip() for c in _TICKET_COLS.split(",")]

    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            # Dense search
            cur.execute(dense_sql, [dense_vec, dense_vec] + filter_params + [per_k])
            dense_hits = [_row_to_ticket_dict(r, col_names + ["similarity"]) for r in cur.fetchall()]

            # FTS search
            try:
                cur.execute(fts_sql, [query_text] + filter_params + [query_text, per_k])
                fts_hits = [_row_to_ticket_dict(r, col_names + ["similarity"]) for r in cur.fetchall()]
            except Exception as e:
                logger.warning(f"FTS search failed for tickets ({chunk_type_filter}): {e}")
                conn.rollback()
                fts_hits = []
    finally:
        pool.putconn(conn)

    return _rrf_merge(dense_hits, fts_hits, id_field="chunk_id")


def _search_docs(
    pool: psycopg2.pool.ThreadedConnectionPool,
    dense_vec: list,
    query_text: str,
    per_k: int,
) -> list[dict]:
    """Hybrid search on doc_chunks."""
    dense_sql = f"""
        SELECT {_DOC_COLS},
               1 - (embedding <=> %s::halfvec) AS similarity
        FROM doc_chunks
        ORDER BY embedding <=> %s::halfvec
        LIMIT %s
    """
    fts_sql = f"""
        SELECT {_DOC_COLS},
               ts_rank(ts_text, plainto_tsquery('english', %s)) AS similarity
        FROM doc_chunks
        WHERE ts_text @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT %s
    """

    col_names = [c.strip() for c in _DOC_COLS.split(",")]

    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            cur.execute(dense_sql, [dense_vec, dense_vec, per_k])
            dense_hits = [_row_to_doc_dict(r, col_names + ["similarity"]) for r in cur.fetchall()]

            try:
                cur.execute(fts_sql, [query_text, query_text, per_k])
                fts_hits = [_row_to_doc_dict(r, col_names + ["similarity"]) for r in cur.fetchall()]
            except Exception as e:
                logger.warning(f"FTS search failed for docs: {e}")
                conn.rollback()
                fts_hits = []
    finally:
        pool.putconn(conn)

    # Use url or issue_key as dedup key (fallback to row index)
    merged = _rrf_merge(dense_hits, fts_hits, id_field="url") if all(h.get("url") for h in dense_hits + fts_hits) else _rrf_merge(dense_hits, fts_hits, id_field="issue_key")
    return merged


def _search_docs_safe(
    pool: psycopg2.pool.ThreadedConnectionPool,
    dense_vec: list,
    query_text: str,
    per_k: int,
) -> list[dict]:
    """Doc search with a stable dedup key even when url/issue_key may be null."""
    dense_sql = f"""
        SELECT {_DOC_COLS},
               1 - (embedding <=> %s::halfvec) AS similarity,
               id::text AS _row_id
        FROM doc_chunks
        ORDER BY embedding <=> %s::halfvec
        LIMIT %s
    """
    fts_sql = f"""
        SELECT {_DOC_COLS},
               ts_rank(ts_text, plainto_tsquery('english', %s)) AS similarity,
               id::text AS _row_id
        FROM doc_chunks
        WHERE ts_text @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT %s
    """

    col_names = [c.strip() for c in _DOC_COLS.split(",")]
    all_cols = col_names + ["similarity", "_row_id"]

    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            cur.execute(dense_sql, [dense_vec, dense_vec, per_k])
            dense_hits = [_row_to_doc_dict(r, all_cols) for r in cur.fetchall()]

            try:
                cur.execute(fts_sql, [query_text, query_text, per_k])
                fts_hits = [_row_to_doc_dict(r, all_cols) for r in cur.fetchall()]
            except Exception as e:
                logger.warning(f"FTS search failed for docs: {e}")
                conn.rollback()
                fts_hits = []
    finally:
        pool.putconn(conn)

    merged = _rrf_merge(dense_hits, fts_hits, id_field="_row_id")
    # Strip internal key
    for h in merged:
        h.pop("_row_id", None)
    return merged


def _search_ryuk(
    pool: psycopg2.pool.ThreadedConnectionPool,
    dense_vec: list,
    query_text: str,
    per_k: int,
) -> list[dict]:
    """Dense search on ryuk_chunks. FTS as secondary signal."""
    dense_sql = f"""
        SELECT {_RYUK_COLS},
               1 - (embedding <=> %s::halfvec) AS similarity,
               id::text AS _row_id
        FROM ryuk_chunks
        ORDER BY embedding <=> %s::halfvec
        LIMIT %s
    """
    fts_sql = f"""
        SELECT {_RYUK_COLS},
               ts_rank(ts_text, plainto_tsquery('english', %s)) AS similarity,
               id::text AS _row_id
        FROM ryuk_chunks
        WHERE ts_text @@ plainto_tsquery('english', %s)
        ORDER BY similarity DESC
        LIMIT %s
    """

    col_names = [c.strip() for c in _RYUK_COLS.split(",")]
    all_cols = col_names + ["similarity", "_row_id"]

    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            cur.execute(dense_sql, [dense_vec, dense_vec, per_k])
            dense_hits = [_row_to_ryuk_dict(r, all_cols) for r in cur.fetchall()]

            try:
                cur.execute(fts_sql, [query_text, query_text, per_k])
                fts_hits = [_row_to_ryuk_dict(r, all_cols) for r in cur.fetchall()]
            except Exception as e:
                logger.warning(f"FTS search failed for ryuk: {e}")
                conn.rollback()
                fts_hits = []
    finally:
        pool.putconn(conn)

    merged = _rrf_merge(dense_hits, fts_hits, id_field="_row_id")
    for h in merged:
        h.pop("_row_id", None)
    return merged


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    pool: psycopg2.pool.ThreadedConnectionPool,
    dense_vector: list,
    query_text: str,          # replaces sparse_vector — used for tsvector FTS
    top_k: int = 20,
    filters: Optional[dict] = None,
    query_type: str = "ticket_search",
) -> list[dict]:
    """
    Guaranteed-lane hybrid retrieval against pgvector.
    All 4 searches run in parallel via ThreadPoolExecutor.
    query_type controls slot allocation (ticket_search: 5+5, product_question: 2+8).
    """
    ticket_slots, docs_slots = _LANE_SLOTS.get(query_type, _DEFAULT_SLOTS)
    per_k = max(top_k, 20)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search_tickets, pool, "resolution",   dense_vector, query_text, per_k, filters): "resolutions",
            executor.submit(_search_tickets, pool, "problem",      dense_vector, query_text, per_k, filters): "problems",
            executor.submit(_search_docs_safe, pool,               dense_vector, query_text, per_k):          "docs",
            executor.submit(_search_ryuk, pool,                    dense_vector, query_text, per_k):          "ryuk",
        }
        raw: dict = {}
        for future in as_completed(futures):
            label = futures[future]
            try:
                raw[label] = future.result()
            except Exception as e:
                logger.error(f"Search failed on '{label}': {e}")
                raw[label] = []

    ticket_candidates = _merge_ticket_lane(raw.get("resolutions", []), raw.get("problems", []), top_k=ticket_slots)
    docs_candidates   = _rank_docs_lane(raw.get("docs", []), top_k=docs_slots)
    ryuk_candidates   = _ryuk_lane(raw.get("ryuk", []), top_k=1)

    merged = ticket_candidates + docs_candidates + ryuk_candidates
    logger.info(
        f"Lanes [{query_type}]: {len(ticket_candidates)} tickets + {len(docs_candidates)} docs "
        f"+ {len(ryuk_candidates)} ryuk = {len(merged)} candidates → reranker"
    )
    return merged


def fetch_ticket_chunks(
    pool: psycopg2.pool.ThreadedConnectionPool,
    ticket_id: str,
) -> list[dict]:
    """
    Fetch all sibling chunks (problem + investigation + resolution) for a ticket.
    pgvector replacement for _fetch_all_ticket_chunks(qdrant, ticket_id) in nodes.py.
    Single table scan — much simpler than searching two Qdrant collections.
    """
    sql = f"""
        SELECT {_TICKET_COLS}
        FROM ticket_chunks
        WHERE ticket_id = %s
        LIMIT 10
    """
    col_names = [c.strip() for c in _TICKET_COLS.split(",")]

    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, [ticket_id])
            rows = cur.fetchall()
            return [_row_to_ticket_dict(r, col_names) for r in rows]
    except Exception as e:
        logger.warning(f"Failed to fetch ticket chunks for {ticket_id}: {e}")
        return []
    finally:
        pool.putconn(conn)


def fetch_related_tickets(
    pool: psycopg2.pool.ThreadedConnectionPool,
    cluster_id: str,
    exclude_ticket_id: str,
) -> list[dict]:
    """
    Fetch sibling tickets in the same root-cause cluster.
    pgvector replacement for _fetch_related(qdrant, chunks) in nodes.py.
    Returns up to 4 tickets with ticket_id, subject, resolution_type.
    """
    sql = """
        SELECT DISTINCT ON (ticket_id)
            ticket_id, subject, resolution_type
        FROM ticket_chunks
        WHERE cluster_id = %s
          AND chunk_type = 'resolution'
          AND ticket_id <> %s
        LIMIT 5
    """
    conn = _get_conn(pool)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, [cluster_id, exclude_ticket_id])
            rows = cur.fetchall()
            return [
                {"ticket_id": r[0], "subject": r[1], "resolution_type": r[2]}
                for r in rows
            ]
    except Exception as e:
        logger.warning(f"Failed to fetch related tickets for cluster {cluster_id}: {e}")
        return []
    finally:
        pool.putconn(conn)
