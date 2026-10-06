"""
Hybrid retrieval: dense + BM25 sparse vectors, guaranteed lanes.
Lane A (tickets): ticket_resolutions + ticket_problems → RRF → top N
Lane B (docs):    support_docs → score × source_multiplier → top N
Lane C (ryuk):    ryuk_docs → always top-1 (reranker decides final relevance)
All lanes always searched. query_type shifts A/B slot allocation, not source exclusion.
Lane C is a fixed +1 slot that cannot be competed away by tickets or docs.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    Prefetch,
    SearchParams,
    SparseVector,
)

logger = logging.getLogger(__name__)

RRF_K = 60
RESOLUTION_WEIGHT = 0.65
PROBLEM_WEIGHT = 0.35  # within the ticket lane (sums to 1.0 inside lane)

# Per-source multipliers within support_docs
_SOURCE_MULTIPLIERS = {
    "confluence": 2.0,
    "support": 1.8,   # support.plivo.com — curated official FAQs
    "slack": 1.5,
    "docs": 1.5,
}
_JIRA_HIGH_VALUE_PROJECTS = {"VT", "VPT"}

# Slot allocation per query_type: (ticket_slots, docs_slots)
_LANE_SLOTS = {
    "ticket_search":    (5, 5),
    "product_question": (2, 8),
}
_DEFAULT_SLOTS = (5, 5)


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


_TICKET_MIN_DATE = "2025-01-01"  # ISO prefix — lexicographic comparison works for ISO strings


def _merge_ticket_lane(
    resolution_results: list[dict],
    problem_results: list[dict],
    top_k: int,
) -> list[dict]:
    """RRF merge within the ticket lane. Deduped by ticket_id. Drops pre-2025 tickets."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    # Apply date cutoff in Python — created_at is an ISO string so lexicographic >= works.
    resolution_results = [h for h in resolution_results if h.get("created_at", "") >= _TICKET_MIN_DATE]
    problem_results    = [h for h in problem_results    if h.get("created_at", "") >= _TICKET_MIN_DATE]

    for rank, hit in enumerate(resolution_results):
        key = hit["ticket_id"]
        recency = _recency_boost(hit.get("created_at", ""))
        scores[key] = scores.get(key, 0) + _rrf_score(rank) * RESOLUTION_WEIGHT * recency
        payloads[key] = hit

    for rank, hit in enumerate(problem_results):
        key = hit["ticket_id"]
        recency = _recency_boost(hit.get("created_at", ""))
        scores[key] = scores.get(key, 0) + _rrf_score(rank) * PROBLEM_WEIGHT * recency
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
    """Score docs by similarity × source_multiplier. Returns top_k."""
    for hit in docs_results:
        hit["rrf_score"] = round(hit.get("similarity", 0) * _source_multiplier(hit), 4)
        hit["lane"] = "docs"
    docs_results.sort(key=lambda h: h["rrf_score"], reverse=True)
    return docs_results[:top_k]


def _ryuk_lane(ryuk_results: list[dict], top_k: int = 1) -> list[dict]:
    """
    Lane C: always include the top Ryuk navigation result.
    The cross-encoder reranker decides final relevance — no score threshold here.
    Returns empty list if ryuk_docs collection is absent or empty (safe before indexing).
    """
    for hit in ryuk_results:
        hit["lane"] = "ryuk"
    ryuk_results.sort(key=lambda h: h.get("similarity", 0), reverse=True)
    return ryuk_results[:top_k]


def _qdrant_hits_to_dicts(hits) -> list[dict]:
    return [
        {**point.payload, "similarity": point.score}
        for point in hits
        if point.payload
    ]


def retrieve(
    client: QdrantClient,
    dense_vector: list[float],
    sparse_vector: dict,
    top_k: int = 20,
    filters: dict[str, str] = None,
    query_type: str = "ticket_search",
) -> list[dict]:
    """
    Guaranteed-lane hybrid retrieval.
    Always searches both ticket collections and support_docs in parallel.
    query_type controls slot allocation (ticket_search: 5+5, product_question: 2+8).
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Only surface tickets created in 2025 or later.
    # created_at is stored as an ISO string — Qdrant Range requires numeric payload,
    # so we filter in Python after retrieval (lexicographic ISO comparison is correct).
    _TICKET_MIN_DATE = "2025-01-01"

    ticket_slots, docs_slots = _LANE_SLOTS.get(query_type, _DEFAULT_SLOTS)
    per_collection_k = max(top_k, 20)  # fetch enough to have good candidates in each lane

    qdrant_filter = None
    if filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    sparse_vec = SparseVector(
        indices=sparse_vector["indices"],
        values=sparse_vector["values"],
    )

    def _search(collection: str) -> list[dict]:
        try:
            results = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=per_collection_k,
                        filter=qdrant_filter,
                        params=SearchParams(hnsw_ef=512),
                    ),
                    Prefetch(
                        query=sparse_vec,
                        using="sparse",
                        limit=per_collection_k,
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=per_collection_k,
            )
            return _qdrant_hits_to_dicts(results.points)
        except Exception as e:
            logger.error(f"Search failed on '{collection}': {e}")
            return []

    # All 4 searches in parallel (ticket_resolutions, ticket_problems, support_docs, ryuk_docs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search, "ticket_resolutions"): "resolutions",
            executor.submit(_search, "ticket_problems"): "problems",
            executor.submit(_search, "support_docs"): "docs",
            executor.submit(_search, "ryuk_docs"): "ryuk",
        }
        raw = {}
        for future in as_completed(futures):
            raw[futures[future]] = future.result()

    ticket_candidates = _merge_ticket_lane(raw["resolutions"], raw["problems"], top_k=ticket_slots)
    docs_candidates = _rank_docs_lane(raw["docs"], top_k=docs_slots)
    ryuk_candidates = _ryuk_lane(raw.get("ryuk", []), top_k=1)

    merged = ticket_candidates + docs_candidates + ryuk_candidates
    logger.info(
        f"Lanes [{query_type}]: {len(ticket_candidates)} tickets + {len(docs_candidates)} docs "
        f"+ {len(ryuk_candidates)} ryuk = {len(merged)} candidates → reranker"
    )
    return merged
