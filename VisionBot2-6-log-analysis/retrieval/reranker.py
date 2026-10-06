"""
Cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
Re-scores top-K retrieved chunks against the original query.
Returns top-N after reranking.

NOTE: BAAI/bge-reranker-v2-m3 was trialled but reverted — 568M params on CPU
produced 100+ second rerank times (vs ~350ms for MiniLM). Not viable without GPU.
MiniLM (22M params) is fast and sufficient for production on CPU.
Scores are in roughly -10 to +10 range; confidence normalisation uses sigmoid.
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info(f"Loading reranker model: {MODEL_NAME}")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank chunks against the query using a cross-encoder.
    Returns top_n chunks sorted by reranker score descending.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"Reranking failed: {e} — returning original order")
        return chunks[:top_n]

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate: tickets by ticket_id, docs by chunk_id
    # Ensures top_n means top_n unique sources
    seen: set[str] = set()
    results = []
    for chunk, score in scored:
        key = chunk.get("ticket_id") or chunk.get("chunk_id") or chunk.get("image_path")
        if key in seen:
            continue
        seen.add(key)
        c = dict(chunk)
        c["rerank_score"] = round(float(score), 4)
        results.append(c)
        if len(results) == top_n:
            break

    logger.info(f"Reranked {len(chunks)} → {len(results)} unique-ticket chunks (top_n={top_n})")
    return results
