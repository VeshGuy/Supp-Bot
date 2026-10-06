"""FastAPI server — streaming query endpoint, feedback, health."""

import asyncio
import json
import logging
import re as _re
import time
from datetime import datetime, timezone
from pathlib import Path

_UUID_RE_SERVER = _re.compile(
    r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
    _re.IGNORECASE,
)
_COMMENT_BLOCK_RE = _re.compile(r'\[(Public|Internal)\s*\|[^\]]+\]', _re.IGNORECASE)


def _extract_context_uuids(injected: str) -> list:
    """
    Extract UUIDs from a live ticket thread, tagged by source (customer vs internal note).
    Returns [{uuid, source}] in reverse-chronological order (most recent first).
    Deduplicates by UUID — if the same UUID appears in multiple comments, keeps
    the most recent occurrence and its source.

    The injected context format uses [Public | timestamp] and [Internal | timestamp]
    markers to distinguish customer messages from agent internal notes.
    """
    if not injected:
        return []

    # re.split with a capturing group interleaves markers and text blocks:
    # [pre_text, 'Public', text0, 'Internal', text1, ...]
    parts = _re.split(r'\[(Public|Internal)\s*\|[^\]]+\]', injected)
    markers = _COMMENT_BLOCK_RE.findall(injected)  # returns just the captured group

    # Actually findall on the split pattern — re.split already gave us the groups in parts.
    # Re-extract just the visibility word from each marker position in parts.
    vis_words = [p for i, p in enumerate(parts) if i % 2 == 1]  # odd indices = captured groups

    seen: dict = {}  # uuid → (source, order) — higher order = more recent
    for i, vis in enumerate(vis_words):
        block_idx = 2 * (i + 1)
        if block_idx >= len(parts):
            break
        text_block = parts[block_idx]
        source = "customer" if vis.strip().lower().startswith("public") else "internal"
        for match in _UUID_RE_SERVER.finditer(text_block):
            uuid = match.group(0).lower()
            # Keep most recent occurrence (highest i wins)
            if uuid not in seen or seen[uuid][1] < i:
                seen[uuid] = (source, i)

    # Sort descending by chronological order → most recent first
    sorted_items = sorted(seen.items(), key=lambda x: -x[1][1])
    return [{"uuid": uuid, "source": source} for uuid, (source, _) in sorted_items]

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from api.config import settings
from api.cost_tracker import log_query_cost
from api.models import FeedbackRequest, HealthResponse, QueryRequest, QueryResponse
from graph.graph import build_graph
from ingestion.embedder import Embedder
from knowledge import facts_db
from retrieval.hybrid_retriever_pg import create_pool, fetch_ticket_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

LOG_FILE = Path("logs/query_logs.jsonl")
ANSWER_LOG_FILE = Path("logs/answer_log.jsonl")

# Version tag — bump this when the pipeline changes so answer_log.jsonl
# entries from different configurations can be compared side by side.
# v1 = gpt-4o-mini HyDE + LLM compression
# v2 = gpt-4o-mini HyDE + parent-child expansion (no LLM compress for ticket_search)
# v3 = gpt-4o HyDE + no LLM compression at all
PIPELINE_VERSION = "v4-split-classify-hyde-dynamic-kb"

# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(title="SupportIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for internal ngrok access; protected by API token
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────

_openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
_pg_pool = create_pool(settings.postgres_url)
_embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
_graph = build_graph(_openai_client, _pg_pool, _embedder, settings)

# Initialise dynamic facts KB — creates table if needed, enables hot-reload from DB
facts_db.init(_pg_pool, _openai_client)

# ── Auth ──────────────────────────────────────────────────────────────────────

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != settings.api_secret_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


# ── Logging ───────────────────────────────────────────────────────────────────

def log_query(session_id: str, question: str, final_state: dict, latency_ms: int):
    timings = final_state.get("timings") or {}
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "intent": final_state.get("intent", ""),
        "voice_related": final_state.get("is_voice_related", True),
        "confidence": round(final_state.get("confidence_score", 0.0), 3),
        "citations": len(final_state.get("citations") or []),
        "retry_count": final_state.get("retry_count", 0),
        "steps_ms": {k: v for k, v in timings.items() if v},
        "total_ms": latency_ms,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    # Also print a compact summary to stdout for live monitoring
    steps = " → ".join(f"{k.replace('_ms','')}:{v}" for k, v in timings.items() if v)
    logger.info(f"[{session_id[:8]}] {question[:60]!r} | {steps} | total:{latency_ms}ms | conf:{entry['confidence']}")


def log_answer(session_id: str, question: str, final_state: dict, latency_ms: int):
    """
    Write the full answer to answer_log.jsonl with pipeline version tag.
    Allows before/after comparison when pipeline parameters change.
    Each entry includes PIPELINE_VERSION so different runs are distinguishable.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "session_id": session_id,
        "question": question,
        "answer": final_state.get("answer", ""),
        "reasoning": final_state.get("reasoning", ""),
        "suggested_action": final_state.get("suggested_action", ""),
        "confidence_score": round(final_state.get("confidence_score", 0.0), 3),
        "confidence_factors": final_state.get("confidence_factors", {}),
        "intent": final_state.get("intent", ""),
        "query_type": final_state.get("query_type", ""),
        "citation_ids": [
            c.get("ticket_id") for c in (final_state.get("citations") or []) if c.get("ticket_id")
        ],
        "latency_ms": latency_ms,
    }
    with open(ANSWER_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Endpoints ─────────────────────────────────────────────────────────────────
# Routes registered under both / and /api so dev proxy and prod build both work

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse, include_in_schema=False)
async def health():
    try:
        import psycopg2
        conn = _pg_pool.getconn()
        _pg_pool.putconn(conn)
        pg_status = "ok"
    except Exception:
        pg_status = "error"
    return HealthResponse(status="ok", qdrant=pg_status)


@app.post("/query")
@app.post("/api/query", include_in_schema=False)
async def query(
    request: QueryRequest,
    _token: str = Depends(verify_token),
):
    """
    Streaming SSE endpoint.
    Streams answer tokens, then sends a final JSON metadata chunk.
    """
    start = time.monotonic()

    initial_state = {
        "question": request.question,
        "session_id": request.session_id,
        "chat_history": request.prior_context,
        "metadata_filters": request.metadata_filters,
        "active_ticket_id": request.active_ticket_id,
        "injected_context": None,
        "is_voice_related": True,
        "is_ambiguous": False,
        "awaiting_clarification": False,
        "confidence_score": 0.0,
        "confidence_factors": {},
        "citations": [],
        "suggested_action": "",
        "related_tickets": [],
        "error": None,
        "cost_entries": [],
    }

    config = {"configurable": {"thread_id": request.session_id}}

    async def event_stream():
        final_state = {}
        try:
            async for event in _graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                if kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        # Stream the extracted answer text word-by-word for a smooth UX
        # (generate_answer uses JSON mode so raw tokens would expose JSON structure)
        answer = final_state.get("answer", "")
        if answer:
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.015)  # ~65 words/sec

        # Append UUID note if ticket had identifiable UUIDs and we're not waiting for clarification
        _injected = final_state.get("injected_context") or ""
        _context_uuids = _extract_context_uuids(_injected)
        if _context_uuids and answer and not final_state.get("awaiting_clarification"):
            customer_uuids = [u["uuid"] for u in _context_uuids if u["source"] == "customer"]
            internal_uuids = [u["uuid"] for u in _context_uuids if u["source"] == "internal"]
            parts = []
            if customer_uuids:
                parts.append("Customer: " + ", ".join(f"`{u}`" for u in customer_uuids))
            if internal_uuids:
                parts.append("Agent notes: " + ", ".join(f"`{u}`" for u in internal_uuids))
            note = "\n\n---\n**UUIDs found in this ticket:** " + " · ".join(parts) + " — reply with a UUID if you'd like a log investigation."
            yield f"data: {json.dumps({'type': 'token', 'content': note})}\n\n"

        latency_ms = int((time.monotonic() - start) * 1000)

        # Send metadata as final chunk
        metadata = {
            "type": "done",
            "citations": final_state.get("citations", []),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "confidence_factors": final_state.get("confidence_factors", {}),
            "suggested_action": final_state.get("suggested_action", ""),
            "related_tickets": final_state.get("related_tickets", []),
            "reasoning": final_state.get("reasoning", ""),
            "intent": final_state.get("intent", ""),
            "awaiting_clarification": final_state.get("awaiting_clarification", False),
            "wants_log_investigation": final_state.get("wants_log_investigation", False),
            "latency_ms": latency_ms,
            # UUIDs found in the injected live ticket thread, tagged by source.
            # [{uuid, source}] in reverse-chrono order (most recent first).
            # Used by Slack bot to offer targeted log investigation.
            "context_uuids": _context_uuids,
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        log_query(request.session_id, request.question, final_state, latency_ms)
        log_answer(request.session_id, request.question, final_state, latency_ms)
        log_query_cost(
            session_id=request.session_id,
            question=request.question,
            query_type=final_state.get("query_type", ""),
            cost_entries=final_state.get("cost_entries") or [],
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
@app.post("/api/feedback", include_in_schema=False)
async def feedback(
    request: FeedbackRequest,
    _token: str = Depends(verify_token),
):
    """
    Record agent feedback (helpful / not helpful).
    In V2 this will update ticket retrieval weights in Qdrant.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": request.session_id,
        "question": request.question,
        "rating": request.rating,
        "issue_type": request.issue_type,
        "comment": request.comment,
        "ticket_ids": request.ticket_ids,
    }
    with open("logs/feedback_logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Low rating with a comment → attempt to extract a standing fact (fire-and-forget)
    if request.rating is not None and request.rating <= 4 and request.comment:
        feedback_id = entry["ts"]
        # Retrieve the bot's answer from the answer log to give the extractor full context
        answer_excerpt = ""
        try:
            lines = Path("logs/answer_log.jsonl").read_text().splitlines()
            for line in reversed(lines[-50:]):
                rec = json.loads(line)
                if rec.get("session_id") == request.session_id:
                    answer_excerpt = rec.get("answer", "")[:600]
                    break
        except Exception:
            pass
        asyncio.create_task(
            facts_db.extract_and_store_fact(
                question=request.question or "",
                answer=answer_excerpt,
                comment=request.comment,
                feedback_id=feedback_id,
            )
        )

    logger.info(f"Feedback logged: rating={request.rating} for session {request.session_id}")
    return {"status": "ok"}


@app.get("/tickets/{ticket_id}")
@app.get("/api/tickets/{ticket_id}", include_in_schema=False)
async def get_ticket(
    ticket_id: str,
    _token: str = Depends(verify_token),
):
    """Fetch full ticket payload from pgvector by ticket_id."""
    try:
        chunks = fetch_ticket_chunks(_pg_pool, ticket_id)
        if chunks:
            return chunks[0]
    except Exception as e:
        logger.error(f"Ticket fetch error: {e}")

    raise HTTPException(status_code=404, detail="Ticket not found")


# ── Admin: facts KB management ────────────────────────────────────────────────

@app.get("/admin/facts")
async def list_facts(_token: str = Depends(verify_token)):
    """List all facts (active + draft). Draft facts have active=false."""
    conn = _pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, fact_text, source, feedback_id, active, created_at "
                "FROM facts ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
        return [
            {
                "id": r[0], "fact_text": r[1], "source": r[2],
                "feedback_id": r[3], "active": r[4], "created_at": str(r[5]),
            }
            for r in rows
        ]
    finally:
        _pg_pool.putconn(conn)


@app.post("/admin/facts/{fact_id}/activate")
async def activate_fact(fact_id: int, _token: str = Depends(verify_token)):
    """Activate a draft fact so it is injected into future prompts."""
    conn = _pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE facts SET active=true WHERE id=%s", (fact_id,))
        conn.commit()
    finally:
        _pg_pool.putconn(conn)
    return {"status": "activated", "id": fact_id}


@app.delete("/admin/facts/{fact_id}")
async def delete_fact(fact_id: int, _token: str = Depends(verify_token)):
    """Delete a fact (draft or active)."""
    conn = _pg_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE id=%s", (fact_id,))
        conn.commit()
    finally:
        _pg_pool.putconn(conn)
    return {"status": "deleted", "id": fact_id}


# ── Serve React frontend (must be last) ───────────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
