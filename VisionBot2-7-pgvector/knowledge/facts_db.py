"""
Dynamic facts knowledge base backed by PostgreSQL.

Merges file-based facts (knowledge/facts.md) with DB-stored facts
auto-extracted from low-rating agent feedback.

Usage:
  # At server startup:
  facts_db.init(pg_pool, openai_client)

  # In any node:
  active_facts = facts_db.get_facts()

  # After low-rating feedback (fire-and-forget):
  asyncio.create_task(facts_db.extract_and_store_fact(...))

Admin endpoints in api/server.py allow activating / deleting draft facts.
Auto-extracted facts start as active=false (draft) — a human must activate them.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_FACTS_FILE = Path(__file__).parent / "facts.md"
_CACHE_TTL = 300  # seconds — refresh DB facts every 5 minutes

# Module-level singletons — set by init()
_pool = None
_openai_client = None

# In-memory cache of DB facts (string, ready to inject)
_db_facts_cache: str = ""
_db_facts_cache_ts: float = 0.0


def init(pool, openai_client) -> None:
    """Call once at server startup."""
    global _pool, _openai_client
    _pool = pool
    _openai_client = openai_client
    _ensure_table()


def _ensure_table() -> None:
    if not _pool:
        return
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS facts (
                        id          SERIAL PRIMARY KEY,
                        fact_text   TEXT        NOT NULL,
                        source      TEXT        DEFAULT 'manual',
                        feedback_id TEXT,
                        active      BOOLEAN     DEFAULT false,
                        created_at  TIMESTAMPTZ DEFAULT now()
                    )
                """)
            conn.commit()
        finally:
            _pool.putconn(conn)
        logger.info("Facts table ready")
    except Exception as e:
        logger.error(f"Facts table init failed: {e}")


def get_facts() -> str:
    """
    Returns merged facts string ready for prompt injection.
    Format: bullet list from facts.md + active DB facts.
    Refreshes DB portion every 5 minutes. Falls back to file-only on DB error.
    """
    global _db_facts_cache, _db_facts_cache_ts

    # Always read file-based facts fresh (they're small and rarely change)
    file_facts = ""
    try:
        file_facts = _FACTS_FILE.read_text().strip()
    except FileNotFoundError:
        pass

    if not _pool:
        return file_facts

    # Refresh DB cache if stale
    if time.time() - _db_facts_cache_ts > _CACHE_TTL:
        try:
            conn = _pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT fact_text FROM facts WHERE active=true ORDER BY created_at"
                    )
                    rows = cur.fetchall()
                _db_facts_cache = "\n".join(f"- {r[0]}" for r in rows)
                _db_facts_cache_ts = time.time()
            finally:
                _pool.putconn(conn)
        except Exception as e:
            logger.warning(f"Facts DB query failed (using cached): {e}")

    parts = [p for p in [file_facts, _db_facts_cache] if p]
    return "\n".join(parts)


# ── Fact extraction from feedback ─────────────────────────────────────────────

_EXTRACT_SYSTEM = """You extract factual corrections from Plivo support agent feedback.

If the agent's comment implies a specific fact the bot should remember for future answers,
return it as one clear sentence (max 30 words). Otherwise return null.

Good facts:
- "CPS = -1 means voice is deliberately blocked, not an unlimited rate limit"
- "US toll-free numbers require carrier verification before they can send — not instant"
- "Jio escalations require a circuit ID (their name for trunk identifier), not just auth_id"

Not a fact (too vague): "the answer was wrong", "needs more detail", "didn't help"

Return JSON only: {"fact": "<sentence>" or null}"""


async def extract_and_store_fact(
    question: str,
    answer: str,
    comment: str,
    feedback_id: str,
) -> None:
    """
    Fire-and-forget coroutine. Call via asyncio.create_task().
    Extracts a fact from low-rating feedback and stores it as a draft (active=false).
    A human must activate it via POST /admin/facts/{id}/activate.
    """
    if not _pool or not _openai_client or not comment.strip():
        return

    try:
        response = await _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": (
                    f"Question: {question}\n\n"
                    f"Bot answer (excerpt): {answer[:600]}\n\n"
                    f"Agent feedback: {comment}"
                )},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        fact = result.get("fact")
        if not fact:
            logger.debug(f"No fact extracted from feedback {feedback_id}")
            return

        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO facts (fact_text, source, feedback_id, active) "
                    "VALUES (%s, 'feedback', %s, false)",
                    (fact, feedback_id),
                )
            conn.commit()
            logger.info(f"Draft fact stored from feedback {feedback_id}: {fact[:80]}")
        finally:
            _pool.putconn(conn)

    except Exception as e:
        logger.warning(f"Fact extraction failed for feedback {feedback_id}: {e}")
