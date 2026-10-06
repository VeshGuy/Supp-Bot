"""
Semantic answer cache — keyed by normalised question hash, TTL 24h.

Same question → same cached result, bypassing retrieval + LLM entirely.
Cache is stored in logs/answer_cache.jsonl (already in .gitignore via logs/).

Lookup strategy:
  1. Normalise question (lowercase, collapse whitespace)
  2. MD5 hash of "query_type::normalised_question"
  3. Scan cache file for matching non-expired entry
  4. Return cached result dict, or None on miss
"""

import hashlib
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = Path("logs/answer_cache.jsonl")
TTL_HOURS = 24


def _normalise(question: str) -> str:
    return " ".join(question.lower().strip().split())


def _cache_key(question: str, query_type: str) -> str:
    text = f"{query_type}::{_normalise(question)}"
    return hashlib.md5(text.encode()).hexdigest()


def get_cached(question: str, query_type: str) -> Optional[dict]:
    """Return cached answer dict if a fresh entry exists, else None."""
    if not CACHE_FILE.exists():
        return None

    key = _cache_key(question, query_type)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS)

    try:
        with open(CACHE_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("key") != key:
                    continue
                ts = datetime.fromisoformat(entry["ts"])
                if ts > cutoff:
                    logger.info(f"[cache] HIT  key={key[:8]} question={question[:60]!r}")
                    return entry["result"]
    except Exception as e:
        logger.warning(f"[cache] Read error: {e}")

    return None


def _compact_cache() -> None:
    """Rewrite cache file keeping only non-expired entries. Called when file exceeds 500KB."""
    if not CACHE_FILE.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(hours=TTL_HOURS)
    try:
        with open(CACHE_FILE) as f:
            lines = f.readlines()
        fresh = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if datetime.fromisoformat(entry["ts"]) > cutoff:
                    fresh.append(line)
            except Exception:
                pass  # Drop malformed lines
        with open(CACHE_FILE, "w") as f:
            f.write("\n".join(fresh) + ("\n" if fresh else ""))
        logger.info(f"[cache] Compacted {len(lines)} → {len(fresh)} entries")
    except Exception as e:
        logger.warning(f"[cache] Compact error: {e}")


def set_cached(question: str, query_type: str, result: dict) -> None:
    """Append result to cache file. Compacts expired entries when file exceeds 500KB."""
    key = _cache_key(question, query_type)
    entry = {
        "key": key,
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question[:300],
        "query_type": query_type,
        "result": result,
    }
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        # Compact before writing if file is large (removes expired entries, bounds file size)
        if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 500_000:
            _compact_cache()
        with open(CACHE_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"[cache] STORE key={key[:8]} question={question[:60]!r}")
    except Exception as e:
        logger.warning(f"[cache] Write error: {e}")
