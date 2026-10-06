"""
Export Slack channel messages and index into Qdrant support_docs.
Channels: #support-important-updates, #api-product-announcements
Date range: 2020-01-01 to today
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVectorParams, SparseIndexParams, VectorParams, Distance

from api.config import settings
from ingestion.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path(__file__).parent / "slack_checkpoint.jsonl"
COLLECTION = "support_docs"
SINCE_DATE = "2020-01-01"

CHANNELS = [
    "support-important-updates",
    "api-product-announcements",
]


class SlackClient:
    def __init__(self, token: str):
        self.token = token
        self.base = "https://slack.com/api"

    def _get(self, endpoint: str, params: dict) -> dict:
        resp = requests.get(
            f"{self.base}/{endpoint}",
            headers={"Authorization": f"Bearer {self.token}"},
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Slack API error: {data.get('error')}")
        return data

    def list_channels(self) -> list[dict]:
        channels = []
        cursor = None
        while True:
            params = {"limit": 200, "types": "public_channel,private_channel"}
            if cursor:
                params["cursor"] = cursor
            data = self._get("conversations.list", params)
            channels.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return channels

    def get_channel_history(self, channel_id: str, oldest: float) -> list[dict]:
        messages = []
        cursor = None
        while True:
            params = {"channel": channel_id, "oldest": oldest, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("conversations.history", params)
            messages.extend(data.get("messages", []))
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
            time.sleep(0.5)  # Rate limit
        return messages

    def get_thread_replies(self, channel_id: str, thread_ts: str) -> list[dict]:
        try:
            data = self._get("conversations.replies", {"channel": channel_id, "ts": thread_ts})
            return data.get("messages", [])[1:]  # Skip parent message
        except Exception:
            return []


def chunk_messages(messages: list[dict], channel_name: str, channel_id: str) -> list[dict]:
    """
    Group messages into chunks of ~20 messages each.
    Thread replies are bundled with their parent.
    """
    chunks = []
    buffer = []

    for msg in messages:
        if msg.get("subtype"):  # Skip join/leave/bot messages
            continue
        text = msg.get("text", "").strip()
        if not text or len(text) < 20:
            continue

        ts = float(msg.get("ts", 0))
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        user = msg.get("user", "unknown")
        reply_count = msg.get("reply_count", 0)

        entry = f"[{dt}] {text}"
        if reply_count > 0:
            entry += f" ({reply_count} replies)"
        buffer.append(entry)

        if len(buffer) >= 20:
            chunks.append(_make_chunk(buffer, channel_name, channel_id))
            buffer = []

    if buffer:
        chunks.append(_make_chunk(buffer, channel_name, channel_id))

    return chunks


def _make_chunk(buffer: list[str], channel_name: str, channel_id: str) -> dict:
    # Extract date from first message for recency
    first_date = buffer[0][1:11] if buffer else "2020-01-01"
    text = f"#{channel_name} updates:\n\n" + "\n".join(buffer)
    return {
        "source": "slack",
        "channel": channel_name,
        "channel_id": channel_id,
        "created_at": first_date + "T00:00:00+00:00",
        "text": text[:3000],
    }


def load_checkpoint() -> set:
    if not CHECKPOINT_FILE.exists():
        return set()
    done = set()
    with open(CHECKPOINT_FILE) as f:
        for line in f:
            done.add(json.loads(line)["channel_id"])
    return done


async def main():
    token = settings.slack_bot_token if hasattr(settings, "slack_bot_token") else None
    if not token:
        # Try reading directly from .env
        import os
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN not set in .env")

    slack = SlackClient(token)
    oldest = datetime.strptime(SINCE_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()

    # Resolve channel names to IDs
    logger.info("Fetching channel list...")
    all_channels = slack.list_channels()
    channel_map = {c["name"]: c["id"] for c in all_channels}

    target = {}
    for name in CHANNELS:
        cid = channel_map.get(name)
        if cid:
            target[name] = cid
            logger.info(f"Found #{name} → {cid}")
        else:
            logger.warning(f"Channel #{name} not found — check bot is invited")

    if not target:
        raise RuntimeError("No target channels found")

    # Embed + index
    embedder = Embedder(api_key=settings.openai_api_key, dense_model=settings.embedding_model)
    qdrant = QdrantClient(path=settings.qdrant_path)

    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())},
        )

    done_channels = load_checkpoint()
    total_upserted = 0

    for channel_name, channel_id in target.items():
        if channel_id in done_channels:
            logger.info(f"#{channel_name} already indexed, skipping")
            continue

        logger.info(f"Fetching #{channel_name} from {SINCE_DATE}...")
        messages = slack.get_channel_history(channel_id, oldest)
        logger.info(f"  {len(messages)} messages fetched")

        chunks = chunk_messages(messages, channel_name, channel_id)
        logger.info(f"  {len(chunks)} chunks created")

        # Embed + upsert
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embedded = await embedder.embed_chunks([{"text": c["text"]} for c in batch])

            points = []
            for j, (chunk, emb) in enumerate(zip(batch, embedded)):
                point_id = abs(hash(f"slack_{channel_id}_{i}_{j}")) % (2**31)
                points.append(PointStruct(
                    id=point_id,
                    vector={"dense": emb["dense_vector"], "sparse": emb["sparse_vector"]},
                    payload=chunk,
                ))

            qdrant.upsert(collection_name=COLLECTION, points=points)
            total_upserted += len(points)
            logger.info(f"  Upserted {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        # Checkpoint
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(json.dumps({"channel_id": channel_id, "channel": channel_name}) + "\n")

    logger.info(f"Done. {total_upserted} Slack chunks indexed into '{COLLECTION}'")


if __name__ == "__main__":
    asyncio.run(main())
