"""
Generate dense (OpenAI) and sparse (BM25 via FastEmbed) embeddings for chunks.
"""

import asyncio
import logging
from typing import Any

from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DENSE_BATCH = 100
SPARSE_MODEL = "Qdrant/bm25"


class Embedder:
    def __init__(self, api_key: str, dense_model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.dense_model = dense_model
        # Sparse model (BM25 via fastembed) — used only by ingestion scripts writing to Qdrant.
        # NOT loaded eagerly because:
        #   1. The query path (pgvector) uses Postgres FTS instead of BM25 sparse vectors.
        #   2. Loading SparseTextEmbedding at server startup costs ~200MB RAM + a few seconds
        #      on every uvicorn restart for zero query-path benefit.
        # Lazy-loaded on first call to _sparse_batch (ingestion only).
        # To restore eager loading (e.g. for a new Qdrant deployment), uncomment:
        # self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        self._sparse_model = None  # lazy — loaded on first call to _sparse_batch

    # text-embedding-3-small limit is 8192 tokens (~30k chars conservatively)
    _MAX_CHARS = 28000

    async def _dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with OpenAI. Truncates texts exceeding token limit."""
        truncated = [t[:self._MAX_CHARS] if len(t) > self._MAX_CHARS else t for t in texts]
        response = await self.client.embeddings.create(
            model=self.dense_model,
            input=truncated,
        )
        return [item.embedding for item in response.data]

    def _sparse_batch(self, texts: list[str]) -> list[dict]:
        """Generate BM25 sparse vectors via FastEmbed. Lazy-loads the model on first call."""
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        results = list(self._sparse_model.embed(texts))
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in results
        ]

    async def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Add dense_vector and sparse_vector to each chunk dict.
        Processes in batches to respect API limits.
        """
        texts = [c["text"] for c in chunks]

        # Dense embeddings in batches
        dense_vectors = []
        for i in range(0, len(texts), DENSE_BATCH):
            batch = texts[i : i + DENSE_BATCH]
            embeddings = await self._dense_batch(batch)
            dense_vectors.extend(embeddings)
            logger.info(f"Dense embedded {min(i + DENSE_BATCH, len(texts))}/{len(texts)}")

        # Sparse embeddings (CPU, fast)
        sparse_vectors = self._sparse_batch(texts)
        logger.info(f"Sparse embedded {len(texts)} chunks")

        # Attach to chunks
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
            chunk["dense_vector"] = dense
            chunk["sparse_vector"] = sparse

        return chunks

    def embed_query(self, text: str) -> tuple[list[float], dict]:
        """Embed a single query text. Returns (dense, sparse)."""
        dense = asyncio.run(self._dense_batch([text]))[0]
        sparse = self._sparse_batch([text])[0]
        return dense, sparse

    async def embed_query_async(self, text: str) -> tuple[list[float], dict]:
        """Async version of embed_query."""
        dense = (await self._dense_batch([text]))[0]
        sparse = self._sparse_batch([text])[0]
        return dense, sparse

    async def embed_dense_async(self, text: str) -> list[float]:
        """Embed a single text, dense vector only. Skips BM25 sparse computation."""
        return (await self._dense_batch([text]))[0]
