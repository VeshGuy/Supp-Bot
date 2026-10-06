# SupportIQ — Architecture & Design Decisions

An internal AI assistant for Plivo support agents. Answers questions about resolved
support tickets with cited sources and suggested actions. Scoped to **voice-related issues**.

---

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | GPT-4o + GPT-4o-mini | GPT-4o for final answer quality; mini for classification, HyDE, compression — 10x cheaper for high-frequency steps |
| Embeddings | text-embedding-3-small (1536-dim) | Best price/performance for semantic search; 1536-dim balances quality and storage |
| Sparse retrieval | FastEmbed BM25 | Keyword matching catches exact error codes, ticket IDs, product names that semantic search misses |
| Vector DB | Qdrant (local file mode) | No Docker/server needed; handles hybrid dense+sparse natively; scales to 500k+ chunks on disk |
| Reranker | ms-marco-MiniLM-L-6-v2 | Local cross-encoder; re-scores top-20 candidates against original query; no API latency |
| Agent framework | LangGraph | Native streaming, conditional edges for ambiguity/retry loops, built-in MemorySaver for multi-turn |
| Backend | FastAPI | Async-native; SSE streaming built-in; minimal overhead |
| Frontend | React + TypeScript + Tailwind | Fast to build; Vite for dev speed |

---

## Data Sources

| Source | Collection | Retrieval Weight |
|--------|-----------|-----------------|
| Ticket resolutions | `ticket_resolutions` | 65% |
| Ticket problems | `ticket_problems` | 25% |
| Plivo website docs | `support_docs` | 10% |

**Why two ticket collections?**
Support agents ask in two modes: *"what caused X?"* (matches problem chunks) and
*"how do I fix X?"* (matches resolution chunks). A single embedding per ticket dilutes
both signals. Splitting gives each a clean, focused vector. Resolution chunks get higher
weight because agents care more about fixes than problem descriptions.

---

## Ingestion Pipeline

```
Zendesk API → Summarize (GPT-4o-mini) → Chunk (3 per ticket) → Embed → Qdrant → Cluster
```

### Why Summarization?
Raw Zendesk tickets are noisy conversations — greetings, repeated context, internal
notes. Summarization extracts structured fields that serve two purposes:

1. **Metadata for filtering**: `product`, `issue_type`, `region`, `resolution_type`
   stored as Qdrant payload — enables pre-filters (e.g. voice-only queries)
2. **Clean resolution chunk**: `resolution_summary + suggested_action` from the
   summarizer becomes chunk_2 — far higher quality than raw last-message text

Results cached to `summaries_cache.jsonl` — incremental runs skip already-summarized
tickets.

### Why 3 Chunks per Ticket?
| Chunk | Content | Weight |
|-------|---------|--------|
| chunk_0 (problem) | Subject + customer's first 2 messages | Standard |
| chunk_1 (investigation) | Agent diagnostic exchanges | Standard |
| chunk_2 (resolution) | resolution_summary + suggested_action | 1.3x boost |

Each chunk gets its own embedding — precision over averaging. Deduplication by
`ticket_id` in the reranker ensures only one chunk per ticket reaches the final answer.

### Clustering
After upsert, resolution chunks are clustered by cosine similarity (threshold 0.85).
Tickets in the same cluster share a `cluster_id` payload — surfaced in the UI as
"Related Tickets" to show agents similar cases they might not have queried for.

---

## Retrieval Pipeline

```
Query → HyDE → Hybrid Search (dense + BM25) → RRF merge → Cross-encoder rerank → Top 5
```

### Why HyDE (Hypothetical Document Embedding)?
Instead of embedding the raw query *"SIP registration failing with 403"*, GPT-4o-mini
first generates a hypothetical resolution: *"This is typically caused by incorrect
credentials or IP not whitelisted. Resolution: update auth settings in the portal."*

We embed the hypothetical answer and search for that. This matches **answer-space to
answer-space** rather than question-space to answer-space — significantly better
retrieval for technical support queries where questions and answers use different
vocabulary.

### Why Hybrid (Dense + BM25)?
- **Dense (semantic)**: catches conceptual matches — "call disconnects" ↔ "audio drops"
- **BM25 (keyword)**: catches exact matches — error codes, SIP response codes, carrier
  names, ticket IDs

RRF (Reciprocal Rank Fusion) merges both ranked lists without needing score normalization:
`score = 1/(k + rank_dense) + 1/(k + rank_sparse)`, k=60

### Why Cross-encoder Reranking?
Bi-encoder embeddings (used for retrieval) optimize for speed — they can't compare
query and document jointly. Cross-encoders read both together, producing much more
accurate relevance scores. Top-20 → top-5 via ms-marco-MiniLM-L-6-v2 (local, fast).

---

## Agent (LangGraph)

```
classify_intent → [ambiguous?] → ask_clarification
      ↓
generate_hyde → retrieve → rerank → compress → check_confidence
      ↓ [low confidence, retry < 2]                    ↓
   retry with raw query                         generate_answer (GPT-4o)
```

### Nodes
| Node | Model | Purpose |
|------|-------|---------|
| classify_intent | GPT-4o-mini | Detect intent + ambiguity |
| ask_clarification | — | Return question to user if ambiguous |
| generate_hyde | GPT-4o-mini | Hypothetical resolution for better retrieval |
| retrieve | Qdrant | Hybrid search top-20 |
| rerank | Cross-encoder (local) | Top-20 → top-5 |
| compress | GPT-4o-mini ×5 | Extract relevant sentences from each chunk |
| check_confidence | Scoring | Multi-factor: retrieval sim + CSAT + recency + consistency |
| generate_answer | GPT-4o | CoT: evaluate → synthesize → cite → suggest action |

### Confidence Scoring
```
confidence = 0.40 × retrieval_similarity
           + 0.25 × resolution_consistency
           + 0.20 × csat_score (normalized)
           + 0.15 × recency
```
Scores below 0.5 trigger a retry (max 2) with the raw query instead of HyDE embedding.
Displayed in UI as green (≥0.80) / yellow (0.50–0.79) / red (<0.50).

### Multi-turn Memory
LangGraph `MemorySaver` checkpointer keyed by `session_id`. Each query carries full
chat history — agents can ask follow-up questions without repeating context.
In-memory is sufficient for ~20 concurrent users.

---

## Why Not Fine-tuning?

1. **Knowledge is dynamic** — new tickets resolved daily. Fine-tuning bakes knowledge
   into weights; you'd need to retrain constantly.
2. **Citations are impossible** — fine-tuned models can't cite specific ticket IDs.
   RAG retrieval is inherently traceable.
3. **The bottleneck is retrieval, not reasoning** — bad answers come from wrong chunks
   retrieved, not from the LLM's reasoning ability. HyDE + hybrid + reranking addresses
   the real problem.

Fine-tuning a small model for *intent classification only* is a reasonable future
optimization for cost/latency, but not for launch.

---

## Scope

Currently indexed: **voice-related tickets** (filtered by `product = "voice"` at
retrieval time). Expanding to SMS, SIP trunking, and other products is a config change
— no architectural work required.

Future data sources (post-launch):
- Confluence internal docs
- Jira engineering tickets (links support issues to root cause fixes)
- Slack support channel threads
