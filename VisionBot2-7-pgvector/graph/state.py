"""LangGraph shared state schema."""

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    question: str
    session_id: str
    chat_history: list[dict]        # [{"role": "user"|"assistant", "content": "..."}]
    metadata_filters: dict[str, str]  # Optional: {"product": "voice_api", "region": "US"}

    # Reasoning
    intent: str                     # troubleshoot | explain | find-similar | summarize
    query_type: str                 # ticket_search | product_question
    is_voice_related: bool          # False = off-topic, short-circuit before retrieval
    is_ambiguous: bool
    needs_clarification: bool       # True when query is clear but missing diagnostic details
    clarification_question: str     # Question to ask user when ambiguous or needs_clarification
    awaiting_clarification: bool    # True when graph is paused for user input
    retry_count: int                # Prevent infinite retry loop (max 2)

    # Pre-diagnosis
    pre_diagnosis: Optional[dict]   # Structured signals extracted before retrieval
                                    # Keys: error_code, failure_point, direction, onset_pattern, product_area

    # Retrieval
    hyde_text: str                  # Hypothetical resolution text
    hyde_dense: list[float]         # Embedding of hyde_text
    hyde_sparse: dict               # BM25 sparse vector
    retrieved_chunks: list[dict]    # Top 20 from hybrid search
    reranked_chunks: list[dict]     # Top 5 after cross-encoder
    compressed_chunks: list[dict]   # Top 5 with compressed_text

    # Confidence
    confidence_score: float         # 0.0 - 1.0
    confidence_factors: dict        # Breakdown for display

    # Live ticket injection (when user pastes a Zendesk URL)
    injected_context: Optional[str]     # Full ticket thread fetched live from Zendesk
    live_ticket_id: Optional[str]       # ID of the live ticket being discussed — excluded from retrieval results
    active_ticket_id: Optional[str]     # Ticket being discussed in this session — set by
                                        # Slack bot, persists across all turns in the thread
    url_only_ticket: Optional[bool]     # True when agent pasted URL with no question — ask what they want
    wants_log_investigation: Optional[bool]  # True when agent wants CDR/log analysis on specific UUIDs

    # Output
    answer: str
    reasoning: str
    citations: list[dict]
    suggested_action: str
    related_tickets: list[dict]     # Cluster siblings
    error: Optional[str]

    # Observability
    timings: dict[str, int]         # Step name → elapsed ms
    cost_entries: list[dict]        # Per-step LLM cost entries (make_entry dicts)
