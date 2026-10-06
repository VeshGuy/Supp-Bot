"""
LangGraph agent graph definition.
Wires all nodes together with conditional edges.
"""

import logging
from functools import partial

import psycopg2.pool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI

from graph.state import AgentState
from graph.nodes import (
    fetch_ticket_context_node,
    classify_intent,
    conversational_reply_node,
    generate_hyde_node,
    retrieve_node,
    rerank_node,
    compress_node,
    check_confidence_node,
    generate_answer_node,
)

logger = logging.getLogger(__name__)


def build_graph(
    openai_client: AsyncOpenAI,
    pg_pool: psycopg2.pool.ThreadedConnectionPool,
    embedder,
    settings,
) -> StateGraph:
    """Build and compile the LangGraph agent."""

    # ── Wrap nodes with dependencies ─────────────────────────────────────────

    async def node_fetch_ticket(state):
        return await fetch_ticket_context_node(state, settings=settings)

    async def node_classify(state):
        return await classify_intent(
            state,
            client=openai_client,
            model=settings.llm_mini_model,  # fast, reliable routing decision only
        )

    async def node_hyde(state):
        return await generate_hyde_node(
            state,
            client=openai_client,
            embedder=embedder,
            model=settings.llm_hyde_model,  # capable model for hypothetical generation
        )

    def node_retrieve(state):
        return retrieve_node(state, pg_pool=pg_pool, top_k=settings.top_k_retrieve)

    def node_rerank(state):
        return rerank_node(state, top_n=settings.top_k_rerank)

    async def node_compress(state):
        return await compress_node(
            state,
            client=openai_client,
            model=settings.llm_mini_model,
            pg_pool=pg_pool,
        )

    def node_confidence(state):
        return check_confidence_node(state, threshold=settings.confidence_threshold)

    async def node_answer(state):
        return await generate_answer_node(
            state,
            client=openai_client,
            model=settings.llm_model,
            pg_pool=pg_pool,
        )

    async def node_conversational(state):
        return await conversational_reply_node(
            state,
            client=openai_client,
            model=settings.llm_model,
        )

    def node_reject_off_topic(state):
        """Short-circuit for non-Plivo queries — no retrieval, no LLM cost."""
        return {
            "answer": "I can only help with Plivo product questions — Voice API (call failures, SIP, WebRTC, audio, DTMF), SMS/Messaging (delivery failures, 10DLC, DLT, sender IDs), Phone Numbers, and SIP Trunking. What Plivo issue can I help you with?",
            "citations": [],
            "suggested_action": "",
            "related_tickets": [],
            "confidence_score": 0.0,
            "confidence_factors": {},
        }

    def node_ask_clarification(state):
        """Return the clarification question — graph pauses here for user input."""
        question = state.get("clarification_question", "Could you provide more details?")
        # Append to chat_history so the next turn sees non-empty history.
        # Without this, route_after_classify sees no_history=True every turn and
        # keeps asking clarification forever even after the user answers.
        history = list(state.get("chat_history") or [])
        history.append({"role": "user", "content": state.get("question", "")})
        history.append({"role": "assistant", "content": question})
        return {
            "answer": question,
            "awaiting_clarification": True,
            "chat_history": history,
            "citations": [],
            "suggested_action": "",
            "related_tickets": [],
        }

    # ── Routing functions ─────────────────────────────────────────────────────

    def node_fetch_error(state):
        """Surface Zendesk fetch failures to the agent instead of silently continuing."""
        return {
            "answer": state.get("error", "⚠️ Could not load the ticket. Please check the URL and try again."),
            "citations": [],
            "suggested_action": "",
            "related_tickets": [],
            "confidence_score": 0.0,
            "confidence_factors": {},
        }

    def route_after_classify(state) -> str:
        # Zendesk fetch failed — surface the error directly instead of giving a generic answer
        if state.get("error"):
            return "fetch_error"
        # Live ticket injected — agent is asking about their specific ticket.
        # Always proceed regardless of is_voice_related classification.
        if state.get("injected_context"):
            if state.get("url_only_ticket"):
                return "ask_clarification"
            if state.get("query_type") == "conversational" or state.get("intent") == "conversational":
                return "conversational_reply"
            return "generate_hyde"
        if not state.get("is_voice_related", True):
            return "reject_off_topic"
        # Conversational follow-ups skip the full RAG pipeline entirely
        if state.get("query_type") == "conversational" or state.get("intent") == "conversational":
            return "conversational_reply"
        # Ask for clarification on first turn only — if chat_history exists the agent
        # has already responded to a clarification request, so proceed with retrieval
        no_history = not state.get("chat_history")
        if (state.get("is_ambiguous") or state.get("needs_clarification")) and no_history:
            return "ask_clarification"
        return "generate_hyde"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)

    graph.add_node("fetch_ticket_context", node_fetch_ticket)
    graph.add_node("fetch_error", node_fetch_error)
    graph.add_node("classify_intent", node_classify)
    graph.add_node("conversational_reply", node_conversational)
    graph.add_node("reject_off_topic", node_reject_off_topic)
    graph.add_node("ask_clarification", node_ask_clarification)
    graph.add_node("generate_hyde", node_hyde)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("compress", node_compress)
    graph.add_node("check_confidence", node_confidence)
    graph.add_node("generate_answer", node_answer)

    # Entry: always check for Zendesk URL/ID first (no-op if none present)
    graph.set_entry_point("fetch_ticket_context")
    graph.add_edge("fetch_ticket_context", "classify_intent")

    # Edges
    graph.add_conditional_edges("classify_intent", route_after_classify, {
        "fetch_error": "fetch_error",
        "reject_off_topic": "reject_off_topic",
        "conversational_reply": "conversational_reply",
        "ask_clarification": "ask_clarification",
        "generate_hyde": "generate_hyde",
    })
    graph.add_edge("fetch_error", END)
    graph.add_edge("reject_off_topic", END)
    graph.add_edge("conversational_reply", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("generate_hyde", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "check_confidence")
    graph.add_edge("check_confidence", "generate_answer")  # Retry loop removed
    graph.add_edge("generate_answer", END)

    # Compile with in-memory checkpointer (sufficient for 20 users)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
