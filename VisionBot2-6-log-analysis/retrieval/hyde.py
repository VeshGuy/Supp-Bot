"""
HyDE — Hypothetical Document Embedding.
GPT-4o-mini generates a hypothetical resolution for the query.
We embed the hypothetical answer (not the raw query) for retrieval.
This matches answer-space to answer-space in the embedding.
"""

import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior technical support engineer.
Given a support question, write a concise hypothetical resolution — exactly as it would
appear in the resolution notes of a solved support ticket.

Write 2-3 sentences maximum. Be technical and specific.
Do NOT say "I" or address the user. Write as if documenting the resolution.
Example format: "The issue was caused by X. Resolution: Y. Suggested action: Z." """


async def generate_hyde(client: AsyncOpenAI, question: str, model: str = "gpt-4o-mini") -> str:
    """Generate a hypothetical ticket resolution for the given question."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        hypothetical = response.choices[0].message.content.strip()
        logger.info(f"HyDE generated: {hypothetical[:80]}...")
        return hypothetical
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e} — falling back to raw query")
        return question
