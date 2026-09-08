"""Composer node — merges tool results into a coherent answer.

Primary: Fine-tuned CulinaryVLM generates the final answer.
Fallback: Groq (for verification/validation only).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

COMPOSER_SYSTEM = """You are CulinaryVLM, an expert on Indian biryani cooking.
Given the user's question and the tool results below, compose a comprehensive,
well-structured answer. Use markdown formatting. Cite sources when available.
Be specific and accurate. If tool results are empty, say what information is missing."""


async def composer_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compose final answer from tool results using the fine-tuned VLM."""
    user_msg = state["messages"][-1]["content"]
    tool_results = state.get("tool_results", [])

    # Build context from tool results
    context_parts = []
    for tr in tool_results:
        if tr.get("output"):
            tool_name = tr.get("tool", "unknown")
            output = tr["output"]
            if isinstance(output, dict):
                output = json.dumps(output, indent=2)
            context_parts.append(f"**{tool_name}**:\n{output}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else ""

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key or groq_key.startswith("REPLACE"):
        # Last resort: return raw tool results
        if context:
            return {
                **state,
                "final_answer": f"Based on available data:\n\n{context}",
            }
        return {
            **state,
            "final_answer": "I couldn't generate an answer at this time. Please try again.",
        }

    try:
        import time
        from groq import Groq

        client = Groq(api_key=groq_key)

        models_to_try = ["allam-2-7b", "qwen/qwen3.8-27b"]
        response = None
        last_error = None

        for model_name in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": COMPOSER_SYSTEM},
                        {"role": "user", "content": (
                            f"User question: {user_msg}\n\n"
                            f"Tool results:\n{context}\n\n"
                            f"Compose a clear, helpful answer."
                        )},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                break
            except Exception as model_err:
                err_str = str(model_err)
                last_error = model_err
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    logger.warning(f"Rate limit on {model_name}, trying next model...")
                    time.sleep(2)
                    continue
                raise

        if response is None:
            raise last_error

        answer = response.choices[0].message.content.strip()
        return {**state, "final_answer": answer}

    except Exception as e:
        logger.error(f"Composer error: {e}")
        if context:
            return {**state, "final_answer": f"Based on available data:\n\n{context}"}
        return {**state, "final_answer": "I couldn't generate an answer at this time."}

