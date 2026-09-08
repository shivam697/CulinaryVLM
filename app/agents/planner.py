"""Planner node — decides which tools to call."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


PLANNER_SYSTEM = """You are CulinaryVLM, an AI assistant specializing in Indian biryani cooking.
You have access to these tools:
1. recipe_lookup — Get canonical recipe for a biryani style
2. recipe_compare — Compare two biryani styles
3. search_segments — Semantic search over video segments
4. video_metadata — Get video information and stats
5. transcript_search — Search ASR transcripts
6. youtube_link — Get YouTube video link

Given the user's question, decide which tools to call and in what order.
Output a JSON object with:
- "plan": brief reasoning (1-2 sentences)
- "tools": array of {"tool": "tool_name", "input": {key: value}}
- "needs_vlm": boolean (true only if the question requires analyzing video frames)

Only use tools that are relevant. For simple recipe questions, recipe_lookup is sufficient.
For comparisons, use recipe_compare. For finding specific cooking techniques, use search_segments."""


async def planner_node(state: dict[str, Any]) -> dict[str, Any]:
    """Plan which tools to call based on user message."""
    user_msg = state["messages"][-1]["content"]

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key or groq_key.startswith("REPLACE"):
        return {
            **state,
            "plan": "No API key — using rule-based routing",
            "tool_results": [],
            "needs_vlm": False,
        }

    try:
        from groq import Groq

        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        plan_text = response.choices[0].message.content.strip()
        plan = json.loads(plan_text)

        return {
            **state,
            "plan": plan.get("plan", ""),
            "tool_results": [{"tool": t["tool"], "input": t.get("input", {}), "output": None}
                            for t in plan.get("tools", [])],
            "needs_vlm": plan.get("needs_vlm", False),
        }

    except Exception as e:
        logger.error(f"Planner error: {e}")
        # Fallback: simple keyword routing
        tools = []
        lower = user_msg.lower()

        if "compare" in lower or " vs " in lower:
            tools.append({"tool": "recipe_compare", "input": {}, "output": None})
        elif any(w in lower for w in ["recipe", "spice", "step", "how", "make", "cook"]):
            tools.append({"tool": "recipe_lookup", "input": {}, "output": None})
        elif any(w in lower for w in ["search", "find", "segment", "technique"]):
            tools.append({"tool": "search_segments", "input": {}, "output": None})
        else:
            tools.append({"tool": "recipe_lookup", "input": {}, "output": None})

        return {
            **state,
            "plan": f"Fallback routing (planner error: {e})",
            "tool_results": tools,
            "needs_vlm": False,
        }
