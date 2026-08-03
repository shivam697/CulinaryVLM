"""LangGraph StateGraph — wires planner → tools → composer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_tool_executors(config: dict) -> dict[str, callable]:
    """Build tool executor map from service functions."""
    from app.services.recipes import load_canonical_recipes, get_recipe, compare_recipes
    from app.services.videos import load_all_videos, get_category_stats

    recipe_dir = PROJECT_ROOT / "configs" / "canonical_recipes"
    recipes = load_canonical_recipes(recipe_dir)

    def recipe_lookup(input_data: dict) -> Any:
        category = input_data.get("category", "")
        if not category:
            return {"categories": sorted(recipes.keys())}
        r = get_recipe(recipes, category)
        return r if r else {"error": f"Recipe not found: {category}"}

    def recipe_compare(input_data: dict) -> Any:
        return compare_recipes(
            recipes,
            input_data.get("category_a", ""),
            input_data.get("category_b", ""),
            input_data.get("aspect"),
        )

    def search_segments(input_data: dict) -> Any:
        return {"message": "FAISS search requires index to be loaded. Start the API server first."}

    def video_metadata(input_data: dict) -> Any:
        return get_category_stats(PROJECT_ROOT)

    def transcript_search(input_data: dict) -> Any:
        return {"message": "Transcript search requires transcripts from Stage 2."}

    def youtube_link(input_data: dict) -> Any:
        vid = input_data.get("video_id", "")
        return {"url": f"https://www.youtube.com/watch?v={vid}", "video_id": vid}

    return {
        "recipe_lookup": recipe_lookup,
        "recipe_compare": recipe_compare,
        "search_segments": search_segments,
        "video_metadata": video_metadata,
        "transcript_search": transcript_search,
        "youtube_link": youtube_link,
    }


async def tool_executor_node(state: dict[str, Any], tool_map: dict) -> dict[str, Any]:
    """Execute planned tools and record traces."""
    tool_results = state.get("tool_results", [])
    traces = []

    for tr in tool_results:
        tool_name = tr.get("tool", "")
        tool_input = tr.get("input", {})

        executor = tool_map.get(tool_name)
        if not executor:
            tr["output"] = {"error": f"Unknown tool: {tool_name}"}
            traces.append({"tool_name": tool_name, "input": tool_input, "output": tr["output"], "duration_ms": 0})
            continue

        start = time.time()
        try:
            result = executor(tool_input)
            tr["output"] = result
        except Exception as e:
            tr["output"] = {"error": str(e)}
        elapsed = (time.time() - start) * 1000

        traces.append({
            "tool_name": tool_name,
            "input": tool_input,
            "output": tr["output"],
            "duration_ms": round(elapsed, 1),
        })

    return {**state, "tool_results": tool_results, "trace": traces}


def build_agent_graph(config: dict):
    """
    Build the LangGraph StateGraph.

    Flow: planner → tool_executor → composer → END
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error("langgraph not installed")
        return None

    from app.agents.state import AgentState
    from app.agents.planner import planner_node
    from app.agents.composer import composer_node

    tool_map = _get_tool_executors(config)

    async def executor(state):
        return await tool_executor_node(state, tool_map)

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", executor)
    graph.add_node("composer", composer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool_executor")
    graph.add_edge("tool_executor", "composer")
    graph.add_edge("composer", END)

    return graph.compile()
