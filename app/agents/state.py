"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """LangGraph state for the CulinaryVLM agent."""

    # Conversation
    messages: list[dict[str, str]]  # [{"role": "user"/"assistant", "content": "..."}]
    session_id: str

    # Planning
    plan: str  # Agent's reasoning about which tools to call

    # Execution
    tool_results: list[dict[str, Any]]  # Results from tool calls
    needs_vlm: bool  # Whether VLM inference is needed

    # Output
    final_answer: str  # Composed final answer
    trace: list[dict[str, Any]]  # Tool execution trace for transparency
