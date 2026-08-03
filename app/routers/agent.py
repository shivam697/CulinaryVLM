"""Agent router — LangGraph-powered multi-tool reasoning (optional)."""

from __future__ import annotations

import uuid
import logging

from fastapi import APIRouter, Request, HTTPException

from app.models.schemas import AgentRequest, AgentResponse, ToolTrace

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agent/query", response_model=AgentResponse)
async def agent_query(request: Request, body: AgentRequest):
    """
    Send a natural language query to the LangGraph agent.

    The agent plans which tools to call, executes them,
    and composes a coherent answer with source attribution.

    Only available when USE_AGENT_LAYER=true and agent deps are installed.
    """
    graph = request.app.state.agent_graph

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent layer is not enabled. "
                "Set USE_AGENT_LAYER=true in .env and install requirements-agent.txt"
            ),
        )

    session_id = body.session_id or str(uuid.uuid4())

    try:
        # Run the LangGraph StateGraph
        initial_state = {
            "messages": [{"role": "user", "content": body.message}],
            "session_id": session_id,
            "plan": "",
            "tool_results": [],
            "needs_vlm": False,
            "final_answer": "",
            "trace": [],
        }

        final_state = await graph.ainvoke(initial_state)

        return AgentResponse(
            answer=final_state.get("final_answer", "I could not generate an answer."),
            session_id=session_id,
            tool_traces=[
                ToolTrace(
                    tool_name=t.get("tool_name", ""),
                    input=t.get("input", {}),
                    output=t.get("output"),
                    duration_ms=t.get("duration_ms", 0),
                )
                for t in final_state.get("trace", [])
            ],
            plan=final_state.get("plan", ""),
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
