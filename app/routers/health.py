"""Health check router."""

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        faiss_loaded=request.app.state.faiss_index is not None,
        recipes_loaded=len(request.app.state.recipes),
        agent_enabled=request.app.state.agent_graph is not None,
    )
