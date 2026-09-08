"""Search router — semantic search over segments."""

from fastapi import APIRouter, Request

from app.models.schemas import SearchRequest, SearchResponse, SearchResult

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_segments(request: Request, body: SearchRequest):
    """
    Semantic search over video segments using FAISS.

    Find cooking technique segments similar to the query text.
    """
    faiss_index = request.app.state.faiss_index

    if faiss_index is None:
        return SearchResponse(
            query=body.query,
            total_results=0,
            results=[],
        )

    from app.services.retrieval import encode_query

    query_embedding = encode_query(body.query)
    raw_results = faiss_index.search(query_embedding, top_k=body.top_k)

    # Filter by category if specified
    if body.category:
        raw_results = [
            r for r in raw_results
            if r.get("category", "").lower() == body.category.lower()
        ]

    results = [
        SearchResult(
            segment_id=r.get("segment_id", ""),
            video_id=r.get("video_id", ""),
            category=r.get("category", ""),
            action=r.get("action", ""),
            description=r.get("description", ""),
            score=r.get("score", 0.0),
            start_time=r.get("start_time", 0.0),
            end_time=r.get("end_time", 0.0),
            video_title=r.get("video_title", ""),
            video_url=r.get("video_url", ""),
        )
        for r in raw_results
    ]

    return SearchResponse(
        query=body.query,
        total_results=len(results),
        results=results,
    )

