"""Videos router — browse and filter videos."""

from fastapi import APIRouter, Query, Request
from typing import Optional

from app.models.schemas import VideoListResponse, VideoSummary
from app.services.videos import load_all_videos, get_video_by_id, get_category_stats

router = APIRouter()


@router.get("/videos", response_model=VideoListResponse)
async def list_videos(
    request: Request,
    category: Optional[str] = Query(None, description="Filter by category"),
    language: Optional[str] = Query(None, description="Filter by language code"),
    confidence: Optional[str] = Query(None, description="Filter by confidence level"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List videos with optional filters."""
    videos = load_all_videos(request.app.state.project_root)

    # Apply filters
    if category:
        videos = [v for v in videos if v.get("assigned_category", "").lower() == category.lower()]
    if language:
        videos = [v for v in videos if v.get("estimated_narration_language", "") == language]
    if confidence:
        videos = [v for v in videos if v.get("confidence", "").lower() == confidence.lower()]

    total = len(videos)
    paged = videos[offset:offset + limit]

    return VideoListResponse(
        total=total,
        videos=[
            VideoSummary(
                video_id=v["video_id"],
                title=v["title"],
                url=v["url"],
                category=v.get("assigned_category", "Unknown"),
                confidence=v.get("confidence", "Unknown"),
                estimated_language=v.get("estimated_narration_language", "unknown"),
                priority=v.get("pipeline_priority", 3),
            )
            for v in paged
        ],
    )


@router.get("/videos/{video_id}")
async def get_video(request: Request, video_id: str):
    """Get a single video by ID."""
    video = get_video_by_id(request.app.state.project_root, video_id)
    if not video:
        return {"error": f"Video not found: {video_id}"}
    return video


@router.get("/videos/stats/overview")
async def video_stats(request: Request):
    """Get video dataset statistics."""
    return get_category_stats(request.app.state.project_root)
