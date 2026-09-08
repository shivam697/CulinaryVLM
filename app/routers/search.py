"""Search router — keyword search over segments (no ML model required)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request

from app.models.schemas import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SEGMENTS_CACHE: list[dict] | None = None


def _load_segments() -> list[dict]:
    """Load segments metadata once and cache in memory."""
    global _SEGMENTS_CACHE
    if _SEGMENTS_CACHE is not None:
        return _SEGMENTS_CACHE

    pkl_path = PROJECT_ROOT / "datasets" / "faiss" / "segments_metadata.pkl"
    if not pkl_path.exists():
        logger.warning("segments_metadata.pkl not found — search unavailable")
        _SEGMENTS_CACHE = []
        return _SEGMENTS_CACHE

    try:
        with open(pkl_path, "rb") as f:
            _SEGMENTS_CACHE = pickle.load(f)
        logger.info(f"Loaded {len(_SEGMENTS_CACHE)} segments for keyword search")
    except Exception as e:
        logger.error(f"Failed to load segments: {e}")
        _SEGMENTS_CACHE = []

    return _SEGMENTS_CACHE


def _keyword_score(segment: dict, query_tokens: list[str]) -> float:
    """Score a segment by how many query tokens appear in its text fields."""
    def to_str(v) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v) if v else ""

    text = " ".join([
        to_str(segment.get("action")),
        to_str(segment.get("description")),
        to_str(segment.get("cooking_technique")),
        to_str(segment.get("cooking_stage")),
        to_str(segment.get("category")),
        to_str(segment.get("canonical_action")),
    ]).lower()

    if not text.strip():
        return 0.0

    matches = sum(1 for token in query_tokens if token in text)
    phrase_boost = 2.0 if " ".join(query_tokens) in text else 0.0
    return (matches + phrase_boost) / (len(query_tokens) + 1)


@router.post("/search", response_model=SearchResponse)
async def search_segments(request: Request, body: SearchRequest):
    """
    Keyword search over video segments.
    Searches action, description, technique and category fields directly.
    """
    segments = _load_segments()

    if not segments:
        return SearchResponse(query=body.query, total_results=0, results=[])

    query_tokens = body.query.lower().split()
    if not query_tokens:
        return SearchResponse(query=body.query, total_results=0, results=[])

    # Score all segments
    scored = []
    for seg in segments:
        # Apply category filter first if specified
        if body.category and seg.get("category", "").lower() != body.category.lower():
            continue
        score = _keyword_score(seg, query_tokens)
        if score > 0:
            scored.append((score, seg))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: body.top_k]

    results = [
        SearchResult(
            segment_id=seg.get("segment_id", ""),
            video_id=seg.get("video_id", ""),
            category=seg.get("category", ""),
            action=seg.get("action", ""),
            description=seg.get("description", ""),
            score=round(score, 3),
            start_time=seg.get("start_time", 0.0),
            end_time=seg.get("end_time", 0.0),
            video_title=seg.get("video_title", ""),
            video_url=seg.get("video_url", ""),
        )
        for score, seg in top
    ]

    return SearchResponse(
        query=body.query,
        total_results=len(results),
        results=results,
    )
