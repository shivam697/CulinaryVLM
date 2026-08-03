"""Video metadata service."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_video_selection(project_root: Path) -> list[dict[str, Any]]:
    """Load pipeline video selection."""
    path = project_root / "datasets" / "categorized" / "pipeline_selection.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos", [])


def load_all_videos(project_root: Path) -> list[dict[str, Any]]:
    """Load all video metadata (enriched)."""
    path = project_root / "datasets" / "categorized" / "video_metadata_enriched.json"
    if not path.exists():
        # Fall back to basic metadata
        path = project_root / "datasets" / "categorized" / "video_metadata.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("videos", [])


def get_video_by_id(
    project_root: Path,
    video_id: str,
) -> dict[str, Any] | None:
    """Get a single video by ID."""
    videos = load_all_videos(project_root)
    for v in videos:
        if v.get("video_id") == video_id:
            return v
    return None


def get_videos_by_category(
    project_root: Path,
    category: str,
) -> list[dict[str, Any]]:
    """Get all videos in a category."""
    videos = load_all_videos(project_root)
    return [v for v in videos if v.get("assigned_category", "").lower() == category.lower()]


def get_category_stats(project_root: Path) -> dict[str, Any]:
    """Get category distribution stats."""
    videos = load_all_videos(project_root)
    from collections import Counter
    cats = Counter(v.get("assigned_category", "Unknown") for v in videos)
    langs = Counter(v.get("estimated_narration_language", "unknown") for v in videos)
    confs = Counter(v.get("confidence", "Unknown") for v in videos)

    return {
        "total_videos": len(videos),
        "categories": dict(cats.most_common()),
        "languages": dict(langs.most_common()),
        "confidence_levels": dict(confs.most_common()),
    }
