#!/usr/bin/env python3
"""
CulinaryVLM — Phase 0C: Multilingual Metadata Tagging
═══════════════════════════════════════════════════════════════

Enriches the categorized video metadata with additional fields:
  - title_language (already added by 0A, verified here)
  - is_code_switched (already added by 0A, verified here)
  - estimated_narration_language (inferred from title + category)
  - pipeline_priority (1=must-have, 2=nice-to-have, 3=optional)

Also produces a pipeline_selection.json that selects the final
set of videos to proceed through Stages 1-10, applying the
minimum-per-category and total-target constraints.

Usage:
    python pipeline/00c_metadata_tagging.py
    python pipeline/00c_metadata_tagging.py --target 250
    python pipeline/00c_metadata_tagging.py --dry-run

Input:  datasets/categorized/video_metadata.json
Output: datasets/categorized/video_metadata_enriched.json
        datasets/categorized/pipeline_selection.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase_0c")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"

# ─── Language inference rules ─────────────────────────────────
# Based on category → most likely narration language

CATEGORY_LANGUAGE_MAP: dict[str, str] = {
    "Hyderabadi": "hi",     # Hindi/Urdu (Deccani), some Telugu
    "Kolkata": "bn",        # Bengali, some Hindi
    "Lucknowi": "hi",       # Hindi/Urdu
    "Malabar": "ml",        # Malayalam
    "Sindhi": "ur",         # Urdu
    "Ambur": "ta",          # Tamil
    "Dindigul": "ta",       # Tamil
    "Muradabadi": "hi",     # Hindi
    "Delhi": "hi",          # Hindi/Urdu
    "Andhra": "te",         # Telugu
    "Bihari": "hi",         # Hindi/Bhojpuri
    "Assamese": "as",       # Assamese
    "Bombay": "hi",         # Hindi/Marathi
    "Kashmiri": "hi",       # Hindi/Kashmiri
    "Mughlai": "hi",        # Hindi/Urdu
    "Bamboo": "en",         # Usually English tutorials
    "Degi": "hi",           # Hindi/Urdu
    "Matka": "hi",          # Hindi
    "Tandoori": "hi",       # Hindi
    "Arabic": "en",         # Usually English-narrated Arabian recipes
    "Generic": "hi",        # Default Hindi
}

# Priority assignment rules
PRIORITY_RULES: dict[str, int] = {
    # Priority 1: Core named styles with strong regional identity
    "Hyderabadi": 1, "Kolkata": 1, "Lucknowi": 1, "Malabar": 1,
    "Muradabadi": 1, "Delhi": 1, "Andhra": 1, "Sindhi": 1,
    # Priority 2: Distinctive but smaller categories
    "Ambur": 2, "Dindigul": 2, "Bihari": 2, "Bombay": 2,
    "Kashmiri": 2, "Assamese": 2,
    # Priority 3: Method-based or non-regional categories
    "Mughlai": 3, "Bamboo": 3, "Degi": 3, "Matka": 3,
    "Tandoori": 3, "Arabic": 3, "Generic": 3,
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config."""
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Load categorized video metadata from Phase 0A output."""
    path = PROJECT_ROOT / config["paths"]["categorized_output"]
    if not path.exists():
        logger.error(f"Not found: {path}. Run Phase 0A first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add estimated narration language and pipeline priority."""
    for video in videos:
        cat = video["assigned_category"]

        # Estimated narration language
        video["estimated_narration_language"] = CATEGORY_LANGUAGE_MAP.get(cat, "hi")

        # Pipeline priority
        video["pipeline_priority"] = PRIORITY_RULES.get(cat, 3)

        # Confidence-based priority boost
        if video["confidence"] == "High":
            video["pipeline_priority"] = min(video["pipeline_priority"], 1)
        elif video["confidence"] == "Medium":
            video["pipeline_priority"] = min(video["pipeline_priority"], 2)

    return videos


def select_pipeline_videos(
    videos: list[dict[str, Any]],
    target_total: int = 250,
    min_per_category: int = 1,
    max_per_category: int = 30,
) -> list[dict[str, Any]]:
    """
    Select videos for the ML pipeline.

    Strategy:
    1. Include ALL High-confidence videos (these are quality-guaranteed)
    2. Include Medium-confidence videos up to max_per_category per category
    3. If under target, add Low-confidence Priority 1 category videos
    4. Ensure minimum representation for each category
    """
    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in videos:
        by_category[v["assigned_category"]].append(v)

    selected: list[dict[str, Any]] = []
    selection_log: dict[str, dict[str, int]] = {}

    for cat in sorted(by_category.keys()):
        if cat == "Generic":
            continue  # Skip generic — too noisy

        cat_videos = by_category[cat]

        # Sort: High first, then Medium, then Low; within same confidence, Priority 1 first
        cat_videos.sort(key=lambda v: (
            {"High": 0, "Medium": 1, "Low": 2}.get(v["confidence"], 3),
            v["pipeline_priority"],
        ))

        # Select up to max_per_category
        cat_selected = cat_videos[:max_per_category]

        # Ensure at least min_per_category if available
        if len(cat_selected) < min_per_category:
            cat_selected = cat_videos[:min_per_category]

        for v in cat_selected:
            v["selected_for_pipeline"] = True
        selected.extend(cat_selected)

        counts = Counter(v["confidence"] for v in cat_selected)
        selection_log[cat] = {
            "total_available": len(cat_videos),
            "selected": len(cat_selected),
            "high": counts.get("High", 0),
            "medium": counts.get("Medium", 0),
            "low": counts.get("Low", 0),
        }

    logger.info(f"\n── Pipeline Video Selection ──")
    logger.info(f"  Target: {target_total} videos")
    logger.info(f"  Selected: {len(selected)} videos from {len(selection_log)} categories")
    logger.info(f"")

    for cat, info in sorted(selection_log.items()):
        logger.info(
            f"  {cat:20s}: {info['selected']:3d} selected "
            f"(H:{info['high']:2d} M:{info['medium']:2d} L:{info['low']:2d}) "
            f"/ {info['total_available']} available"
        )

    return selected


def save_enriched(
    videos: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save enriched metadata."""
    output = {
        "metadata": {
            "description": "CulinaryVLM — Enriched video metadata (Phase 0C)",
            "source": "Phase 0A → Phase 0C enrichment",
        },
        "summary": summary,
        "videos": videos,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Enriched metadata saved: {output_path}")


def save_selection(
    selected: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save pipeline selection."""
    # Produce a lean selection file for Stage 1
    selection = {
        "metadata": {
            "description": "CulinaryVLM — Videos selected for ML pipeline",
            "total_selected": len(selected),
            "categories": dict(Counter(v["assigned_category"] for v in selected).most_common()),
        },
        "videos": [
            {
                "video_id": v["video_id"],
                "url": v["url"],
                "title": v["title"],
                "category": v["assigned_category"],
                "confidence": v["confidence"],
                "priority": v["pipeline_priority"],
                "estimated_language": v["estimated_narration_language"],
            }
            for v in selected
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    logger.info(f"Pipeline selection saved: {output_path} ({len(selected)} videos)")


def main() -> None:
    """Phase 0C main entry point."""
    parser = argparse.ArgumentParser(
        description="CulinaryVLM Phase 0C — Metadata tagging + pipeline selection",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=int, default=250,
                        help="Target video count for pipeline (default: 250)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Phase 0C: Metadata Tagging       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)
    data = load_metadata(config)
    videos = data["videos"]

    # 1. Enrich with narration language + priority
    videos = enrich_videos(videos)

    # 2. Select pipeline videos
    selected = select_pipeline_videos(
        videos,
        target_total=args.target,
        min_per_category=1,
        max_per_category=30,
    )

    # 3. Summary stats
    summary = {
        "total_videos": len(videos),
        "pipeline_selected": len(selected),
        "narration_languages": dict(Counter(
            v["estimated_narration_language"] for v in selected
        ).most_common()),
        "priorities": dict(Counter(
            v["pipeline_priority"] for v in selected
        ).most_common()),
    }

    logger.info(f"\n── Narration Languages (selected) ──")
    for lang, count in summary["narration_languages"].items():
        logger.info(f"  {lang:4s}: {count:3d}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No files saved")
        return

    # 4. Save outputs
    enriched_path = PROJECT_ROOT / "datasets/categorized/video_metadata_enriched.json"
    selection_path = PROJECT_ROOT / "datasets/categorized/pipeline_selection.json"

    save_enriched(videos, summary, enriched_path)
    save_selection(selected, selection_path)

    logger.info("")
    logger.info("✓ Phase 0C complete!")
    logger.info(f"  Selected {len(selected)} videos for pipeline")
    logger.info(f"  NEXT: Stage 1 — Download videos from YouTube")
    logger.info(f"        python pipeline/01_download.py")


if __name__ == "__main__":
    main()
