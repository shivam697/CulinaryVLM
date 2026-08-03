#!/usr/bin/env python3
"""
CulinaryVLM — Phase 0A: Sub-categorize 474 "Other" videos
═══════════════════════════════════════════════════════════════

Reads Chicken_Biryani_Classified.xlsx, parses all 573 videos,
applies multi-tier NLP keyword classification to the 474
uncategorized "Other" videos, and outputs a unified JSON
with every video assigned a category + confidence.

Usage:
    python pipeline/00a_categorize.py
    python pipeline/00a_categorize.py --config configs/config.yaml
    python pipeline/00a_categorize.py --dry-run

Input:  Chicken_Biryani_Classified.xlsx
Output: datasets/categorized/video_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import openpyxl
import yaml

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase_0a")

# ─── Constants ────────────────────────────────────────────────

DEFAULT_CONFIG = "configs/config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    path = PROJECT_ROOT / config_path
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use last 11 chars if URL structure is unusual
    return url.strip().split("=")[-1][:11] if url else "unknown"


def detect_title_language(title: str) -> str:
    """
    Detect the primary script/language of a video title.
    Returns one of: 'hindi', 'telugu', 'bengali', 'malayalam',
    'urdu', 'english', 'mixed', 'unknown'.
    """
    script_counts: dict[str, int] = {
        "devanagari": 0,
        "telugu": 0,
        "bengali": 0,
        "malayalam": 0,
        "arabic": 0,  # covers Urdu
        "latin": 0,
    }

    for char in title:
        try:
            name = unicodedata.name(char, "").upper()
        except ValueError:
            continue

        if "DEVANAGARI" in name:
            script_counts["devanagari"] += 1
        elif "TELUGU" in name:
            script_counts["telugu"] += 1
        elif "BENGALI" in name:
            script_counts["bengali"] += 1
        elif "MALAYALAM" in name:
            script_counts["malayalam"] += 1
        elif "ARABIC" in name:
            script_counts["arabic"] += 1
        elif "LATIN" in name:
            script_counts["latin"] += 1

    total_script_chars = sum(script_counts.values())
    if total_script_chars == 0:
        return "unknown"

    # Find dominant non-Latin script
    non_latin = {k: v for k, v in script_counts.items() if k != "latin" and v > 0}

    if not non_latin:
        return "english"

    dominant = max(non_latin, key=non_latin.get)  # type: ignore[arg-type]
    dominant_count = non_latin[dominant]

    # If non-Latin script is present with Latin, it's code-switched/mixed
    if script_counts["latin"] > 0 and dominant_count > 0:
        # Map script names to language names
        script_to_lang = {
            "devanagari": "hindi",
            "telugu": "telugu",
            "bengali": "bengali",
            "malayalam": "malayalam",
            "arabic": "urdu",
        }
        return script_to_lang.get(dominant, "mixed")

    script_to_lang = {
        "devanagari": "hindi",
        "telugu": "telugu",
        "bengali": "bengali",
        "malayalam": "malayalam",
        "arabic": "urdu",
    }
    return script_to_lang.get(dominant, "unknown")


def is_code_switched(title: str) -> bool:
    """Check if a title mixes scripts (e.g., Hindi + English)."""
    scripts_found: set[str] = set()
    for char in title:
        try:
            name = unicodedata.name(char, "").upper()
        except ValueError:
            continue
        for script in ["DEVANAGARI", "TELUGU", "BENGALI", "MALAYALAM", "ARABIC", "LATIN"]:
            if script in name:
                scripts_found.add(script)
                break

    # Code-switched if we see 2+ different scripts
    return len(scripts_found) >= 2


def parse_excel(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse the Excel file and return a list of video records.
    Each record has: title, url, original_category, original_confidence.
    """
    cat_config = config["categorization"]
    excel_path = PROJECT_ROOT / config["paths"]["dataset_excel"]

    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        sys.exit(1)

    logger.info(f"Loading Excel: {excel_path}")
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb[cat_config["excel_sheet_name"]]

    videos: list[dict[str, Any]] = []
    sections = cat_config["named_sections"]

    for section in sections:
        section_name = section["name"]
        start = section["start_row"]
        end = section["end_row"]

        count = 0
        for row in ws.iter_rows(min_row=start, max_row=end, values_only=True):
            # row: (#, Recipe Name, YouTube URL, Confidence Score)
            title = row[1]
            url = row[2]
            confidence = row[3]

            if not url or not isinstance(url, str) or "youtube.com" not in url:
                continue

            videos.append({
                "title": str(title).strip() if title else "",
                "url": url.strip(),
                "original_category": section_name,
                "original_confidence": str(confidence).strip() if confidence else "Low",
            })
            count += 1

        logger.info(f"  Section '{section_name}': {count} videos parsed")

    wb.close()
    logger.info(f"Total videos parsed: {len(videos)}")
    return videos


def categorize_other_videos(
    videos: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Apply multi-tier NLP keyword classification to "Other" videos.

    Tier 1 — Regional keywords (→ High confidence)
    Tier 2 — Cooking method keywords (→ Medium confidence)
    Tier 3 — Script-based language inference (→ Medium confidence)
    Tier 4 — No match (→ "Generic", Low confidence)
    """
    cat_config = config["categorization"]
    keyword_cats = cat_config["keyword_categories"]
    lang_patterns = cat_config.get("language_patterns", {})

    stats: dict[str, int] = {}

    for video in videos:
        if video["original_category"] != "Other":
            # Already in a named section — keep as-is
            video["assigned_category"] = video["original_category"]
            video["confidence"] = video["original_confidence"]
            video["keywords_matched"] = []
            video["categorization_tier"] = "original"
            stats[video["original_category"]] = stats.get(video["original_category"], 0) + 1
            continue

        title_lower = video["title"].lower()
        matched = False
        keywords_found: list[str] = []

        # ── Tier 1 & 2: Keyword matching (priority order) ────
        for kw_cat in keyword_cats:
            category = kw_cat["category"]
            confidence = kw_cat["confidence"]
            patterns = kw_cat["patterns"]

            for pattern in patterns:
                try:
                    if re.search(pattern, title_lower):
                        video["assigned_category"] = category
                        video["confidence"] = confidence
                        keywords_found.append(pattern)
                        matched = True
                        tier = "tier1_keyword" if confidence == "High" else "tier2_method"
                        video["categorization_tier"] = tier
                        break
                except re.error:
                    # Invalid regex — try as literal
                    if pattern in title_lower:
                        video["assigned_category"] = category
                        video["confidence"] = confidence
                        keywords_found.append(pattern)
                        matched = True
                        video["categorization_tier"] = "tier1_keyword"
                        break

            if matched:
                break

        # ── Tier 3: Script-based language detection ───────────
        if not matched:
            title_lang = detect_title_language(video["title"])

            for lang_key, lang_config in lang_patterns.items():
                inferred = lang_config.get("inferred_category")
                if inferred is None:
                    continue

                # Check if detected language matches this pattern
                lang_map = {
                    "telugu": "telugu",
                    "bengali": "bengali",
                    "malayalam": "malayalam",
                    "arabic_urdu": "urdu",
                }
                expected_lang = lang_map.get(lang_key)
                if expected_lang and title_lang == expected_lang:
                    video["assigned_category"] = inferred
                    video["confidence"] = "Medium"
                    keywords_found.append(f"script:{lang_key}")
                    video["categorization_tier"] = "tier3_script"
                    matched = True
                    break

        # ── Tier 4: No match → Generic ───────────────────────
        if not matched:
            video["assigned_category"] = "Generic"
            video["confidence"] = "Low"
            video["categorization_tier"] = "tier4_generic"

        video["keywords_matched"] = keywords_found
        stats[video["assigned_category"]] = stats.get(video["assigned_category"], 0) + 1

    # Print categorization summary
    logger.info("═" * 60)
    logger.info("CATEGORIZATION SUMMARY")
    logger.info("═" * 60)
    for cat in sorted(stats.keys()):
        logger.info(f"  {cat:20s}: {stats[cat]:4d} videos")
    logger.info(f"  {'TOTAL':20s}: {sum(stats.values()):4d} videos")
    logger.info("═" * 60)

    return videos


def enrich_metadata(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add derived metadata fields to every video:
    - video_id (from YouTube URL)
    - title_language
    - is_code_switched
    """
    for video in videos:
        video["video_id"] = extract_video_id(video["url"])
        video["title_language"] = detect_title_language(video["title"])
        video["is_code_switched"] = is_code_switched(video["title"])

    return videos


def deduplicate(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate videos by URL.
    Keeps the first occurrence (named-section version preferred
    over Other-section version).
    """
    seen_urls: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicates = 0

    for video in videos:
        url_normalized = video["url"].strip().lower()
        if url_normalized in seen_urls:
            duplicates += 1
            continue
        seen_urls.add(url_normalized)
        unique.append(video)

    if duplicates > 0:
        logger.warning(f"Removed {duplicates} duplicate URLs")
    else:
        logger.info("No duplicate URLs found")

    return unique


def compute_summary_stats(videos: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics for the categorized dataset."""
    from collections import Counter

    categories = Counter(v["assigned_category"] for v in videos)
    confidences = Counter(v["confidence"] for v in videos)
    languages = Counter(v["title_language"] for v in videos)
    tiers = Counter(v["categorization_tier"] for v in videos)
    code_switched = sum(1 for v in videos if v["is_code_switched"])

    return {
        "total_videos": len(videos),
        "categories": dict(categories.most_common()),
        "confidence_distribution": dict(confidences.most_common()),
        "title_languages": dict(languages.most_common()),
        "categorization_tiers": dict(tiers.most_common()),
        "code_switched_count": code_switched,
        "high_confidence_count": confidences.get("High", 0),
        "pipeline_ready_count": confidences.get("High", 0) + confidences.get("Medium", 0),
    }


def save_output(
    videos: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save categorized videos and summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "description": "CulinaryVLM — Categorized video dataset",
            "source": "Chicken_Biryani_Classified.xlsx",
            "total_videos": summary["total_videos"],
            "pipeline_ready_count": summary["pipeline_ready_count"],
        },
        "summary": summary,
        "videos": videos,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Output saved to: {output_path}")
    logger.info(f"  Total videos:        {summary['total_videos']}")
    logger.info(f"  High confidence:     {summary['high_confidence_count']}")
    logger.info(f"  Pipeline-ready (H+M):{summary['pipeline_ready_count']}")


def main() -> None:
    """Phase 0A main entry point."""
    parser = argparse.ArgumentParser(
        description="CulinaryVLM Phase 0A — Sub-categorize video dataset",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to config.yaml (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and categorize but don't save output",
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Phase 0A: Video Categorization   ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    # 1. Load config
    config = load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    # 2. Parse Excel
    videos = parse_excel(config)

    # 3. Deduplicate by URL
    videos = deduplicate(videos)

    # 4. Categorize "Other" videos
    videos = categorize_other_videos(videos, config)

    # 5. Enrich with metadata (language, video_id, code-switching)
    videos = enrich_metadata(videos)

    # 6. Compute summary statistics
    summary = compute_summary_stats(videos)

    # 7. Print detailed stats
    logger.info("")
    logger.info("── Confidence Distribution ──")
    for conf, count in summary["confidence_distribution"].items():
        logger.info(f"  {conf:8s}: {count:4d}")

    logger.info("")
    logger.info("── Title Languages ──")
    for lang, count in summary["title_languages"].items():
        logger.info(f"  {lang:12s}: {count:4d}")

    logger.info("")
    logger.info("── Categorization Tiers ──")
    for tier, count in summary["categorization_tiers"].items():
        logger.info(f"  {tier:18s}: {count:4d}")

    # 8. Save output
    if args.dry_run:
        logger.info("[DRY RUN] Skipping file save")
        # Print a few sample entries
        logger.info("")
        logger.info("── Sample Entries (first 3 Other → categorized) ──")
        other_recategorized = [
            v for v in videos
            if v["original_category"] == "Other" and v["assigned_category"] != "Generic"
        ]
        for v in other_recategorized[:3]:
            logger.info(f"  {v['title'][:60]:60s} → {v['assigned_category']} ({v['confidence']})")
    else:
        output_path = PROJECT_ROOT / config["paths"]["categorized_output"]
        save_output(videos, summary, output_path)

    logger.info("")
    logger.info("✓ Phase 0A complete!")
    logger.info(f"  NEXT: Review output, then run Phase 0B (canonical recipes)")


if __name__ == "__main__":
    main()
