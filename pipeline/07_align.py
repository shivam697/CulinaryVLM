#!/usr/bin/env python3
"""
CulinaryVLM — Stage 7: Multimodal Alignment (BGE + DTW)
═══════════════════════════════════════════════════════════

Aligns verified video segments to canonical recipe steps using
BGE embeddings and Dynamic Time Warping (DTW).

Runs on: MacBook CPU or GPU cluster

Usage:
    python pipeline/07_align.py
    python pipeline/07_align.py --resume

Input:  datasets/verified/verified_segments.json
        configs/canonical_recipes/{category}.json
Output: datasets/alignments/{video_id}.json
"""

from __future__ import annotations

import argparse, json, logging, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_7")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_dtw(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Dynamic Time Warping on a cost matrix. Returns optimal path."""
    n, m = cost_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost_matrix[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [(dtw[i - 1, j - 1], i - 1, j - 1), (dtw[i - 1, j], i - 1, j), (dtw[i, j - 1], i, j - 1)]
        _, i, j = min(candidates)
    path.reverse()
    return path


def align_video_to_recipe(
    video_segments: list[dict],
    recipe_steps: list[dict],
    embedder,
) -> list[dict[str, Any]]:
    """Align video segments to canonical recipe steps via DTW."""
    seg_texts = [f"{s.get('action', '')} {s.get('description', '')}" for s in video_segments]
    step_texts = [f"{s.get('action', '')} {s.get('description', '')}" for s in recipe_steps]

    if not seg_texts or not step_texts:
        return []

    seg_emb = embedder.encode(seg_texts, normalize_embeddings=True)
    step_emb = embedder.encode(step_texts, normalize_embeddings=True)

    # Cosine distance matrix
    similarity = seg_emb @ step_emb.T
    cost_matrix = 1 - similarity

    path = compute_dtw(cost_matrix)

    alignments = []
    for seg_idx, step_idx in path:
        alignments.append({
            "segment_index": seg_idx,
            "segment_action": video_segments[seg_idx].get("action", ""),
            "segment_start": video_segments[seg_idx].get("start_time", 0),
            "segment_end": video_segments[seg_idx].get("end_time", 0),
            "canonical_step": recipe_steps[step_idx].get("step_number", step_idx + 1),
            "canonical_action": recipe_steps[step_idx].get("action", ""),
            "similarity": float(similarity[seg_idx, step_idx]),
            "is_defining": recipe_steps[step_idx].get("is_defining_step", False),
        })

    return alignments


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 7 — Multimodal Alignment")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 7: Multimodal Alignment    ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    verified_path = PROJECT_ROOT / config["paths"]["verified"] / "verified_segments.json"
    recipe_dir = PROJECT_ROOT / "configs" / "canonical_recipes"
    align_dir = PROJECT_ROOT / config["paths"]["alignments"]

    if not verified_path.exists():
        logger.error(f"Not found: {verified_path}. Run Stage 6 first.")
        sys.exit(1)

    with open(verified_path) as f:
        data = json.load(f)
    all_segments = [s for s in data.get("segments", []) if s.get("verified", True)]

    # Group by video
    by_video: dict[str, list] = defaultdict(list)
    video_cats: dict[str, str] = {}
    for s in all_segments:
        by_video[s["video_id"]].append(s)
        video_cats[s["video_id"]] = s["category"]

    logger.info(f"Verified segments: {len(all_segments)} across {len(by_video)} videos")

    if args.dry_run:
        logger.info("[DRY RUN] Would align:")
        for vid, segs in list(by_video.items())[:10]:
            logger.info(f"  {video_cats[vid]:15s} | {vid} | {len(segs)} segments")
        return

    # Load embedder
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except ImportError:
        logger.error("sentence-transformers not installed")
        sys.exit(1)

    # Load canonical recipes
    recipes: dict[str, list] = {}
    for fp in recipe_dir.glob("*.json"):
        with open(fp) as f:
            r = json.load(f)
        recipes[r["category"]] = r.get("steps", [])

    stats = {"aligned": 0, "skipped": 0, "no_recipe": 0}
    align_dir.mkdir(parents=True, exist_ok=True)

    for vid, segments in by_video.items():
        cat = video_cats[vid]
        out_path = align_dir / f"{vid}.json"

        if args.resume and out_path.exists():
            stats["skipped"] += 1
            continue

        if cat not in recipes:
            stats["no_recipe"] += 1
            continue

        alignments = align_video_to_recipe(segments, recipes[cat], embedder)

        with open(out_path, "w") as f:
            json.dump({
                "video_id": vid, "category": cat,
                "num_alignments": len(alignments), "alignments": alignments,
            }, f, indent=2)

        stats["aligned"] += 1
        logger.info(f"  ✓ {cat:15s} | {vid} | {len(alignments)} alignments")

    logger.info(f"\n{'═'*50}\nALIGNMENT: {stats}\n{'═'*50}")
    logger.info("✓ Stage 7 complete! NEXT: Stage 8 — python pipeline/08_viddiff.py")


if __name__ == "__main__":
    main()
