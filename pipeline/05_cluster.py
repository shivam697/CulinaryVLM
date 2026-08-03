#!/usr/bin/env python3
"""
CulinaryVLM — Stage 5: Action Clustering + Temporal Merge
═══════════════════════════════════════════════════════════

Clusters similar action segments across videos using sentence embeddings
and agglomerative clustering. Reduces redundant segments.

Runs on: MacBook CPU (no GPU needed)

Usage:
    python pipeline/05_cluster.py
    python pipeline/05_cluster.py --threshold 0.3
    python pipeline/05_cluster.py --dry-run

Input:  datasets/segments/{video_id}.json
Output: datasets/clusters/clustered_segments.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_5")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_all_segments(seg_dir: Path) -> list[dict[str, Any]]:
    """Load all segment files."""
    segments = []
    for filepath in sorted(seg_dir.glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            seg["video_id"] = data["video_id"]
            seg["category"] = data["category"]
            segments.append(seg)
    return segments


def embed_actions(segments: list[dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode segment actions into embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed")
        sys.exit(1)

    model = SentenceTransformer(model_name)
    texts = [
        f"{s.get('action', '')} — {s.get('description', '')}" for s in segments
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings)


def cluster_segments(
    embeddings: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Agglomerative clustering with cosine distance."""
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def build_cluster_summaries(
    segments: list[dict],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Build cluster summaries with representative segments."""
    clusters: dict[int, list[dict]] = defaultdict(list)
    for seg, label in zip(segments, labels):
        seg["cluster_id"] = int(label)
        clusters[int(label)].append(seg)

    summaries = []
    for cluster_id in sorted(clusters.keys()):
        members = clusters[cluster_id]
        # Pick the member with the longest description as representative
        representative = max(members, key=lambda s: len(s.get("description", "")))
        actions = Counter(s.get("action", "") for s in members)

        summaries.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "representative_action": representative.get("action", ""),
            "representative_description": representative.get("description", ""),
            "action_distribution": dict(actions.most_common(5)),
            "categories": dict(Counter(s["category"] for s in members).most_common()),
            "video_ids": list(set(s["video_id"] for s in members)),
        })

    return {
        "total_segments": len(segments),
        "total_clusters": len(summaries),
        "reduction_pct": round((1 - len(summaries) / max(len(segments), 1)) * 100, 1),
        "clusters": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 5 — Action Clustering")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 5: Action Clustering       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seg_dir = PROJECT_ROOT / config["paths"]["segments"]
    out_dir = PROJECT_ROOT / config["paths"]["clusters"]

    segments = load_all_segments(seg_dir)
    logger.info(f"Loaded {len(segments)} segments from {seg_dir}")

    if not segments:
        logger.error("No segments found. Run Stage 4 first.")
        sys.exit(1)

    if args.dry_run:
        cats = Counter(s["category"] for s in segments)
        logger.info("[DRY RUN] Segment distribution:")
        for cat, count in cats.most_common():
            logger.info(f"  {cat:15s}: {count}")
        return

    # Embed
    logger.info(f"Embedding {len(segments)} segment actions...")
    embeddings = embed_actions(segments)

    # Cluster
    logger.info(f"Clustering with threshold={args.threshold}...")
    labels = cluster_segments(embeddings, threshold=args.threshold)

    # Build summaries
    result = build_cluster_summaries(segments, labels)
    logger.info(f"Segments: {result['total_segments']} → Clusters: {result['total_clusters']}")
    logger.info(f"Reduction: {result['reduction_pct']}%")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clustered_segments.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")

    # Also save per-segment labels
    labeled_path = out_dir / "segment_labels.json"
    labeled = [
        {"video_id": s["video_id"], "step_number": s.get("step_number"), "cluster_id": s["cluster_id"]}
        for s in segments
    ]
    with open(labeled_path, "w") as f:
        json.dump(labeled, f, indent=2)

    logger.info("✓ Stage 5 complete! NEXT: Stage 6 — python pipeline/06_verify.py")


if __name__ == "__main__":
    main()
