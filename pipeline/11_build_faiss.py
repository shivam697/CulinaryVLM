#!/usr/bin/env python3
"""
CulinaryVLM — Stage 11: Build FAISS Index
═══════════════════════════════════════════

Builds a FAISS index over all segment embeddings for fast retrieval.

Runs on: MacBook CPU

Usage:
    python pipeline/11_build_faiss.py

Input:  datasets/verified/verified_segments.json
Output: datasets/faiss/segments.index
        datasets/faiss/segments_metadata.pkl
"""

from __future__ import annotations

import argparse, json, logging, pickle, sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_11")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 11 — FAISS Index")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 11: Build FAISS Index      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    verified_path = PROJECT_ROOT / config["paths"]["verified"] / "verified_segments.json"
    faiss_dir = PROJECT_ROOT / config["paths"].get("faiss_index", "datasets/faiss")

    if not verified_path.exists():
        logger.error(f"Not found: {verified_path}. Run Stage 6 first.")
        sys.exit(1)

    with open(verified_path) as f:
        data = json.load(f)
    segments = [s for s in data.get("segments", []) if s.get("verified", True)]
    logger.info(f"Segments to index: {len(segments)}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would build index with {len(segments)} vectors using {args.model}")
        return

    if not segments:
        logger.error("No verified segments. Run stages 4-6 first.")
        sys.exit(1)

    # Embed
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed")
        sys.exit(1)

    logger.info(f"Embedding with {args.model}...")
    model = SentenceTransformer(args.model)

    # Build rich embedding text — falls back gracefully for v1.0 segments
    texts = []
    for s in segments:
        parts = [
            s.get("action", ""),
            s.get("description", ""),
        ]
        # v2.0 enrichments — append only if non-empty
        for key in ("cooking_technique", "cooking_stage", "style",
                    "rice_type", "protein", "canonical_action"):
            val = s.get(key, "")
            if val:
                parts.append(f"{key}: {val}")

        ingredient_state = s.get("ingredient_state", {})
        if isinstance(ingredient_state, dict) and ingredient_state:
            parts.append("state: " + ", ".join(f"{k}: {v}" for k, v in ingredient_state.items()))
        elif isinstance(ingredient_state, str) and ingredient_state:
            parts.append(f"state: {ingredient_state}")

        visible_objects = s.get("visible_objects", [])
        if visible_objects:
            parts.append("objects: " + ", ".join(visible_objects))

        texts.append(" — ".join(p for p in parts if p))

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)

    logger.info(f"Embedding shape: {embeddings.shape}")

    # Build FAISS index
    try:
        import faiss
    except ImportError:
        logger.error("faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
    index.add(embeddings)
    logger.info(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # Save
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_dir / "segments.index"))

    # Save metadata
    metadata = []
    for s in segments:
        meta: dict[str, Any] = {
            # Existing keys — unchanged
            "video_id": s.get("video_id", ""),
            "category": s.get("category", ""),
            "action": s.get("action", ""),
            "description": s.get("description", ""),
            "start_time": s.get("start_time", 0),
            "end_time": s.get("end_time", 0),
            "segment_id": f"{s.get('video_id', '')}_{s.get('step_number', 0)}",
            # v2.0 additive keys
            "cooking_technique": s.get("cooking_technique", ""),
            "cooking_stage": s.get("cooking_stage", ""),
            "style": s.get("style", s.get("category", "")),
            "rice_type": s.get("rice_type", ""),
            "protein": s.get("protein", ""),
            "canonical_action": s.get("canonical_action", ""),
            "visible_objects": s.get("visible_objects", []),
            "ingredient_state": s.get("ingredient_state", {}),
        }
        metadata.append(meta)

    with open(faiss_dir / "segments_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Also save as JSON for inspection
    with open(faiss_dir / "segments_metadata.json", "w") as f:
        json.dump({"schema_version": "2.0", "metadata": metadata}, f, indent=2)

    logger.info(f"Saved: {faiss_dir}/segments.index + segments_metadata.pkl")
    logger.info("✓ Stage 11 complete! FAISS index ready for the API.")
    logger.info("  NEXT: Start API with: python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
