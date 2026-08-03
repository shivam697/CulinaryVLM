#!/usr/bin/env python3
"""
CulinaryVLM — Stage 4: Video Segmentation (VLM-based)
═══════════════════════════════════════════════════════

Uses InternVL2-8B (4-bit quantized) to segment cooking videos
into procedural steps with temporal boundaries.

MUST RUN ON GPU CLUSTER (Gpu7-80G queue).

Usage:
    python pipeline/04_segment.py
    python pipeline/04_segment.py --resume --batch-size 4
    python pipeline/04_segment.py --category Hyderabadi --limit 5
    python pipeline/04_segment.py --dry-run

Input:  datasets/raw_videos/{category}/{video_id}.mp4
        datasets/transcripts/{video_id}.json
Output: datasets/segments/{video_id}.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"

# ─── Paper's exact segmentation prompt ────────────────
SEGMENTATION_PROMPT = """You are analyzing a cooking video frame sequence. For each distinct cooking step visible in these frames, provide:

1. step_number: Sequential step number
2. action: Brief action name (e.g., "Marinating chicken", "Frying onions")
3. description: Detailed description of what is happening
4. start_frame: Approximate start frame index
5. end_frame: Approximate end frame index
6. ingredients_visible: List of ingredients visible in the frames
7. cooking_technique: The technique being used (e.g., "frying", "boiling", "layering")
8. confidence: Your confidence in this segment (0.0-1.0)

Output as a JSON array of step objects. Be precise about temporal boundaries."""


def load_config(config_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frames(video_path: Path, fps: float = 0.5) -> list:
    """Extract frames from video at given FPS rate."""
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python not installed. Run: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(video_fps / fps)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Resize for VLM input (448x448)
            frame_resized = cv2.resize(frame, (448, 448))
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps,
                "image": frame_resized,
            })
        frame_idx += 1

    cap.release()
    total_duration = frame_idx / video_fps
    logger.info(f"  Extracted {len(frames)} frames from {total_duration:.0f}s video")
    return frames


def segment_with_vlm(
    frames: list,
    model,
    tokenizer,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """
    Run InternVL2-8B on frame batches to identify cooking steps.

    Processes frames in sliding windows of 8 frames.
    """
    import torch
    from PIL import Image
    import numpy as np

    segments = []
    window_size = 8
    stride = 4

    for i in range(0, len(frames), stride):
        window = frames[i:i + window_size]
        if len(window) < 2:
            break

        # Convert frames to PIL images
        images = []
        for f in window:
            img = Image.fromarray(f["image"][:, :, ::-1])  # BGR to RGB
            images.append(img)

        start_time = window[0]["timestamp"]
        end_time = window[-1]["timestamp"]

        try:
            # Build prompt with frame context
            prompt = (
                f"These are {len(images)} frames from a biryani cooking video, "
                f"spanning {start_time:.1f}s to {end_time:.1f}s.\n\n"
                f"{SEGMENTATION_PROMPT}"
            )

            # Model inference (InternVL2 specific)
            with torch.no_grad():
                response = model.chat(
                    tokenizer,
                    pixel_values=None,  # Will be set by chat method
                    question=prompt,
                    generation_config={"max_new_tokens": 1024, "temperature": 0.1},
                    images=images,
                )

            # Parse JSON response
            try:
                steps = json.loads(response)
                if isinstance(steps, list):
                    for step in steps:
                        step["start_time"] = start_time
                        step["end_time"] = end_time
                        step["window_start"] = i
                    segments.extend(steps)
            except json.JSONDecodeError:
                # Try to extract structured info from text
                segments.append({
                    "action": "unknown",
                    "description": response[:200],
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": 0.3,
                })

        except Exception as e:
            logger.warning(f"  VLM error at {start_time:.1f}s: {e}")

        # Free GPU memory
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return segments


def merge_overlapping_segments(
    segments: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge overlapping segments from sliding window."""
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda s: s.get("start_time", 0))

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        # Check overlap
        overlap = min(last["end_time"], seg["end_time"]) - max(last["start_time"], seg["start_time"])
        duration = max(seg["end_time"] - seg["start_time"], 0.1)

        if overlap / duration > overlap_threshold and last.get("action") == seg.get("action"):
            # Merge: extend end time, keep higher confidence
            last["end_time"] = max(last["end_time"], seg["end_time"])
            if seg.get("confidence", 0) > last.get("confidence", 0):
                last["description"] = seg.get("description", last["description"])
        else:
            merged.append(seg)

    # Renumber steps
    for i, seg in enumerate(merged, 1):
        seg["step_number"] = i

    return merged


def save_segments(
    video_id: str,
    category: str,
    segments: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Save segment JSON."""
    output = {
        "video_id": video_id,
        "category": category,
        "num_segments": len(segments),
        "segments": segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{video_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 4 — Video Segmentation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 4: Video Segmentation      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)

    # Load video selection
    sel_path = PROJECT_ROOT / "datasets/categorized/pipeline_selection.json"
    with open(sel_path) as f:
        videos = json.load(f)["videos"]

    if args.category:
        videos = [v for v in videos if v["category"] == args.category]
    if args.limit:
        videos = videos[:args.limit]

    raw_dir = PROJECT_ROOT / config["paths"]["raw_videos"]
    seg_dir = PROJECT_ROOT / config["paths"]["segments"]

    # Find downloaded videos
    available = []
    for v in videos:
        cat_dir = raw_dir / v["category"].lower().replace(" ", "_")
        for ext in [".mp4", ".mkv", ".webm"]:
            p = cat_dir / f"{v['video_id']}{ext}"
            if p.exists():
                available.append((v, p))
                break

    logger.info(f"Selected: {len(videos)}, Available: {len(available)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would segment:")
        for v, p in available[:10]:
            logger.info(f"  {v['category']:15s} | {v['video_id']}")
        return

    if not available:
        logger.error("No downloaded videos found. Run Stage 1 first.")
        sys.exit(1)

    # Load VLM model
    logger.info("Loading InternVL2-8B (4-bit)...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = config["pipeline"]["segmentation"]["model"]
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        logger.error("This stage requires GPU. Run on the IIIT Delhi cluster.")
        sys.exit(1)

    # Process videos
    stats = {"success": 0, "skipped": 0, "error": 0}

    for i, (video, vpath) in enumerate(available, 1):
        vid = video["video_id"]
        cat = video["category"]

        logger.info(f"\n[{i}/{len(available)}] {cat} | {vid}")

        out_path = seg_dir / f"{vid}.json"
        if args.resume and out_path.exists():
            logger.info("  ⊘ Already segmented (resume)")
            stats["skipped"] += 1
            continue

        try:
            frames = extract_frames(vpath, fps=0.5)
            if not frames:
                stats["error"] += 1
                continue

            raw_segments = segment_with_vlm(frames, model, tokenizer, device)
            merged = merge_overlapping_segments(raw_segments)

            save_segments(vid, cat, merged, seg_dir)
            logger.info(f"  ✓ {len(merged)} segments")
            stats["success"] += 1

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            stats["error"] += 1

    logger.info(f"\n{'═'*50}")
    logger.info(f"SEGMENTATION SUMMARY: {stats}")
    logger.info(f"{'═'*50}")
    logger.info("✓ Stage 4 complete! NEXT: Stage 5 — python pipeline/05_cluster.py")


if __name__ == "__main__":
    main()
