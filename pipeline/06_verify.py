#!/usr/bin/env python3
"""
CulinaryVLM — Stage 6: Verification (Gemini Flash)
═══════════════════════════════════════════════════════

Verifies each segment using Gemini 1.5 Flash yes/no classification.
Rate-limited to 14 RPM. Runs overnight on MacBook.

Usage:
    python pipeline/06_verify.py
    python pipeline/06_verify.py --resume --rpm 10

Input:  datasets/segments/{video_id}.json
Output: datasets/verified/verified_segments.json
"""

from __future__ import annotations

import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_6")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_segment_gemini(segment: dict, api_key: str) -> dict:
    """Ask Gemini Flash if this segment is a valid cooking step."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            f"Is this a valid cooking step in a biryani recipe?\n"
            f"Action: {segment.get('action', '')}\n"
            f"Description: {segment.get('description', '')}\n\n"
            f"Answer ONLY 'yes' or 'no', then a brief reason."
        )

        response = model.generate_content(prompt)
        text = response.text.strip().lower()
        is_valid = text.startswith("yes")

        return {"verified": is_valid, "reason": response.text.strip()[:200], "status": "success"}
    except Exception as e:
        return {"verified": None, "reason": str(e)[:200], "status": "error"}


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 6 — Verification")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--rpm", type=int, default=14, help="Requests per minute (Gemini rate limit)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 6: Segment Verification    ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    seg_dir = PROJECT_ROOT / config["paths"]["segments"]
    out_dir = PROJECT_ROOT / config["paths"]["verified"]
    out_path = out_dir / "verified_segments.json"

    # Load all segments
    all_segments = []
    for fp in sorted(seg_dir.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            seg["video_id"] = data["video_id"]
            seg["category"] = data["category"]
            all_segments.append(seg)

    logger.info(f"Total segments to verify: {len(all_segments)}")

    if args.dry_run:
        est_minutes = len(all_segments) / args.rpm
        logger.info(f"[DRY RUN] Estimated time: {est_minutes:.0f} min ({est_minutes/60:.1f} hours) at {args.rpm} RPM")
        return

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("REPLACE"):
        logger.error("GEMINI_API_KEY not set. Add it to .env")
        sys.exit(1)

    # Resume support
    existing = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = {s["video_id"] + str(s.get("step_number")): s for s in json.load(f).get("segments", [])}

    delay = 60.0 / args.rpm
    stats = {"verified": 0, "rejected": 0, "error": 0, "skipped": 0}

    for i, seg in enumerate(all_segments, 1):
        key = seg["video_id"] + str(seg.get("step_number"))
        if key in existing:
            seg.update(existing[key])
            stats["skipped"] += 1
            continue

        logger.info(f"[{i}/{len(all_segments)}] {seg['category']} | {seg.get('action', '')[:40]}")
        result = verify_segment_gemini(seg, api_key)
        seg.update(result)

        if result["verified"] is True:
            stats["verified"] += 1
        elif result["verified"] is False:
            stats["rejected"] += 1
        else:
            stats["error"] += 1

        # Rate limit
        time.sleep(delay)

        # Checkpoint every 50
        if i % 50 == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"segments": all_segments[:i]}, f, indent=2)
            logger.info(f"  Checkpoint saved ({i}/{len(all_segments)})")

    # Final save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"total": len(all_segments), "stats": stats, "segments": all_segments}, f, indent=2)

    logger.info(f"\n{'═'*50}\nVERIFICATION: {stats}\n{'═'*50}")
    logger.info("✓ Stage 6 complete! NEXT: Stage 7 — python pipeline/07_align.py")


if __name__ == "__main__":
    main()
