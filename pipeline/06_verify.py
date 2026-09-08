#!/usr/bin/env python3
"""
CulinaryVLM — Stage 6: Verification (Ollama)
═══════════════════════════════════════════════

Verifies each segment using local Ollama (qwen2.5:32b-instruct)
with structured JSON classification (v2.0).  Returns cooking_stage,
ingredient_state, confidence, missing_information alongside the
yes/no verdict.  Falls back to old yes/no text parsing if Ollama
doesn't return valid JSON.

Usage:
    python pipeline/06_verify.py
    python pipeline/06_verify.py --resume
    python pipeline/06_verify.py --dry-run

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

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.utils.ollama_client import check_ollama, ollama_chat  # noqa: E402


def _parse_yes_no_fallback(text: str) -> dict:
    """Fallback parser for non-JSON responses (old yes/no format)."""
    lower = text.strip().lower()
    is_valid = lower.startswith("yes")
    return {
        "verified": is_valid,
        "cooking_stage": "",
        "ingredient_state": "",
        "confidence": 0.7 if is_valid else 0.3,
        "missing_information": [],
        "reason": text.strip()[:200],
        "status": "success_fallback",
    }


def verify_segment_ollama(segment: dict, host: str, model: str) -> dict:
    """Ask local Ollama if this segment is a valid cooking step.

    Returns a dict containing at minimum:
        - verified: bool | None  (backward-compatible with 07_align.py)
        - reason: str
    Plus v2.0 fields: cooking_stage, ingredient_state, confidence,
    missing_information.
    """
    try:
        prompt = (
            f"Is this a valid cooking step in a biryani recipe?\n"
            f"Action: {segment.get('action', '')}\n"
            f"Description: {segment.get('description', '')}\n"
            f"Technique: {segment.get('cooking_technique', '')}\n\n"
            f"Respond with ONLY a JSON object (no markdown, no code fences):\n"
            f'{{\n'
            f'  "verified": true/false,\n'
            f'  "cooking_stage": "prep|marination|rice_cooking|meat_cooking|layering|dum|serving|other",\n'
            f'  "ingredient_state": "brief description of ingredient states visible",\n'
            f'  "confidence": 0.0-1.0,\n'
            f'  "missing_information": ["list of info that would help verify"],\n'
            f'  "reason": "brief explanation"\n'
            f'}}'
        )

        raw_text = ollama_chat(
            host=host,
            model=model,
            system="You are a cooking step verification assistant. Respond with valid JSON only.",
            user=prompt,
            num_predict=512,
            temperature=0.1,
            json_format=True,
        )

        if not raw_text:
            return {
                "verified": None,
                "cooking_stage": "",
                "ingredient_state": "",
                "confidence": 0.0,
                "missing_information": [],
                "reason": "Empty response from Ollama",
                "status": "error",
            }

        # Strip markdown code fences if model wraps in ```json ... ```
        json_text = raw_text
        if json_text.startswith("```"):
            lines = json_text.split("\n")
            json_text = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        try:
            parsed = json.loads(json_text)
            # Validate and normalize the structured response
            verified_val = parsed.get("verified")
            if isinstance(verified_val, str):
                verified_val = verified_val.lower() in ("true", "yes", "1")
            elif not isinstance(verified_val, bool):
                verified_val = None

            return {
                "verified": verified_val,
                "cooking_stage": str(parsed.get("cooking_stage", "")),
                "ingredient_state": str(parsed.get("ingredient_state", "")),
                "confidence": float(parsed.get("confidence", 0.5)),
                "missing_information": list(parsed.get("missing_information", [])),
                "reason": str(parsed.get("reason", ""))[:200],
                "status": "success",
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            # Didn't return valid JSON — fall back to old yes/no parsing
            logger.debug(f"  JSON parse failed, falling back to yes/no: {raw_text[:80]}")
            return _parse_yes_no_fallback(raw_text)

    except Exception as e:
        return {
            "verified": None,
            "cooking_stage": "",
            "ingredient_state": "",
            "confidence": 0.0,
            "missing_information": [],
            "reason": str(e)[:200],
            "status": "error",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 6 — Verification")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="qwen2.5:32b-instruct",
                        help="Ollama model name (default: qwen2.5:32b-instruct)")
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
        logger.info(f"[DRY RUN] Would verify {len(all_segments)} segments using Ollama {args.model}")
        return

    # Validate Ollama connectivity
    check_ollama(args.host, args.model)

    # Resume support
    existing = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = {
                s["video_id"] + str(s.get("step_number")): s
                for s in json.load(f).get("segments", [])
                if s.get("status") != "error"
            }

    stats = {"verified": 0, "rejected": 0, "error": 0, "skipped": 0}

    for i, seg in enumerate(all_segments, 1):
        key = seg["video_id"] + str(seg.get("step_number"))
        if key in existing:
            seg.update(existing[key])
            stats["skipped"] += 1
            continue

        logger.info(f"[{i}/{len(all_segments)}] {seg['category']} | {seg.get('action', '')[:40]}")
        result = verify_segment_ollama(seg, args.host, args.model)
        seg.update(result)

        if result["verified"] is True:
            stats["verified"] += 1
        elif result["verified"] is False:
            stats["rejected"] += 1
        else:
            stats["error"] += 1

        # Checkpoint every 50
        if i % 50 == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"segments": all_segments[:i]}, f, indent=2)
            logger.info(f"  Checkpoint saved ({i}/{len(all_segments)})")

    # Final save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"schema_version": "2.0", "total": len(all_segments), "stats": stats, "segments": all_segments}, f, indent=2)

    logger.info(f"\n{'═'*50}\nVERIFICATION: {stats}\n{'═'*50}")
    logger.info("✓ Stage 6 complete! NEXT: Stage 7 — python pipeline/07_align.py")


if __name__ == "__main__":
    main()
