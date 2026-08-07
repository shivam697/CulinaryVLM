#!/usr/bin/env python3
"""
CulinaryVLM — Stage 6: Verification (Gemini Flash)
═══════════════════════════════════════════════

Verifies each segment using Gemini 1.5 Flash with structured JSON
classification (v2.0).  Returns cooking_stage, ingredient_state,
confidence, missing_information alongside the yes/no verdict.
Falls back to old yes/no text parsing if Gemini doesn't return
valid JSON.  Rate-limited to 14 RPM. Runs overnight on MacBook.

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


def _parse_yes_no_fallback(text: str) -> dict:
    """Fallback parser for non-JSON Gemini responses (old yes/no format)."""
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


def verify_segment_gemini(segment: dict, api_key: str) -> dict:
    """Ask Gemini Flash if this segment is a valid cooking step.

    Returns a dict containing at minimum:
        - verified: bool | None  (backward-compatible with 07_align.py)
        - reason: str
    Plus v2.0 fields: cooking_stage, ingredient_state, confidence,
    missing_information.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-flash-latest")

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

        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                break
            except Exception as retry_err:
                if "429" in str(retry_err) and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 10  # 10s, 20s, 40s
                    logger.warning(f"  429 rate limit, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                raise

        # Try to parse as JSON first (v2.0 structured response)
        json_text = raw_text
        # Strip markdown code fences if Gemini wraps in ```json ... ```
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
            # Gemini didn't return valid JSON — fall back to old yes/no parsing
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


def verify_segment_groq(segment: dict, api_key: str) -> dict:
    """Ask Groq (Llama 3.3 70B, falling back to Llama 3.1 8B) if this segment is a valid cooking step."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

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

        models_to_try = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        max_retries = 2
        raw_text = ""
        last_err = None
        for model_name in models_to_try:
            success = False
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    raw_text = response.choices[0].message.content.strip()
                    success = True
                    break
                except Exception as retry_err:
                    last_err = retry_err
                    if "429" in str(retry_err) and attempt < max_retries - 1:
                        wait = (2 ** attempt) * 10
                        logger.warning(f"  429 rate limit on {model_name}, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    break
            if success:
                break
            else:
                logger.warning(f"  {model_name} exhausted, falling back to next model")
        if not raw_text:
            raise last_err

        json_text = raw_text
        if json_text.startswith("```"):
            lines = json_text.split("\n")
            json_text = "\n".join(l for l in lines if not l.strip().startswith("```"))

        try:
            parsed = json.loads(json_text)
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
    parser.add_argument("--rpm", type=int, default=25, help="Requests per minute (Groq rate limit)")
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

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("REPLACE"):
        logger.error("GROQ_API_KEY not set. Add it to .env")
        sys.exit(1)

    # Resume support
    existing = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            existing = {
                s["video_id"] + str(s.get("step_number")): s
                for s in json.load(f).get("segments", [])
                if s.get("status") != "error"
            }

    delay = 60.0 / args.rpm
    stats = {"verified": 0, "rejected": 0, "error": 0, "skipped": 0}

    for i, seg in enumerate(all_segments, 1):
        key = seg["video_id"] + str(seg.get("step_number"))
        if key in existing:
            seg.update(existing[key])
            stats["skipped"] += 1
            continue

        logger.info(f"[{i}/{len(all_segments)}] {seg['category']} | {seg.get('action', '')[:40]}")
        result = verify_segment_groq(seg, api_key)
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
        json.dump({"schema_version": "2.0", "total": len(all_segments), "stats": stats, "segments": all_segments}, f, indent=2)

    logger.info(f"\n{'═'*50}\nVERIFICATION: {stats}\n{'═'*50}")
    logger.info("✓ Stage 6 complete! NEXT: Stage 7 — python pipeline/07_align.py")


if __name__ == "__main__":
    main()
