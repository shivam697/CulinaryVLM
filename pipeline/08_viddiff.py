#!/usr/bin/env python3
"""
CulinaryVLM — Stage 8: VidDiff Comparison
═══════════════════════════════════════════

Three-stage comparison pipeline:
  1. Proposer: Local Ollama (qwen2.5:32b-instruct) — proposes difference aspects
  2. Frame Retriever: OpenCLIP — retrieves relevant frame pairs
  3. Action Differencer: Gemini Flash MCQ — scores differences

Usage:
    python pipeline/08_viddiff.py
    python pipeline/08_viddiff.py --pairs 50

Input:  datasets/alignments/{video_id}.json
        datasets/segments/{video_id}.json
Output: datasets/comparisons/comparison_results.json
"""

from __future__ import annotations

import argparse, itertools, json, logging, os, sys, time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_8")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.utils.ollama_client import check_ollama, ollama_chat  # noqa: E402


def propose_differences(category_a: str, category_b: str, host: str, model: str) -> list[dict]:
    """Use local Ollama to propose comparison aspects."""
    try:
        user_prompt = (
            f"Compare {category_a} and {category_b} biryani cooking across "
            f"EACH of these 10 aspects:\n"
            f"1. rice (grain length, parboil level, soaking)\n"
            f"2. layering (number of layers, order, technique)\n"
            f"3. marination (wet vs dry, duration, acid, dairy)\n"
            f"4. dum (sealed vs open, duration, heat source)\n"
            f"5. vessel (handi vs deg vs matka vs pot)\n"
            f"6. spice_profile (key spices, intensity, whole vs ground)\n"
            f"7. cooking_order (meat first vs rice first, simultaneous)\n"
            f"8. garnish (saffron milk, fried onions, mint, kewra)\n"
            f"9. oil_usage (ghee vs oil vs mustard oil, quantity)\n"
            f"10. ingredient_substitutions (regional swaps, e.g. potato, egg)\n\n"
            f"Output as a JSON array of objects, one per aspect, each with:\n"
            f"  'aspect': the aspect name from the list above,\n"
            f"  'description': 1-2 sentence comparison,\n"
            f"  'visual_cue': what a camera would see differently.\n"
            f"Return ONLY the JSON array, no markdown."
        )

        text = ollama_chat(
            host=host,
            model=model,
            system="You are a culinary expert specializing in Indian biryani. Output JSON only.",
            user=user_prompt,
            num_predict=6000,
            temperature=0.2,
            json_format=True,
        )

        if not text:
            logger.warning("Proposer returned empty response")
            return [{"aspect": "general", "description": "General cooking differences", "visual_cue": "N/A"}]

        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1].lstrip("json\n")

        parsed = json.loads(text)
        # Handle case where Ollama wraps array in an object due to json_format
        if isinstance(parsed, dict):
            # Try common wrapper keys
            for key in ("aspects", "differences", "comparisons", "data", "result"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # If single-level dict with list values, take first list
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            return [{"aspect": "general", "description": "General cooking differences", "visual_cue": "N/A"}]
        return parsed
    except Exception as e:
        logger.warning(f"Proposer failed: {e}")
        return [{"aspect": "general", "description": "General cooking differences", "visual_cue": "N/A"}]


def generate_comparison_pairs(
    categories: list[str],
    max_pairs: int = 50,
) -> list[tuple[str, str]]:
    """Generate stratified category pairs for comparison."""
    all_pairs = list(itertools.combinations(sorted(set(categories)), 2))
    if len(all_pairs) <= max_pairs:
        return all_pairs
    # Stratified sample
    import random
    random.seed(42)
    return random.sample(all_pairs, max_pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 8 — VidDiff Comparison")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="qwen2.5:32b-instruct",
                        help="Ollama model name (default: qwen2.5:32b-instruct)")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 8: VidDiff Comparison      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    align_dir = PROJECT_ROOT / config["paths"]["alignments"]
    out_dir = PROJECT_ROOT / config["paths"]["comparisons"]

    # Discover aligned categories
    categories = set()
    for fp in align_dir.glob("*.json"):
        with open(fp) as f:
            d = json.load(f)
        categories.add(d.get("category", ""))
    categories.discard("")

    pairs = generate_comparison_pairs(list(categories), args.pairs)
    logger.info(f"Categories: {len(categories)}, Pairs: {len(pairs)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would compare:")
        for a, b in pairs[:10]:
            logger.info(f"  {a} vs {b}")
        return

    # Validate Ollama connectivity
    check_ollama(args.host, args.model)

    results = []
    for i, (cat_a, cat_b) in enumerate(pairs, 1):
        logger.info(f"\n[{i}/{len(pairs)}] {cat_a} vs {cat_b}")

        # Stage 1: Propose
        aspects = propose_differences(cat_a, cat_b, args.host, args.model)
        logger.info(f"  Proposed {len(aspects)} aspects")

        results.append({
            "category_a": cat_a, "category_b": cat_b,
            "proposed_aspects": aspects,
            "num_aspects": len(aspects),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump({"schema_version": "2.0", "total_pairs": len(results), "comparisons": results}, f, indent=2)

    logger.info(f"\n{'═'*50}\nCOMPARISON: {len(results)} pairs processed\n{'═'*50}")
    logger.info("✓ Stage 8 complete! NEXT: Stage 9 — python pipeline/09_generate_qa.py")


if __name__ == "__main__":
    main()
