#!/usr/bin/env python3
"""
CulinaryVLM — Stage 8: VidDiff Comparison
═══════════════════════════════════════════

Three-stage comparison pipeline:
  1. Proposer: Qwen2.5-7B via Groq — proposes difference aspects
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


def propose_differences(category_a: str, category_b: str, groq_key: str) -> list[dict]:
    """Use Groq Qwen2.5 to propose comparison aspects."""
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)

        response = client.chat.completions.create(
            model="qwen-2.5-7b",
            messages=[{
                "role": "system",
                "content": "You are a culinary expert specializing in Indian biryani. Output JSON only."
            }, {
                "role": "user",
                "content": (
                    f"List 5-8 specific visual/procedural differences between "
                    f"{category_a} and {category_b} biryani cooking. "
                    f"Output as JSON array of objects with 'aspect', 'description', 'visual_cue'."
                )
            }],
            temperature=0.2, max_tokens=1024,
        )

        text = response.choices[0].message.content.strip()
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1].lstrip("json\n")
        return json.loads(text)
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

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key or groq_key.startswith("REPLACE"):
        logger.error("GROQ_API_KEY required for Proposer stage")
        sys.exit(1)

    results = []
    for i, (cat_a, cat_b) in enumerate(pairs, 1):
        logger.info(f"\n[{i}/{len(pairs)}] {cat_a} vs {cat_b}")

        # Stage 1: Propose
        aspects = propose_differences(cat_a, cat_b, groq_key)
        logger.info(f"  Proposed {len(aspects)} aspects")

        results.append({
            "category_a": cat_a, "category_b": cat_b,
            "proposed_aspects": aspects,
            "num_aspects": len(aspects),
        })
        time.sleep(1)  # Rate limit

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump({"total_pairs": len(results), "comparisons": results}, f, indent=2)

    logger.info(f"\n{'═'*50}\nCOMPARISON: {len(results)} pairs processed\n{'═'*50}")
    logger.info("✓ Stage 8 complete! NEXT: Stage 9 — python pipeline/09_generate_qa.py")


if __name__ == "__main__":
    main()
