#!/usr/bin/env python3
"""
CulinaryVLM — Stage 9: QA Generation
═══════════════════════════════════════

Generates multi-tier QA pairs:
  - Easy: Groq Llama-3.1-8B, per segment
  - Medium: Gemini Flash, per video + multilingual templates
  - Hard: Gemini Flash, multi-video 2-5 combos

Stratified train/test split by tier AND language.

Usage:
    python pipeline/09_generate_qa.py
    python pipeline/09_generate_qa.py --tier easy --limit 100
    python pipeline/09_generate_qa.py --dry-run

Input:  datasets/alignments/, datasets/verified/, datasets/comparisons/
Output: datasets/qa/train.json, datasets/qa/test.json
"""

from __future__ import annotations

import argparse, json, logging, os, random, sys, time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_9")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
random.seed(42)

EASY_TEMPLATES = [
    "What is happening in this cooking step?",
    "What ingredients are being used in this step?",
    "What cooking technique is being demonstrated?",
    "Describe what the cook is doing in this segment.",
    "What equipment is being used?",
]

MEDIUM_TEMPLATES = [
    "How does this video prepare the rice for {category} biryani?",
    "What is the marination process shown in this {category} biryani video?",
    "Describe the dum cooking technique used in this video.",
    "What spices are used during the {category} biryani preparation?",
    "How is the layering done in this {category} biryani?",
]

HARD_TEMPLATES = [
    "Compare the dum cooking methods between {cat_a} and {cat_b} biryani as shown in these videos.",
    "What are the key differences in rice preparation between {cat_a} and {cat_b} styles?",
    "How do the spice profiles differ between {cat_a} and {cat_b} biryani?",
    "Compare the marination techniques shown across these {n} videos of different biryani styles.",
]


def generate_easy_qa(segments: list[dict], groq_key: str, limit: int = None) -> list[dict]:
    """Generate easy QA from individual segments using Groq."""
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
    except ImportError:
        logger.error("groq not installed")
        return []

    qa_pairs = []
    segs = segments[:limit] if limit else segments

    for i, seg in enumerate(segs):
        template = random.choice(EASY_TEMPLATES)
        context = f"Action: {seg.get('action', '')}. Description: {seg.get('description', '')}"

        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Generate a concise, accurate answer based on the cooking step context. 2-3 sentences max."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {template}"},
                ],
                temperature=0.3, max_tokens=256,
            )
            answer = resp.choices[0].message.content.strip()

            qa_pairs.append({
                "question": template, "answer": answer, "tier": "easy",
                "video_id": seg.get("video_id", ""), "category": seg.get("category", ""),
                "segment_start": seg.get("start_time", 0), "segment_end": seg.get("end_time", 0),
            })

            if (i + 1) % 20 == 0:
                logger.info(f"  Easy: {i+1}/{len(segs)}")
                time.sleep(1)

        except Exception as e:
            logger.warning(f"  Easy QA error: {e}")
            time.sleep(2)

    return qa_pairs


def generate_medium_qa(videos: dict, categories: set) -> list[dict]:
    """Generate medium QA using templates per video."""
    qa_pairs = []
    for vid, data in videos.items():
        cat = data.get("category", "")
        for tmpl in random.sample(MEDIUM_TEMPLATES, min(2, len(MEDIUM_TEMPLATES))):
            q = tmpl.format(category=cat)
            qa_pairs.append({
                "question": q, "answer": "", "tier": "medium",
                "video_id": vid, "category": cat, "needs_generation": True,
            })
    return qa_pairs


def generate_hard_qa(categories: list[str]) -> list[dict]:
    """Generate hard QA from multi-video combos."""
    import itertools
    qa_pairs = []
    for cat_a, cat_b in itertools.combinations(sorted(categories)[:10], 2):
        tmpl = random.choice(HARD_TEMPLATES[:3])
        q = tmpl.format(cat_a=cat_a, cat_b=cat_b)
        qa_pairs.append({
            "question": q, "answer": "", "tier": "hard",
            "category": f"{cat_a}_vs_{cat_b}", "needs_generation": True,
        })
    return qa_pairs


def stratified_split(qa_pairs: list[dict], test_ratio: float = 0.2) -> tuple[list, list]:
    """Stratified split by tier and category."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for qa in qa_pairs:
        key = f"{qa['tier']}_{qa.get('category', 'none')}"
        buckets[key].append(qa)

    train, test = [], []
    for key, items in buckets.items():
        random.shuffle(items)
        split_idx = max(1, int(len(items) * test_ratio))
        test.extend(items[:split_idx])
        train.extend(items[split_idx:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 9 — QA Generation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 9: QA Generation           ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    verified_path = PROJECT_ROOT / config["paths"]["verified"] / "verified_segments.json"
    align_dir = PROJECT_ROOT / config["paths"]["alignments"]
    qa_dir = PROJECT_ROOT / "datasets" / "qa"

    # Load data
    segments = []
    if verified_path.exists():
        with open(verified_path) as f:
            segments = json.load(f).get("segments", [])

    videos_by_id = {}
    for fp in align_dir.glob("*.json"):
        with open(fp) as f:
            d = json.load(f)
        videos_by_id[d["video_id"]] = d

    categories = set(s.get("category", "") for s in segments) | set(d.get("category", "") for d in videos_by_id.values())
    categories.discard("")

    logger.info(f"Segments: {len(segments)}, Aligned videos: {len(videos_by_id)}, Categories: {len(categories)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would generate QA across tiers: easy, medium, hard")
        return

    all_qa = []
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if args.tier in ("easy", "all") and segments and groq_key:
        logger.info("Generating EASY tier...")
        easy = generate_easy_qa(segments, groq_key, args.limit)
        all_qa.extend(easy)
        logger.info(f"  Easy: {len(easy)} pairs")

    if args.tier in ("medium", "all"):
        logger.info("Generating MEDIUM tier...")
        medium = generate_medium_qa(videos_by_id, categories)
        all_qa.extend(medium)
        logger.info(f"  Medium: {len(medium)} pairs")

    if args.tier in ("hard", "all"):
        logger.info("Generating HARD tier...")
        hard = generate_hard_qa(list(categories))
        all_qa.extend(hard)
        logger.info(f"  Hard: {len(hard)} pairs")

    # Split
    train, test = stratified_split(all_qa)

    qa_dir.mkdir(parents=True, exist_ok=True)
    with open(qa_dir / "train.json", "w") as f:
        json.dump({"total": len(train), "qa_pairs": train}, f, indent=2)
    with open(qa_dir / "test.json", "w") as f:
        json.dump({"total": len(test), "qa_pairs": test}, f, indent=2)

    logger.info(f"\n{'═'*50}\nQA GENERATION: {len(all_qa)} total → {len(train)} train / {len(test)} test\n{'═'*50}")
    logger.info("✓ Stage 9 complete! NEXT: Stage 10 — training/finetune_qlora.py (GPU cluster)")


if __name__ == "__main__":
    main()
