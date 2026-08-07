#!/usr/bin/env python3
"""
CulinaryVLM — Stage 9: QA Generation
═══════════════════════════════════════

Generates multi-tier QA pairs:
  - Easy: Groq Llama-3.1-8B, per segment
  - Medium: Gemini Flash, per video + multilingual templates
  - Hard: Gemini Flash, multi-video 2-5 combos
  - Expert (v2.0): cooking science, substitutions, failure analysis,
    regional adaptation questions

Stratified train/test split by tier AND language.

Usage:
    python pipeline/09_generate_qa.py
    python pipeline/09_generate_qa.py --tier easy --limit 100
    python pipeline/09_generate_qa.py --tier expert
    python pipeline/09_generate_qa.py --dry-run

Input:  datasets/alignments/, datasets/verified/, datasets/comparisons/
        configs/canonical_recipes/{category}.json
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

EXPERT_TEMPLATES = [
    # Cooking science
    "Why is the rice cooked to exactly 70% in {category} biryani? What would happen if it were fully cooked before layering?",
    "Explain the science behind sealing the pot with dough during dum. How does this affect moisture, pressure, and flavor development?",
    "Why does {category} biryani use {fat_type} instead of other fats? How does the smoke point and flavor profile affect the final dish?",
    # Substitutions
    "If basmati rice is unavailable, what substitutions could work for {category} biryani, and what adjustments would be needed?",
    "How would substituting yogurt with lemon juice in {category} biryani marination affect the meat texture and flavor?",
    "What happens if you replace ghee with oil in {category} biryani? Which stages are most affected?",
    # Failure analysis
    "What are the signs that the rice in {category} biryani has been overcooked during parboiling, and how can this be recovered?",
    "If the dum seal breaks during cooking, what impact does this have and how should the cook respond?",
    "Why might the bottom layer burn in {category} biryani, and what preventive measures are shown in expert cooking videos?",
    # Regional adaptation
    "How would you adapt {cat_a} biryani technique for {cat_b} regional ingredients and taste preferences?",
    "What are the key compromises when cooking {category} biryani outside its region of origin, and how do diaspora cooks handle them?",
    "Compare how {cat_a} and {cat_b} biryani handle the same ingredient differently due to regional climate and tradition.",
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
                # Existing keys — unchanged
                "question": template, "answer": answer, "tier": "easy",
                "video_id": seg.get("video_id", ""), "category": seg.get("category", ""),
                "segment_start": seg.get("start_time", 0), "segment_end": seg.get("end_time", 0),
                # v2.0 additive keys
                "instruction": template,
                "context": context,
                "canonical_recipe_ref": f"configs/canonical_recipes/{seg.get('category', '').lower()}.json",
                "video_metadata_ref": {
                    "video_id": seg.get("video_id", ""),
                    "segment_action": seg.get("action", ""),
                    "cooking_technique": seg.get("cooking_technique", ""),
                    "cooking_stage": seg.get("cooking_stage", ""),
                },
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


def generate_expert_qa(
    categories: list[str],
    recipes: dict[str, dict[str, Any]] | None = None,
) -> list[dict]:
    """Generate expert-tier QA: cooking science, substitutions, failure
    analysis, and regional adaptation questions.

    Follows the same pattern as generate_hard_qa().  Uses canonical recipe
    metadata when available to fill template placeholders.
    """
    import itertools

    if recipes is None:
        recipes = {}
    qa_pairs: list[dict] = []
    sorted_cats = sorted(categories)[:10]

    for cat in sorted_cats:
        recipe = recipes.get(cat, {})
        # Determine fat type for science templates
        fat_type = "ghee"
        for feat in recipe.get("distinguishing_features", []):
            fl = feat.lower()
            if "mustard oil" in fl:
                fat_type = "mustard oil"
                break
            elif "coconut" in fl:
                fat_type = "coconut oil"
                break

        # Single-category expert templates (science, substitution, failure)
        single_templates = [t for t in EXPERT_TEMPLATES if "{cat_a}" not in t and "{cat_b}" not in t]
        for tmpl in random.sample(single_templates, min(3, len(single_templates))):
            q = tmpl.format(category=cat, fat_type=fat_type, n=len(sorted_cats))
            qa_pairs.append({
                "question": q, "answer": "", "tier": "expert",
                "category": cat, "needs_generation": True,
                "canonical_recipe_ref": f"configs/canonical_recipes/{cat.lower()}.json",
                "expert_subtopic": _classify_expert_subtopic(tmpl),
            })

    # Cross-category expert templates (regional adaptation)
    cross_templates = [t for t in EXPERT_TEMPLATES if "{cat_a}" in t and "{cat_b}" in t]
    for cat_a, cat_b in itertools.combinations(sorted_cats, 2):
        if random.random() > 0.3:  # Sample ~30% of pairs to keep count manageable
            continue
        tmpl = random.choice(cross_templates)
        q = tmpl.format(cat_a=cat_a, cat_b=cat_b, category=cat_a, fat_type="ghee", n=2)
        qa_pairs.append({
            "question": q, "answer": "", "tier": "expert",
            "category": f"{cat_a}_vs_{cat_b}", "needs_generation": True,
            "expert_subtopic": "regional_adaptation",
        })

    return qa_pairs


def _classify_expert_subtopic(template: str) -> str:
    """Classify an expert template into a subtopic."""
    tl = template.lower()
    if any(w in tl for w in ["science", "why", "explain", "affect"]):
        return "cooking_science"
    elif any(w in tl for w in ["substitut", "replace", "unavailable"]):
        return "substitution"
    elif any(w in tl for w in ["fail", "burn", "break", "overcook", "recover", "sign"]):
        return "failure_analysis"
    elif any(w in tl for w in ["adapt", "region", "diaspora", "compromise"]):
        return "regional_adaptation"
    return "general"


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
    parser.add_argument("--tier", choices=["easy", "medium", "hard", "expert", "all"], default="all")
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

    # Load canonical recipes for expert tier
    recipe_dir = PROJECT_ROOT / "configs" / "canonical_recipes"
    recipes: dict[str, dict] = {}
    for fp in recipe_dir.glob("*.json"):
        try:
            with open(fp) as f:
                r = json.load(f)
            recipes[r.get("category", fp.stem.title())] = r
        except (json.JSONDecodeError, OSError):
            pass

    if args.dry_run:
        logger.info("[DRY RUN] Would generate QA across tiers: easy, medium, hard, expert")
        logger.info(f"  Canonical recipes loaded: {len(recipes)}")
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

    if args.tier in ("expert", "all"):
        logger.info("Generating EXPERT tier...")
        expert = generate_expert_qa(list(categories), recipes=recipes)
        all_qa.extend(expert)
        logger.info(f"  Expert: {len(expert)} pairs")

    # Split
    train, test = stratified_split(all_qa)

    qa_dir.mkdir(parents=True, exist_ok=True)
    with open(qa_dir / "train.json", "w") as f:
        json.dump({"schema_version": "2.0", "total": len(train), "qa_pairs": train}, f, indent=2)
    with open(qa_dir / "test.json", "w") as f:
        json.dump({"schema_version": "2.0", "total": len(test), "qa_pairs": test}, f, indent=2)

    logger.info(f"\n{'═'*50}\nQA GENERATION: {len(all_qa)} total → {len(train)} train / {len(test)} test\n{'═'*50}")
    logger.info("✓ Stage 9 complete! NEXT: Stage 10 — training/finetune_qlora.py (GPU cluster)")


if __name__ == "__main__":
    main()
