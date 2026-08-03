#!/usr/bin/env python3
"""
CulinaryVLM — Phase 0B: Canonical Recipe Extension
═══════════════════════════════════════════════════════════════

Generates canonical recipe templates for each biryani category
using Groq Llama-3.1-70B. These recipes serve as the alignment
reference for Stage 7 (multimodal alignment with DTW).

Format follows the paper's Appendix A structure + an added
ingredient_aliases field for multilingual matching.

Usage:
    python pipeline/00b_canonical_recipes.py
    python pipeline/00b_canonical_recipes.py --categories Hyderabadi Delhi
    python pipeline/00b_canonical_recipes.py --dry-run
    python pipeline/00b_canonical_recipes.py --offline   # use built-in fallback recipes

Requires: GROQ_API_KEY in .env or environment
Input:  datasets/categorized/video_metadata.json
Output: configs/canonical_recipes/{category}.json
"""

from __future__ import annotations

import argparse
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

# ─── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase_0b")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"

# ─── Rate limit constants ────────────────────────────────────
GROQ_RPM_LIMIT = 30   # requests per minute (free tier)
GROQ_DELAY = 2.5       # seconds between requests (safe margin)

# ─── Prompt Template ─────────────────────────────────────────

CANONICAL_RECIPE_PROMPT = """You are an expert in Indian regional cuisine, specifically biryani preparation methods.

Generate a canonical (reference/standard) recipe for **{category} Chicken Biryani**.

The recipe must capture the DEFINING procedural steps that distinguish {category} style from other styles.
Focus on: what makes this style unique in technique, layering, spice profile, and cooking method.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "category": "{category}",
  "full_name": "{category} Chicken Biryani",
  "region": "<geographic region of origin>",
  "distinguishing_features": [
    "<3-5 features that make this style unique>"
  ],
  "steps": [
    {{
      "step_number": 1,
      "action": "<main cooking action verb phrase>",
      "description": "<detailed 1-2 sentence description>",
      "duration_minutes": <estimated time or null>,
      "is_defining_step": <true if this step is unique to this style>,
      "ingredients_used": ["<ingredient1>", "<ingredient2>"]
    }}
  ],
  "ingredient_aliases": {{
    "<english_ingredient>": {{
      "hi": "<Hindi name>",
      "te": "<Telugu name or null>",
      "ml": "<Malayalam name or null>",
      "bn": "<Bengali name or null>",
      "ur": "<Urdu name or null>"
    }}
  }},
  "key_spices": ["<spice1>", "<spice2>"],
  "rice_type": "<type of rice used>",
  "cooking_vessel": "<vessel typically used>",
  "dum_method": "<open/sealed dum/no dum/other>",
  "typical_duration_minutes": <total estimated cooking time>,
  "notes": "<any additional notes about regional variations>"
}}

IMPORTANT:
- Include 8-15 steps covering: marination, rice preparation, layering, dum/cooking, resting
- ingredient_aliases should cover at least 10 key ingredients with multilingual names
- Focus on PROCEDURAL accuracy — this will be used to align with real cooking video segments
- If a language translation is unknown, use null"""


# ─── Built-in Fallback Recipes ───────────────────────────────
# These are used when --offline is set or Groq API is unavailable.
# They cover the minimum needed categories.

FALLBACK_RECIPES: dict[str, dict[str, Any]] = {
    "Hyderabadi": {
        "category": "Hyderabadi",
        "full_name": "Hyderabadi Chicken Biryani",
        "region": "Telangana / Andhra Pradesh (Hyderabad)",
        "distinguishing_features": [
            "Kacchi (raw) method — raw marinated meat layered with parboiled rice",
            "Sealed dum cooking with dough-sealed lid (purdah)",
            "Generous use of saffron, fried onions (birista), and mint",
            "Basmati rice soaked and parboiled with whole spices",
            "Green chili-mint-coriander paste as key flavoring"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken with yogurt, ginger-garlic paste, green chili paste, red chili powder, turmeric, biryani masala, fried onions, mint, coriander, lemon juice, and salt for 2-6 hours.", "duration_minutes": 180, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "green chilies", "mint", "coriander", "fried onions", "biryani masala"]},
            {"step_number": 2, "action": "Soak basmati rice", "description": "Wash and soak long-grain basmati rice in water for 30-45 minutes.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": ["basmati rice"]},
            {"step_number": 3, "action": "Parboil rice", "description": "Boil rice in water with whole spices (bay leaf, cardamom, cloves, cinnamon, star anise) until 70% cooked. Drain.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cloves", "cinnamon", "star anise", "salt"]},
            {"step_number": 4, "action": "Fry onions for birista", "description": "Thinly slice onions and deep fry until dark golden brown and crispy. Reserve for layering.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["onions", "oil"]},
            {"step_number": 5, "action": "Layer marinated chicken", "description": "Spread raw marinated chicken evenly at the bottom of a heavy-bottomed pot (handi).", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["marinated chicken"]},
            {"step_number": 6, "action": "Layer parboiled rice", "description": "Spread parboiled rice evenly over the chicken layer.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["parboiled rice"]},
            {"step_number": 7, "action": "Add saffron and garnish", "description": "Drizzle saffron milk, ghee, fried onions, mint leaves, and coriander over the rice layer.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "milk", "ghee", "fried onions", "mint", "coriander"]},
            {"step_number": 8, "action": "Seal with dough (purdah)", "description": "Seal the lid with wheat flour dough to trap steam. This is the defining Hyderabadi dum technique.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["wheat flour dough"]},
            {"step_number": 9, "action": "Cook on dum", "description": "Place sealed pot on high heat for 5 minutes, then reduce to very low heat for 25-35 minutes. Do not open.", "duration_minutes": 35, "is_defining_step": True, "ingredients_used": []},
            {"step_number": 10, "action": "Rest and serve", "description": "Turn off heat, let rest for 10 minutes without opening. Break seal, gently mix layers, serve with raita and mirchi ka salan.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": []}
        ],
        "ingredient_aliases": {
            "chicken": {"hi": "मुर्गा/चिकन", "te": "కోడి", "ml": "കോഴി", "bn": "মুরগি", "ur": "مرغ"},
            "basmati rice": {"hi": "बासमती चावल", "te": "బాస్మతి బియ్యం", "ml": "ബസ്മതി അരി", "bn": "বাসমতী চাল", "ur": "باسمتی چاول"},
            "yogurt": {"hi": "दही", "te": "పెరుగు", "ml": "തൈര്", "bn": "দই", "ur": "دہی"},
            "onion": {"hi": "प्याज", "te": "ఉల్లిపాయ", "ml": "ഉള്ളി", "bn": "পেঁয়াজ", "ur": "پیاز"},
            "ghee": {"hi": "घी", "te": "నెయ్యి", "ml": "നെയ്യ്", "bn": "ঘি", "ur": "گھی"},
            "saffron": {"hi": "केसर", "te": "కుంకుమపువ్వు", "ml": "കുങ്കുമപ്പൂവ്", "bn": "জাফরান", "ur": "زعفران"},
            "mint": {"hi": "पुदीना", "te": "పుదీనా", "ml": "പുതിന", "bn": "পুদিনা", "ur": "پودینہ"},
            "coriander": {"hi": "धनिया", "te": "కొత్తిమీర", "ml": "മല്ലി", "bn": "ধনে", "ur": "دھنیا"},
            "green chili": {"hi": "हरी मिर्च", "te": "పచ్చి మిర్చి", "ml": "പച്ചമുളക്", "bn": "কাঁচা লঙ্কা", "ur": "ہری مرچ"},
            "ginger-garlic paste": {"hi": "अदरक-लहसुन पेस्ट", "te": "అల్లం-వెల్లుల్లి పేస్ట్", "ml": "ഇഞ്ചി-വെളുത്തുള്ളി പേസ്റ്റ്", "bn": "আদা-রসুন বাটা", "ur": "ادرک لہسن پیسٹ"}
        },
        "key_spices": ["saffron", "cardamom", "cloves", "cinnamon", "star anise", "mace", "nutmeg", "bay leaf", "biryani masala"],
        "rice_type": "Long-grain Basmati",
        "cooking_vessel": "Heavy-bottomed handi (round pot)",
        "dum_method": "Sealed dum with dough (purdah)",
        "typical_duration_minutes": 120,
        "notes": "Hyderabadi biryani uses the kacchi (raw) method where uncooked marinated meat cooks entirely via dum steam. The Nizam's kitchen legacy. Always served with mirchi ka salan and raita."
    },
}


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config."""
    path = PROJECT_ROOT / config_path
    if not path.exists():
        logger.error(f"Config not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_video_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Load categorized video metadata from Phase 0A output."""
    path = PROJECT_ROOT / config["paths"]["categorized_output"]
    if not path.exists():
        logger.error(f"Video metadata not found: {path}")
        logger.error("Run Phase 0A first: python pipeline/00a_categorize.py")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_categories_with_counts(metadata: dict[str, Any]) -> dict[str, int]:
    """Get categories that have enough videos for recipe generation."""
    from collections import Counter
    cats = Counter(v["assigned_category"] for v in metadata["videos"])
    # Only generate recipes for categories with >= 1 video
    # (even 1-video categories like Ambur and Dindigul deserve a canonical recipe)
    return dict(cats.most_common())


def generate_recipe_groq(
    category: str,
    api_key: str,
    model: str = "llama-3.1-70b-versatile",
) -> dict[str, Any] | None:
    """Call Groq API to generate a canonical recipe."""
    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return None

    client = Groq(api_key=api_key)
    prompt = CANONICAL_RECIPE_PROMPT.format(category=category)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Indian regional cuisine. Return ONLY valid JSON, no markdown formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            logger.error(f"Empty response for {category}")
            return None

        recipe = json.loads(content)
        return recipe

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {category}: {e}")
        logger.error(f"Raw content: {content[:200]}...")  # type: ignore[possibly-undefined]
        return None
    except Exception as e:
        logger.error(f"Groq API error for {category}: {e}")
        return None


def validate_recipe(recipe: dict[str, Any], category: str) -> list[str]:
    """Validate recipe structure and return list of issues."""
    issues: list[str] = []

    required_fields = [
        "category", "full_name", "region", "steps",
        "ingredient_aliases", "key_spices", "rice_type",
        "cooking_vessel", "dum_method",
    ]
    for field in required_fields:
        if field not in recipe:
            issues.append(f"Missing required field: {field}")

    if "steps" in recipe:
        steps = recipe["steps"]
        if not isinstance(steps, list):
            issues.append("'steps' is not a list")
        elif len(steps) < 5:
            issues.append(f"Too few steps: {len(steps)} (minimum 5)")

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"Step {i+1} is not a dict")
                continue
            for key in ["step_number", "action", "description"]:
                if key not in step:
                    issues.append(f"Step {i+1} missing '{key}'")

    if "ingredient_aliases" in recipe:
        aliases = recipe["ingredient_aliases"]
        if not isinstance(aliases, dict):
            issues.append("'ingredient_aliases' is not a dict")
        elif len(aliases) < 5:
            issues.append(f"Too few ingredient aliases: {len(aliases)} (minimum 5)")

    if recipe.get("category", "").lower() != category.lower():
        issues.append(f"Category mismatch: got '{recipe.get('category')}', expected '{category}'")

    return issues


def save_recipe(recipe: dict[str, Any], category: str, output_dir: Path) -> None:
    """Save a canonical recipe to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = category.lower().replace("/", "_").replace(" ", "_") + ".json"
    path = output_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ Saved: {path.name} ({len(recipe.get('steps', []))} steps, "
                f"{len(recipe.get('ingredient_aliases', {}))} aliases)")


def main() -> None:
    """Phase 0B main entry point."""
    parser = argparse.ArgumentParser(
        description="CulinaryVLM Phase 0B — Generate canonical recipes",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without calling API")
    parser.add_argument("--offline", action="store_true",
                        help="Use built-in fallback recipes (no API call)")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only generate for these categories")
    parser.add_argument("--min-videos", type=int, default=1,
                        help="Minimum videos in category to generate recipe (default: 1)")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Phase 0B: Canonical Recipes      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)
    metadata = load_video_metadata(config)
    output_dir = PROJECT_ROOT / config["paths"]["canonical_recipes"]

    # Get categories to process
    all_cats = get_categories_with_counts(metadata)
    logger.info(f"Found {len(all_cats)} categories in dataset")

    # Filter
    target_cats = {}
    for cat, count in all_cats.items():
        if cat == "Generic":
            continue  # Skip generic — no canonical recipe for "generic biryani"
        if count < args.min_videos:
            continue
        if args.categories and cat not in args.categories:
            continue
        target_cats[cat] = count

    logger.info(f"Generating recipes for {len(target_cats)} categories:")
    for cat, count in sorted(target_cats.items()):
        logger.info(f"  {cat:20s}: {count:4d} videos")

    if args.dry_run:
        logger.info("[DRY RUN] Would generate recipes for the above categories")
        logger.info("  Pass --offline to use built-in fallback recipes")
        logger.info("  Or set GROQ_API_KEY to use Groq Llama-3.1-70B")
        return

    # Check for Groq API key
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    use_groq = bool(groq_api_key) and not args.offline

    if use_groq:
        logger.info("Using Groq Llama-3.1-70B for recipe generation")
        model = config.get("agent", {}).get("llm_model", "llama-3.1-70b-versatile")
    else:
        if not args.offline:
            logger.warning("GROQ_API_KEY not set — falling back to built-in recipes")
            logger.warning("Set GROQ_API_KEY or use --offline flag")
        logger.info("Using built-in fallback recipes")

    # Generate recipes
    generated = 0
    failed = 0
    skipped = 0

    for cat in sorted(target_cats.keys()):
        logger.info(f"\n  Processing: {cat}")

        # Check if recipe already exists
        filename = cat.lower().replace("/", "_").replace(" ", "_") + ".json"
        existing = output_dir / filename
        if existing.exists():
            logger.info(f"  ⊘ Already exists: {filename} (use --force to overwrite)")
            skipped += 1
            continue

        if use_groq:
            recipe = generate_recipe_groq(cat, groq_api_key, model)
            time.sleep(GROQ_DELAY)  # Rate limit
        elif cat in FALLBACK_RECIPES:
            recipe = FALLBACK_RECIPES[cat]
        else:
            logger.warning(f"  ⊘ No fallback recipe for '{cat}' — skipping (need Groq API)")
            skipped += 1
            continue

        if recipe is None:
            failed += 1
            continue

        # Validate
        issues = validate_recipe(recipe, cat)
        if issues:
            logger.warning(f"  ⚠ Validation issues for {cat}:")
            for issue in issues:
                logger.warning(f"    - {issue}")
            # Save anyway — issues are warnings, not blockers
            # (the LLM may use slightly different category casing etc.)
            if "category" in recipe:
                recipe["category"] = cat  # Fix casing

        save_recipe(recipe, cat, output_dir)
        generated += 1

    logger.info("")
    logger.info("═" * 50)
    logger.info(f"  Generated: {generated}")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Skipped:   {skipped}")
    logger.info("═" * 50)
    logger.info("")
    logger.info("✓ Phase 0B complete!")
    logger.info("  NEXT: Run Phase 0C (metadata tagging), then Stage 1 (download)")


if __name__ == "__main__":
    main()
