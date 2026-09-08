#!/usr/bin/env python3
"""
CulinaryVLM — Phase 0B: Canonical Recipe Extension
═══════════════════════════════════════════════════════════════

Generates canonical recipe templates for each biryani category
using a local Ollama server (qwen2.5:32b-instruct). These recipes
serve as the alignment reference for Stage 7 (multimodal alignment
with DTW).

Format follows the paper's Appendix A structure + an added
ingredient_aliases field for multilingual matching.

Usage:
    python pipeline/00b_canonical_recipes.py
    python pipeline/00b_canonical_recipes.py --categories Hyderabadi Delhi
    python pipeline/00b_canonical_recipes.py --dry-run
    python pipeline/00b_canonical_recipes.py --offline   # use built-in fallback recipes

Requires: Ollama server running with qwen2.5:32b-instruct
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

# ─── Ollama client import ────────────────────────────────────

sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.utils.ollama_client import check_ollama, ollama_chat  # noqa: E402

# ─── Prompt Template ─────────────────────────────────────────

CANONICAL_RECIPE_PROMPT = """You are an expert in Indian regional cuisine, specifically biryani preparation methods.

Generate a canonical (reference/standard) recipe for **{category} Chicken Biryani**.

The recipe must capture the DEFINING procedural steps that distinguish {category} style from other styles.
Focus on: what makes this style unique in technique, layering, spice profile, and cooking method.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "schema_version": "2.0",
  "category": "{category}",
  "full_name": "{category} Chicken Biryani",
  "region": "<geographic region of origin>",
  "protein": "chicken",
  "utensil_hierarchy": ["<primary vessel>", "<secondary utensil>", "<etc>"],
  "marination_strategy": "<brief description: wet/dry, duration, acid+dairy components>",
  "spice_intensity": "<one of: low, medium, medium-high, high>",
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
      "ingredients_used": ["<ingredient1>", "<ingredient2>"],
      "visible_objects": ["<object visible in video at this step>"],
      "ingredient_state": {{"<ingredient>": "<physical state, e.g. '70% cooked', 'raw marinated'>"}},
      "expected_visual_features": ["<what a camera would see, e.g. 'golden fried onions'>"],
      "expected_audio": "<dominant sound, e.g. 'sizzling oil', 'bubbling water'>",
      "common_mistakes": ["<typical error a home cook might make>"],
      "recovery_actions": ["<how to fix the mistake>"]
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
- If a language translation is unknown, use null
- For each step, include visible_objects (what a camera sees), ingredient_state (physical state of each ingredient AT THIS POINT), expected_visual_features, expected_audio, common_mistakes, and recovery_actions
- spice_intensity should reflect the overall heat/spice level: low, medium, medium-high, or high
- utensil_hierarchy lists ALL utensils needed in order of importance
- marination_strategy summarizes the marination approach concisely"""


# ─── Built-in Fallback Recipes ───────────────────────────────
# These are used when --offline is set or Ollama is unavailable.
# They cover the minimum needed categories.

FALLBACK_RECIPES: dict[str, dict[str, Any]] = {
    "Hyderabadi": {
        "schema_version": "2.0",
        "category": "Hyderabadi",
        "full_name": "Hyderabadi Chicken Biryani",
        "region": "Telangana / Andhra Pradesh (Hyderabad)",
        "protein": "chicken",
        "utensil_hierarchy": ["handi (heavy-bottomed round pot)", "tawa/kadhai (for frying birista)", "strainer", "mixing bowl", "rolling pin (for dough seal)"],
        "marination_strategy": "wet yogurt-based, 2-6 hrs, acid (lemon juice) + dairy (yogurt) + aromatics (mint, fried onions)",
        "spice_intensity": "medium-high",
        "distinguishing_features": [
            "Kacchi (raw) method — raw marinated meat layered with parboiled rice",
            "Sealed dum cooking with dough-sealed lid (purdah)",
            "Generous use of saffron, fried onions (birista), and mint",
            "Basmati rice soaked and parboiled with whole spices",
            "Green chili-mint-coriander paste as key flavoring"
        ],
        "steps": [
            {"step_number": 1, "action": "Marinate chicken", "description": "Marinate chicken with yogurt, ginger-garlic paste, green chili paste, red chili powder, turmeric, biryani masala, fried onions, mint, coriander, lemon juice, and salt for 2-6 hours.", "duration_minutes": 180, "is_defining_step": True, "ingredients_used": ["chicken", "yogurt", "ginger-garlic paste", "green chilies", "mint", "coriander", "fried onions", "biryani masala"],
             "visible_objects": ["mixing bowl", "chicken pieces", "yogurt", "spice powders", "mint leaves", "fried onions"],
             "ingredient_state": {"chicken": "raw, cut into pieces", "yogurt": "whisked", "onions": "deep-fried golden (birista)"},
             "expected_visual_features": ["red-orange marinated chicken", "thick yogurt coating", "visible mint and fried onions mixed in"],
             "expected_audio": "mixing and squelching sounds",
             "common_mistakes": ["insufficient marination time (under 2 hrs)", "not scoring chicken pieces", "using too little yogurt"],
             "recovery_actions": ["extend marination time in refrigerator", "score pieces and re-coat generously"]},
            {"step_number": 2, "action": "Soak basmati rice", "description": "Wash and soak long-grain basmati rice in water for 30-45 minutes.", "duration_minutes": 30, "is_defining_step": False, "ingredients_used": ["basmati rice"],
             "visible_objects": ["bowl", "rice", "water"],
             "ingredient_state": {"rice": "dry, uncooked, soaking in water"},
             "expected_visual_features": ["white rice grains submerged in clear water", "water turning slightly starchy"],
             "expected_audio": "water pouring, gentle swishing",
             "common_mistakes": ["not washing rice enough (residual starch)", "soaking too long (grains break)"],
             "recovery_actions": ["drain and rinse again before cooking", "reduce parboil time if over-soaked"]},
            {"step_number": 3, "action": "Parboil rice", "description": "Boil rice in water with whole spices (bay leaf, cardamom, cloves, cinnamon, star anise) until 70% cooked. Drain.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": ["basmati rice", "bay leaf", "cardamom", "cloves", "cinnamon", "star anise", "salt"],
             "visible_objects": ["large pot", "boiling water", "rice", "whole spices floating", "strainer"],
             "ingredient_state": {"rice": "70% cooked — grains elongated but still firm in center", "water": "starchy, rolling boil"},
             "expected_visual_features": ["white elongated grains", "rolling boil", "whole spices floating on surface"],
             "expected_audio": "vigorous bubbling, water boiling",
             "common_mistakes": ["overcooking rice past 70%", "not draining immediately", "forgetting whole spices"],
             "recovery_actions": ["spread rice on flat tray to stop cooking", "rinse briefly with cold water to halt cooking"]},
            {"step_number": 4, "action": "Fry onions for birista", "description": "Thinly slice onions and deep fry until dark golden brown and crispy. Reserve for layering.", "duration_minutes": 15, "is_defining_step": True, "ingredients_used": ["onions", "oil"],
             "visible_objects": ["kadhai/deep pan", "sliced onions", "hot oil", "slotted spoon"],
             "ingredient_state": {"onions": "thinly sliced, turning dark golden brown and crispy", "oil": "hot, shimmering"},
             "expected_visual_features": ["dark golden-brown crispy onion strands", "oil bubbling around onions"],
             "expected_audio": "continuous sizzling, crackling",
             "common_mistakes": ["uneven slicing (some burn, some stay raw)", "removing too early (pale, not crispy)", "oil not hot enough"],
             "recovery_actions": ["use mandoline for even slices", "fry in smaller batches", "drain on paper towels immediately"]},
            {"step_number": 5, "action": "Layer marinated chicken", "description": "Spread raw marinated chicken evenly at the bottom of a heavy-bottomed pot (handi).", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["marinated chicken"],
             "visible_objects": ["handi", "marinated chicken pieces", "marinade"],
             "ingredient_state": {"chicken": "raw marinated, coated in red-orange marinade"},
             "expected_visual_features": ["red-orange chicken pieces arranged at bottom of handi", "marinade pooling"],
             "expected_audio": "placement sounds, wet slapping",
             "common_mistakes": ["uneven distribution (some spots cook faster)", "discarding excess marinade"],
             "recovery_actions": ["rearrange pieces evenly", "pour all marinade over the chicken"]},
            {"step_number": 6, "action": "Layer parboiled rice", "description": "Spread parboiled rice evenly over the chicken layer.", "duration_minutes": 5, "is_defining_step": False, "ingredients_used": ["parboiled rice"],
             "visible_objects": ["handi with chicken", "parboiled rice being spread", "spoon"],
             "ingredient_state": {"rice": "70% cooked, drained, still warm", "chicken": "raw marinated, underneath"},
             "expected_visual_features": ["white rice layer covering the red chicken layer", "visible whole spices in rice"],
             "expected_audio": "gentle placing sounds",
             "common_mistakes": ["pressing rice down (compacts, prevents steam circulation)", "uneven layer thickness"],
             "recovery_actions": ["gently fluff rice with fork to loosen", "redistribute for even coverage"]},
            {"step_number": 7, "action": "Add saffron and garnish", "description": "Drizzle saffron milk, ghee, fried onions, mint leaves, and coriander over the rice layer.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["saffron", "milk", "ghee", "fried onions", "mint", "coriander"],
             "visible_objects": ["saffron milk (golden)", "ghee", "fried onions", "mint leaves", "coriander leaves"],
             "ingredient_state": {"saffron": "dissolved in warm milk, deep golden", "ghee": "melted, liquid", "onions": "crispy fried birista"},
             "expected_visual_features": ["golden saffron streaks on white rice", "green mint and coriander specks", "brown fried onion strands"],
             "expected_audio": "drizzling liquid, sprinkling",
             "common_mistakes": ["using too little saffron", "not warming milk before adding saffron"],
             "recovery_actions": ["add more saffron milk if rice looks too plain", "warm milk first then steep saffron 10 min"]},
            {"step_number": 8, "action": "Seal with dough (purdah)", "description": "Seal the lid with wheat flour dough to trap steam. This is the defining Hyderabadi dum technique.", "duration_minutes": 5, "is_defining_step": True, "ingredients_used": ["wheat flour dough"],
             "visible_objects": ["handi with rice", "dough rope", "lid"],
             "ingredient_state": {"dough": "pliable wheat flour dough rope pressed around lid edge"},
             "expected_visual_features": ["dough strip sealing the gap between lid and pot rim", "no steam escaping"],
             "expected_audio": "pressing/sealing sounds, then silence (steam trapped)",
             "common_mistakes": ["gaps in dough seal (steam escapes, rice dries)", "dough too dry (cracks)"],
             "recovery_actions": ["press dough firmly and patch any gaps", "add a drop of water to dough if cracking"]},
            {"step_number": 9, "action": "Cook on dum", "description": "Place sealed pot on high heat for 5 minutes, then reduce to very low heat for 25-35 minutes. Do not open.", "duration_minutes": 35, "is_defining_step": True, "ingredients_used": [],
             "visible_objects": ["sealed handi on stove", "tawa underneath (optional, for heat diffusion)"],
             "ingredient_state": {"rice": "cooking from 70% to fully done via steam", "chicken": "cooking from raw to fully done via trapped steam"},
             "expected_visual_features": ["sealed pot on stove", "no visible steam escaping", "slight aroma building"],
             "expected_audio": "initial hissing/steam sounds, then very quiet on low heat",
             "common_mistakes": ["opening lid during dum (releases steam)", "heat too high throughout (bottom burns)", "insufficient dum time"],
             "recovery_actions": ["never open — trust the process", "place a tawa under handi for heat diffusion", "extend dum time by 5-10 min if unsure"]},
            {"step_number": 10, "action": "Rest and serve", "description": "Turn off heat, let rest for 10 minutes without opening. Break seal, gently mix layers, serve with raita and mirchi ka salan.", "duration_minutes": 10, "is_defining_step": False, "ingredients_used": [],
             "visible_objects": ["sealed handi", "serving dish", "raita", "mirchi ka salan"],
             "ingredient_state": {"rice": "fully cooked, fluffy, saffron-streaked", "chicken": "fully cooked, tender, falling off bone"},
             "expected_visual_features": ["bicolor rice (white and saffron-gold)", "tender chicken visible when mixing", "steam rising when seal broken"],
             "expected_audio": "seal cracking, steam release, gentle mixing",
             "common_mistakes": ["mixing too vigorously (breaks rice grains)", "not resting (flavors not settled)"],
             "recovery_actions": ["use flat spatula and fold gently from edges", "always rest minimum 5-10 min"]}
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


def generate_recipe_ollama(
    category: str,
    host: str,
    model: str,
) -> dict[str, Any] | None:
    """Call local Ollama server to generate a canonical recipe."""
    prompt = CANONICAL_RECIPE_PROMPT.format(category=category)

    content = ollama_chat(
        host=host,
        model=model,
        system="You are an expert in Indian regional cuisine. Return ONLY valid JSON, no markdown formatting.",
        user=prompt,
        num_predict=4000,
        temperature=0.3,
        json_format=True,
    )

    if not content:
        logger.error(f"Empty response for {category}")
        return None

    try:
        recipe = json.loads(content)
        return recipe
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {category}: {e}")
        logger.error(f"Raw content: {content[:200]}...")
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
    """Save a canonical recipe to JSON file atomically.

    Writes to a temporary file first, then uses os.replace() so that
    the final path is either the old file or the complete new file —
    never a half-written or missing file.
    """
    import tempfile

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = category.lower().replace("/", "_").replace(" ", "_") + ".json"
    path = output_dir / filename

    # Ensure schema_version is always present
    recipe.setdefault("schema_version", "2.0")

    # Write to temp file in same directory, then atomic rename
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)  # atomic on same filesystem
    except Exception:
        # Clean up temp file on failure; old file stays intact
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

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
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing recipe files (re-generate with current schema)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="qwen2.5:32b-instruct",
                        help="Ollama model name (default: qwen2.5:32b-instruct)")
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
        logger.info("  Or ensure Ollama is running with the target model")
        return

    # Check Ollama connectivity (unless offline)
    use_ollama = not args.offline
    if use_ollama:
        check_ollama(args.host, args.model)
        logger.info(f"Using Ollama {args.model} for recipe generation")
    else:
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
        if existing.exists() and not args.force:
            logger.info(f"  ⊘ Already exists: {filename} (use --force to overwrite)")
            skipped += 1
            continue
        elif existing.exists() and args.force:
            logger.info(f"  ♻ Overwriting: {filename} (--force)")

        if use_ollama:
            recipe = generate_recipe_ollama(cat, args.host, args.model)
        elif cat in FALLBACK_RECIPES:
            recipe = FALLBACK_RECIPES[cat]
        else:
            logger.warning(f"  ⊘ No fallback recipe for '{cat}' — skipping (need Ollama)")
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
