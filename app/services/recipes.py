"""Recipe service — loads and queries canonical recipes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_canonical_recipes(recipe_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all canonical recipe JSONs into memory."""
    recipes: dict[str, dict[str, Any]] = {}
    if not recipe_dir.exists():
        logger.warning(f"Recipe directory not found: {recipe_dir}")
        return recipes

    for filepath in sorted(recipe_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                recipe = json.load(f)
            category = recipe.get("category", filepath.stem.title())
            recipes[category] = recipe
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    return recipes


def get_recipe(recipes: dict, category: str) -> dict[str, Any] | None:
    """Get canonical recipe by category name (case-insensitive)."""
    # Exact match
    if category in recipes:
        return recipes[category]
    # Case-insensitive
    for key, recipe in recipes.items():
        if key.lower() == category.lower():
            return recipe
    return None


def list_recipes(recipes: dict) -> list[dict[str, Any]]:
    """List all recipes sorted by category."""
    return [recipes[k] for k in sorted(recipes.keys())]


def compare_recipes(
    recipes: dict,
    category_a: str,
    category_b: str,
    aspect: str | None = None,
) -> dict[str, Any]:
    """
    Compare two canonical recipes and return structured differences.
    """
    recipe_a = get_recipe(recipes, category_a)
    recipe_b = get_recipe(recipes, category_b)

    if not recipe_a or not recipe_b:
        missing = category_a if not recipe_a else category_b
        return {"error": f"Recipe not found: {missing}"}

    differences = []

    # Compare specific aspects
    compare_fields = [
        ("rice_type", "Rice Type"),
        ("cooking_vessel", "Cooking Vessel"),
        ("dum_method", "Dum Method"),
        ("typical_duration_minutes", "Duration (minutes)"),
    ]

    for field, label in compare_fields:
        if aspect and aspect.lower() not in field.lower() and aspect.lower() not in label.lower():
            continue
        val_a = recipe_a.get(field, "N/A")
        val_b = recipe_b.get(field, "N/A")
        if str(val_a) != str(val_b):
            differences.append({
                "aspect": label,
                "category_a_value": str(val_a),
                "category_b_value": str(val_b),
                "explanation": "",
            })

    # Compare key spices
    if not aspect or "spice" in aspect.lower():
        spices_a = set(recipe_a.get("key_spices", []))
        spices_b = set(recipe_b.get("key_spices", []))
        unique_a = spices_a - spices_b
        unique_b = spices_b - spices_a
        if unique_a or unique_b:
            differences.append({
                "aspect": "Key Spices",
                "category_a_value": ", ".join(sorted(spices_a)),
                "category_b_value": ", ".join(sorted(spices_b)),
                "explanation": (
                    f"Unique to {category_a}: {', '.join(sorted(unique_a)) or 'none'}. "
                    f"Unique to {category_b}: {', '.join(sorted(unique_b)) or 'none'}."
                ),
            })

    # Compare distinguishing features
    if not aspect or "feature" in aspect.lower() or "distinguish" in aspect.lower():
        feat_a = recipe_a.get("distinguishing_features", [])
        feat_b = recipe_b.get("distinguishing_features", [])
        if feat_a or feat_b:
            differences.append({
                "aspect": "Distinguishing Features",
                "category_a_value": "; ".join(feat_a[:3]),
                "category_b_value": "; ".join(feat_b[:3]),
                "explanation": "",
            })

    # Compare step counts
    if not aspect or "step" in aspect.lower():
        steps_a = len(recipe_a.get("steps", []))
        steps_b = len(recipe_b.get("steps", []))
        defining_a = sum(1 for s in recipe_a.get("steps", []) if s.get("is_defining_step"))
        defining_b = sum(1 for s in recipe_b.get("steps", []) if s.get("is_defining_step"))
        differences.append({
            "aspect": "Recipe Steps",
            "category_a_value": f"{steps_a} total, {defining_a} defining",
            "category_b_value": f"{steps_b} total, {defining_b} defining",
            "explanation": "",
        })

    summary = (
        f"{category_a} biryani uses {recipe_a.get('rice_type', 'rice')} with "
        f"{recipe_a.get('dum_method', 'dum cooking')}, while {category_b} biryani uses "
        f"{recipe_b.get('rice_type', 'rice')} with {recipe_b.get('dum_method', 'dum cooking')}."
    )

    return {
        "category_a": category_a,
        "category_b": category_b,
        "differences": differences,
        "summary": summary,
    }
