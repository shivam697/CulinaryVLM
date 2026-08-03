"""Recipes router — browse and compare canonical recipes."""

from fastapi import APIRouter, Request
from typing import Optional

from app.models.schemas import RecipeListResponse, CanonicalRecipe, CompareRequest, CompareResponse
from app.services.recipes import get_recipe, list_recipes, compare_recipes

router = APIRouter()


@router.get("/recipes", response_model=RecipeListResponse)
async def get_all_recipes(request: Request):
    """List all canonical recipes."""
    recipes = list_recipes(request.app.state.recipes)
    return RecipeListResponse(
        total=len(recipes),
        recipes=[CanonicalRecipe(**r) for r in recipes],
    )


@router.get("/recipes/{category}")
async def get_recipe_by_category(request: Request, category: str):
    """Get a specific canonical recipe."""
    recipe = get_recipe(request.app.state.recipes, category)
    if not recipe:
        return {"error": f"Recipe not found: {category}"}
    return recipe
