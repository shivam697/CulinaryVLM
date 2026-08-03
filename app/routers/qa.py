"""QA router — question answering about biryani styles."""

from fastapi import APIRouter, Request

from app.models.schemas import QARequest, QAResponse
from app.services.recipes import get_recipe

router = APIRouter()


@router.post("/qa", response_model=QAResponse)
async def answer_question(request: Request, body: QARequest):
    """
    Answer questions about biryani preparation.

    Uses canonical recipes and (optionally) VLM inference.
    """
    recipes = request.app.state.recipes
    question = body.question.lower()

    # Extract relevant category from question
    matched_recipe = None
    for category in recipes:
        if category.lower() in question:
            matched_recipe = get_recipe(recipes, category)
            break

    if not matched_recipe:
        # Try to answer from general knowledge
        return QAResponse(
            question=body.question,
            answer=(
                "I can answer questions about specific biryani styles. "
                "Try asking about Hyderabadi, Kolkata, Lucknowi, Malabar, "
                "Sindhi, Muradabadi, Delhi, Andhra, or any of the 20 regional styles."
            ),
            sources=[],
            confidence=0.5,
            model_used="rule_based",
        )

    # Build answer from canonical recipe
    recipe = matched_recipe
    category = recipe["category"]

    # Determine what aspect the question is about
    if any(w in question for w in ["spice", "masala", "seasoning"]):
        answer = (
            f"{category} biryani uses these key spices: "
            f"{', '.join(recipe.get('key_spices', []))}. "
            f"{recipe.get('notes', '')}"
        )
        aspect = "spices"
    elif any(w in question for w in ["rice", "grain", "chawal"]):
        answer = (
            f"{category} biryani uses {recipe.get('rice_type', 'Basmati rice')}. "
            f"The rice is typically cooked in a {recipe.get('cooking_vessel', 'heavy pot')} "
            f"using the {recipe.get('dum_method', 'dum')} method."
        )
        aspect = "rice"
    elif any(w in question for w in ["dum", "cook", "method", "technique"]):
        answer = (
            f"{category} biryani uses the {recipe.get('dum_method', 'dum')} method "
            f"in a {recipe.get('cooking_vessel', 'heavy pot')}. "
            f"Total cooking time is approximately {recipe.get('typical_duration_minutes', 'unknown')} minutes. "
            f"{recipe.get('notes', '')}"
        )
        aspect = "method"
    elif any(w in question for w in ["step", "process", "how", "make", "prepare"]):
        steps = recipe.get("steps", [])
        defining = [s for s in steps if s.get("is_defining_step")]
        step_text = "\n".join(
            f"  {s['step_number']}. {s['action']}: {s['description']}"
            for s in (defining if defining else steps[:5])
        )
        answer = f"{category} biryani key steps:\n{step_text}"
        aspect = "steps"
    elif any(w in question for w in ["differ", "special", "unique", "feature", "what is"]):
        features = recipe.get("distinguishing_features", [])
        answer = (
            f"{category} biryani is known for:\n"
            + "\n".join(f"  • {f}" for f in features)
        )
        aspect = "features"
    else:
        answer = (
            f"{recipe.get('full_name', category + ' biryani')} ({recipe.get('region', 'India')}). "
            f"Uses {recipe.get('rice_type', 'rice')} with {recipe.get('dum_method', 'dum cooking')}. "
            f"Key spices: {', '.join(recipe.get('key_spices', [])[:5])}. "
            f"{recipe.get('notes', '')}"
        )
        aspect = "general"

    return QAResponse(
        question=body.question,
        answer=answer,
        sources=[{
            "type": "canonical_recipe",
            "category": category,
            "aspect": aspect,
        }],
        confidence=0.85,
        model_used="rule_based_recipe",
    )
