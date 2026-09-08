"""Compare router — compare biryani styles side by side."""

from fastapi import APIRouter, Request

from app.models.schemas import CompareRequest, CompareResponse, ComparisonDifference
from app.services.recipes import compare_recipes

router = APIRouter()


@router.post("/compare", response_model=CompareResponse)
async def compare_styles(request: Request, body: CompareRequest):
    """
    Compare two biryani styles and return structured differences.

    Compares rice type, dum method, spices, cooking vessel,
    distinguishing features, and step counts.
    """
    result = compare_recipes(
        request.app.state.recipes,
        body.category_a,
        body.category_b,
        body.aspect,
    )

    if "error" in result:
        return CompareResponse(
            category_a=body.category_a,
            category_b=body.category_b,
            differences=[],
            summary=result["error"],
        )

    return CompareResponse(
        category_a=result["category_a"],
        category_b=result["category_b"],
        differences=[
            ComparisonDifference(**d) for d in result["differences"]
        ],
        summary=result.get("summary", ""),
    )

