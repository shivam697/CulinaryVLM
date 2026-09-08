"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Videos ──────────────────────────────────────────

class VideoSummary(BaseModel):
    video_id: str
    title: str
    url: str
    category: str
    confidence: str
    estimated_language: str
    priority: int


class VideoDetail(VideoSummary):
    segments: list["SegmentSummary"] = []
    transcript_available: bool = False
    alignment_available: bool = False


class VideoListResponse(BaseModel):
    total: int
    videos: list[VideoSummary]


# ─── Segments ────────────────────────────────────────

class SegmentSummary(BaseModel):
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    action: str
    description: str = ""
    canonical_step: Optional[str] = None
    confidence: float = 0.0


# ─── Recipes ─────────────────────────────────────────

class RecipeStep(BaseModel):
    step_number: int
    action: str
    description: str
    duration_minutes: Optional[int] = None
    is_defining_step: bool = False
    ingredients_used: list[str] = []


class CanonicalRecipe(BaseModel):
    category: str
    full_name: str
    region: str
    distinguishing_features: list[str] = []
    steps: list[RecipeStep] = []
    ingredient_aliases: dict[str, dict[str, Optional[str]]] = {}
    key_spices: list[str] = []
    rice_type: str = ""
    cooking_vessel: str = ""
    dum_method: str = ""
    typical_duration_minutes: Optional[int] = None
    notes: str = ""


class RecipeListResponse(BaseModel):
    total: int
    recipes: list[CanonicalRecipe]


# ─── Search ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    category: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    segment_id: str
    video_id: str
    category: str
    action: str
    description: str
    score: float
    start_time: float
    end_time: float
    video_title: str = ""
    video_url: str = ""


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]


# ─── QA ──────────────────────────────────────────────

class QARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    category: Optional[str] = None
    use_vlm: bool = False


class QAResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]] = []
    confidence: float = 0.0
    model_used: str = ""


# ─── Compare ────────────────────────────────────────

class CompareRequest(BaseModel):
    category_a: str
    category_b: str
    aspect: Optional[str] = None  # e.g., "dum_method", "rice", "spices"


class ComparisonDifference(BaseModel):
    aspect: str
    category_a_value: str
    category_b_value: str
    explanation: str = ""


class CompareResponse(BaseModel):
    category_a: str
    category_b: str
    differences: list[ComparisonDifference]
    summary: str = ""


# ─── Agent ───────────────────────────────────────────

class AgentRequest(BaseModel):
    message: str = Field(..., min_length=2, max_length=2000)
    session_id: Optional[str] = None


class ToolTrace(BaseModel):
    tool_name: str
    input: dict[str, Any] = {}
    output: Any = None
    duration_ms: float = 0


class AgentResponse(BaseModel):
    answer: str
    session_id: str
    tool_traces: list[ToolTrace] = []
    plan: str = ""


# ─── Health ──────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    faiss_loaded: bool = False
    recipes_loaded: int = 0
    agent_enabled: bool = False
