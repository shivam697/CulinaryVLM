"""
CulinaryVLM — FastAPI Application Entry Point
═══════════════════════════════════════════════

Main application factory. Registers routers, CORS, startup events.
Agent layer is feature-flagged via config.yaml → agent.enabled.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("culinary_vlm")


def load_app_config() -> dict:
    """Load project config."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    config = load_app_config()
    app.state.config = config
    app.state.project_root = PROJECT_ROOT

    logger.info("CulinaryVLM API starting up...")

    # Load FAISS index if available
    faiss_path = PROJECT_ROOT / config["paths"].get("faiss_index", "datasets/faiss")
    if (faiss_path / "segments.index").exists():
        from app.services.retrieval import load_faiss_index
        app.state.faiss_index = load_faiss_index(faiss_path)
        logger.info(f"FAISS index loaded: {faiss_path}")

        # Pre-load embedding model so first search request doesn't timeout
        try:
            from sentence_transformers import SentenceTransformer
            app.state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model pre-loaded (all-MiniLM-L6-v2)")
        except Exception as e:
            app.state.embed_model = None
            logger.warning(f"Could not pre-load embedding model: {e}")
    else:
        app.state.faiss_index = None
        app.state.embed_model = None
        logger.warning("FAISS index not found — /search will be unavailable")

    # Load canonical recipes
    from app.services.recipes import load_canonical_recipes
    app.state.recipes = load_canonical_recipes(
        PROJECT_ROOT / "configs" / "canonical_recipes"
    )
    logger.info(f"Loaded {len(app.state.recipes)} canonical recipes")

    # Optionally load agent graph
    agent_enabled = config.get("agent", {}).get("enabled", False)
    use_agent = os.environ.get("USE_AGENT_LAYER", "false").lower() == "true"
    if agent_enabled and use_agent:
        try:
            from app.agents.graph import build_agent_graph
            app.state.agent_graph = build_agent_graph(config)
            logger.info("Agent layer enabled (LangGraph)")
        except ImportError:
            logger.warning(
                "Agent dependencies not installed (pip install -r requirements-agent.txt). "
                "Agent layer disabled."
            )
            app.state.agent_graph = None
    else:
        app.state.agent_graph = None
        logger.info("Agent layer disabled (config or env)")

    yield

    logger.info("CulinaryVLM API shutting down...")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="CulinaryVLM API",
        description=(
            "Multilingual Indian Cooking Video Intelligence API. "
            "Search cooking techniques, ask questions about biryani preparation, "
            "and compare regional styles."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from app.routers import health, videos, recipes, search, qa, compare

    app.include_router(health.router, tags=["Health"])
    app.include_router(videos.router, prefix="/api/v1", tags=["Videos"])
    app.include_router(recipes.router, prefix="/api/v1", tags=["Recipes"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(qa.router, prefix="/api/v1", tags=["QA"])
    app.include_router(compare.router, prefix="/api/v1", tags=["Compare"])

    # Conditionally include agent router
    try:
        config = load_app_config()
        agent_enabled = config.get("agent", {}).get("enabled", False)
        use_agent = os.environ.get("USE_AGENT_LAYER", "false").lower() == "true"
        if agent_enabled and use_agent:
            from app.routers import agent
            app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])
    except Exception:
        pass

    return app


app = create_app()
