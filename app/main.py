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
    import asyncio
    import concurrent.futures

    config = load_app_config()
    app.state.config = config
    app.state.project_root = PROJECT_ROOT
    app.state.faiss_index = None
    app.state.embed_model = None
    app.state.agent_graph = None

    logger.info("CulinaryVLM API starting up (fast path)...")

    # 1. Canonical recipes — fast (small JSON files, ~1MB)
    from app.services.recipes import load_canonical_recipes
    app.state.recipes = load_canonical_recipes(
        PROJECT_ROOT / "configs" / "canonical_recipes"
    )
    logger.info(f"Loaded {len(app.state.recipes)} canonical recipes")

    # 2. Agent graph — fast (just wires graph nodes, no model downloads)
    agent_enabled = config.get("agent", {}).get("enabled", False)
    use_agent = os.environ.get("USE_AGENT_LAYER", "false").lower() == "true"
    logger.info(f"Agent config: enabled={agent_enabled}, USE_AGENT_LAYER={use_agent}")
    if agent_enabled and use_agent:
        try:
            from app.agents.graph import build_agent_graph
            app.state.agent_graph = build_agent_graph(config)
            if app.state.agent_graph is not None:
                logger.info("Agent layer enabled (LangGraph)")
            else:
                logger.warning("Agent graph build returned None")
        except ImportError as e:
            logger.warning(f"Agent deps not installed: {e}")
        except Exception as e:
            logger.error(f"Agent init failed: {e}")
    else:
        logger.info("Agent layer disabled")

    # 3. FAISS index + embedding model in background (slow: binary read + model download)
    def _background_init():
        try:
            faiss_path = PROJECT_ROOT / config["paths"].get("faiss_index", "datasets/faiss")
            if (faiss_path / "segments.index").exists():
                from app.services.retrieval import load_faiss_index
                app.state.faiss_index = load_faiss_index(faiss_path)
                logger.info(f"[BG] FAISS index loaded")
            else:
                logger.warning("[BG] FAISS index not found")
        except Exception as e:
            logger.error(f"[BG] FAISS load failed: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            app.state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[BG] Embedding model loaded")
        except Exception as e:
            logger.warning(f"[BG] Embedding model failed: {e}")

        logger.info("[BG] Background init complete.")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=1),
        _background_init
    )

    logger.info("CulinaryVLM API ready.")
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

    # Always register all routers including agent
    from app.routers import health, videos, recipes, search, qa, compare
    from app.routers import agent

    app.include_router(health.router, tags=["Health"])
    app.include_router(videos.router, prefix="/api/v1", tags=["Videos"])
    app.include_router(recipes.router, prefix="/api/v1", tags=["Recipes"])
    app.include_router(search.router, prefix="/api/v1", tags=["Search"])
    app.include_router(qa.router, prefix="/api/v1", tags=["QA"])
    app.include_router(compare.router, prefix="/api/v1", tags=["Compare"])
    app.include_router(agent.router, prefix="/api/v1", tags=["Agent"])

    return app


app = create_app()
