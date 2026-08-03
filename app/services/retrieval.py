"""FAISS-based semantic retrieval service."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Wrapper around FAISS index with metadata."""

    def __init__(self, index, metadata: list[dict[str, Any]]):
        self.index = index
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        """Search for nearest segments."""
        if self.index is None:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(1 / (1 + dist))  # Convert distance to similarity
            results.append(meta)

        return results


def load_faiss_index(faiss_dir: Path) -> FAISSIndex | None:
    """Load FAISS index and metadata from disk."""
    try:
        import faiss

        index_path = faiss_dir / "segments.index"
        meta_path = faiss_dir / "segments_metadata.pkl"

        if not index_path.exists():
            logger.warning(f"FAISS index not found: {index_path}")
            return None

        index = faiss.read_index(str(index_path))
        logger.info(f"FAISS index loaded: {index.ntotal} vectors")

        metadata = []
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

        return FAISSIndex(index, metadata)

    except ImportError:
        logger.warning("faiss-cpu not installed — search unavailable")
        return None
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        return None


def encode_query(text: str, model=None) -> np.ndarray:
    """
    Encode a text query into an embedding vector.

    Uses sentence-transformers MiniLM if no model is provided.
    """
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return np.zeros(384, dtype=np.float32)

    embedding = model.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)
