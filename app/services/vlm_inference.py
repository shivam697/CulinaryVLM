"""
VLM Inference Service — calls the fine-tuned CulinaryVLM model
hosted on HuggingFace Spaces (ZeroGPU).

The model is Llama-3.2-11B-Vision-Instruct fine-tuned with QLoRA
on biryani recipe data.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client = None


def get_vlm_client():
    """Lazy-initialize the Gradio client for the HF Space."""
    global _client
    if _client is None:
        try:
            from gradio_client import Client

            hf_token = os.environ.get("HF_TOKEN", "")
            space_id = os.environ.get(
                "VLM_SPACE_ID", "shivamminde/culinary-vlm"
            )
            _client = Client(space_id, hf_token=hf_token or None)
            logger.info(f"Connected to VLM Space: {space_id}")
        except Exception as e:
            logger.error(f"Failed to connect to VLM Space: {e}")
            _client = None
    return _client


async def ask_vlm(
    question: str,
    category: Optional[str] = None,
) -> dict:
    """
    Send a question to the fine-tuned CulinaryVLM model.

    Args:
        question: The user's question about biryani cooking.
        category: Optional biryani style (e.g., "Hyderabadi", "Kolkata").

    Returns:
        dict with 'answer', 'model_used', 'confidence'.
    """
    client = get_vlm_client()

    if client is None:
        return {
            "answer": "VLM model is currently unavailable. Please try again later.",
            "model_used": "unavailable",
            "confidence": 0.0,
        }

    try:
        result = client.predict(
            question=question,
            category=category or "",
            api_name="/predict",
        )

        return {
            "answer": result,
            "model_used": "culinary-vlm-qlora (Llama-3.2-11B-Vision fine-tuned)",
            "confidence": 0.90,
        }

    except Exception as e:
        logger.error(f"VLM inference error: {e}")
        return {
            "answer": f"Model inference error: {str(e)}",
            "model_used": "error",
            "confidence": 0.0,
        }
