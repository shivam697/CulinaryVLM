"""
CulinaryVLM — Ollama Client Utility
════════════════════════════════════

Shared helper for all pipeline stages that need LLM text generation.
Talks to a local Ollama server via its HTTP API.

Usage:
    from pipeline.utils.ollama_client import check_ollama, ollama_chat
"""

from __future__ import annotations

import logging
import sys
import time

import requests

logger = logging.getLogger("ollama_client")

MAX_RETRIES = 3
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:32b-instruct"


def check_ollama(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> None:
    """Ensure Ollama server is running and the target model is pulled.

    Exits the process if the server is unreachable or the model is missing.
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=600)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.error(
            f"Cannot reach Ollama at {host} — start server with 'ollama serve'. "
            f"Error: {e}"
        )
        sys.exit(1)

    tags = [m["name"] for m in r.json().get("models", [])]
    if not any(model == t or model == t.split(":")[0] for t in tags):
        logger.error(f"Model '{model}' not found on server. Available: {tags}")
        logger.error(f"Run: ollama pull {model}")
        sys.exit(1)

    logger.info(f"Ollama reachable at {host}, model '{model}' ready.")


def ollama_chat(
    host: str,
    model: str,
    system: str,
    user: str,
    num_predict: int = 4000,
    temperature: float = 0.3,
    json_format: bool = False,
) -> str | None:
    """Execute a chat completion request to local Ollama server.

    Parameters
    ----------
    host : str
        Ollama server URL (e.g. ``http://localhost:11434``).
    model : str
        Model tag (e.g. ``qwen2.5:32b-instruct``).
    system : str
        System prompt.
    user : str
        User prompt.
    num_predict : int
        Max tokens to generate.
    temperature : float
        Sampling temperature.
    json_format : bool
        If True, sets ``"format": "json"`` so Ollama forces JSON output.

    Returns
    -------
    str | None
        The assistant's response text, or None after all retries fail.
    """
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "think": False,
    }
    if json_format:
        payload["format"] = "json"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{host}/api/chat", json=payload, timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = (data.get("message", {}).get("content") or "").strip()
            if answer:
                return answer
            logger.warning(
                f"Empty response on attempt {attempt}/{MAX_RETRIES}"
            )
            time.sleep(2)
        except requests.RequestException as e:
            logger.warning(
                f"Ollama request error on attempt {attempt}/{MAX_RETRIES}: {e}"
            )
            time.sleep(2 * attempt)

    return None
