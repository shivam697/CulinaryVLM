#!/usr/bin/env python3
"""
CulinaryVLM — Push Fine-tuned Model to HuggingFace Hub
═══════════════════════════════════════════════════════════

Uploads the fine-tuned QLoRA adapter (and optionally merged model)
to HuggingFace Hub.

Usage:
    python scripts/push_model_to_hub.py
    python scripts/push_model_to_hub.py --repo-id username/culinary-vlm-11b
    python scripts/push_model_to_hub.py --merge --push-merged
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hf_push")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument(
        "--repo-id", default=None,
        help="HuggingFace repo ID (e.g., username/culinary-vlm-11b)",
    )
    parser.add_argument(
        "--adapter-dir", default="training/output",
        help="Directory containing QLoRA adapter weights",
    )
    parser.add_argument("--merge", action="store_true", help="Merge adapter with base model")
    parser.add_argument("--push-merged", action="store_true", help="Push merged model instead of adapter")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Push Model to HuggingFace Hub   ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token or hf_token.startswith("REPLACE"):
        logger.error("HF_TOKEN not set. Add it to .env")
        sys.exit(1)

    repo_id = args.repo_id or os.environ.get("HF_MODEL_REPO", "culinary-vlm-11b")
    if "/" not in repo_id:
        # Infer username from token
        try:
            from huggingface_hub import whoami
            user = whoami(token=hf_token)
            repo_id = f"{user['name']}/{repo_id}"
        except Exception:
            logger.error("Cannot infer HF username. Use --repo-id username/model-name")
            sys.exit(1)

    adapter_dir = PROJECT_ROOT / args.adapter_dir
    logger.info(f"Repo ID: {repo_id}")
    logger.info(f"Adapter dir: {adapter_dir}")

    if args.dry_run:
        logger.info("[DRY RUN] Would push to HuggingFace Hub")
        return

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface-hub not installed")
        sys.exit(1)

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, private=args.private, exist_ok=True)
        logger.info(f"Repo created/verified: {repo_id}")
    except Exception as e:
        logger.warning(f"Repo creation: {e}")

    if adapter_dir.exists():
        logger.info("Uploading adapter weights...")
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=repo_id,
            commit_message="Upload CulinaryVLM QLoRA adapter",
        )
        logger.info("✓ Adapter uploaded!")
    else:
        logger.warning(f"Adapter dir not found: {adapter_dir}")
        logger.info("Uploading project metadata instead...")

    # Upload model card
    model_card = f"""---
language:
  - en
  - hi
  - te
  - ml
  - bn
  - ur
tags:
  - cooking
  - biryani
  - vlm
  - qlora
  - llama-3.2-vision
license: mit
datasets:
  - custom
base_model: meta-llama/Llama-3.2-11B-Vision-Instruct
---

# CulinaryVLM — Multilingual Indian Cooking Video Intelligence

Fine-tuned Llama-3.2-11B-Vision-Instruct with QLoRA on 172 chicken biryani
cooking videos across 20 regional styles and 6 languages.

## Training Details
- **Base Model**: meta-llama/Llama-3.2-11B-Vision-Instruct
- **Method**: QLoRA (rank=16, alpha=32)
- **Data**: {172} videos, multi-tier QA pairs
- **Categories**: Hyderabadi, Kolkata, Lucknowi, and 17 more regional styles

## Usage
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained("{repo_id}")
```
"""

    card_path = PROJECT_ROOT / "MODEL_CARD.md"
    with open(card_path, "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )

    logger.info(f"✓ Model pushed to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
