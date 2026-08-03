#!/usr/bin/env python3
"""
CulinaryVLM — Stage 10: QLoRA Fine-tuning
═══════════════════════════════════════════

Fine-tunes Llama-3.2-11B-Vision-Instruct with QLoRA using Unsloth.
MUST RUN ON GPU CLUSTER (Gpu5-40g or Gpu6-40g).

Usage:
    python training/finetune_qlora.py
    python training/finetune_qlora.py --epochs 3 --lr 2e-4

Input:  datasets/qa/train.json, datasets/qa/test.json
Output: models/culinary_vlm_qlora/ → pushed to HF Hub
"""

from __future__ import annotations

import argparse, json, logging, os, sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("stage_10")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def format_qa_for_training(qa_pairs: list[dict]) -> list[dict]:
    """Convert QA pairs to instruction-tuning format."""
    formatted = []
    for qa in qa_pairs:
        if qa.get("needs_generation") or not qa.get("answer"):
            continue
        formatted.append({
            "instruction": qa["question"],
            "input": f"Category: {qa.get('category', 'biryani')}",
            "output": qa["answer"],
        })
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 10 — QLoRA Fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 10: QLoRA Fine-tuning      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    train_path = PROJECT_ROOT / "datasets" / "qa" / "train.json"
    test_path = PROJECT_ROOT / "datasets" / "qa" / "test.json"

    if not train_path.exists():
        logger.error(f"Not found: {train_path}. Run Stage 9 first.")
        sys.exit(1)

    with open(train_path) as f:
        train_data = json.load(f).get("qa_pairs", [])
    with open(test_path) as f:
        test_data = json.load(f).get("qa_pairs", [])

    train_formatted = format_qa_for_training(train_data)
    test_formatted = format_qa_for_training(test_data)

    logger.info(f"Train: {len(train_formatted)}, Test: {len(test_formatted)}")

    if args.dry_run:
        logger.info("[DRY RUN] Config:")
        logger.info(f"  Model: meta-llama/Llama-3.2-11B-Vision-Instruct")
        logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
        logger.info(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
        return

    # --- Unsloth QLoRA Training ---
    try:
        from unsloth import FastLanguageModel
        import torch
    except ImportError:
        logger.error("unsloth not installed. Run on GPU cluster with: pip install unsloth")
        sys.exit(1)

    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    logger.info(f"Loading {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Format dataset
    from datasets import Dataset

    def format_prompt(example):
        return {
            "text": (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        }

    train_ds = Dataset.from_list(train_formatted).map(format_prompt)

    # Training
    from transformers import TrainingArguments
    from trl import SFTTrainer

    output_dir = PROJECT_ROOT / "models" / "culinary_vlm_qlora"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=False, bf16=True,
            logging_steps=10,
            output_dir=str(output_dir),
            save_strategy="epoch",
            report_to="none",
        ),
    )

    logger.info("Starting training...")
    trainer.train()

    # Save
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info(f"Model saved: {output_dir / 'final'}")

    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token and not hf_token.startswith("REPLACE"):
            model.push_to_hub("shivam-biryani/culinary-vlm-qlora", token=hf_token)
            logger.info("Pushed to HuggingFace Hub")

    logger.info("✓ Stage 10 complete! NEXT: Stage 11 — python pipeline/11_build_faiss.py")


if __name__ == "__main__":
    main()
