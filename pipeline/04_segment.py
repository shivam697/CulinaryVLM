#!/usr/bin/env python3
"""
CulinaryVLM — Stage 4: Video Segmentation (VLM-based)
═══════════════════════════════════════════════════════

Uses InternVL2-8B (bfloat16) to segment cooking videos
into procedural steps with temporal boundaries.

MUST RUN ON GPU CLUSTER (Gpu7-80G queue).

Usage:
    python pipeline/04_segment.py
    python pipeline/04_segment.py --resume --batch-size 4
    python pipeline/04_segment.py --category Hyderabadi --limit 5
    python pipeline/04_segment.py --dry-run

Input:  datasets/raw_videos/{category}/{video_id}.mp4
        datasets/transcripts/{video_id}.json
Output: datasets/segments/{video_id}.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_4")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"

# ─── Paper's exact segmentation prompt (v2.0: enriched) ─────────
SEGMENTATION_PROMPT = """You are analyzing a cooking video frame sequence. For each distinct cooking step visible in these frames, provide:

1. step_number: Sequential step number
2. action: Brief action name (e.g., "Marinating chicken", "Frying onions")
3. description: Detailed description of what is happening
4. start_frame: Approximate start frame index
5. end_frame: Approximate end frame index
6. ingredients_visible: List of ingredients visible in the frames
7. cooking_technique: The technique being used (e.g., "frying", "boiling", "layering")
8. confidence: Your confidence in this segment (0.0-1.0)
9. visible_utensil: List of utensils/tools visible (e.g., ["spatula", "ladle", "tongs"])
10. vessel: The main cooking vessel visible (e.g., "handi", "kadhai", "pot", "tawa")
11. ingredient_state: Dict mapping visible ingredients to their physical state (e.g., {"onions": "golden brown", "rice": "70% cooked"})
12. rice_state: State of rice if visible (e.g., "raw", "soaking", "parboiling", "70% cooked", "fully cooked", "layered", "not visible")
13. meat_state: State of meat if visible (e.g., "raw", "marinated", "searing", "partially cooked", "fully cooked", "not visible")
14. flame_level: Estimated flame/heat level (e.g., "high", "medium", "low", "off", "not visible")
15. oil_amount: Estimated oil/ghee amount (e.g., "dry", "light", "moderate", "deep", "not visible")
16. steam_visibility: Whether steam is visible (e.g., "none", "light", "moderate", "heavy")
17. cooking_stage: Overall stage of the cooking process (e.g., "prep", "marination", "rice_cooking", "meat_cooking", "layering", "dum", "serving")

Output as a JSON array of step objects. Be precise about temporal boundaries."""


# ─── InternVL2 image preprocessing (from official model card) ────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    """Build the InternVL2 image transform pipeline."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split image into tiles based on best-fit aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _preprocess_pil_image(pil_image, input_size=448, max_num=1):
    """Convert a PIL Image to an InternVL2 pixel_values tensor.

    Args:
        pil_image: PIL.Image in RGB
        input_size: tile size (448 for InternVL2)
        max_num: max tiles per image (1 for speed, up to 12 for quality)

    Returns:
        pixel_values: tensor of shape (num_tiles, 3, 448, 448)
    """
    import torch

    transform = _build_transform(input_size=input_size)
    tiles = _dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num,
    )
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values



def load_config(config_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frames(video_path: Path, fps: float = 0.5) -> list:
    """Extract frames from video at given FPS rate."""
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python not installed. Run: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(video_fps / fps)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Resize for VLM input (448x448)
            frame_resized = cv2.resize(frame, (448, 448))
            frames.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / video_fps,
                "image": frame_resized,
            })
        frame_idx += 1

    cap.release()
    total_duration = frame_idx / video_fps
    logger.info(f"  Extracted {len(frames)} frames from {total_duration:.0f}s video")
    return frames


def segment_with_vlm(
    frames: list,
    model,
    tokenizer,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """
    Run InternVL2-8B on frame batches to identify cooking steps.

    Processes frames in sliding windows of 8 frames.  Each PIL image is
    preprocessed into a pixel_values tensor via InternVL2's official
    dynamic_preprocess pipeline, then passed as a positional arg to
    model.chat() alongside a num_patches_list.
    """
    import torch
    from PIL import Image

    segments = []
    window_size = 8
    stride = 4
    generation_config = dict(max_new_tokens=1024, temperature=0.1, do_sample=False)
    _first_error_logged = False

    for i in range(0, len(frames), stride):
        window = frames[i:i + window_size]
        if len(window) < 2:
            break

        # Convert BGR numpy frames to RGB PIL images
        pil_images = []
        for f in window:
            img = Image.fromarray(f["image"][:, :, ::-1])  # BGR to RGB
            pil_images.append(img)

        start_time = window[0]["timestamp"]
        end_time = window[-1]["timestamp"]

        pixel_values_list = None
        pixel_values = None
        try:
            # ── Preprocess images into pixel_values tensor ──
            pixel_values_list = []
            num_patches_list = []
            for img in pil_images:
                pv = _preprocess_pil_image(img, max_num=1)  # 1 tile per frame for speed
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])

            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                dtype=torch.bfloat16, device=device,
            )

            # ── Build prompt with <image> tokens (one per frame) ──
            image_tokens = "\n".join(
                f"Frame-{j+1}: <image>" for j in range(len(pil_images))
            )
            prompt = (
                f"{image_tokens}\n\n"
                f"These are {len(pil_images)} frames from a biryani cooking video, "
                f"spanning {start_time:.1f}s to {end_time:.1f}s.\n\n"
                f"{SEGMENTATION_PROMPT}"
            )

            # ── Model inference (InternVL2 positional API) ──
            with torch.no_grad():
                response = model.chat(
                    tokenizer,           # arg 1: tokenizer
                    pixel_values,        # arg 2: preprocessed tensor
                    prompt,              # arg 3: question with <image> tokens
                    generation_config,   # arg 4: generation config
                    num_patches_list=num_patches_list,
                )

            # Parse JSON response
            try:
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

                steps = json.loads(cleaned)
                if isinstance(steps, list):
                    for step in steps:
                        step["start_time"] = start_time
                        step["end_time"] = end_time
                        step["window_start"] = i
                        # Ensure all v2.0 fields present with safe defaults
                        step.setdefault("visible_utensil", [])
                        step.setdefault("vessel", "")
                        step.setdefault("ingredient_state", {})
                        step.setdefault("rice_state", "not visible")
                        step.setdefault("meat_state", "not visible")
                        step.setdefault("flame_level", "not visible")
                        step.setdefault("oil_amount", "not visible")
                        step.setdefault("steam_visibility", "none")
                        step.setdefault("cooking_stage", "")
                    segments.extend(steps)
            except json.JSONDecodeError:
                # Non-JSON response — save as low-confidence fallback
                segments.append({
                    "action": "unknown",
                    "description": response[:200],
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": 0.3,
                    "visible_utensil": [],
                    "vessel": "",
                    "ingredient_state": {},
                    "rice_state": "not visible",
                    "meat_state": "not visible",
                    "flame_level": "not visible",
                    "oil_amount": "not visible",
                    "steam_visibility": "none",
                    "cooking_stage": "",
                })

        except Exception as e:
            logger.warning(f"  VLM error at {start_time:.1f}s: {e}")
            if not _first_error_logged:
                import traceback
                logger.warning(f"  FULL TRACEBACK:\n{traceback.format_exc()}")
                _first_error_logged = True

        # Free GPU memory between windows
        if pixel_values_list is not None:
            del pixel_values_list
        if pixel_values is not None:
            del pixel_values
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return segments


def merge_overlapping_segments(
    segments: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge overlapping segments from sliding window."""
    if not segments:
        return []

    # Sort by start time
    segments.sort(key=lambda s: s.get("start_time", 0))

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        # Check overlap
        overlap = min(last["end_time"], seg["end_time"]) - max(last["start_time"], seg["start_time"])
        duration = max(seg["end_time"] - seg["start_time"], 0.1)

        if overlap / duration > overlap_threshold and last.get("action") == seg.get("action"):
            # Merge: extend end time, keep higher confidence
            last["end_time"] = max(last["end_time"], seg["end_time"])
            if seg.get("confidence", 0) > last.get("confidence", 0):
                last["description"] = seg.get("description", last["description"])
        else:
            merged.append(seg)

    # Renumber steps
    for i, seg in enumerate(merged, 1):
        seg["step_number"] = i

    return merged


def save_segments(
    video_id: str,
    category: str,
    segments: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Save segment JSON."""
    output = {
        "schema_version": "2.0",
        "video_id": video_id,
        "category": category,
        "num_segments": len(segments),
        "segments": segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{video_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filepath


def main() -> None:
    parser = argparse.ArgumentParser(description="CulinaryVLM Stage 4 — Video Segmentation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 4: Video Segmentation      ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)

    # Load video selection
    sel_path = PROJECT_ROOT / "datasets/categorized/pipeline_selection.json"
    with open(sel_path) as f:
        videos = json.load(f)["videos"]

    if args.category:
        videos = [v for v in videos if v["category"] == args.category]
    if args.limit:
        videos = videos[:args.limit]

    raw_dir = PROJECT_ROOT / config["paths"]["raw_videos"]
    seg_dir = PROJECT_ROOT / config["paths"]["segments"]

    # Find downloaded videos
    available = []
    for v in videos:
        cat_dir = raw_dir / v["category"].lower().replace(" ", "_")
        for ext in [".mp4", ".mkv", ".webm"]:
            p = cat_dir / f"{v['video_id']}{ext}"
            if p.exists():
                available.append((v, p))
                break

    logger.info(f"Selected: {len(videos)}, Available: {len(available)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would segment:")
        for v, p in available[:10]:
            logger.info(f"  {v['category']:15s} | {v['video_id']}")
        return

    if not available:
        logger.error("No downloaded videos found. Run Stage 1 first.")
        sys.exit(1)

    # Load VLM model
    logger.info("Loading InternVL2-8B (bfloat16)...")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = config["pipeline"]["segmentation"]["model"]
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()

        from transformers import GenerationMixin, GenerationConfig

        # Diagnostic: show the real submodule structure first
        logger.info(f"  DEBUG: named_children = {[n for n, _ in model.named_children()]}")

        # Robust patch: walk ALL submodules (not just top-level), find any
        # module that defines prepare_inputs_for_generation (i.e. is a causal
        # LM) but doesn't have a working generate() via GenerationMixin, and
        # patch it in place.
        patched = []
        for name, module in model.named_modules():
            if hasattr(module, "prepare_inputs_for_generation") and not isinstance(module, GenerationMixin):
                module.__class__ = type(
                    module.__class__.__name__,
                    (module.__class__, GenerationMixin),
                    {},
                )
                patched.append(name or "<root>")

                # GenerationMixin.generate() requires these attributes that a
                # normally-initialized PreTrainedModel would have but a
                # monkey-patched module may not:
                #   - generation_config: used at line 24 of _prepare_generation_config
                #     (self.generation_config._from_model_config crashes if None)
                #   - main_input_name: used by generate() for input handling
                # (config, device, dtype are already present via nn.Module/PreTrainedModel)
                if getattr(module, "generation_config", None) is None:
                    if hasattr(module, "config"):
                        module.generation_config = GenerationConfig.from_model_config(module.config)
                    else:
                        module.generation_config = GenerationConfig()
                    logger.info(f"  Set generation_config on {name or '<root>'}")

                if not hasattr(module, "main_input_name"):
                    module.main_input_name = "input_ids"
                    logger.info(f"  Set main_input_name on {name or '<root>'}")

        logger.info(f"  DEBUG: patched modules = {patched}")

        # Also patch the top-level model itself if needed (covers the case
        # where model.generate() is called directly rather than via a
        # submodule)
        if hasattr(model, "prepare_inputs_for_generation") and not isinstance(model, GenerationMixin):
            model.__class__ = type(
                model.__class__.__name__, (model.__class__, GenerationMixin), {}
            )
            patched.append("<top-level model>")
            if getattr(model, "generation_config", None) is None:
                if hasattr(model, "config"):
                    model.generation_config = GenerationConfig.from_model_config(model.config)
                else:
                    model.generation_config = GenerationConfig()
                logger.info("  Set generation_config on <top-level model>")
            if not hasattr(model, "main_input_name"):
                model.main_input_name = "input_ids"
                logger.info("  Set main_input_name on <top-level model>")

        if not patched:
            logger.warning("  WARNING: GenerationMixin patch found NOTHING to patch — investigate further")

        # Post-patch verification: specifically check whatever module chat()
        # actually calls .generate() on. Since we've confirmed via source
        # inspection that InternVLChatModel.chat() -> self.generate() ->
        # self.language_model.generate(), verify that path directly:
        if hasattr(model, "language_model"):
            lm = model.language_model
            logger.info(f"  DEBUG POST-PATCH: model.language_model hasattr('generate') = {hasattr(lm, 'generate')}")
            logger.info(f"  DEBUG POST-PATCH: model.language_model.generation_config = {getattr(lm, 'generation_config', 'MISSING')}")
        else:
            logger.warning("  WARNING: model still has no 'language_model' attribute after loading — this may indicate the submodule is under a different name or wrapped differently than expected")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        device = "cuda"
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        logger.error("This stage requires GPU. Run on the IIIT Delhi cluster.")
        sys.exit(1)

    # Process videos
    stats = {"success": 0, "skipped": 0, "error": 0, "empty": 0}

    for i, (video, vpath) in enumerate(available, 1):
        vid = video["video_id"]
        cat = video["category"]

        logger.info(f"\n[{i}/{len(available)}] {cat} | {vid}")

        out_path = seg_dir / f"{vid}.json"
        if args.resume and out_path.exists():
            logger.info("  ⊘ Already segmented (resume)")
            stats["skipped"] += 1
            continue

        try:
            frames = extract_frames(vpath, fps=0.5)
            if not frames:
                stats["error"] += 1
                continue

            raw_segments = segment_with_vlm(frames, model, tokenizer, device)
            merged = merge_overlapping_segments(raw_segments)

            save_segments(vid, cat, merged, seg_dir)

            if merged:
                logger.info(f"  ✓ {len(merged)} segments")
                stats["success"] += 1
            else:
                logger.warning(f"  ⚠ 0 segments produced (all windows failed)")
                stats["empty"] += 1

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            stats["error"] += 1

    logger.info(f"\n{'═'*50}")
    logger.info(f"SEGMENTATION SUMMARY: {stats}")
    if stats['empty'] > 0:
        logger.warning(f"  ⚠ {stats['empty']} videos produced 0 segments — check VLM errors above")
    logger.info(f"{'═'*50}")
    logger.info("✓ Stage 4 complete! NEXT: Stage 5 — python pipeline/05_cluster.py")


if __name__ == "__main__":
    main()
