#!/usr/bin/env python3
"""
CulinaryVLM — Stage 2: ASR + Translation + NLP
═══════════════════════════════════════════════════════════════

Transcribes downloaded videos using WhisperX (word-level timestamps),
translates non-English transcripts to English using NLLB or Groq,
and extracts lightweight NLP features (ingredients, actions, utensils,
temperatures, quantities, ingredient_state) from each transcript segment
using regex + lexicon (zero extra API calls, zero new dependencies).

MUST RUN ON GPU CLUSTER (Gpu7-80G queue).

Usage:
    python pipeline/02_asr.py
    python pipeline/02_asr.py --resume
    python pipeline/02_asr.py --limit 10 --batch-size 8
    python pipeline/02_asr.py --category Hyderabadi
    python pipeline/02_asr.py --dry-run

Run on: IIIT Delhi GPU cluster (L40S 48GB)
Input:  datasets/raw_videos/{category}/{video_id}.mp4
        configs/canonical_recipes/{category}.json (for ingredient vocabulary)
Output: datasets/transcripts/{video_id}.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"

# ─── Cooking Action & Utensil Lexicons (compact, no new deps) ────

COOKING_ACTIONS: set[str] = {
    # Heat / fire
    "fry", "deep fry", "shallow fry", "sauté", "saute", "roast", "bake",
    "grill", "broil", "toast", "char", "smoke", "sear", "flambe",
    # Water / liquid
    "boil", "simmer", "blanch", "parboil", "steam", "poach", "braise",
    "stew", "reduce", "deglaze",
    # Prep
    "chop", "dice", "slice", "mince", "julienne", "grate", "peel",
    "crush", "grind", "pound", "blend", "purée", "puree", "shred",
    "cut", "trim", "score", "deseed",
    # Combine
    "mix", "stir", "fold", "whisk", "beat", "knead", "toss",
    "combine", "incorporate", "emulsify",
    # Apply
    "marinate", "season", "rub", "coat", "baste", "drizzle", "garnish",
    "sprinkle", "dust", "glaze", "temper", "tadka", "baghar",
    # Biryani-specific
    "layer", "dum", "seal", "soak", "drain", "strain", "rinse",
    "wash", "rest", "infuse", "steep", "bloom",
    # General
    "cook", "heat", "warm", "cool", "chill", "freeze",
    "add", "pour", "place", "spread", "arrange", "cover",
    "remove", "transfer", "serve",
}

UTENSIL_LEXICON: set[str] = {
    # Vessels
    "handi", "degchi", "deg", "matka", "pot", "pan", "kadhai", "kadai",
    "wok", "skillet", "saucepan", "tawa", "tava", "griddle",
    "pressure cooker", "cooker", "oven", "tandoor", "microwave",
    "slow cooker", "rice cooker", "instant pot",
    # Tools
    "ladle", "spatula", "spoon", "wooden spoon", "slotted spoon",
    "tongs", "whisk", "rolling pin", "belan", "chakla",
    "knife", "chopping board", "cutting board", "peeler",
    "grater", "mortar", "pestle", "sil batta", "mixer", "blender",
    "food processor", "strainer", "colander", "sieve",
    # Biryani-specific
    "muslin cloth", "potli", "aluminium foil", "foil", "lid",
    "dough", "chapati dough",  # for sealing
    "bamboo", "bamboo trunk", "banana leaf",
}

# Pre-compiled regex patterns for temperatures and quantities
_RE_TEMPERATURE = re.compile(
    r'(\d+\s*(?:°|degree|degrees?)\s*(?:c|f|celsius|fahrenheit)'
    r'|low\s+(?:heat|flame)|medium\s+(?:heat|flame)|high\s+(?:heat|flame)'
    r'|sim\s*(?:mer|ming)?\s+(?:heat|flame))',
    re.IGNORECASE,
)
_RE_QUANTITY = re.compile(
    r'(\d+(?:\.\d+)?\s*(?:'
    r'kg|kilogram|gram|grams|g|lb|lbs|pound|pounds'
    r'|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons'
    r'|litre|liter|litres|liters|ml|millilitre|milliliter'
    r'|pinch|pinches|handful|handfuls'
    r'|piece|pieces|nos|number'
    r'|inch|inches|cm'
    r'|minute|minutes|min|mins|hour|hours|hr|hrs|second|seconds|sec|secs'
    r')s?\b)',
    re.IGNORECASE,
)

# Ingredient state indicators (action phrase → likely state)
_STATE_INDICATORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(?:70|seventy)\s*%?\s*(?:cooked|done)\b', re.I), "70% cooked"),
    (re.compile(r'\b(?:half|50)\s*%?\s*(?:cooked|done|boiled)\b', re.I), "half cooked"),
    (re.compile(r'\bfully\s+(?:cooked|done)\b', re.I), "fully cooked"),
    (re.compile(r'\bgolden\s*(?:brown)?\b', re.I), "golden brown"),
    (re.compile(r'\bcrispy\b', re.I), "crispy"),
    (re.compile(r'\btender\b', re.I), "tender"),
    (re.compile(r'\braw\b', re.I), "raw"),
    (re.compile(r'\bmarinated\b', re.I), "marinated"),
    (re.compile(r'\bsoak(?:ed|ing)?\b', re.I), "soaked"),
    (re.compile(r'\bfried\b', re.I), "fried"),
    (re.compile(r'\bboil(?:ed|ing)?\b', re.I), "boiling/boiled"),
    (re.compile(r'\bsimmer(?:ed|ing)?\b', re.I), "simmering"),
    (re.compile(r'\bmelted\b', re.I), "melted"),
    (re.compile(r'\bchopped\b', re.I), "chopped"),
    (re.compile(r'\bsliced\b', re.I), "sliced"),
    (re.compile(r'\bground\b', re.I), "ground"),
    (re.compile(r'\bpaste\b', re.I), "paste form"),
    (re.compile(r'\bpowder(?:ed)?\b', re.I), "powdered"),
]


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config."""
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ingredient_vocabulary(config: dict[str, Any]) -> set[str]:
    """Build ingredient vocabulary from canonical recipe ingredient_aliases.

    Collects English ingredient names from all canonical recipe files.
    Returns a set of lowercased ingredient strings for fast lookup.
    """
    recipe_dir = PROJECT_ROOT / config["paths"].get(
        "canonical_recipes", "configs/canonical_recipes"
    )
    vocab: set[str] = set()

    if not recipe_dir.exists():
        logger.warning(f"Canonical recipes dir not found: {recipe_dir}")
        return vocab

    for path in recipe_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                recipe = json.load(f)
            for ingredient in recipe.get("ingredient_aliases", {}):
                vocab.add(ingredient.lower())
            # Also pull from key_spices
            for spice in recipe.get("key_spices", []):
                vocab.add(spice.lower())
        except (json.JSONDecodeError, OSError):
            pass

    logger.info(f"Ingredient vocabulary: {len(vocab)} terms from canonical recipes")
    return vocab


def _extract_nlp_features(
    text: str,
    ingredient_vocab: set[str],
) -> dict[str, Any]:
    """Extract cooking-relevant NLP features from a transcript segment.

    Pure regex + lexicon approach — no model loading, no API calls.
    Runs on the text/text_en field of each segment.
    """
    text_lower = text.lower()

    # 1. Detected ingredients (from canonical recipe vocabulary)
    detected_ingredients: list[str] = []
    for ingredient in sorted(ingredient_vocab, key=len, reverse=True):
        # Match whole-word (avoid "rice" matching inside "price")
        if re.search(r'\b' + re.escape(ingredient) + r'\b', text_lower):
            detected_ingredients.append(ingredient)

    # 2. Cooking actions
    cooking_actions: list[str] = []
    for action in sorted(COOKING_ACTIONS):
        # Match the action as a word boundary (verb forms too)
        pattern = r'\b' + re.escape(action)
        if len(action) <= 3:
            pattern += r'\b'  # exact match for short words like "fry", "cut"
        if re.search(pattern, text_lower):
            cooking_actions.append(action)

    # 3. Utensils
    utensils: list[str] = []
    for utensil in sorted(UTENSIL_LEXICON):
        if re.search(r'\b' + re.escape(utensil) + r'\b', text_lower):
            utensils.append(utensil)

    # 4. Temperatures
    temperatures: list[str] = [
        m.group(0).strip() for m in _RE_TEMPERATURE.finditer(text)
    ]

    # 5. Quantities
    quantities: list[str] = [
        m.group(0).strip() for m in _RE_QUANTITY.finditer(text)
    ]

    # 6. Ingredient state (best-effort: which ingredient is in what state)
    ingredient_state: dict[str, str] = {}
    for ingredient in detected_ingredients:
        # Look for state indicators near the ingredient mention
        # Use a tight window (~20 chars) to avoid cross-ingredient contamination
        for m in re.finditer(r'\b' + re.escape(ingredient) + r'\b', text_lower):
            start = max(0, m.start() - 20)
            end = min(len(text_lower), m.end() + 20)
            window = text_lower[start:end]
            for pattern, state in _STATE_INDICATORS:
                if pattern.search(window):
                    ingredient_state[ingredient] = state
                    break  # first match wins for this ingredient

    return {
        "detected_ingredients": detected_ingredients,
        "cooking_actions": cooking_actions,
        "utensils": utensils,
        "temperatures": temperatures,
        "quantities": quantities,
        "ingredient_state": ingredient_state,
    }


def enrich_segments_nlp(
    segments: list[dict[str, Any]],
    ingredient_vocab: set[str],
) -> list[dict[str, Any]]:
    """Add NLP features to each transcript segment.

    Runs on text_en (if available) or text field.
    Adds: detected_ingredients, cooking_actions, utensils,
          temperatures, quantities, ingredient_state.
    """
    for seg in segments:
        # Prefer English text for NLP extraction
        text = seg.get("text_en", seg.get("text", ""))
        features = _extract_nlp_features(text, ingredient_vocab)
        seg.update(features)

    return segments


def load_selection() -> list[dict[str, Any]]:
    """Load pipeline video selection from Phase 0C."""
    path = PROJECT_ROOT / "datasets/categorized/pipeline_selection.json"
    if not path.exists():
        logger.error(f"Not found: {path}. Run Phase 0C first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["videos"]


def find_video_file(video_id: str, category: str, raw_dir: Path) -> Path | None:
    """Find the downloaded video file."""
    cat_dir = raw_dir / category.lower().replace("/", "_").replace(" ", "_")
    for ext in [".mp4", ".mkv", ".webm"]:
        path = cat_dir / f"{video_id}{ext}"
        if path.exists():
            return path
    return None


def check_gpu() -> str:
    """Check for GPU availability and return device string."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS (Apple Silicon)")
            return "mps"
    except ImportError:
        pass

    logger.warning("No GPU found — running on CPU (very slow for ASR)")
    return "cpu"


def load_whisperx_model(
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
):
    """Load WhisperX model."""
    import whisperx

    logger.info(f"Loading WhisperX {model_size} on {device} (compute: {compute_type})")

    # Adjust compute type for non-CUDA devices
    if device != "cuda":
        compute_type = "float32"
        batch_size = 4

    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    return model, batch_size


def transcribe_video(
    video_path: Path,
    model,
    device: str,
    batch_size: int,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """
    Transcribe a single video using WhisperX with word-level timestamps.

    Returns structured transcript with segments and word-level alignment.
    """
    import whisperx

    result: dict[str, Any] = {
        "status": "unknown",
        "language": None,
        "segments": [],
        "word_segments": [],
        "duration": None,
        "error": None,
    }

    try:
        # Step 1: Load audio
        audio = whisperx.load_audio(str(video_path))
        result["duration"] = len(audio) / 16000  # WhisperX uses 16kHz

        # Step 2: Transcribe
        logger.info(f"    Transcribing ({result['duration']:.0f}s audio)...")
        transcript = model.transcribe(
            audio,
            batch_size=batch_size,
            language=None,  # Auto-detect
        )

        result["language"] = transcript.get("language", "unknown")
        logger.info(f"    Detected language: {result['language']}")

        # Step 3: Word-level alignment (non-fatal — unsupported
        # language codes like garbage detections skip gracefully)
        align_model = None
        try:
            logger.info("    Aligning words...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device,
            )
            aligned = whisperx.align(
                transcript["segments"],
                align_model,
                align_metadata,
                audio,
                device=device,
                return_char_alignments=False,
            )
        except Exception as e:
            logger.warning(
                f"    Word alignment skipped (non-fatal) for lang="
                f"{result['language']}: {e}"
            )
            # Fall back to unaligned transcript segments
            aligned = {"segments": transcript["segments"], "word_segments": []}
        finally:
            if align_model is not None:
                del align_model
                gc.collect()

        # Step 4: Speaker diarization (optional, needs HF token)
        if hf_token:
            try:
                logger.info("    Running speaker diarization...")
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=device,
                )
                diarize_segments = diarize_model(audio)
                aligned = whisperx.assign_word_speakers(diarize_segments, aligned)
            except Exception as e:
                logger.warning(f"    Diarization failed (non-fatal): {e}")

        # Step 5: Extract segments
        result["segments"] = [
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker", None),
                "words": seg.get("words", []),
            }
            for seg in aligned.get("segments", [])
        ]

        result["word_segments"] = aligned.get("word_segments", [])
        result["status"] = "success"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        logger.error(f"    Transcription error: {e}")

    return result


def translate_segments(
    segments: list[dict[str, Any]],
    source_lang: str,
    groq_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Translate non-English segments to English.

    Uses Groq API (Llama-3.1-70B) for translation.
    Falls back to returning original text if translation fails.
    """
    if source_lang == "en":
        for seg in segments:
            seg["text_en"] = seg["text"]
        return segments

    if not groq_api_key:
        logger.warning("  No GROQ_API_KEY — skipping translation")
        for seg in segments:
            seg["text_en"] = seg["text"]  # Keep original
        return segments

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
    except ImportError:
        logger.warning("  groq package not installed — skipping translation")
        for seg in segments:
            seg["text_en"] = seg["text"]
        return segments

    lang_names = {
        "hi": "Hindi", "te": "Telugu", "ml": "Malayalam",
        "bn": "Bengali", "ur": "Urdu", "ta": "Tamil",
        "as": "Assamese", "mr": "Marathi", "gu": "Gujarati",
    }
    lang_name = lang_names.get(source_lang, source_lang)

    # Batch translation: combine segments into chunks of ~10
    batch_size = 10
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        texts = [seg["text"] for seg in batch]
        combined = "\n---\n".join(texts)

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a cooking video translator. Translate the following {lang_name} cooking instructions to English. "
                                   f"Preserve cooking terminology. Separate translations with ---. "
                                   f"Keep the same number of segments. Only output translations."
                    },
                    {"role": "user", "content": combined},
                ],
                temperature=0.1,
                max_tokens=4096,
            )

            translated = response.choices[0].message.content.strip().split("---")

            for j, seg in enumerate(batch):
                if j < len(translated):
                    seg["text_en"] = translated[j].strip()
                else:
                    seg["text_en"] = seg["text"]

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            logger.warning(f"  Translation batch failed: {e}")
            for seg in batch:
                seg["text_en"] = seg["text"]

    return segments


def save_transcript(
    video_id: str,
    category: str,
    transcript: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save transcript JSON."""
    output = {
        "schema_version": "2.0",
        "video_id": video_id,
        "category": category,
        "language": transcript["language"],
        "duration_seconds": transcript["duration"],
        "num_segments": len(transcript["segments"]),
        "segments": transcript["segments"],
        "word_segments": transcript["word_segments"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{video_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return filepath


def main() -> None:
    """Stage 2 main entry point."""
    parser = argparse.ArgumentParser(
        description="CulinaryVLM Stage 2 — ASR + Translation",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-size", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 2: ASR + Translation       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)
    videos = load_selection()
    ingredient_vocab = load_ingredient_vocabulary(config)

    # Filter
    if args.category:
        videos = [v for v in videos if v["category"] == args.category]
        logger.info(f"Filtered to '{args.category}': {len(videos)} videos")
    if args.limit:
        videos = videos[:args.limit]
        logger.info(f"Limited to {args.limit} videos")

    # Paths
    raw_dir = PROJECT_ROOT / config["paths"]["raw_videos"]
    transcript_dir = PROJECT_ROOT / config["paths"]["transcripts"]

    # Check which videos are actually downloaded
    available = []
    for v in videos:
        vpath = find_video_file(v["video_id"], v["category"], raw_dir)
        if vpath:
            available.append((v, vpath))
        else:
            if not args.dry_run:
                logger.debug(f"  Not downloaded: {v['video_id']}")

    logger.info(f"Total selected: {len(videos)}, Downloaded: {len(available)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would transcribe:")
        for v, vpath in available[:10]:
            logger.info(f"  {v['category']:15s} | {v['video_id']} | {vpath.name}")
        if len(available) > 10:
            logger.info(f"  ... and {len(available) - 10} more")
        return

    if not available:
        logger.error("No downloaded videos found! Run Stage 1 first.")
        sys.exit(1)

    # Setup
    device = check_gpu()
    groq_key = os.environ.get("GROQ_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if groq_key.startswith("REPLACE"):
        groq_key = ""
    if hf_token.startswith("REPLACE"):
        hf_token = ""

    # Load WhisperX
    model, effective_batch = load_whisperx_model(
        model_size=args.model_size,
        device=device,
        batch_size=args.batch_size,
    )

    # Transcribe loop
    stats = {"success": 0, "skipped": 0, "error": 0}

    for i, (video, vpath) in enumerate(available, 1):
        vid = video["video_id"]
        cat = video["category"]

        logger.info(f"\n[{i}/{len(available)}] {cat} | {vid}")

        # Resume check
        transcript_path = transcript_dir / f"{vid}.json"
        if args.resume and transcript_path.exists():
            logger.info(f"  ⊘ Already transcribed (resume mode)")
            stats["skipped"] += 1
            continue

        # Transcribe
        transcript = transcribe_video(
            vpath, model, device, effective_batch, hf_token or None,
        )

        if transcript["status"] == "success":
            # Translate if not English
            transcript["segments"] = translate_segments(
                transcript["segments"],
                transcript["language"],
                groq_key or None,
            )

            # NLP feature extraction (v2.0 — lightweight, no API calls)
            transcript["segments"] = enrich_segments_nlp(
                transcript["segments"],
                ingredient_vocab,
            )

            saved = save_transcript(vid, cat, transcript, transcript_dir)
            logger.info(
                f"  ✓ {transcript['language']} | "
                f"{len(transcript['segments'])} segments | "
                f"{transcript['duration']:.0f}s"
            )
            stats["success"] += 1
        else:
            stats["error"] += 1

    # Summary
    logger.info("")
    logger.info("═" * 50)
    logger.info("ASR SUMMARY")
    logger.info("═" * 50)
    for key, count in stats.items():
        if count > 0:
            logger.info(f"  {key:20s}: {count}")
    logger.info("═" * 50)
    logger.info("")
    logger.info("✓ Stage 2 complete!")
    logger.info("  NEXT: Stage 3 — Canonical Recipe Generation")
    logger.info("        python pipeline/03_recipe_gen.py")


if __name__ == "__main__":
    main()
