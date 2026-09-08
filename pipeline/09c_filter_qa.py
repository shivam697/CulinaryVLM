#!/usr/bin/env python3
"""
CulinaryVLM — Stage 9C: QA Dataset Quality Filter
═════════════════════════════════════════════════
Filters out unverified, low-confidence, or silent segments from the QA dataset.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("filter")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIN_CONFIDENCE = 0.75
MIN_WORDS = 4

def main():
    verified_path = PROJECT_ROOT / "datasets" / "verified" / "verified_segments.json"
    qa_dir = PROJECT_ROOT / "datasets" / "qa"
    
    if not verified_path.exists():
        logger.error(f"Verified segments not found at {verified_path}")
        return

    with open(verified_path) as f:
        verified_data = json.load(f)

    # 1. Build a registry of high-quality segments and valid videos
    valid_segments = set()
    valid_videos = set()
    total_segs = 0

    for seg in verified_data.get("segments", []):
        total_segs += 1
        vid = seg.get("video_id")
        start = seg.get("start_time")
        text = seg.get("description", seg.get("action", ""))
        
        word_count = len(text.split())
        is_verified = seg.get("verified", False)
        confidence = float(seg.get("confidence", 0.0))

        if is_verified and confidence >= MIN_CONFIDENCE and word_count >= MIN_WORDS:
            valid_segments.add(f"{vid}_{start}")
            valid_videos.add(vid)

    logger.info(f"Quality Check: {len(valid_segments)}/{total_segs} segments passed strict filtering.")

    # 2. Filter Train and Test datasets
    for split in ["train", "test"]:
        file_path = qa_dir / f"{split}.json"
        out_path = qa_dir / f"{split}_filtered.json"
        
        if not file_path.exists():
            continue

        with open(file_path) as f:
            data = json.load(f)

        original_pairs = data.get("qa_pairs", [])
        filtered_pairs = []

        for qa in original_pairs:
            tier = qa.get("tier", "")
            vid = qa.get("video_id", "")
            
            if tier == "easy":
                # Easy tier MUST map directly to a high-quality segment
                start = qa.get("segment_start")
                if f"{vid}_{start}" in valid_segments:
                    filtered_pairs.append(qa)
            elif tier == "medium":
                # Medium tier applies to the whole video; keep if video has valid segments
                if vid in valid_videos:
                    filtered_pairs.append(qa)
            elif tier in ["hard", "expert"]:
                # Hard and Expert are category-level (canonical recipe based) - always keep
                filtered_pairs.append(qa)

        # Update metadata and save
        data["total"] = len(filtered_pairs)
        data["qa_pairs"] = filtered_pairs
        data["filtering_applied"] = {
            "min_confidence": MIN_CONFIDENCE,
            "min_words": MIN_WORDS
        }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        dropped = len(original_pairs) - len(filtered_pairs)
        logger.info(f"[{split.upper()}] Kept {len(filtered_pairs)} pairs. Dropped {dropped} noisy pairs.")

if __name__ == "__main__":
    main()
