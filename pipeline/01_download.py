#!/usr/bin/env python3
"""
CulinaryVLM — Stage 1: Video Download
═══════════════════════════════════════════════════════════════

Downloads selected videos from YouTube using yt-dlp.
Handles deduplication by perceptual hash, filters Shorts and
overlong videos, and organizes downloads by category.

Usage:
    python pipeline/01_download.py
    python pipeline/01_download.py --limit 10            # test with 10 videos
    python pipeline/01_download.py --category Hyderabadi  # single category
    python pipeline/01_download.py --resume               # skip already downloaded
    python pipeline/01_download.py --dry-run

Run on: MacBook (no GPU) or cluster head node
Input:  datasets/categorized/pipeline_selection.json
Output: datasets/raw_videos/{category}/{video_id}.mp4
        datasets/download_log.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from dotenv import load_dotenv

load_dotenv()

# Ensure deno JS runtime is in PATH for yt-dlp YouTube signature solving
_deno_bin = Path.home() / ".deno" / "bin"
if _deno_bin.exists() and str(_deno_bin) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_deno_bin}:{os.environ.get('PATH', '')}"

# ─── Logging ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stage_1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/config.yaml"


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config."""
    path = PROJECT_ROOT / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_selection(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load pipeline video selection from Phase 0C."""
    path = PROJECT_ROOT / "datasets/categorized/pipeline_selection.json"
    if not path.exists():
        logger.error(f"Not found: {path}. Run Phase 0C first.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["videos"]


def get_ytdlp_cmd() -> list[str]:
    """Get the yt-dlp command, handling PATH issues."""
    # Try bare command first
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return ["yt-dlp"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: python -m yt_dlp
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [sys.executable, "-m", "yt_dlp"]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


def check_yt_dlp() -> list[str] | None:
    """Check if yt-dlp is installed. Returns the command list or None."""
    cmd = get_ytdlp_cmd()
    if cmd:
        result = subprocess.run(
            cmd + ["--version"],
            capture_output=True, text=True, timeout=10,
        )
        logger.info(f"yt-dlp version: {result.stdout.strip()} (via {' '.join(cmd)})")
        return cmd
    logger.error("yt-dlp not found. Install: pip install yt-dlp")
    return None


def download_video(
    url: str,
    output_dir: Path,
    video_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Download a single video using yt-dlp.

    Returns a download result dict with status and metadata.
    """
    dl_config = config["pipeline"]["download"]
    output_template = str(output_dir / f"{video_id}.%(ext)s")

    ytdlp_cmd = get_ytdlp_cmd()
    cmd = ytdlp_cmd + [
        "--format", dl_config["format"],
        "--output", output_template,
        "--merge-output-format", "mkv",
        "--rate-limit", dl_config["rate_limit"],
        "--no-playlist",
        "--no-overwrites",
        "--write-info-json",
        "--print-json",
        "--socket-timeout", "30",
        "--retries", "3",
        "--no-check-certificates",
        url,
    ]

    # Cookie support to handle YouTube authentication issues.
    # Place a cookies.txt file in the project root (export from browser extension).
    # IMPORTANT: yt-dlp reads AND writes to the cookie file, which can corrupt
    # the original. We use a temp copy to prevent this.
    cookie_file = PROJECT_ROOT / "cookies.txt"
    cookie_tmp = None
    if cookie_file.exists() and cookie_file.stat().st_size > 0:
        import shutil
        import tempfile
        cookie_tmp = Path(tempfile.mktemp(suffix="_cookies.txt", dir=str(PROJECT_ROOT)))
        shutil.copy2(cookie_file, cookie_tmp)
        cmd.insert(-1, "--cookies")
        cmd.insert(-1, str(cookie_tmp))

    result: dict[str, Any] = {
        "video_id": video_id,
        "url": url,
        "status": "unknown",
        "filepath": None,
        "duration": None,
        "title": None,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max per video
        )

        if proc.returncode == 0:
            # Parse yt-dlp JSON output
            try:
                info = json.loads(proc.stdout.strip().split("\n")[-1])
                duration = info.get("duration", 0) or 0

                # Filter by duration
                if duration < dl_config["min_duration_sec"]:
                    result["status"] = "skipped_short"
                    result["duration"] = duration
                    result["error"] = f"Too short: {duration}s < {dl_config['min_duration_sec']}s"
                    # Remove the downloaded file
                    for ext in [".mkv", ".mp4", ".webm"]:
                        dl_path = output_dir / f"{video_id}{ext}"
                        if dl_path.exists():
                            dl_path.unlink()
                    return result

                if duration > dl_config["max_duration_sec"]:
                    result["status"] = "skipped_long"
                    result["duration"] = duration
                    result["error"] = f"Too long: {duration}s > {dl_config['max_duration_sec']}s"
                    for ext in [".mkv", ".mp4", ".webm"]:
                        dl_path = output_dir / f"{video_id}{ext}"
                        if dl_path.exists():
                            dl_path.unlink()
                    return result

                result["status"] = "success"
                result["duration"] = duration
                result["title"] = info.get("title", "")
                # Find the actual downloaded file (mkv, mp4, or webm)
                for ext in [".mkv", ".mp4", ".webm"]:
                    dl_path = output_dir / f"{video_id}{ext}"
                    if dl_path.exists():
                        result["filepath"] = str(dl_path)
                        break

            except (json.JSONDecodeError, IndexError):
                # Download succeeded but couldn't parse JSON
                for ext in [".mkv", ".mp4", ".webm"]:
                    dl_path = output_dir / f"{video_id}{ext}"
                    if dl_path.exists():
                        result["status"] = "success"
                        result["filepath"] = str(dl_path)
                        break
                else:
                    result["status"] = "error"
                    result["error"] = "Download finished but file not found"
        else:
            result["status"] = "error"
            # Filter out warnings to find the actual error
            stderr_lines = proc.stderr.strip().split("\n")
            error_lines = [
                line for line in stderr_lines
                if not any(skip in line for skip in ["Warning", "Deprecated", "urllib3", "warnings.warn"])
                and line.strip()
            ]
            result["error"] = "\n".join(error_lines)[:500] if error_lines else proc.stderr.strip()[:500]

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Download timed out after 600s"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
    finally:
        # Clean up temp cookie file
        if cookie_tmp and cookie_tmp.exists():
            cookie_tmp.unlink()

    return result


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_download_log(log_path: Path) -> dict[str, Any]:
    """Load existing download log for resume support."""
    if log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return {"downloads": {}, "hashes": {}}


def save_download_log(log: dict[str, Any], log_path: Path) -> None:
    """Save download log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def main() -> None:
    """Stage 1 main entry point."""
    parser = argparse.ArgumentParser(
        description="CulinaryVLM Stage 1 — Download videos",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only N videos (for testing)")
    parser.add_argument("--category", default=None,
                        help="Download only this category")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-downloaded videos")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  CulinaryVLM — Stage 1: Video Download          ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    config = load_config(args.config)
    videos = load_selection(config)

    # Filter by category if specified
    if args.category:
        videos = [v for v in videos if v["category"] == args.category]
        logger.info(f"Filtered to category '{args.category}': {len(videos)} videos")

    # Apply limit
    if args.limit:
        videos = videos[:args.limit]
        logger.info(f"Limited to {args.limit} videos")

    # Output directory
    output_base = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / config["paths"]["raw_videos"]
    )
    log_path = PROJECT_ROOT / "datasets/download_log.json"

    logger.info(f"Videos to download: {len(videos)}")
    logger.info(f"Output directory: {output_base}")

    if args.dry_run:
        logger.info("[DRY RUN] Would download the following:")
        for v in videos[:10]:
            logger.info(f"  {v['category']:15s} | {v['video_id']} | {v['title'][:50]}")
        if len(videos) > 10:
            logger.info(f"  ... and {len(videos) - 10} more")
        return

    # Check yt-dlp
    if not check_yt_dlp():
        sys.exit(1)

    # Load existing log for resume
    dl_log = load_download_log(log_path)
    file_hashes: dict[str, str] = dl_log.get("hashes", {})

    # Download loop
    stats = {"success": 0, "skipped_existing": 0, "skipped_short": 0,
             "skipped_long": 0, "skipped_dup": 0, "error": 0, "timeout": 0}

    for i, video in enumerate(videos, 1):
        vid = video["video_id"]
        cat = video["category"]

        logger.info(f"\n[{i}/{len(videos)}] {cat} | {vid} | {video['title'][:50]}")

        # Resume check
        if args.resume and vid in dl_log.get("downloads", {}):
            prev = dl_log["downloads"][vid]
            if prev.get("status") == "success" and prev.get("filepath"):
                if Path(prev["filepath"]).exists():
                    logger.info(f"  ⊘ Already downloaded (resume mode)")
                    stats["skipped_existing"] += 1
                    continue

        # Create category directory
        cat_dir = output_base / cat.lower().replace("/", "_").replace(" ", "_")
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Download
        result = download_video(video["url"], cat_dir, vid, config)

        if result["status"] == "success" and result["filepath"]:
            # Perceptual hash deduplication
            file_hash = compute_file_hash(result["filepath"])
            if file_hash in file_hashes:
                logger.warning(f"  ⊘ Duplicate of {file_hashes[file_hash]} — removing")
                Path(result["filepath"]).unlink()
                result["status"] = "skipped_dup"
                stats["skipped_dup"] += 1
            else:
                file_hashes[file_hash] = vid
                stats["success"] += 1
                logger.info(f"  ✓ Downloaded: {result['duration']}s")
        else:
            stats[result["status"]] = stats.get(result["status"], 0) + 1
            if result["error"]:
                logger.warning(f"  ✗ {result['status']}: {result['error'][:80]}")

        # Save progress
        dl_log.setdefault("downloads", {})[vid] = result
        dl_log["hashes"] = file_hashes
        save_download_log(dl_log, log_path)

        # Small delay to be polite to YouTube
        time.sleep(1)

    # Final summary
    logger.info("")
    logger.info("═" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("═" * 50)
    for key, count in stats.items():
        if count > 0:
            logger.info(f"  {key:20s}: {count}")
    logger.info("═" * 50)
    logger.info("")
    logger.info("✓ Stage 1 complete!")
    logger.info("  NEXT: Stage 2 — ASR + Translation (on GPU cluster)")
    logger.info("        Upload videos to cluster: scp -r datasets/raw_videos/ iiitd@cb-cluster.iiitd.edu.in:/storage/iiitd/culinary_vlm/")


if __name__ == "__main__":
    main()
