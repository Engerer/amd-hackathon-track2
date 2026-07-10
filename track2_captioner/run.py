from __future__ import annotations

import argparse
import json
from pathlib import Path

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import load_settings
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, discover_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Track 2 captions for a folder of videos.")
    parser.add_argument("--input", type=Path, default=Path("data/videos"), help="Folder containing video clips.")
    parser.add_argument("--output", type=Path, default=Path("outputs/captions.json"), help="Output JSON path.")
    parser.add_argument("--work-dir", type=Path, default=Path("data/frames"), help="Frame extraction folder.")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Maximum sampled frames per video.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip internal accuracy/tone judge calls.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Fireworks; emit placeholder output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    videos = discover_videos(args.input)

    if not videos:
        print(f"No videos found in {args.input}. Add .mp4/.mov/.mkv/.avi/.webm files and run again.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("[]\n", encoding="utf-8")
        return

    pipeline = CaptionPipeline(
        settings=settings,
        work_dir=args.work_dir,
        dry_run=args.dry_run,
        max_frames=args.max_frames,
        run_checks=not args.skip_checks,
    )
    results = []

    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] Processing {video.path.name}")
        results.append(pipeline.process(video))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
