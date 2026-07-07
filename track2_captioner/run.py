from __future__ import annotations

import argparse
import json
from pathlib import Path

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import load_settings
from track2_captioner.transcription import WHISPER_MODELS, transcribe_video
from track2_captioner.video_ingest import discover_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Track 2 captions for a folder of videos.")
    parser.add_argument("--input", type=Path, default=Path("data/videos"), help="Folder containing video clips.")
    parser.add_argument("--output", type=Path, default=Path("outputs/captions.json"), help="Output JSON path.")
    parser.add_argument("--work-dir", type=Path, default=Path("data/frames"), help="Frame extraction folder.")
    parser.add_argument("--transcripts", type=Path, default=None, help="Optional transcript folder.")
    parser.add_argument("--max-frames", type=int, default=5, help="Maximum sampled frames per video.")
    parser.add_argument("--auto-transcribe", action="store_true", help="Generate missing transcripts with Whisper.")
    parser.add_argument("--whisper-model", choices=WHISPER_MODELS, default="base", help="Whisper model size.")
    parser.add_argument("--whisper-language", default="", help="Optional Whisper language code, for example en.")
    parser.add_argument("--force-transcribe", action="store_true", help="Overwrite existing transcript files.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip internal accuracy/tone judge calls.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Fireworks; emit placeholder output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    videos = discover_videos(args.input, args.transcripts)

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
        if args.auto_transcribe and (args.force_transcribe or video.transcript_path is None):
            transcript_dir = args.transcripts or args.input.parent / "transcripts"
            transcript_path = transcribe_video(
                video_path=video.path,
                transcript_dir=transcript_dir,
                model_name=args.whisper_model,
                language=args.whisper_language.strip() or None,
                force=args.force_transcribe,
            )
            video = type(video)(
                video_id=video.video_id,
                path=video.path,
                transcript_path=transcript_path,
            )
        results.append(pipeline.process(video))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
