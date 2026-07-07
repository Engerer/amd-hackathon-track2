from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import load_settings
from track2_captioner.transcription import transcribe_video
from track2_captioner.video_ingest import VIDEO_EXTENSIONS, VideoAsset


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_tasks(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("/input/tasks.json must contain a JSON array.")
    return payload


def download_video(video_url: str, destination_dir: Path, task_id: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if video_url.startswith("file://"):
        local_path = Path(video_url.removeprefix("file://"))
        destination = destination_dir / f"{task_id}{local_path.suffix or '.mp4'}"
        shutil.copyfile(local_path, destination)
        return destination

    local_path = Path(video_url)
    if local_path.exists():
        destination = destination_dir / f"{task_id}{local_path.suffix or '.mp4'}"
        shutil.copyfile(local_path, destination)
        return destination

    suffix = Path(video_url.split("?", 1)[0]).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        suffix = ".mp4"
    destination = destination_dir / f"{task_id}{suffix}"

    with requests.get(video_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def fallback_captions(styles: list[str]) -> dict[str, str]:
    return {
        style: "The video shows a scene with visible subjects and activity."
        for style in styles
    }


def run_harness(input_path: Path, output_path: Path) -> int:
    settings = load_settings()
    dry_run = truthy(os.getenv("TRACK2_DRY_RUN"))
    auto_transcribe = truthy(os.getenv("AUTO_TRANSCRIBE"))
    force_transcribe = truthy(os.getenv("FORCE_TRANSCRIBE"))
    run_checks = truthy(os.getenv("RUN_CHECKS"))
    max_frames = int(os.getenv("TRACK2_MAX_FRAMES", "8"))
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    whisper_language = os.getenv("WHISPER_LANGUAGE", "").strip() or None

    tasks = read_tasks(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="track2_harness_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_dir = temp_dir / "videos"
        transcript_dir = temp_dir / "transcripts"
        pipeline = CaptionPipeline(
            settings=settings,
            work_dir=temp_dir / "frames",
            dry_run=dry_run,
            max_frames=max_frames,
            run_checks=run_checks,
        )

        for task in tasks:
            task_id = str(task.get("task_id", ""))
            video_url = str(task.get("video_url", ""))
            styles = task.get("styles") or DEFAULT_STYLES
            styles = [str(style) for style in styles]

            try:
                video_path = download_video(video_url, video_dir, task_id)
                asset = VideoAsset(video_id=task_id, path=video_path, transcript_path=None)

                if auto_transcribe:
                    transcript_path = transcribe_video(
                        video_path=video_path,
                        transcript_dir=transcript_dir,
                        model_name=whisper_model,
                        language=whisper_language,
                        force=force_transcribe,
                    )
                    asset = VideoAsset(video_id=task_id, path=video_path, transcript_path=transcript_path)

                processed = pipeline.process(asset, styles=styles)
                captions = processed.get("captions", {})
                results.append(
                    {
                        "task_id": task_id,
                        "captions": {
                            style: captions.get(style, fallback_captions([style])[style])
                            for style in styles
                        },
                    }
                )
            except Exception as exc:
                print(f"Task {task_id or '[missing task_id]'} failed: {exc}", file=sys.stderr)
                results.append({"task_id": task_id, "captions": fallback_captions(styles)})

    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    input_path = Path(os.getenv("TRACK2_INPUT", "/input/tasks.json"))
    output_path = Path(os.getenv("TRACK2_OUTPUT", "/output/results.json"))
    raise SystemExit(run_harness(input_path, output_path))


if __name__ == "__main__":
    main()
