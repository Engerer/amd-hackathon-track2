from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import load_settings
from track2_captioner.transcription import transcribe_video
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, VIDEO_EXTENSIONS, VideoAsset


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
DEFAULT_RUNTIME_BUDGET_SECONDS = 570
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120
DEFAULT_TRANSCRIBE_MIN_REMAINING_SECONDS = 150


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def read_tasks(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("/input/tasks.json must contain a JSON array.")
    return payload


def download_video(video_url: str, destination_dir: Path, task_id: str, timeout_seconds: int = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS) -> Path:
    import requests

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

    with requests.get(video_url, stream=True, timeout=(10, timeout_seconds)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def seconds_remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def adaptive_frame_budget(max_frames: int, remaining_seconds: float, tasks_left: int) -> int:
    if remaining_seconds <= 75:
        return min(max_frames, 8)
    if remaining_seconds <= 150:
        return min(max_frames, 12)
    if tasks_left > 1 and remaining_seconds <= 300:
        return min(max_frames, 18)
    return max_frames


def fallback_captions(styles: list[str], observations: dict[str, Any] | None = None) -> dict[str, str]:
    observations = observations or {}
    summary = str(observations.get("summary") or "").strip()
    setting = str(observations.get("setting") or "").strip()
    subjects = ", ".join(str(item) for item in observations.get("subjects", [])[:2])
    actions = ", ".join(str(item) for item in observations.get("actions", [])[:2])
    details = ", ".join(str(item) for item in observations.get("key_objects", [])[:2])
    base = summary or f"The video shows {subjects or 'visible subjects'} with {actions or 'visible activity'}."
    if setting and setting.lower() not in base.lower():
        base = f"{base.rstrip('.')} in {setting}."
    if details and details.lower() not in base.lower():
        base = f"{base.rstrip('.')} with {details}."

    templates = {
        "formal": base,
        "sarcastic": f"{base} Clearly, ordinary visual evidence has never worked harder.",
        "humorous_tech": f"{base} The scene ships its visual update with no rollback needed.",
        "humorous_non_tech": f"{base} It is doing its best to make everyday motion look eventful.",
    }
    return {
        style: templates.get(style, base)
        for style in styles
    }


def run_harness(input_path: Path, output_path: Path) -> int:
    settings = load_settings()
    dry_run = truthy(os.getenv("TRACK2_DRY_RUN"))
    auto_transcribe = truthy(os.getenv("AUTO_TRANSCRIBE"))
    force_transcribe = truthy(os.getenv("FORCE_TRANSCRIBE"))
    run_checks = truthy(os.getenv("RUN_CHECKS"))
    max_frames = positive_int_env("TRACK2_MAX_FRAMES", DEFAULT_MAX_FRAMES)
    runtime_budget = positive_int_env("TRACK2_RUNTIME_BUDGET_SECONDS", DEFAULT_RUNTIME_BUDGET_SECONDS)
    download_timeout = positive_int_env("TRACK2_DOWNLOAD_TIMEOUT_SECONDS", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
    transcribe_min_remaining = positive_int_env(
        "TRACK2_TRANSCRIBE_MIN_REMAINING_SECONDS",
        DEFAULT_TRANSCRIBE_MIN_REMAINING_SECONDS,
    )
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    whisper_language = os.getenv("WHISPER_LANGUAGE", "").strip() or None

    tasks = read_tasks(input_path)
    deadline = time.monotonic() + runtime_budget
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

        for index, task in enumerate(tasks, start=1):
            task_id = str(task.get("task_id", "") or f"task_{index}")
            video_url = str(task.get("video_url", ""))
            styles = task.get("styles") or DEFAULT_STYLES
            styles = [str(style) for style in styles]
            tasks_left = max(1, len(tasks) - index + 1)
            remaining = seconds_remaining(deadline)

            if remaining <= 15:
                print(f"Task {task_id}: runtime budget nearly exhausted; using fallback captions.", file=sys.stderr)
                results.append({"task_id": task_id, "captions": fallback_captions(styles)})
                continue

            try:
                task_max_frames = adaptive_frame_budget(max_frames, remaining, tasks_left)
                pipeline.max_frames = task_max_frames

                if dry_run:
                    asset = VideoAsset(video_id=task_id, path=Path(f"{task_id}.mp4"), transcript_path=None)
                else:
                    task_download_timeout = min(download_timeout, max(15, int(remaining) - 30))
                    video_path = download_video(video_url, video_dir, task_id, timeout_seconds=task_download_timeout)
                    asset = VideoAsset(video_id=task_id, path=video_path, transcript_path=None)

                if auto_transcribe and not dry_run:
                    try:
                        if seconds_remaining(deadline) < transcribe_min_remaining:
                            print(f"Task {task_id}: Whisper skipped to preserve runtime budget.", file=sys.stderr)
                        else:
                            transcript_path = transcribe_video(
                                video_path=asset.path,
                                transcript_dir=transcript_dir,
                                model_name=whisper_model,
                                language=whisper_language,
                                force=force_transcribe,
                            )
                            asset = VideoAsset(video_id=task_id, path=asset.path, transcript_path=transcript_path)
                    except Exception as exc:
                        print(f"Task {task_id}: Whisper skipped: {exc}", file=sys.stderr)

                processed = pipeline.process(asset, styles=styles)
                captions = processed.get("captions", {})
                observations = processed.get("observations", {})
                results.append(
                    {
                        "task_id": task_id,
                        "captions": {
                            style: captions.get(style, fallback_captions([style], observations)[style])
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
