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
from track2_captioner.video_ingest import DEFAULT_FRAME_PROFILE, DEFAULT_MAX_FRAMES, VIDEO_EXTENSIONS, VideoAsset


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
DEFAULT_RUNTIME_TARGET_SECONDS = 540.0
DEFAULT_HARD_DEADLINE_SECONDS = 585.0
REDUCE_FRAMES_AFTER_SECONDS = 450.0
SKIP_STYLE_RETRY_AFTER_SECONDS = 510.0
FALLBACK_CAPTIONS_AFTER_SECONDS = 555.0


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_tasks(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("/input/tasks.json must contain a JSON array.")
    return payload


def float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def download_video(video_url: str, destination_dir: Path, task_id: str, timeout: float = 120.0) -> Path:
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

    with requests.get(video_url, stream=True, timeout=max(5.0, timeout)) as response:
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


def choose_task_frame_budget(
    max_frames: int,
    task_count: int,
    completed_count: int,
    elapsed_seconds: float,
    hard_deadline_seconds: float,
    reduce_frames_after_seconds: float,
) -> int:
    cap = max(1, max_frames)
    if task_count >= 10:
        cap = min(cap, 8)

    remaining_tasks = max(1, task_count - completed_count)
    remaining_seconds = max(0.0, hard_deadline_seconds - elapsed_seconds)
    seconds_per_task = remaining_seconds / remaining_tasks

    if elapsed_seconds >= reduce_frames_after_seconds or seconds_per_task < 45:
        cap = min(cap, 6)
    if seconds_per_task < 30:
        cap = min(cap, 4)
    return max(1, cap)


def run_harness(input_path: Path, output_path: Path) -> int:
    started_at = time.monotonic()
    settings = load_settings()
    dry_run = truthy(os.getenv("TRACK2_DRY_RUN"))
    auto_transcribe = truthy(os.getenv("AUTO_TRANSCRIBE"))
    force_transcribe = truthy(os.getenv("FORCE_TRANSCRIBE"))
    run_checks = truthy(os.getenv("RUN_CHECKS"))
    max_frames = int(os.getenv("TRACK2_MAX_FRAMES", str(DEFAULT_MAX_FRAMES)))
    frame_profile = os.getenv("TRACK2_FRAME_PROFILE", DEFAULT_FRAME_PROFILE)
    enable_style_retry = truthy(os.getenv("TRACK2_ENABLE_STYLE_RETRY", "true"))
    runtime_target_seconds = float_env("TRACK2_RUNTIME_TARGET_SECONDS", DEFAULT_RUNTIME_TARGET_SECONDS)
    hard_deadline_seconds = float_env("TRACK2_HARD_DEADLINE_SECONDS", DEFAULT_HARD_DEADLINE_SECONDS)
    reduce_frames_after = min(REDUCE_FRAMES_AFTER_SECONDS, runtime_target_seconds)
    skip_style_retry_after = min(SKIP_STYLE_RETRY_AFTER_SECONDS, hard_deadline_seconds)
    fallback_captions_after = min(FALLBACK_CAPTIONS_AFTER_SECONDS, hard_deadline_seconds)
    caption_fallback_deadline = started_at + fallback_captions_after
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
            enable_style_retry=enable_style_retry,
        )

        for task_index, task in enumerate(tasks):
            task_started_at = time.monotonic()
            task_id = str(task.get("task_id", ""))
            video_url = str(task.get("video_url", ""))
            styles = task.get("styles") or DEFAULT_STYLES
            styles = [str(style) for style in styles]

            try:
                elapsed = time.monotonic() - started_at
                if elapsed >= hard_deadline_seconds:
                    print(
                        f"Task {task_id}: hard deadline reached before processing; using fallback captions.",
                        file=sys.stderr,
                    )
                    results.append({"task_id": task_id, "captions": fallback_captions(styles)})
                    continue

                if elapsed >= fallback_captions_after:
                    print(
                        f"Task {task_id}: fallback window reached before download; using fallback captions.",
                        file=sys.stderr,
                    )
                    results.append({"task_id": task_id, "captions": fallback_captions(styles)})
                    continue

                if dry_run:
                    video_path = video_dir / f"{task_id}.mp4"
                    download_seconds = 0.0
                else:
                    remaining_for_download = max(5.0, hard_deadline_seconds - elapsed)
                    download_started_at = time.monotonic()
                    video_path = download_video(
                        video_url,
                        video_dir,
                        task_id,
                        timeout=min(120.0, remaining_for_download),
                    )
                    download_seconds = time.monotonic() - download_started_at
                asset = VideoAsset(video_id=task_id, path=video_path, transcript_path=None)

                elapsed = time.monotonic() - started_at
                task_max_frames = choose_task_frame_budget(
                    max_frames=max_frames,
                    task_count=len(tasks),
                    completed_count=task_index,
                    elapsed_seconds=elapsed,
                    hard_deadline_seconds=hard_deadline_seconds,
                    reduce_frames_after_seconds=reduce_frames_after,
                )
                task_enable_style_retry = enable_style_retry and elapsed < skip_style_retry_after
                force_fallback_captions = elapsed >= fallback_captions_after

                if auto_transcribe and elapsed < reduce_frames_after:
                    try:
                        transcript_path = transcribe_video(
                            video_path=video_path,
                            transcript_dir=transcript_dir,
                            model_name=whisper_model,
                            language=whisper_language,
                            force=force_transcribe,
                        )
                        asset = VideoAsset(video_id=task_id, path=video_path, transcript_path=transcript_path)
                    except Exception as exc:
                        print(f"Task {task_id}: Whisper skipped: {exc}", file=sys.stderr)
                elif auto_transcribe:
                    print(f"Task {task_id}: Whisper skipped by runtime guard.", file=sys.stderr)

                processed = pipeline.process(
                    asset,
                    styles=styles,
                    max_frames=task_max_frames,
                    frame_profile=frame_profile,
                    enable_style_retry=task_enable_style_retry,
                    force_fallback_captions=force_fallback_captions,
                    caption_fallback_deadline=caption_fallback_deadline,
                )
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
                timings = processed.get("timings", {})
                task_total = time.monotonic() - task_started_at
                elapsed_total = time.monotonic() - started_at
                print(
                    (
                        f"Task {task_id}: download={download_seconds:.1f}s "
                        f"frames={timings.get('frame_extraction_sec', 0.0):.1f}s "
                        f"vision={timings.get('vision_observation_sec', 0.0):.1f}s "
                        f"captions={timings.get('caption_generation_sec', 0.0):.1f}s "
                        f"task_total={task_total:.1f}s elapsed={elapsed_total:.1f}s "
                        f"max_frames={task_max_frames} retry={task_enable_style_retry}"
                    ),
                    file=sys.stderr,
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
