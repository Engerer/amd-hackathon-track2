from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import load_settings
from track2_captioner.video_ingest import DEFAULT_FRAME_PROFILE, DEFAULT_MAX_FRAMES, VIDEO_EXTENSIONS, VideoAsset


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
DEFAULT_RUNTIME_TARGET_SECONDS = 540.0
DEFAULT_HARD_DEADLINE_SECONDS = 585.0
REDUCE_FRAMES_AFTER_SECONDS = 450.0
SKIP_STYLE_RETRY_AFTER_SECONDS = 510.0
FALLBACK_CAPTIONS_AFTER_SECONDS = 555.0
DEFAULT_MODEL_CALL_RESERVE_SECONDS = 60.0

# Per-clip deadline: maximum wall-clock time for a single video
DEFAULT_PER_CLIP_DEADLINE_SECONDS = 28.0


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_tasks(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input task file not found at {input_path}.")

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Could not read {input_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"{input_path} must contain a JSON array.")
    if not all(isinstance(task, dict) for task in payload):
        raise ValueError(f"{input_path} must contain only JSON objects.")
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
    fallbacks = {
        "formal": "The video presents visible subjects and activity within the scene.",
        "sarcastic": "Visible subjects carry on with the activity, because apparently the scene insists on staying busy.",
        "humorous_tech": "The visible scene keeps its activity running like a process with no scheduled downtime.",
        "humorous_non_tech": "The visible subjects keep things moving, as if standing still simply was not on today's agenda.",
    }
    return {style: fallbacks.get(style, fallbacks["formal"]) for style in styles}


def task_styles(task: dict[str, Any]) -> list[str]:
    styles = task.get("styles")
    if styles is None:
        return list(DEFAULT_STYLES)
    if not isinstance(styles, list) or not styles:
        raise ValueError("Task styles must be a non-empty JSON array.")

    requested: list[str] = []
    for style in styles:
        if style not in DEFAULT_STYLES:
            raise ValueError(f"Unsupported requested style: {style!r}")
        if style not in requested:
            requested.append(style)
    return requested


def write_results_atomic(output_path: Path, results: list[dict[str, Any]]) -> None:
    """Write results using atomic temporary-file replacement.

    Writes to a temporary file first, then atomically replaces the target.
    This prevents partial writes from corrupting output on crash or timeout.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(results, indent=2) + "\n"

    # Write to temp file in the same directory (required for os.replace atomicity)
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=".results_",
            suffix=".tmp",
        )
        temp_path = Path(temp_path_str)
        os.write(temp_fd, content.encode("utf-8"))
        os.close(temp_fd)
        temp_fd = None
        # Atomic replacement
        os.replace(str(temp_path), str(output_path))
    except Exception:
        # Fallback to direct write if atomic fails (e.g., cross-device)
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        output_path.write_text(content, encoding="utf-8")


# Keep old name for backward compatibility
def write_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    write_results_atomic(output_path, results)


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
        cap = min(cap, 5)

    remaining_tasks = max(1, task_count - completed_count)
    remaining_seconds = max(0.0, hard_deadline_seconds - elapsed_seconds)
    seconds_per_task = remaining_seconds / remaining_tasks

    if elapsed_seconds >= reduce_frames_after_seconds or seconds_per_task < 45:
        cap = min(cap, 8)
    if seconds_per_task < 30:
        cap = min(cap, 6)
    return max(1, cap)


def run_harness(input_path: Path, output_path: Path) -> int:
    started_at = time.monotonic()
    settings = load_settings()
    dry_run = truthy(os.getenv("TRACK2_DRY_RUN"))
    run_checks = truthy(os.getenv("RUN_CHECKS"))
    max_frames = int(os.getenv("TRACK2_MAX_FRAMES", str(DEFAULT_MAX_FRAMES)))
    frame_profile = os.getenv("TRACK2_FRAME_PROFILE", DEFAULT_FRAME_PROFILE)
    enable_style_retry = truthy(os.getenv("TRACK2_ENABLE_STYLE_RETRY", "true"))
    runtime_target_seconds = float_env("TRACK2_RUNTIME_TARGET_SECONDS", DEFAULT_RUNTIME_TARGET_SECONDS)
    hard_deadline_seconds = float_env("TRACK2_HARD_DEADLINE_SECONDS", DEFAULT_HARD_DEADLINE_SECONDS)
    per_clip_deadline = float_env("TRACK2_PER_CLIP_DEADLINE_SECONDS", DEFAULT_PER_CLIP_DEADLINE_SECONDS)
    model_call_reserve_seconds = float_env(
        "TRACK2_MODEL_CALL_RESERVE_SECONDS",
        max(DEFAULT_MODEL_CALL_RESERVE_SECONDS, settings.request_timeout_seconds + 10.0),
    )
    reduce_frames_after = min(REDUCE_FRAMES_AFTER_SECONDS, runtime_target_seconds)
    skip_style_retry_after = min(SKIP_STYLE_RETRY_AFTER_SECONDS, hard_deadline_seconds)
    fallback_captions_after = min(
        FALLBACK_CAPTIONS_AFTER_SECONDS,
        max(0.0, hard_deadline_seconds - model_call_reserve_seconds),
    )
    caption_fallback_deadline = started_at + fallback_captions_after

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(input_path)
    output_results: list[dict[str, Any]] = [
        {
            "task_id": str(task.get("task_id", "")),
            "captions": fallback_captions(task_styles(task)),
        }
        for task in tasks
    ]
    write_results_atomic(output_path, output_results)
    if not tasks:
        return 0

    with tempfile.TemporaryDirectory(prefix="track2_harness_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_dir = temp_dir / "videos"
        pipeline = CaptionPipeline(
            settings=settings,
            work_dir=temp_dir / "frames",
            dry_run=dry_run,
            max_frames=max_frames,
            run_checks=run_checks,
            enable_style_retry=enable_style_retry,
        )

        download_pool: ThreadPoolExecutor | None = None
        download_futures: dict[int, Future[Path]] = {}
        prefetch_depth = min(3, len(tasks))

        def submit_download(task_index: int) -> None:
            if download_pool is None or task_index >= len(tasks) or task_index in download_futures:
                return
            task = tasks[task_index]
            download_futures[task_index] = download_pool.submit(
                download_video,
                str(task.get("video_url", "")),
                video_dir,
                str(task.get("task_id", "")),
                min(45.0, per_clip_deadline),
            )

        if not dry_run:
            download_pool = ThreadPoolExecutor(max_workers=prefetch_depth, thread_name_prefix="video-download")
            for prefetch_index in range(prefetch_depth):
                submit_download(prefetch_index)

        for task_index, task in enumerate(tasks):
            task_started_at = time.monotonic()
            clip_deadline = task_started_at + per_clip_deadline
            task_id = str(task.get("task_id", ""))
            video_url = str(task.get("video_url", ""))
            styles = task_styles(task)

            try:
                elapsed = time.monotonic() - started_at
                if elapsed >= hard_deadline_seconds:
                    print(
                        f"Task {task_id}: hard deadline reached before processing; using fallback captions.",
                        file=sys.stderr,
                    )
                    output_results[task_index] = {"task_id": task_id, "captions": fallback_captions(styles)}
                    write_results_atomic(output_path, output_results)
                    continue

                if elapsed >= fallback_captions_after:
                    print(
                        f"Task {task_id}: fallback window reached before download; using fallback captions.",
                        file=sys.stderr,
                    )
                    output_results[task_index] = {"task_id": task_id, "captions": fallback_captions(styles)}
                    write_results_atomic(output_path, output_results)
                    continue

                if dry_run:
                    video_path = video_dir / f"{task_id}.mp4"
                    download_seconds = 0.0
                else:
                    remaining_for_download = max(
                        0.5,
                        min(
                            hard_deadline_seconds - elapsed,
                            clip_deadline - time.monotonic(),
                        ),
                    )
                    download_started_at = time.monotonic()
                    download_future = download_futures.pop(task_index, None)
                    if download_future is None:
                        submit_download(task_index)
                        download_future = download_futures.pop(task_index)
                    submit_download(task_index + prefetch_depth)
                    video_path = download_future.result(timeout=remaining_for_download)
                    download_seconds = time.monotonic() - download_started_at
                asset = VideoAsset(video_id=task_id, path=video_path)

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
                if hard_deadline_seconds - elapsed <= model_call_reserve_seconds:
                    force_fallback_captions = True

                # Per-clip deadline enforcement
                remaining_clip_time = clip_deadline - time.monotonic()
                if remaining_clip_time <= 2.0:
                    force_fallback_captions = True

                processed = pipeline.process(
                    asset,
                    styles=styles,
                    max_frames=task_max_frames,
                    frame_profile=frame_profile,
                    enable_style_retry=task_enable_style_retry,
                    force_fallback_captions=force_fallback_captions,
                    caption_fallback_deadline=min(caption_fallback_deadline, clip_deadline),
                )
                captions = processed.get("captions", {})
                output_results[task_index] = {
                    "task_id": task_id,
                    "captions": {
                        style: captions.get(style, fallback_captions([style])[style])
                        for style in styles
                    },
                }
                write_results_atomic(output_path, output_results)
                timings = processed.get("timings", {})
                task_total = time.monotonic() - task_started_at
                elapsed_total = time.monotonic() - started_at
                print(
                    (
                        f"Task {task_id}: download={download_seconds:.1f}s "
                        f"frames={timings.get('frame_extraction_sec', 0.0):.1f}s "
                        f"evidence={timings.get('evidence_extraction_sec', 0.0):.1f}s "
                        f"captions={timings.get('caption_generation_sec', 0.0):.1f}s "
                        f"task_total={task_total:.1f}s elapsed={elapsed_total:.1f}s "
                        f"max_frames={task_max_frames} retry={task_enable_style_retry}"
                    ),
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"Task {task_id or '[missing task_id]'} failed: {exc}", file=sys.stderr)
                output_results[task_index] = {"task_id": task_id, "captions": fallback_captions(styles)}
                write_results_atomic(output_path, output_results)

        if download_pool is not None:
            download_pool.shutdown(wait=False, cancel_futures=True)

    write_results_atomic(output_path, output_results)
    return 0


def main() -> None:
    input_path = Path(os.getenv("TRACK2_INPUT", "/input/tasks.json"))
    output_path = Path(os.getenv("TRACK2_OUTPUT", "/output/results.json"))
    try:
        exit_code = run_harness(input_path, output_path)
    except Exception as exc:
        print(f"Fatal harness error: {exc}", file=sys.stderr)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_results_atomic(output_path, [])
        except Exception as output_exc:
            print(f"Could not write fallback results to {output_path}: {output_exc}", file=sys.stderr)
        # Return nonzero for genuine fatal failure
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
