from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from track2_captioner.caption_pipeline import CaptionPipeline, FALLBACK_CAPTIONS
from track2_captioner.config import load_settings
from track2_captioner.video_ingest import DEFAULT_MAX_FRAMES, VIDEO_EXTENSIONS, VideoAsset


DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def read_tasks(input_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("/input/tasks.json must contain a JSON array.")
    return payload


def download_video(
    video_url: str,
    destination_dir: Path,
    task_id: str,
    deadline: float | None = None,
) -> Path:
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

    timeout = 120.0
    if deadline is not None:
        timeout = max(0.1, min(timeout, deadline - time.monotonic() - 30.0))
    with requests.get(video_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if deadline is not None and deadline - time.monotonic() < 30.0:
                    raise TimeoutError("Video download stopped to protect the output-write reserve.")
                if chunk:
                    handle.write(chunk)
    return destination


def fallback_captions(styles: list[str]) -> dict[str, str]:
    return {
        style: FALLBACK_CAPTIONS.get(style, FALLBACK_CAPTIONS["formal"])
        for style in styles
    }


def run_harness(input_path: Path, output_path: Path) -> int:
    settings = load_settings()
    deadline = time.monotonic() + settings.runtime_budget_seconds
    dry_run = truthy(os.getenv("TRACK2_DRY_RUN"))
    run_checks = truthy(os.getenv("RUN_CHECKS"))
    max_frames = int(os.getenv("TRACK2_MAX_FRAMES", str(DEFAULT_MAX_FRAMES)))

    tasks = read_tasks(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(tasks)

    with tempfile.TemporaryDirectory(prefix="track2_harness_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_dir = temp_dir / "videos"
        pipeline = None
        if tasks and deadline - time.monotonic() >= 30.0:
            pipeline = CaptionPipeline(
                settings=settings,
                work_dir=temp_dir / "frames",
                dry_run=dry_run,
                max_frames=max_frames,
                run_checks=run_checks,
            )

        def process_task(task: dict[str, Any]) -> dict[str, Any]:
            task_id = ""
            styles = list(DEFAULT_STYLES)
            try:
                task_id = str(task.get("task_id", ""))
                video_url = str(task.get("video_url", ""))
                styles = [str(style) for style in (task.get("styles") or DEFAULT_STYLES)]

                if deadline - time.monotonic() < 30.0:
                    return {"task_id": task_id, "captions": fallback_captions(styles)}

                assert pipeline is not None
                video_path = download_video(video_url, video_dir, task_id, deadline=deadline)
                asset = VideoAsset(video_id=task_id, path=video_path)

                processed = pipeline.process(asset, styles=styles, deadline=deadline)
                captions = processed.get("captions", {})
                return {
                    "task_id": task_id,
                    "captions": {
                        style: captions.get(style, fallback_captions([style])[style])
                        for style in styles
                    },
                }
            except Exception as exc:
                print(f"Task {task_id or '[missing task_id]'} failed: {exc}", file=sys.stderr)
                return {"task_id": task_id, "captions": fallback_captions(styles)}

        workers = min(settings.task_workers, max(1, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="track2-video") as pool:
            futures = {
                pool.submit(process_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()

    completed_results = [result for result in results if result is not None]
    output_path.write_text(json.dumps(completed_results, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    input_path = Path(os.getenv("TRACK2_INPUT", "/input/tasks.json"))
    output_path = Path(os.getenv("TRACK2_OUTPUT", "/output/results.json"))
    raise SystemExit(run_harness(input_path, output_path))


if __name__ == "__main__":
    main()
