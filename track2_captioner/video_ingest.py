from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_FRAME_INTERVAL_SECONDS = 3.0
DEFAULT_MAX_FRAMES = 40
DEFAULT_FRAME_WIDTH = 768


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    path: Path
    transcript_path: Path | None


def discover_videos(input_dir: Path, transcript_dir: Path | None = None) -> list[VideoAsset]:
    if not input_dir.exists():
        return []

    transcript_dir = transcript_dir or input_dir.parent / "transcripts"
    assets: list[VideoAsset] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            transcript_path = transcript_dir / f"{path.stem}.txt"
            assets.append(
                VideoAsset(
                    video_id=path.stem,
                    path=path,
                    transcript_path=transcript_path if transcript_path.exists() else None,
                )
            )
    return assets


def probe_duration_seconds(video_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    return float(duration) if duration else None


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_ffmpeg(command: list[str]) -> bool:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _extract_timeline_frames(
    video_path: Path,
    frame_dir: Path,
    frame_interval_seconds: float,
    max_frames: int,
    width: int,
) -> list[Path]:
    _reset_dir(frame_dir)
    fps = 1 / max(frame_interval_seconds, 0.1)

    output_pattern = frame_dir / "frame_%03d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={width}:-1",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    if not _run_ffmpeg(command):
        return []
    return sorted(frame_dir.glob("frame_*.jpg"))


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    width: int = DEFAULT_FRAME_WIDTH,
    frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    frames = _extract_timeline_frames(
        video_path=video_path,
        frame_dir=frame_dir / "timeline",
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
        width=width,
    )
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return frames
