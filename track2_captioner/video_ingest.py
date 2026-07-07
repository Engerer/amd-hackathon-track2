from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


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


def _extract_uniform_frames(video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[Path]:
    _reset_dir(frame_dir)
    duration = probe_duration_seconds(video_path)

    if duration and duration > 0:
        fps = min(max_frames / duration, 1.0)
    else:
        fps = 0.2

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


def _extract_scene_frames(video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[Path]:
    _reset_dir(frame_dir)
    output_pattern = frame_dir / "frame_%03d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,0.12)',scale={width}:-1",
        "-fps_mode",
        "vfr",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    if not _run_ffmpeg(command):
        return []
    return sorted(frame_dir.glob("frame_*.jpg"))


def extract_frames(video_path: Path, frame_dir: Path, max_frames: int = 10, width: int = 768) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    scene_frames = _extract_scene_frames(video_path, frame_dir / "scene", max_frames, width)
    minimum_scene_frames = max(3, min(max_frames, max_frames // 2))
    if len(scene_frames) >= minimum_scene_frames:
        return scene_frames

    uniform_frames = _extract_uniform_frames(video_path, frame_dir / "uniform", max_frames, width)
    if not uniform_frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return uniform_frames
