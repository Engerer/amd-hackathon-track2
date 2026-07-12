from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MIN_VIDEO_DURATION_SECONDS = 30.0
MAX_VIDEO_DURATION_SECONDS = 120.0
DURATION_TOLERANCE_SECONDS = 0.5
ABSOLUTE_MAX_FRAMES = 5
DEFAULT_MAX_FRAMES = 5
REPRESENTATIVE_CANDIDATE_COUNT = 25


class VideoDurationError(ValueError):
    pass


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    path: Path


def discover_videos(input_dir: Path) -> list[VideoAsset]:
    if not input_dir.exists():
        return []

    assets: list[VideoAsset] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            assets.append(VideoAsset(video_id=path.stem, path=path))
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

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    duration = payload.get("format", {}).get("duration")
    return float(duration) if duration else None


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    if minutes:
        return f"{minutes}m {remainder:.1f}s"
    return f"{remainder:.1f}s"


def validate_video_duration(
    video_path: Path,
    min_seconds: float = MIN_VIDEO_DURATION_SECONDS,
    max_seconds: float = MAX_VIDEO_DURATION_SECONDS,
    require_probe: bool = False,
) -> float | None:
    duration = probe_duration_seconds(video_path)
    if duration is None:
        if require_probe:
            raise VideoDurationError(f"Could not determine duration for {video_path}.")
        return None

    if duration < min_seconds - DURATION_TOLERANCE_SECONDS:
        raise VideoDurationError(
            f"Video duration {format_duration(duration)} is below the Track 2 minimum "
            f"of {format_duration(min_seconds)}."
        )
    if duration > max_seconds + DURATION_TOLERANCE_SECONDS:
        raise VideoDurationError(
            f"Video duration {format_duration(duration)} exceeds the Track 2 maximum "
            f"of {format_duration(max_seconds)}."
        )
    return duration


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


def _sample_timestamps(duration: float | None, max_frames: int) -> list[float]:
    if not duration or duration <= 0 or max_frames <= 0:
        return []
    if max_frames == 1:
        return [max(duration / 2, 0)]

    start = min(0.5, max(duration * 0.05, 0))
    end = max(duration - 0.5, start)
    if end <= start:
        return [max(duration / 2, 0)]

    timestamps: list[float] = []
    seen: set[float] = set()
    for index in range(max_frames):
        position = index / (max_frames - 1)
        timestamp = round(start + ((end - start) * position), 3)
        if timestamp not in seen:
            timestamps.append(timestamp)
            seen.add(timestamp)
    return timestamps


def _extract_anchor_frames(video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[Path]:
    _reset_dir(frame_dir)
    duration = probe_duration_seconds(video_path)
    timestamps = _sample_timestamps(duration, max_frames)
    if not timestamps:
        return []

    frames: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = frame_dir / f"frame_{index:03d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-q:v",
            "3",
            str(output_path),
        ]
        if _run_ffmpeg(command) and output_path.exists():
            frames.append(output_path)
    return frames


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


def _extract_candidate_frames(
    video_path: Path,
    frame_dir: Path,
    candidate_count: int,
    width: int,
) -> list[Path]:
    """Extract a dense chronological pool for representative-frame selection."""
    return _extract_uniform_frames(video_path, frame_dir, candidate_count, width)


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


def compute_dynamic_frame_count(duration_seconds: float | None, max_frames: int) -> int:
    """Kimi receives exactly five chronological visual samples."""
    return min(max(1, max_frames), ABSOLUTE_MAX_FRAMES)


def _average_hash(image_path: Path, hash_size: int = 8) -> int:
    """Compute a perceptual average hash for an image using Pillow."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    return sum(1 << i for i, px in enumerate(pixels) if px >= mean)


def _hamming_distance(hash_a: int, hash_b: int) -> int:
    return bin(hash_a ^ hash_b).count("1")


def _frame_quality(image_path: Path) -> float:
    """Favor sharp, well-exposed, sufficiently contrasted frames."""
    from PIL import Image, ImageFilter, ImageStat

    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((256, 144), Image.Resampling.LANCZOS)
        statistics = ImageStat.Stat(grayscale)
        mean = statistics.mean[0]
        contrast = statistics.stddev[0]
        edge_variance = ImageStat.Stat(grayscale.filter(ImageFilter.FIND_EDGES)).var[0]

    exposure = max(0.0, 1.0 - (abs(mean - 127.5) / 127.5))
    return math.log1p(edge_variance) + (contrast / 32.0) + exposure


def select_representative_frames(
    candidate_frames: list[Path],
    frame_count: int = DEFAULT_MAX_FRAMES,
    diversity_threshold: int = 6,
) -> list[Path]:
    """Choose one high-quality, diverse frame from each temporal bucket."""
    if frame_count <= 0 or len(candidate_frames) < frame_count:
        return []

    try:
        quality = {path: _frame_quality(path) for path in candidate_frames}
        hashes = {path: _average_hash(path) for path in candidate_frames}
    except Exception:
        logger.debug("Representative-frame scoring failed.", exc_info=True)
        return []

    selected: list[Path] = []
    total = len(candidate_frames)
    for bucket_index in range(frame_count):
        start = (bucket_index * total) // frame_count
        end = ((bucket_index + 1) * total) // frame_count
        bucket = candidate_frames[start:end]
        ranked = sorted(bucket, key=lambda path: quality[path], reverse=True)
        diverse = [
            path
            for path in ranked
            if all(
                _hamming_distance(hashes[path], hashes[chosen]) >= diversity_threshold
                for chosen in selected
            )
        ]
        selected.append((diverse or ranked)[0])

    return sorted(selected, key=candidate_frames.index)


def _extract_representative_frames(
    video_path: Path,
    frame_dir: Path,
    frame_count: int,
    width: int,
) -> list[Path]:
    candidate_count = max(REPRESENTATIVE_CANDIDATE_COUNT, frame_count * 5)
    candidates = _extract_candidate_frames(
        video_path,
        frame_dir / "candidates",
        candidate_count,
        width,
    )
    if len(candidates) < frame_count:
        return []

    selected = select_representative_frames(candidates, frame_count)
    if len(selected) != frame_count:
        return []

    output_dir = frame_dir / "selected"
    _reset_dir(output_dir)
    output: list[Path] = []
    for index, source in enumerate(selected, start=1):
        destination = output_dir / f"frame_{index:03d}.jpg"
        shutil.copyfile(source, destination)
        output.append(destination)
    return output


def deduplicate_frames(frame_paths: list[Path], threshold: int = 6) -> list[Path]:
    """Drop near-duplicate frames using perceptual hashing (average hash).

    Frames whose hamming distance to the previously kept frame is below
    *threshold* are considered duplicates and removed.  The first and last
    frames are always kept to preserve temporal coverage.
    """
    if len(frame_paths) <= 2:
        return list(frame_paths)

    try:
        hashes = [(path, _average_hash(path)) for path in frame_paths]
    except Exception:
        logger.debug("Perceptual hashing failed; skipping deduplication.", exc_info=True)
        return list(frame_paths)

    kept: list[Path] = [hashes[0][0]]
    last_hash = hashes[0][1]

    for path, h in hashes[1:-1]:
        if _hamming_distance(last_hash, h) >= threshold:
            kept.append(path)
            last_hash = h

    # Always keep the last frame
    kept.append(hashes[-1][0])
    if len(kept) < len(frame_paths):
        logger.info("Deduplication: kept %d / %d frames.", len(kept), len(frame_paths))
    return kept


def extract_frames(video_path: Path, frame_dir: Path, max_frames: int = DEFAULT_MAX_FRAMES, width: int = 896) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic scaling: adapt frame budget to video duration
    duration = probe_duration_seconds(video_path)
    effective_max = compute_dynamic_frame_count(duration, max_frames)

    representative_frames = _extract_representative_frames(
        video_path,
        frame_dir / "representative",
        effective_max,
        width,
    )
    if len(representative_frames) == effective_max:
        return representative_frames

    anchor_frames = _extract_anchor_frames(video_path, frame_dir / "anchor", effective_max, width)
    if len(anchor_frames) == effective_max:
        return anchor_frames

    scene_frames = _extract_scene_frames(video_path, frame_dir / "scene", effective_max, width)
    if len(scene_frames) == effective_max:
        return scene_frames

    uniform_frames = _extract_uniform_frames(video_path, frame_dir / "uniform", effective_max, width)
    if len(uniform_frames) == effective_max:
        return uniform_frames

    raise RuntimeError(f"Could not extract exactly {effective_max} frames from {video_path}")
