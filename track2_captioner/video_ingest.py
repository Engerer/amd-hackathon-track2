from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MIN_VIDEO_DURATION_SECONDS = 30.0
MAX_VIDEO_DURATION_SECONDS = 240.0
DURATION_TOLERANCE_SECONDS = 0.5
ABSOLUTE_MAX_FRAMES = 20
DEFAULT_MAX_FRAMES = 12
DEFAULT_FRAME_PROFILE = "fast"
FAST_FRAME_PROFILES = {"fast", "storyboard"}


class VideoDurationError(ValueError):
    pass


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    path: Path
    transcript_path: Path | None


@dataclass(frozen=True)
class FrameCandidate:
    timestamp: float
    score: float
    sharpness: float
    brightness: float
    motion: float
    hash_value: int


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


def compute_dynamic_frame_count(
    duration_seconds: float | None,
    max_frames: int,
    frame_profile: str = DEFAULT_FRAME_PROFILE,
) -> int:
    """Scale frame count by video duration while preserving enough evidence for judging."""
    cap = min(max_frames, ABSOLUTE_MAX_FRAMES)
    if duration_seconds is None or duration_seconds <= 0:
        return cap
    profile = (frame_profile or DEFAULT_FRAME_PROFILE).lower()
    if profile in FAST_FRAME_PROFILES:
        if duration_seconds <= 60:
            return min(8, cap)
        return min(12, cap)
    if profile != "balanced":
        return cap
    if duration_seconds <= 45:
        return min(14, cap)
    if duration_seconds <= 75:
        return min(16, cap)
    return min(18, cap)


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


def _normalize_score(value: float, scale: float) -> float:
    if value <= 0:
        return 0.0
    return min(value / scale, 1.0)


def _candidate_hash(gray: Any) -> int:
    import cv2

    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = small >= mean
    value = 0
    for index, enabled in enumerate(bits.flatten()):
        if bool(enabled):
            value |= 1 << index
    return value


def _candidate_score(gray: Any, previous_gray: Any | None) -> tuple[float, float, float, float]:
    import cv2

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if previous_gray is None:
        motion = 0.0
    else:
        motion = float(cv2.absdiff(gray, previous_gray).mean())

    brightness_score = max(0.0, 1.0 - (abs(brightness - 127.0) / 127.0))
    sharpness_score = _normalize_score(sharpness, 500.0)
    motion_score = _normalize_score(motion, 45.0)
    score = (sharpness_score * 0.35) + (brightness_score * 0.25) + (motion_score * 0.40)
    return score, sharpness, brightness, motion


def _compute_candidate_count(duration: float | None, final_count: int) -> int:
    if duration is None or duration <= 0:
        return min(72, max(final_count * 4, 48))
    return min(72, max(final_count * 4, math.ceil(duration * 0.5)))


def _closest_candidate(candidates: list[FrameCandidate], timestamp: float) -> FrameCandidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: abs(candidate.timestamp - timestamp))


def _add_candidate_once(selected: dict[float, FrameCandidate], candidate: FrameCandidate | None) -> None:
    if candidate is not None:
        selected[candidate.timestamp] = candidate


def _select_adaptive_candidates(
    candidates: list[FrameCandidate],
    duration: float,
    final_count: int,
) -> list[FrameCandidate]:
    selected: dict[float, FrameCandidate] = {}
    _add_candidate_once(selected, _closest_candidate(candidates, min(0.5, duration * 0.05)))
    _add_candidate_once(selected, _closest_candidate(candidates, duration / 2))
    _add_candidate_once(selected, _closest_candidate(candidates, max(duration - 0.5, 0)))

    segment_count = min(final_count, 6 if duration > 75 else 5)
    for index in range(segment_count):
        start = duration * (index / segment_count)
        end = duration * ((index + 1) / segment_count)
        segment = [
            candidate for candidate in candidates
            if start <= candidate.timestamp <= end
        ]
        if segment:
            _add_candidate_once(selected, max(segment, key=lambda candidate: candidate.score))

    min_gap = max(duration / max(final_count * 2.5, 1), 0.75)
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= final_count:
            break
        too_close = any(abs(candidate.timestamp - kept.timestamp) < min_gap for kept in selected.values())
        too_similar = any(_hamming_distance(candidate.hash_value, kept.hash_value) < 5 for kept in selected.values())
        if not too_close and not too_similar:
            selected[candidate.timestamp] = candidate

    if len(selected) < final_count:
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if len(selected) >= final_count:
                break
            selected.setdefault(candidate.timestamp, candidate)

    return sorted(selected.values(), key=lambda candidate: candidate.timestamp)[:final_count]


def _extract_adaptive_frames_opencv(
    video_path: Path,
    frame_dir: Path,
    max_frames: int,
    width: int,
    frame_profile: str,
) -> list[Path]:
    import cv2

    _reset_dir(frame_dir)
    duration = probe_duration_seconds(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    try:
        if duration is None or duration <= 0:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
        if duration is None or duration <= 0:
            return []

        final_count = compute_dynamic_frame_count(duration, max_frames, frame_profile)
        candidate_count = _compute_candidate_count(duration, final_count)
        timestamps = _sample_timestamps(duration, candidate_count)
        if not timestamps:
            return []

        candidates: list[FrameCandidate] = []
        previous_gray: Any | None = None
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            resized = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            score, sharpness, brightness, motion = _candidate_score(gray, previous_gray)
            candidates.append(
                FrameCandidate(
                    timestamp=timestamp,
                    score=score,
                    sharpness=sharpness,
                    brightness=brightness,
                    motion=motion,
                    hash_value=_candidate_hash(gray),
                )
            )
            previous_gray = gray

        selected = _select_adaptive_candidates(candidates, duration, final_count)
        if not selected:
            return []

        frames: list[Path] = []
        for index, candidate in enumerate(selected, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, candidate.timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, original_width = frame.shape[:2]
            if original_width > 0:
                target_height = max(1, round(height * (width / original_width)))
                frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
            output_path = frame_dir / f"frame_{index:03d}.jpg"
            if cv2.imwrite(str(output_path), frame):
                frames.append(output_path)
        return frames
    finally:
        capture.release()


def _extract_timestamp_frames_opencv(
    video_path: Path,
    frame_dir: Path,
    max_frames: int,
    width: int,
) -> list[Path]:
    import cv2

    _reset_dir(frame_dir)
    duration = probe_duration_seconds(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    try:
        if duration is None or duration <= 0:
            fps = capture.get(cv2.CAP_PROP_FPS) or 0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
        timestamps = _sample_timestamps(duration, max_frames)
        if not timestamps:
            return []

        frames: list[Path] = []
        for index, timestamp in enumerate(timestamps, start=1):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, original_width = frame.shape[:2]
            if original_width > 0:
                target_height = max(1, round(height * (width / original_width)))
                frame = cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)
            output_path = frame_dir / f"frame_{index:03d}.jpg"
            if cv2.imwrite(str(output_path), frame):
                frames.append(output_path)
        return frames
    finally:
        capture.release()


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    width: int = 768,
    frame_profile: str = DEFAULT_FRAME_PROFILE,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration_seconds(video_path)
    effective_max = compute_dynamic_frame_count(duration, max_frames, frame_profile)
    profile = (frame_profile or DEFAULT_FRAME_PROFILE).lower()

    if profile in FAST_FRAME_PROFILES:
        try:
            fast_frames = _extract_timestamp_frames_opencv(
                video_path,
                frame_dir / "fast",
                effective_max,
                width,
            )
            if fast_frames:
                logger.info("Fast timestamp selection: kept %d frames.", len(fast_frames))
                return deduplicate_frames(fast_frames)
        except Exception:
            logger.warning("Fast OpenCV frame selection failed; using ffmpeg fallback.", exc_info=True)

        anchor_frames = _extract_anchor_frames(video_path, frame_dir / "anchor", effective_max, width)
        if anchor_frames:
            return deduplicate_frames(anchor_frames)

    try:
        adaptive_frames = _extract_adaptive_frames_opencv(
            video_path,
            frame_dir / "adaptive",
            effective_max,
            width,
            frame_profile,
        )
        if adaptive_frames:
            logger.info("Adaptive selection: kept %d frames.", len(adaptive_frames))
            return adaptive_frames
    except Exception:
        logger.warning("Adaptive OpenCV frame selection failed; using ffmpeg fallback.", exc_info=True)

    scene_frames = _extract_scene_frames(video_path, frame_dir / "scene", effective_max, width)
    minimum_scene_frames = max(3, min(effective_max, effective_max // 2))
    if len(scene_frames) >= minimum_scene_frames:
        return deduplicate_frames(scene_frames)

    minimum_frames = max(1, min(effective_max, 3))
    anchor_frames = _extract_anchor_frames(video_path, frame_dir / "anchor", effective_max, width)
    if len(anchor_frames) >= minimum_frames:
        return deduplicate_frames(anchor_frames)

    uniform_frames = _extract_uniform_frames(video_path, frame_dir / "uniform", effective_max, width)
    if uniform_frames:
        return deduplicate_frames(uniform_frames)

    if anchor_frames:
        return deduplicate_frames(anchor_frames)

    if not uniform_frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
