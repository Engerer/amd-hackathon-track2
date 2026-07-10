from __future__ import annotations
# noinspection PyUnresolvedReferences  – field used by VideoAsset below

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MIN_VIDEO_DURATION_SECONDS = 30.0
MAX_VIDEO_DURATION_SECONDS = 120.0
DURATION_TOLERANCE_SECONDS = 0.5
ABSOLUTE_MAX_FRAMES = 5
DEFAULT_MAX_FRAMES = 5
DEFAULT_FRAME_PROFILE = "hybrid"
FAST_FRAME_PROFILES = {"fast", "storyboard"}
HYBRID_FRAME_PROFILES = {"hybrid", "accuracy"}


class VideoDurationError(ValueError):
    pass


@dataclass(frozen=True)
class VideoAsset:
    video_id: str
    path: Path
    segments: list[Path] | None = field(default=None)


@dataclass(frozen=True)
class FrameCandidate:
    timestamp: float
    score: float
    sharpness: float
    brightness: float
    motion: float
    hash_value: int


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


def _timestamped_frame_name(index: int, timestamp: float) -> str:
    return f"frame_{index:03d}_t{max(timestamp, 0.0):09.3f}.jpg"


def _extract_anchor_frames(video_path: Path, frame_dir: Path, max_frames: int, width: int) -> list[Path]:
    _reset_dir(frame_dir)
    duration = probe_duration_seconds(video_path)
    timestamps = _sample_timestamps(duration, max_frames)
    if not timestamps:
        return []

    frames: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = frame_dir / _timestamped_frame_name(index, timestamp)
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
    if profile in FAST_FRAME_PROFILES or profile in HYBRID_FRAME_PROFILES:
        return min(5, cap)
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
    sharp_motion_score = motion_score * (0.35 + (sharpness_score * 0.65))
    score = (sharpness_score * 0.50) + (brightness_score * 0.20) + (sharp_motion_score * 0.30)
    return score, sharpness, brightness, motion


def _compute_candidate_count(
    duration: float | None,
    final_count: int,
    frame_profile: str,
) -> int:
    profile = (frame_profile or DEFAULT_FRAME_PROFILE).lower()
    if profile in HYBRID_FRAME_PROFILES:
        duration_candidates = math.ceil(duration * 0.3) if duration and duration > 0 else 0
        return min(42, max(final_count * 2, duration_candidates))
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
    """Select 5 frames with deliberate allocation (beginning, end, middle, scene change, detail) and enforce visual diversity via perceptual hash deduplication."""
    final_count = min(final_count, 5)
    if not candidates:
        return []

    selected: list[FrameCandidate] = []

    def is_duplicate(cand: FrameCandidate) -> bool:
        for s in selected:
            if _hamming_distance(cand.hash_value, s.hash_value) < 8:
                return True
        return False

    # 1. Beginning candidate: closest to 0.05 * duration
    beg_target = 0.05 * duration
    for c in sorted(candidates, key=lambda c: abs(c.timestamp - beg_target)):
        if not is_duplicate(c):
            selected.append(c)
            break

    # 2. End candidate: closest to 0.95 * duration
    end_target = 0.95 * duration
    for c in sorted(candidates, key=lambda c: abs(c.timestamp - end_target)):
        if not is_duplicate(c):
            selected.append(c)
            break

    # 3. Middle candidate: closest to 0.50 * duration
    mid_target = 0.50 * duration
    for c in sorted(candidates, key=lambda c: abs(c.timestamp - mid_target)):
        if not is_duplicate(c):
            selected.append(c)
            break

    # 4. Strongest scene change: highest motion
    for c in sorted(candidates, key=lambda c: c.motion, reverse=True):
        if c not in selected and not is_duplicate(c):
            selected.append(c)
            break

    # 5. Strongest action/detail: highest score
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if c not in selected and not is_duplicate(c):
            selected.append(c)
            break

    # Fallback to timeline anchors if still lacking frames due to strict deduplication
    if len(selected) < final_count:
        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if len(selected) >= final_count:
                break
            if c not in selected:
                selected.append(c)

    return sorted(selected, key=lambda c: c.timestamp)[:final_count]


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
        candidate_count = _compute_candidate_count(duration, final_count, frame_profile)
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
            output_path = frame_dir / _timestamped_frame_name(index, candidate.timestamp)
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
            output_path = frame_dir / _timestamped_frame_name(index, timestamp)
            if cv2.imwrite(str(output_path), frame):
                frames.append(output_path)
        return frames
    finally:
        capture.release()


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    max_frames: int = DEFAULT_MAX_FRAMES,
    width: int = 896,
    frame_profile: str = DEFAULT_FRAME_PROFILE,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration_seconds(video_path)
    effective_max = compute_dynamic_frame_count(duration, max_frames, frame_profile)
    profile = (frame_profile or DEFAULT_FRAME_PROFILE).lower()

    if profile in HYBRID_FRAME_PROFILES:
        try:
            hybrid_frames = _extract_adaptive_frames_opencv(
                video_path,
                frame_dir / "hybrid",
                effective_max,
                width,
                frame_profile,
            )
            if len(hybrid_frames) == effective_max:
                logger.info("Hybrid selection: kept %d frames.", len(hybrid_frames))
                return hybrid_frames
            if hybrid_frames:
                logger.warning(
                    "Hybrid selection returned %d / %d frames; using timeline fallback.",
                    len(hybrid_frames),
                    effective_max,
                )
        except Exception:
            logger.warning("Hybrid OpenCV selection failed; using timeline fallback.", exc_info=True)

    if profile in FAST_FRAME_PROFILES or profile in HYBRID_FRAME_PROFILES:
        try:
            fast_frames = _extract_timestamp_frames_opencv(
                video_path,
                frame_dir / "fast",
                effective_max,
                width,
            )
            if len(fast_frames) == effective_max:
                logger.info("Timeline selection: kept %d frames.", len(fast_frames))
                return fast_frames
            if fast_frames:
                logger.warning(
                    "Timeline selection returned %d / %d frames; using ffmpeg fallback.",
                    len(fast_frames),
                    effective_max,
                )
        except Exception:
            logger.warning("Fast OpenCV frame selection failed; using ffmpeg fallback.", exc_info=True)

        anchor_frames = _extract_anchor_frames(video_path, frame_dir / "anchor", effective_max, width)
        if len(anchor_frames) == effective_max:
            return anchor_frames

    try:
        adaptive_frames = _extract_adaptive_frames_opencv(
            video_path,
            frame_dir / "adaptive",
            effective_max,
            width,
            frame_profile,
        )
        if len(adaptive_frames) == effective_max:
            logger.info("Adaptive selection: kept %d frames.", len(adaptive_frames))
            return adaptive_frames
    except Exception:
        logger.warning("Adaptive OpenCV frame selection failed; using ffmpeg fallback.", exc_info=True)

    scene_frames = _extract_scene_frames(video_path, frame_dir / "scene", effective_max, width)
    minimum_scene_frames = effective_max
    if len(scene_frames) >= minimum_scene_frames:
        return scene_frames

    minimum_frames = effective_max
    anchor_frames = _extract_anchor_frames(video_path, frame_dir / "anchor", effective_max, width)
    if len(anchor_frames) >= minimum_frames:
        return anchor_frames

    uniform_frames = _extract_uniform_frames(video_path, frame_dir / "uniform", effective_max, width)
    if len(uniform_frames) >= effective_max:
        return uniform_frames

    if len(anchor_frames) >= effective_max:
        return anchor_frames

    raise RuntimeError(
        f"Could not extract the required {effective_max} frames from {video_path}"
    )


def split_video_segments(video_path: Path, max_segment_seconds: float = 60.0) -> list[Path]:
    """Split a video into segments of at most *max_segment_seconds* each.

    If the video is already short enough, returns the original path unchanged.
    """
    duration = probe_duration_seconds(video_path)
    if duration is None or duration <= max_segment_seconds:
        return [video_path]

    segment_count = math.ceil(duration / max_segment_seconds)
    output_dir = video_path.parent / f"{video_path.stem}_segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for i in range(segment_count):
        start = i * max_segment_seconds
        output_path = output_dir / f"{video_path.stem}_seg{i:03d}{video_path.suffix}"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{max_segment_seconds:.3f}",
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(output_path),
        ]
        if _run_ffmpeg(command) and output_path.exists():
            segments.append(output_path)
    return segments


def extract_crop_frames(frames: list[Path], output_dir: Path) -> list[Path]:
    """Create centre-cropped versions (50 % width × 50 % height) of each frame."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    crops: list[Path] = []
    for frame_path in frames:
        with Image.open(frame_path) as img:
            w, h = img.size
            crop_w, crop_h = w // 2, h // 2
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            cropped = img.crop((left, top, left + crop_w, top + crop_h))
            out_path = output_dir / f"crop_{frame_path.name}"
            cropped.save(out_path)
            crops.append(out_path)
    return crops


def extract_ocr_frames(
    video_path: Path,
    frame_dir: Path,
    max_frames: int = 3,
    width: int = 1280,
) -> list[Path]:
    """Extract a small number of high-resolution frames for OCR at 1/3, 1/2 and 2/3 of the video."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration_seconds(video_path)
    if duration is None or duration <= 0:
        return []

    positions = [1 / 3, 1 / 2, 2 / 3]
    timestamps = [duration * p for p in positions[:max_frames]]

    frames: list[Path] = []
    for i, ts in enumerate(timestamps):
        output_path = frame_dir / f"ocr_frame_{i:03d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ts:.3f}",
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


def extract_motion_clip_frames(
    video_path: Path,
    frame_dir: Path,
    max_clips: int = 2,
    fps: float = 3.0,
) -> list[Path]:
    """Extract bursts of frames around detected scene changes.

    Uses ffmpeg scene-change detection (``select='gt(scene,0.15)'``) to
    locate transitions, then captures 3 frames at *fps* rate starting
    0.5 s before each transition.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Detect scene-change timestamps
    detect_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "json",
        "-f",
        "lavfi",
        f"movie='{str(video_path)}',select='gt(scene\\,0.15)'",
    ]
    try:
        result = subprocess.run(detect_cmd, check=True, capture_output=True, text=True)
        probe_data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []

    scene_timestamps: list[float] = []
    for frame_info in probe_data.get("frames", []):
        pts = frame_info.get("pts_time")
        if pts is not None:
            scene_timestamps.append(float(pts))
    scene_timestamps = scene_timestamps[:max_clips]
    if not scene_timestamps:
        return []

    all_frames: list[Path] = []
    for clip_idx, scene_ts in enumerate(scene_timestamps):
        burst_start = max(0.0, scene_ts - 0.5)
        for frame_idx in range(3):
            ts = burst_start + frame_idx / fps
            output_path = frame_dir / f"motion_{clip_idx:03d}_{frame_idx:03d}.jpg"
            command = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(output_path),
            ]
            if _run_ffmpeg(command) and output_path.exists():
                all_frames.append(output_path)
    return all_frames
