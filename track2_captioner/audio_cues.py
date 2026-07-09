from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from track2_captioner.transcription import has_audio_stream


@dataclass(frozen=True)
class AudioCues:
    has_audio: bool
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    sampled_seconds: float = 0.0

    @property
    def likely_quiet(self) -> bool:
        if not self.has_audio:
            return True
        return self.mean_volume_db is not None and self.mean_volume_db < -42.0


def analyze_audio_cues(
    video_path: Path,
    sample_seconds: float = 20.0,
    timeout_seconds: float = 18.0,
) -> AudioCues:
    if not has_audio_stream(video_path):
        return AudioCues(has_audio=False)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-t",
        f"{max(sample_seconds, 1.0):.1f}",
        "-i",
        str(video_path),
        "-vn",
        "-sn",
        "-dn",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return AudioCues(has_audio=True, sampled_seconds=sample_seconds)

    output = f"{result.stdout}\n{result.stderr}"
    mean_volume = _parse_db(output, "mean_volume")
    max_volume = _parse_db(output, "max_volume")
    return AudioCues(
        has_audio=True,
        mean_volume_db=mean_volume,
        max_volume_db=max_volume,
        sampled_seconds=sample_seconds,
    )


def format_audio_context(cues: AudioCues, transcript: str = "") -> str:
    parts: list[str] = []
    if not cues.has_audio:
        parts.append("No audio stream was detected.")
    elif cues.likely_quiet:
        parts.append("An audio stream is present, but the sampled audio is very quiet.")
    else:
        parts.append("An audio stream is present.")

    if cues.mean_volume_db is not None:
        parts.append(f"Sampled mean volume is about {cues.mean_volume_db:.1f} dB.")
    if cues.max_volume_db is not None:
        parts.append(f"Sampled peak volume is about {cues.max_volume_db:.1f} dB.")

    transcript = transcript.strip()
    if transcript:
        parts.append(f"Speech transcript: {transcript}")
    return " ".join(parts)


def _parse_db(output: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*(-?\d+(?:\.\d+)?)\s*dB", output)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
