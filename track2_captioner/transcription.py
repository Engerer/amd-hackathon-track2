from __future__ import annotations

import functools
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Shared thread pool for async transcription (1 worker — Whisper is GPU/CPU-bound)
_TRANSCRIPTION_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


def transcribe_video(
    video_path: Path,
    transcript_dir: Path,
    model_name: str = "base",
    language: str | None = None,
    force: bool = False,
) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    output_path = transcript_dir / f"{video_path.stem}.txt"

    if output_path.exists() and not force:
        return output_path

    if not has_audio_stream(video_path):
        output_path.write_text("", encoding="utf-8")
        return output_path

    try:
        return _transcribe_with_python_whisper(video_path, output_path, model_name, language)
    except ImportError:
        return _transcribe_with_whisper_cli(video_path, transcript_dir, output_path, model_name, language)


def has_audio_stream(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def transcribe_video_async(
    video_path: Path,
    transcript_dir: Path,
    model_name: str = "base",
    language: str | None = None,
    force: bool = False,
) -> Future[Path]:
    """Submit transcription to a background thread and return a Future.

    Use this from the Streamlit app to keep the UI responsive while Whisper
    runs.  Call ``future.result()`` to block until the transcript is ready.
    """
    return _TRANSCRIPTION_POOL.submit(
        transcribe_video,
        video_path,
        transcript_dir,
        model_name,
        language,
        force,
    )


@functools.lru_cache(maxsize=4)
def _load_whisper_model(model_name: str):  # type: ignore[no-untyped-def]
    """Load and cache a Whisper model so it is reused across videos in a batch."""
    import whisper

    return whisper.load_model(model_name)


def _transcribe_with_python_whisper(
    video_path: Path,
    output_path: Path,
    model_name: str,
    language: str | None,
) -> Path:
    model = _load_whisper_model(model_name)
    options: dict[str, object] = {"fp16": False}
    if language:
        options["language"] = language
    result = model.transcribe(str(video_path), **options)
    text = str(result.get("text", "")).strip()
    output_path.write_text(text + "\n", encoding="utf-8")
    return output_path


def _transcribe_with_whisper_cli(
    video_path: Path,
    transcript_dir: Path,
    output_path: Path,
    model_name: str,
    language: str | None,
) -> Path:
    whisper_command = shutil.which("whisper")
    if not whisper_command:
        raise RuntimeError(
            "Whisper is not installed. Install it with: "
            "pip install -r requirements-whisper.txt"
        )

    command = [
        whisper_command,
        str(video_path),
        "--model",
        model_name,
        "--output_format",
        "txt",
        "--output_dir",
        str(transcript_dir),
    ]
    if language:
        command.extend(["--language", language])

    subprocess.run(command, check=True, capture_output=True, text=True)
    if not output_path.exists():
        raise RuntimeError(f"Whisper finished but did not create {output_path}")
    return output_path
