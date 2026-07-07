from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


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

    try:
        return _transcribe_with_python_whisper(video_path, output_path, model_name, language)
    except ImportError:
        return _transcribe_with_whisper_cli(video_path, transcript_dir, output_path, model_name, language)


def _transcribe_with_python_whisper(
    video_path: Path,
    output_path: Path,
    model_name: str,
    language: str | None,
) -> Path:
    import whisper

    model = whisper.load_model(model_name)
    options = {"fp16": False}
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
