from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import streamlit as st

from track2_captioner.caption_pipeline import CaptionPipeline
from track2_captioner.config import Settings, load_settings
from track2_captioner.transcription import WHISPER_MODELS, transcribe_video
from track2_captioner.video_ingest import VIDEO_EXTENSIONS, VideoAsset, discover_videos, probe_duration_seconds


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DEFAULT_VIDEO_DIR = DATA_DIR / "videos"
DEFAULT_TRANSCRIPT_DIR = DATA_DIR / "transcripts"
OUTPUT_DIR = APP_ROOT / "outputs"
WEB_RUN_DIR = OUTPUT_DIR / "web_runs"
LATEST_OUTPUT = OUTPUT_DIR / "latest_web_results.json"

STYLE_LABELS = {
    "formal": "Formal",
    "sarcastic": "Sarcastic",
    "humorous_tech": "Humorous-tech",
    "humorous_non_tech": "Humorous non-tech",
}
DEFAULT_FRAME_COUNT = 10
FRAME_OPTIONS = [5, 8, 10, 12, 16]
MIN_VIDEO_SECONDS = 30
MAX_VIDEO_SECONDS = 120


def compact_model_name(model: str) -> str:
    return model.rsplit("/", 1)[-1] if model else "Not set"


def safe_filename(name: str) -> str:
    cleaned = Path(name).name
    return re.sub(r"[^A-Za-z0-9._ -]", "_", cleaned).strip() or "uploaded_file"


def ensure_dirs() -> None:
    for path in [DEFAULT_VIDEO_DIR, DEFAULT_TRANSCRIPT_DIR, OUTPUT_DIR, WEB_RUN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_uploaded_files(files: list[Any], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for uploaded_file in files:
        target = destination / safe_filename(uploaded_file.name)
        target.write_bytes(uploaded_file.getbuffer())
        saved.append(target)
    return saved


def assets_from_uploads(video_files: list[Any], transcript_files: list[Any]) -> tuple[list[VideoAsset], Path]:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = WEB_RUN_DIR / run_id
    video_dir = run_dir / "videos"
    transcript_dir = run_dir / "transcripts"

    saved_videos = [
        path for path in write_uploaded_files(video_files, video_dir)
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    write_uploaded_files(transcript_files, transcript_dir)

    assets = []
    for path in sorted(saved_videos):
        transcript_path = transcript_dir / f"{path.stem}.txt"
        assets.append(
            VideoAsset(
                video_id=path.stem,
                path=path,
                transcript_path=transcript_path if transcript_path.exists() else None,
            )
        )
    return assets, run_dir


def maybe_transcribe_asset(
    asset: VideoAsset,
    transcript_dir: Path,
    model_name: str,
    language: str,
    force: bool,
) -> VideoAsset:
    transcript_path = transcribe_video(
        video_path=asset.path,
        transcript_dir=transcript_dir,
        model_name=model_name,
        language=language.strip() or None,
        force=force,
    )
    return VideoAsset(
        video_id=asset.video_id,
        path=asset.path,
        transcript_path=transcript_path,
    )


def caption_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": item.get("video_id", ""),
            "captions": item.get("captions", {}),
        }
        for item in results
        if "error" not in item
    ]


def count_passes(results: list[dict[str, Any]]) -> tuple[int, int]:
    passed = 0
    total = 0
    for item in results:
        checks = item.get("checks", {})
        for check in checks.values():
            total += 2
            if check.get("accuracy") == "pass":
                passed += 1
            if check.get("tone") == "pass":
                passed += 1
    return passed, total


def format_duration(seconds: float) -> str:
    minutes, remaining = divmod(round(seconds), 60)
    return f"{minutes}:{remaining:02d}"


def split_assets_by_duration(assets: list[VideoAsset]) -> tuple[list[VideoAsset], list[tuple[VideoAsset, float, str]]]:
    accepted: list[VideoAsset] = []
    skipped: list[tuple[VideoAsset, float, str]] = []

    for asset in assets:
        duration = probe_duration_seconds(asset.path)
        if duration is not None and duration < MIN_VIDEO_SECONDS:
            skipped.append((asset, duration, "too short"))
        elif duration is not None and duration > MAX_VIDEO_SECONDS:
            skipped.append((asset, duration, "too long"))
        else:
            accepted.append(asset)

    return accepted, skipped


def render_storyboard(frame_paths: list[str]) -> None:
    frames = [Path(path) for path in frame_paths if Path(path).is_file()]
    if not frames:
        return

    st.caption(f"Storyboard: {len(frames)} sampled frames")
    columns = st.columns(min(5, len(frames)))
    for index, frame in enumerate(frames):
        with columns[index % len(columns)]:
            st.image(str(frame), caption=f"Frame {index + 1}", use_container_width=True)


def render_result(item: dict[str, Any]) -> None:
    video_id = item.get("video_id", "video")
    with st.expander(video_id, expanded=True):
        if "error" in item:
            st.error(item["error"])
            return

        left, right = st.columns([0.85, 1.35], gap="large")
        with left:
            source_text = item.get("source_path", "")
            source = Path(source_text) if source_text else None
            if source and source.is_file():
                st.video(str(source))
            render_storyboard(item.get("frames", []))
            with st.expander("Observations", expanded=False):
                frame_meta = {
                    "frame_count": item.get("frame_count", 0),
                    "sampling_strategy": item.get("sampling_strategy", "unknown"),
                }
                st.caption(json.dumps(frame_meta))
                st.json(item.get("observations", {}), expanded=False)

        with right:
            captions = item.get("captions", {})
            checks = item.get("checks", {})
            tabs = st.tabs([STYLE_LABELS[key] for key in STYLE_LABELS])
            for tab, key in zip(tabs, STYLE_LABELS):
                with tab:
                    st.write(captions.get(key, ""))
                    check = checks.get(key, {})
                    if check:
                        status_cols = st.columns(2)
                        status_cols[0].metric("Accuracy", check.get("accuracy", "n/a"))
                        status_cols[1].metric("Tone", check.get("tone", "n/a"))
                        if check.get("notes"):
                            st.caption(check["notes"])


def main() -> None:
    st.set_page_config(
        page_title="Track 2 Caption Studio",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    ensure_dirs()

    defaults = load_settings()
    if "results" not in st.session_state:
        st.session_state.results = []

    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 12px 14px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    default_backend = "Proxy" if defaults.proxy_url else "Direct Fireworks"
    frame_options = sorted(set(FRAME_OPTIONS + [DEFAULT_FRAME_COUNT]))
    source_mode = "Upload"
    dry_run = not bool(defaults.api_key or defaults.proxy_url)
    max_frames = DEFAULT_FRAME_COUNT
    run_checks = False
    auto_transcribe = False
    whisper_model = WHISPER_MODELS[1] if len(WHISPER_MODELS) > 1 else WHISPER_MODELS[0]
    whisper_language = ""
    force_transcribe = False
    backend = default_backend
    api_key = os.getenv("FIREWORKS_API_KEY", "")
    proxy_url = defaults.proxy_url
    proxy_token = defaults.proxy_token
    model = defaults.model
    judge_model = defaults.judge_model

    st.title("Track 2 Caption Studio")
    st.caption(f"{compact_model_name(model)} | smart {DEFAULT_FRAME_COUNT}-frame sampling | checks off")

    with st.sidebar:
        st.header("Preset")
        with st.expander("Advanced", expanded=False):
            source_mode = st.radio("Source", ["Upload", "data/videos"], horizontal=True)
            max_frames = st.select_slider("Frame budget", options=frame_options, value=DEFAULT_FRAME_COUNT)
            dry_run = st.toggle("Dry run", value=dry_run)
            run_checks = st.toggle("Quality checks", value=False)

            auto_transcribe = st.toggle("Whisper", value=False)
            if auto_transcribe:
                whisper_model = st.selectbox("Whisper model", WHISPER_MODELS, index=1)
                whisper_language = st.text_input("Language", value="", placeholder="optional, e.g. en")
                force_transcribe = st.toggle("Overwrite transcripts", value=False)

            backend = st.radio(
                "Backend",
                ["Proxy", "Direct Fireworks"],
                index=0 if default_backend == "Proxy" else 1,
                disabled=dry_run,
            )
            if backend == "Proxy":
                proxy_url = st.text_input("Proxy URL", value=defaults.proxy_url, disabled=dry_run)
                proxy_token = st.text_input("Proxy token", value=defaults.proxy_token, type="password", disabled=dry_run)
            else:
                api_key = st.text_input("Fireworks API key", value=api_key, type="password", disabled=dry_run)

            model = st.text_input("Caption model", value=defaults.model, disabled=dry_run)
            if run_checks:
                judge_model = st.text_input("Judge model", value=defaults.judge_model, disabled=dry_run)

        st.metric("Model", compact_model_name(model))
        st.metric("Frames", max_frames)
        st.metric("Length", f"{format_duration(MIN_VIDEO_SECONDS)}-{format_duration(MAX_VIDEO_SECONDS)}")
        st.metric("Sampling", "Smart")
        st.metric("Checks", "On" if run_checks else "Off")
        st.caption("Proxy connected" if proxy_url and backend == "Proxy" else "Direct API" if backend == "Direct Fireworks" else "Dry run")

    input_col, run_col = st.columns([1.35, 0.65], gap="large")
    with input_col:
        st.subheader("Videos")
        if source_mode == "Upload":
            video_files = st.file_uploader(
                "Upload videos",
                type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)],
                accept_multiple_files=True,
            )
            video_files = list(video_files or [])
            with st.expander("Transcripts", expanded=False):
                transcript_files = st.file_uploader(
                    "Upload matching .txt files",
                    type=["txt"],
                    accept_multiple_files=True,
                )
                transcript_files = list(transcript_files or [])
            existing_assets: list[VideoAsset] = []
        else:
            video_files = []
            transcript_files = []
            existing_assets = discover_videos(DEFAULT_VIDEO_DIR.resolve(), DEFAULT_TRANSCRIPT_DIR.resolve())
            st.write([asset.path.name for asset in existing_assets] or "No files in data/videos.")

    with run_col:
        st.subheader("Run")
        results = st.session_state.results
        passed, total = count_passes(results)
        queued_count = len(video_files) if source_mode == "Upload" else len(existing_assets)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Queued", queued_count)
        metric_cols[1].metric("Results", len(results))
        metric_cols[2].metric("Checks", f"{passed}/{total}" if total else "0/0")

        run_clicked = st.button("Generate captions", type="primary", use_container_width=True)
        clear_clicked = st.button("Clear results", use_container_width=True)

    if clear_clicked:
        st.session_state.results = []
        st.rerun()

    if run_clicked:
        if source_mode == "Upload":
            if not video_files:
                st.warning("Add at least one video.")
                return
            assets, run_dir = assets_from_uploads(video_files, transcript_files)
            transcript_dir = run_dir / "transcripts"
            work_dir = run_dir / "frames"
        else:
            assets = existing_assets
            transcript_dir = DEFAULT_TRANSCRIPT_DIR
            work_dir = DATA_DIR / "frames"

        if not assets:
            st.warning("No supported video files found.")
            return

        assets, skipped_assets = split_assets_by_duration(assets)
        if skipped_assets:
            skipped = ", ".join(
                f"{asset.path.name} ({format_duration(duration)}, {reason})"
                for asset, duration, reason in skipped_assets
            )
            st.warning(
                f"Skipped videos outside {format_duration(MIN_VIDEO_SECONDS)}-"
                f"{format_duration(MAX_VIDEO_SECONDS)}: {skipped}"
            )

        if not assets:
            st.warning(
                f"No videos between {format_duration(MIN_VIDEO_SECONDS)} and "
                f"{format_duration(MAX_VIDEO_SECONDS)} to process."
            )
            return

        if not dry_run and backend == "Direct Fireworks" and not api_key:
            st.error("Fireworks API key is required.")
            return

        if not dry_run and backend == "Proxy" and not proxy_url:
            st.error("Proxy URL is required.")
            return

        settings = Settings(
            api_key=api_key if backend == "Direct Fireworks" else "",
            model=model,
            judge_model=judge_model,
            base_url=defaults.base_url,
            proxy_url=proxy_url if backend == "Proxy" else "",
            proxy_token=proxy_token if backend == "Proxy" else "",
        )
        pipeline = CaptionPipeline(
            settings=settings,
            work_dir=work_dir,
            dry_run=dry_run,
            max_frames=max_frames,
            run_checks=run_checks,
        )

        progress = st.progress(0)
        current = st.empty()
        results = []
        for index, asset in enumerate(assets, start=1):
            current.write(f"Processing {asset.video_id}")
            try:
                if auto_transcribe and (force_transcribe or asset.transcript_path is None):
                    current.write(f"Transcribing {asset.video_id} with Whisper")
                    asset = maybe_transcribe_asset(
                        asset=asset,
                        transcript_dir=transcript_dir,
                        model_name=whisper_model,
                        language=whisper_language,
                        force=force_transcribe,
                    )
                    current.write(f"Processing {asset.video_id}")
                results.append(pipeline.process(asset))
            except Exception as exc:
                results.append(
                    {
                        "video_id": asset.video_id,
                        "source_path": str(asset.path),
                        "error": str(exc),
                    }
                )
            progress.progress(index / len(assets))

        st.session_state.results = results
        LATEST_OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        st.rerun()

    if st.session_state.results:
        st.divider()
        st.subheader("Results")
        full_json = json.dumps(st.session_state.results, indent=2)
        captions_json = json.dumps(caption_only(st.session_state.results), indent=2)

        export_cols = st.columns([1, 1, 2])
        export_cols[0].download_button(
            "Download full JSON",
            data=full_json,
            file_name="captions_full.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Download captions JSON",
            data=captions_json,
            file_name="captions.json",
            mime="application/json",
            use_container_width=True,
        )
        export_cols[2].caption(str(LATEST_OUTPUT))

        for item in st.session_state.results:
            render_result(item)
    else:
        st.divider()
        st.subheader("Queue")
        if source_mode == "Upload":
            names = [file.name for file in video_files] if video_files else []
            st.write(names or "No upload selected.")
        else:
            st.write([asset.path.name for asset in existing_assets] or "No files in data/videos.")


if __name__ == "__main__":
    main()
