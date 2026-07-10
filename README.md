# AMD ACT II Track 2: Video Captioning Agent

Track 2 submission for the AMD Developer Hackathon ACT II.

The agent captions each video in four required styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

Submission image:

```text
engeraaa/amd-track2-captioner:latest
```

GitHub repo:

```text
https://github.com/Engerer/amd-hackathon-track2
```

## How It Works

The pipeline samples timeline evidence instead of processing every video frame. By default it keeps 10 uniformly distributed timeline anchors and adds 5 sharp, salient frames selected from a lightweight OpenCV candidate scan.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> OpenCV preserves timeline anchors and scores extra candidates for sharpness, exposure, and motion
 -> resize each frame to 768px width
 -> build one overview storyboard and select three higher-resolution detail frames
 -> Qwen3.7 Plus receives the overview and details directly
 -> Qwen3.7 Plus writes all requested style captions in one multimodal JSON call
 -> Docker writes /output/results.json
```

The Qwen3.7 call receives image frames and returns captions directly:

```json
{
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
  },
  "visual_facts": ["brief factual details used for grounding"]
}
```

The prompt establishes one shared factual core before generating all four styles, so humor changes tone without changing the visible subject, action, or setting. Optional style repair remains available for local experiments but is disabled in the timed submission.

## Defaults

- Direct multimodal model: `accounts/fireworks/models/qwen3p7-plus`
- Caption model setting: also `accounts/fireworks/models/qwen3p7-plus` for compatibility
- Frame sampling: hybrid timeline anchors plus sharp salient candidates
- Frame cap: `15` total frames per video by default, hard-capped at `20`
- Frame width: `768px`
- Model input: one 15-frame overview plus three high-resolution detail images
- Whisper audio transcription: off by default in Docker
- Volume-only audio cues: off by default in Docker
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: `15` is a total-frame cap, not 15 FPS. Perceptual deduplication is not applied to the timed path, so fixed-camera clips retain their complete timeline coverage.

## Quick Start

Clone the repo:

```powershell
git clone https://github.com/Engerer/amd-hackathon-track2.git
cd amd-hackathon-track2
```

Create the Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Run the web app:

```powershell
.\.venv\Scripts\streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

## Use The Web App

1. Open the app.
2. Upload one or more videos.
3. Click **Generate captions**.
4. Review the sampled storyboard frames, observations, and four captions for each video.
5. Download the JSON if needed.

Advanced settings are in the collapsed sidebar. Most users can leave them unchanged.

## Deploy The Demo

The repository is ready for Streamlit Community Cloud:

- Repository: `Engerer/amd-hackathon-track2`
- Branch: `master`
- Main file: `app.py`
- Python dependencies: `requirements.txt`
- System dependency: `ffmpeg` from `packages.txt`

Set this root-level Streamlit secret so the hosted demo uses the same model proxy as the submission image:

```toml
MODEL_PROXY_URL = "https://track2-fireworks-proxy.proxide-track2.workers.dev"
```

## Run A Dry Test

Dry-run mode checks the input/output flow without spending Fireworks credits.

```powershell
$env:TRACK2_DRY_RUN="1"
$env:TRACK2_INPUT="sample_input/tasks.json"
$env:TRACK2_OUTPUT="sample_output/results.json"
.\.venv\Scripts\python -m track2_captioner.harness
Remove-Item Env:TRACK2_DRY_RUN,Env:TRACK2_INPUT,Env:TRACK2_OUTPUT
```

Check:

```text
sample_output/results.json
```

## Run The Real Local Harness

This uses the configured model/proxy and spends Fireworks credits.

```powershell
$env:TRACK2_INPUT="sample_input/tasks.json"
$env:TRACK2_OUTPUT="sample_output/results.json"
.\.venv\Scripts\python -m track2_captioner.harness
Remove-Item Env:TRACK2_INPUT,Env:TRACK2_OUTPUT
```

## Optional Whisper

Whisper can transcribe video audio and pass the transcript into the caption pipeline.

Install optional dependencies:

```powershell
.\.venv\Scripts\pip install -r requirements-whisper.txt
```

Enable it for harness runs:

```powershell
$env:AUTO_TRANSCRIBE="true"
$env:WHISPER_MODEL="base"
```

The submitted Docker image keeps `AUTO_TRANSCRIBE=off` and does not install Whisper, avoiding the large Torch dependency and CPU transcription cost. Local experiments can still enable it explicitly.

## Docker

Build locally:

```powershell
docker build --build-arg MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev --build-arg FIREWORKS_MODEL=accounts/fireworks/models/qwen3p7-plus -t amd-track2-captioner:local .
```

Run locally:

```powershell
docker run --rm -v "${PWD}\sample_input:/input:ro" -v "${PWD}\docker_sample_output:/output" amd-track2-captioner:local
```

Push final image:

```powershell
docker tag amd-track2-captioner:local engeraaa/amd-track2-captioner:latest
docker push engeraaa/amd-track2-captioner:latest
```

Public submission image:

```text
engeraaa/amd-track2-captioner:latest
```

## Hackathon I/O Contract

The evaluator provides:

```text
/input/tasks.json
```

Example input:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/example.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
```

The container writes:

```text
/output/results.json
```

Example output:

```json
[
  {
    "task_id": "v1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

## Environment Variables

Useful variables:

```text
TRACK2_RUNTIME_TARGET_SECONDS=540
TRACK2_HARD_DEADLINE_SECONDS=585
TRACK2_FRAME_PROFILE=hybrid
TRACK2_MAX_FRAMES=15
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_AUDIO_CUES=false
TRACK2_AUDIO_CUE_SECONDS=20
TRACK2_MAX_TRANSCRIBED_CLIPS=3
TRACK2_TRANSCRIBE_MAX_DURATION_SECONDS=90
TRACK2_TRANSCRIBE_BEFORE_SECONDS=360
TRACK2_DRY_RUN=true
RUN_CHECKS=true
AUTO_TRANSCRIBE=off
WHISPER_MODEL=tiny
WHISPER_LANGUAGE=en
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_CAPTION_MAX_TOKENS=700
FIREWORKS_CREATIVE_TEMPERATURE=0.45
FIREWORKS_MAX_RETRIES=1
FIREWORKS_REQUEST_TIMEOUT_SECONDS=28
```

Recommended final settings:

```text
RUN_CHECKS=false
AUTO_TRANSCRIBE=off
TRACK2_FRAME_PROFILE=hybrid
TRACK2_MAX_FRAMES=15
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_AUDIO_CUES=false
TRACK2_MAX_TRANSCRIBED_CLIPS=0
TRACK2_TRANSCRIBE_MAX_DURATION_SECONDS=90
FIREWORKS_CAPTION_MAX_TOKENS=700
FIREWORKS_CREATIVE_TEMPERATURE=0.45
FIREWORKS_MAX_RETRIES=1
FIREWORKS_REQUEST_TIMEOUT_SECONDS=28
```

## Prompt Tuning

The production multimodal prompt, factual-core schema, and concise style templates live in `track2_captioner/caption_pipeline.py`. The files under `prompts/` support optional local judging and legacy single-style workflows.

## Security

Do not commit secrets.

The Fireworks API key should not be stored in:

- `.env` committed to Git
- Dockerfile
- README
- Discord messages
- screenshots

Current setup:

```text
Docker container
 -> Cloudflare Worker proxy
 -> Fireworks API
```

The Fireworks API key is stored as a Cloudflare Worker secret.

## Final Checklist

- Docker image is public
- Docker image has `linux/amd64`
- Docker image is under 10GB
- Container reads `/input/tasks.json`
- Container writes `/output/results.json`
- JSON is valid
- All four styles are present
- No hardcoded sample answers
- Fireworks key is not exposed

## Known Issues

- Gemma deployment currently fails with `payment method is required`.
- This version assumes `qwen3p7-plus` accepts image inputs on the configured Fireworks endpoint.
- Humor prompts can still invent small details; prompt tuning should focus on reducing hallucination.
