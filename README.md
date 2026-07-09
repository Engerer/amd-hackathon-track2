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

The pipeline samples timeline evidence instead of processing every video frame. By default it uses OpenCV to scan low-resolution candidate frames, then keeps a diverse 14-18 frame evidence pack with beginning, middle, and end coverage.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> OpenCV scores candidate frames for motion, scene change, sharpness, brightness, and diversity
 -> resize each frame to 768px width
 -> Qwen3.7 Plus receives the sampled frames directly
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

If one style is missing or too weak, the pipeline retries a direct multimodal repair call with the frames again while protecting the 10-minute runtime.

## Defaults

- Direct multimodal model: `accounts/fireworks/models/qwen3p7-plus`
- Caption model setting: also `accounts/fireworks/models/qwen3p7-plus` for compatibility
- Frame sampling: adaptive OpenCV candidate selection
- Frame cap: `18` total frames per video by default, hard-capped at `20`
- Frame width: `768px`
- Whisper audio transcription: off by default in Docker
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: `18` is a safety cap, not 18 FPS. A 30-45 second video gives up to 14 selected frames, 45-75 seconds gives up to 16, and 75-120 seconds gives up to 18.

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

The submitted Docker image uses `AUTO_TRANSCRIBE=conditional` with `WHISPER_MODEL=tiny`.
It only attempts Whisper for a small number of clips when audio is present and enough runtime remains.
Videos without an audio stream skip Whisper and continue through the visual caption pipeline.

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
TRACK2_FRAME_PROFILE=fast
TRACK2_MAX_FRAMES=12
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_AUDIO_CUES=true
TRACK2_AUDIO_CUE_SECONDS=20
TRACK2_MAX_TRANSCRIBED_CLIPS=3
TRACK2_TRANSCRIBE_MAX_DURATION_SECONDS=90
TRACK2_TRANSCRIBE_BEFORE_SECONDS=360
TRACK2_DRY_RUN=true
RUN_CHECKS=true
AUTO_TRANSCRIBE=conditional
WHISPER_MODEL=tiny
WHISPER_LANGUAGE=en
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/qwen3p7-plus
FIREWORKS_CAPTION_MAX_TOKENS=700
FIREWORKS_MAX_RETRIES=1
FIREWORKS_REQUEST_TIMEOUT_SECONDS=60
```

Recommended final settings:

```text
RUN_CHECKS=false
AUTO_TRANSCRIBE=conditional
TRACK2_FRAME_PROFILE=fast
TRACK2_MAX_FRAMES=12
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_AUDIO_CUES=true
TRACK2_MAX_TRANSCRIBED_CLIPS=3
TRACK2_TRANSCRIBE_MAX_DURATION_SECONDS=90
FIREWORKS_CAPTION_MAX_TOKENS=700
FIREWORKS_MAX_RETRIES=1
FIREWORKS_REQUEST_TIMEOUT_SECONDS=60
```

## Prompt Tuning

Prompt files live in:

```text
prompts/
```

Tune prompts when captions hallucinate details, miss the required style, or become too long.

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
