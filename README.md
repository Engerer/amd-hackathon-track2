# AMD ACT II Track 2: Video Captioning Agent

Track 2 submission for the AMD Developer Hackathon ACT II.

The agent captions each video in four required styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

Submission image:

```text
somnuskai/amd-track2-captioner:latest
```

GitHub repo:

```text
https://github.com/nish0203/amd-track2-captioner
```

## How It Works

The pipeline samples frames instead of processing every video frame. By default it extracts one timeline frame every 3 seconds, capped at 40 frames for cost and runtime control.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> ffmpeg extracts one frame every 3 seconds
 -> resize each frame to 768px width
 -> Kimi K2.6 creates factual observations
 -> Kimi K2.6 writes four styled captions
 -> Docker writes /output/results.json
```

The first Kimi call receives image frames and creates observations:

```json
{
  "setting": "...",
  "subjects": ["..."],
  "actions": ["..."],
  "sequence": ["beginning", "middle", "end"],
  "visible_text": ["..."],
  "audio_or_speech": ["..."],
  "uncertainties": ["..."]
}
```

The caption calls use those observations only, so they are cheaper than sending frames again.

## Defaults

- Model: `accounts/fireworks/models/kimi-k2p6`
- Frame sampling: `1` frame every `3` seconds
- Frame cap: `40` total frames per video
- Frame width: `768px`
- Whisper audio transcription: off by default
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: `40` is a safety cap, not 40 FPS. A 30-second video gives about 10 frames, and a 2-minute video gives about 40 frames.

## Quick Start

Clone the repo:

```powershell
git clone https://github.com/nish0203/amd-track2-captioner.git
cd amd-track2-captioner
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

For final submission, Whisper is off by default to keep runtime and Docker size smaller.

## Docker

Build locally:

```powershell
docker build --build-arg MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev --build-arg FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6 -t amd-track2-captioner:local .
```

Run locally:

```powershell
docker run --rm -v "${PWD}\sample_input:/input:ro" -v "${PWD}\docker_sample_output:/output" amd-track2-captioner:local
```

Push final image:

```powershell
docker tag amd-track2-captioner:local somnuskai/amd-track2-captioner:latest
docker push somnuskai/amd-track2-captioner:latest
```

Public submission image:

```text
somnuskai/amd-track2-captioner:latest
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
TRACK2_MAX_FRAMES=40
TRACK2_DRY_RUN=true
RUN_CHECKS=true
AUTO_TRANSCRIBE=true
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6
```

Recommended final settings:

```text
RUN_CHECKS=false
AUTO_TRANSCRIBE=false
TRACK2_MAX_FRAMES=40
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
- DeepSeek V4 does not support image input, so it cannot be the main video-understanding model.
- Humor prompts can still invent small details; prompt tuning should focus on reducing hallucination.
