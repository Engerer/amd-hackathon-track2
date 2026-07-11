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

The production path uses four frames and one Kimi K2.6 call per clip. For every 30-120 second video it preserves beginning, middle, and end coverage, then adds the strongest non-duplicate motion or salience frame.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> OpenCV preserves three timeline anchors and selects one extra candidate for sharpness, exposure, and motion
 -> perceptual hashing prevents the salience frame from duplicating an anchor
 -> resize each frame to 896px width (896x504 for 16:9 video)
 -> one multimodal Kimi K2.6 call receives four chronological pristine frames and their timestamps
 -> that same call grounds the scene and returns every requested style in schema-constrained JSON
 -> Docker writes /output/results.json
```

The prompt asks Kimi to identify a shared factual nucleus internally, preserve it across all styles, and vary only the tone. This avoids the information loss and factual drift of intermediate evidence and candidate rewrites:

```json
{
  "captions": {
    "formal": "...",
    "sarcastic": "...",
    "humorous_tech": "...",
    "humorous_non_tech": "..."
  }
}
```

The single call receives a deadline-derived HTTP timeout. Schema-constrained output requires every requested style; parsing and caption cleanup are local and never trigger a second model call.

## Defaults

- Multimodal caption model: `accounts/fireworks/models/kimi-k2p6`
- Model calls: exactly one under normal inference
- Frame sampling: four hybrid timeline and salience frames
- Frame cap: exactly `4` total frames for 30-120 second clips
- Frame width: `896px` (`896x504` for 16:9 video)
- Model input: four pristine frame images with timestamps supplied as adjacent text metadata
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: Kimi receives exactly four images in one request. The images are not obscured by timestamp banners; timestamps and total duration are supplied as adjacent text metadata.

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
4. Review the sampled timestamped frames, observations, and four captions for each video.
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

## Docker

Build locally:

```powershell
docker buildx build --platform linux/amd64 --provenance=false --sbom=false --load --build-arg MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev --build-arg FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6 -t amd-track2-captioner:local .
```

Publish the challenge image with the same single-platform settings:

```powershell
docker buildx build --platform linux/amd64 --provenance=false --sbom=false --push -t engeraaa/amd-track2-captioner:latest .
```

Disabling provenance and SBOM attestations keeps `latest` as one plain `linux/amd64` manifest for compatibility with minimal challenge image pullers.

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
TRACK2_MAX_FRAMES=4
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_PER_CLIP_DEADLINE_SECONDS=28
TRACK2_STAGE_DEADLINE_DIRECT=23
TRACK2_DRY_RUN=true
RUN_CHECKS=true
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/kimi-k2p6
FIREWORKS_CAPTION_MAX_TOKENS=800
FIREWORKS_CREATIVE_TEMPERATURE=0.45
FIREWORKS_MAX_RETRIES=0
FIREWORKS_REQUEST_TIMEOUT_SECONDS=28
```

Recommended final settings:

```text
RUN_CHECKS=false
TRACK2_FRAME_PROFILE=hybrid
TRACK2_MAX_FRAMES=4
TRACK2_MODEL_CALL_RESERVE_SECONDS=75
TRACK2_ENABLE_STYLE_RETRY=false
TRACK2_PER_CLIP_DEADLINE_SECONDS=28
TRACK2_STAGE_DEADLINE_DIRECT=23
FIREWORKS_CAPTION_MAX_TOKENS=500
FIREWORKS_CREATIVE_TEMPERATURE=0.2
FIREWORKS_MAX_RETRIES=0
FIREWORKS_REQUEST_TIMEOUT_SECONDS=28
```

## Prompt Tuning

The production direct-caption prompt lives in `track2_captioner/caption_pipeline.py`. It instructs Kimi to ground a shared factual nucleus from four chronological frames, then render that same evidence in every requested style. A dynamic Fireworks-supported JSON schema requires exactly the requested style keys.

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

## Accuracy Notes

- Ordered frames can establish broad temporal change, but four samples cannot prove every event between timestamps.
- The prompt therefore forbids unsupported intent, causality, dialogue, and off-screen events.
- Humor changes framing only; visible subjects, setting, and action must remain consistent across all styles.
