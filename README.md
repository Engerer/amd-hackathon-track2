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

The pipeline sends exactly three silent representative frames covering the beginning, middle, and end of each clip.

```text
up to two video tasks concurrently
 -> video URL
 -> download video
 -> ffprobe checks duration
 -> ffmpeg samples chronological candidates at 2 FPS across the full clip
 -> local sharpness/exposure/contrast/diversity scoring selects one frame per temporal third
 -> resize each frame to 896px width
 -> four Kimi K2.6 calls run in parallel, one for each requested style
 -> every Kimi call receives the same three frames plus its own style system prompt
 -> the four returned sentences are recombined into the Track 2 captions object
 -> Docker writes /output/results.json
```

Each style is grounded directly in the same three locally selected visual samples. There is no audio processing, transcript, judge call, or repair call.

The harness preserves input ordering even when the two video workers finish out of
order. One 540-second runtime budget is shared across all tasks. Below 30 seconds
it emits conservative style-specific
fallbacks. This leaves the final minute of the evaluator's 10-minute limit for
startup, last-resort output completion, and writing valid JSON.

## Defaults

- Vision model: `accounts/fireworks/models/kimi-k2p6`
- Caption model: `accounts/fireworks/models/kimi-k2p6`
- Frame sampling: FFmpeg extracts candidates at `2 FPS`
- Frame selector: local quality and perceptual-diversity scoring, then timeline anchors
- Frame cap: `3` total frames per video
- Frame width: `896px`
- Reasoning: enabled at `medium` effort for Kimi
- Caption completion budget: `4000` tokens, including reasoning
- Concurrent video tasks: `2` (each retains four parallel style calls)
- Global processing budget: `540` seconds
- Model attempts: `2` total; retry only 429, 500/502/503/504, reset connections, and deadline-safe timeouts
- Audio/transcription: disabled and not installed
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: FFmpeg produces 60-240 candidates for the allowed 30-120 second clips. Selection stays local, and each of the four parallel Kimi calls receives only the same three selected images.

Important: medium reasoning requires the checked-in Cloudflare Worker configuration
with `MAX_TOKENS = 4000`. Redeploying an older 1,000-token Worker can silently
exhaust the reasoning budget and break Kimi output.

The retired eight-video judging set and a repeatable five-point scoring rubric are in [`benchmarks/`](benchmarks/README.md). Use that set to compare prompt and frame-selection revisions across all 32 clip/style combinations.

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
TRACK2_MAX_FRAMES=3
TRACK2_TASK_WORKERS=2
TRACK2_RUNTIME_BUDGET_SECONDS=540
TRACK2_DRY_RUN=true
RUN_CHECKS=true
FIREWORKS_MAX_ATTEMPTS=2
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/kimi-k2p6
```

Recommended final settings:

```text
RUN_CHECKS=false
TRACK2_MAX_FRAMES=3
TRACK2_TASK_WORKERS=2
TRACK2_RUNTIME_BUDGET_SECONDS=540
FIREWORKS_MAX_ATTEMPTS=2
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
- Kimi K2.6 is used for every caption because all four style calls require direct image grounding.
- Humor prompts can still invent small details; prompt tuning should focus on reducing hallucination.
