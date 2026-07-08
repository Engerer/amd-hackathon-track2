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

The pipeline samples timeline evidence instead of processing every video frame. By default it combines evenly spaced anchor frames with a smaller scene-change sample, keeps approximate timestamps for the observation prompt, and stays under a 32-frame safety cap for accuracy and runtime control.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> ffmpeg extracts timeline anchor frames plus scene-change frames
 -> resize each frame to 768px width
 -> Kimi K2.6 creates detailed factual observations from frames and frame timing metadata
 -> GLM 5.2 writes four style-specific captions from the observations
 -> Docker writes /output/results.json
```

The first Kimi call receives image frames and creates observations:

```json
{
  "summary": "...",
  "setting": "...",
  "subjects": ["..."],
  "key_objects": ["..."],
  "actions": ["..."],
  "timeline": ["beginning: ...", "middle: ...", "end: ..."],
  "visible_text": ["..."],
  "audio_or_speech": ["..."],
  "uncertainties": ["..."]
}
```

The caption calls use those observations only, so they are cheaper than sending frames again.

## Defaults

- Vision model: `accounts/fireworks/models/kimi-k2p6`
- Caption model: `accounts/fireworks/models/glm-5p2`
- Frame sampling: adaptive anchor + scene-change frames
- Frame cap: `32` total frames per video
- Frame width: `768px`
- Whisper audio transcription: on by default in Docker with the `tiny` model
- Internal judge checks: off by default
- Fireworks key: stored in a Cloudflare Worker secret, not in the repo

Important: `32` is a safety cap, not 32 FPS. The harness can reduce the frame budget when the runtime deadline is tight.

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

Dry-run mode does not download remote videos, so it is safe for quick contract checks.

Run the local test suite:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
```

Run the sample contract eval helper:

```powershell
.\.venv\Scripts\python scripts\eval_examples.py
```

To spend credits and call the configured backend, add `--real`:

```powershell
.\.venv\Scripts\python scripts\eval_examples.py --real
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

The submitted Docker image installs Whisper and enables `AUTO_TRANSCRIBE=true` with `WHISPER_MODEL=tiny`.
Videos without an audio stream skip Whisper and continue through the visual caption pipeline.

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
TRACK2_MAX_FRAMES=32
TRACK2_DRY_RUN=true
TRACK2_RUNTIME_BUDGET_SECONDS=570
TRACK2_DOWNLOAD_TIMEOUT_SECONDS=120
TRACK2_TRANSCRIBE_MIN_REMAINING_SECONDS=150
RUN_CHECKS=true
AUTO_TRANSCRIBE=true
WHISPER_MODEL=tiny
WHISPER_LANGUAGE=en
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6
FIREWORKS_CAPTION_MODEL=accounts/fireworks/models/glm-5p2
```

Recommended final settings:

```text
RUN_CHECKS=false
AUTO_TRANSCRIBE=true
TRACK2_MAX_FRAMES=32
```

Cloudflare Worker proxy tuning:

```text
MAX_TOKENS=1000
MAX_TEMPERATURE=0.9
```

The proxy forwards `response_format` so JSON-mode observation and judge calls stay reliable.

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
- Local dry-run contract eval passes
- No hardcoded sample answers
- Fireworks key is not exposed

## Known Issues

- Gemma deployment currently fails with `payment method is required`.
- GLM 5.2 and DeepSeek V4 do not support image input on Fireworks, so they cannot be the main video-understanding model. They can be used after Kimi turns frames into text observations.
- Humor prompts can still invent small details; prompt tuning should focus on reducing hallucination.
