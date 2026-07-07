# AMD ACT II Track 2: Video Captioning Agent

Track 2 submission for the AMD Developer Hackathon ACT II.

The agent captions each video in four required styles:

- `formal`
- `sarcastic`
- `humorous_tech`
- `humorous_non_tech`

Current submission image:

```text
somnuskai/amd-track2-captioner:latest
```

GitHub repo:

```text
https://github.com/nish0203/amd-track2-captioner
```

## How It Works

The pipeline does not process every video frame. It samples a small number of frames across the video to control cost and runtime.

```text
video URL
 -> download video
 -> ffprobe checks duration
 -> ffmpeg extracts sampled frames
 -> resize each frame to 768px width
 -> Kimi K2.6 creates factual observations
 -> Kimi K2.6 writes four styled captions
 -> Docker writes /output/results.json
```

The observation step creates structured facts like:

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

Then the style prompts generate captions from those observations. Only the first Kimi call sends image frames; later caption calls are text-only and cheaper.

## Current Defaults

- Model: `accounts/fireworks/models/kimi-k2p6`
- Frame sampling: `10` total frames per video
- Frame width: `768px`
- Whisper audio transcription: off by default
- Internal judge checks: off by default
- Fireworks key: stored in Cloudflare Worker secret, not in the repo

Important: `10` means **10 total sampled frames**, not 10 FPS.

## What Teammates Need To Do

Each teammate should pick one lane.

### Person 1: App And Demo Testing

Run the web app, upload clips, check whether captions make sense, and collect screenshots for the final submission.

Commands:

```powershell
git clone https://github.com/nish0203/amd-track2-captioner.git
cd amd-track2-captioner
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

Do not paste any API key into Discord or GitHub.

### Person 2: Caption Quality

Tune prompts inside:

```text
prompts/
```

Focus on:

- reducing hallucinated details
- keeping captions accurate
- making each style clearly different
- keeping captions short and punchy

Main files:

```text
prompts/formal.txt
prompts/sarcastic.txt
prompts/humorous-tech.txt
prompts/humorous-non-tech.txt
prompts/perception_system.txt
```

### Person 3: Docker And Submission

Verify the Docker image works exactly like the hackathon evaluator expects.

Run:

```powershell
docker run --rm -v "${PWD}\sample_input:/input:ro" -v "${PWD}\sample_output:/output" somnuskai/amd-track2-captioner:latest
```

Check:

```text
sample_output/results.json
```

The file must be valid JSON and contain all requested styles.

### Person 4: Research And Model Testing

Compare model options and report whether they support image input.

Current status:

- Kimi K2.6 works and supports image input.
- DeepSeek V4 works for text but does not support image input.
- Gemma 4 E4B needs Fireworks deployment and currently fails without payment method.

Useful question to answer:

```text
Does this model support image input through Fireworks chat completions?
```

### Person 5: Final Materials

Prepare submission assets:

- project description
- demo video script
- screenshots
- explanation of the pipeline
- final Docker image name
- GitHub repo link

Short project explanation:

```text
We sample frames from each video, turn those frames into structured observations with Kimi K2.6, then generate four captions from the same factual observations. This keeps the captions accurate while still matching the required formal, sarcastic, humorous-tech, and humorous-non-tech styles.
```

## Local Development

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Optional Whisper support:

```powershell
.\.venv\Scripts\pip install -r requirements-whisper.txt
```

Run the web app:

```powershell
.\.venv\Scripts\streamlit run app.py
```

Run a dry test without spending Fireworks credits:

```powershell
$env:TRACK2_DRY_RUN="1"
$env:TRACK2_INPUT="sample_input/tasks.json"
$env:TRACK2_OUTPUT="sample_output/results.json"
.\.venv\Scripts\python -m track2_captioner.harness
Remove-Item Env:TRACK2_DRY_RUN,Env:TRACK2_INPUT,Env:TRACK2_OUTPUT
```

Run the real local harness:

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

Submission image:

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

Our container writes:

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
TRACK2_MAX_FRAMES=10
TRACK2_DRY_RUN=true
RUN_CHECKS=true
AUTO_TRANSCRIBE=true
WHISPER_MODEL=base
WHISPER_LANGUAGE=en
MODEL_PROXY_URL=https://track2-fireworks-proxy.proxide-track2.workers.dev
FIREWORKS_MODEL=accounts/fireworks/models/kimi-k2p6
```

For final submission, keep:

```text
RUN_CHECKS=false
AUTO_TRANSCRIBE=false
TRACK2_MAX_FRAMES=10
```

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
- README and GitHub are updated

## Known Issues

- Gemma deployment currently fails with `payment method is required`.
- DeepSeek V4 does not support image input, so it cannot be the main video-understanding model.
- Humor prompts can still invent small details; prompt tuning should focus on reducing hallucination.
