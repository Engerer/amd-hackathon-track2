# AMD ACT II Track 2: Video Captioning

Starter pipeline for Track 2 of the AMD Developer Hackathon: ACT II.

The goal is to generate one caption or summary per video in four styles:

- Formal
- Sarcastic
- Humorous-tech
- Humorous-non-tech

The strategy is simple: extract a small set of frames, turn the video into factual observations first, then rewrite the same facts into each required tone. This keeps the funny captions from drifting away from what actually happened in the clip.

## Quick Start

1. Create an environment file:

```bash
cp .env.example .env
```

2. Add your Fireworks API key to `.env`.

For the public Docker submission, do not bake a Fireworks API key into the image. Use the Firebase proxy in `firebase_proxy/`, then set:

```text
MODEL_PROXY_URL=https://us-central1-your-project.cloudfunctions.net/fireworksChat
FIREWORKS_API_KEY=
```

3. Put official hackathon clips in:

```text
data/videos/
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

To enable automatic audio transcription with Whisper, install the optional extras:

```bash
pip install -r requirements-whisper.txt
```

Whisper also needs `ffmpeg` available on your machine. The Docker image installs it automatically.

5. Start the web app:

```bash
streamlit run app.py
```

6. Run a dry test without calling Fireworks from the CLI:

```bash
python -m track2_captioner.run --input data/videos --output outputs/captions.json --dry-run
```

7. Run the real pipeline from the CLI:

```bash
python -m track2_captioner.run --input data/videos --output outputs/captions.json
```

8. Run with Whisper auto-transcription:

```bash
python -m track2_captioner.run --input data/videos --output outputs/captions.json --auto-transcribe --whisper-model base
```

Whisper writes transcripts to `data/transcripts/<video-name>.txt`. Existing transcripts are reused unless `--force-transcribe` is passed.

## Submission Harness

The participant guide says Track 2 is evaluated as a Docker batch agent. The submitted image must:

- read `/input/tasks.json`
- download each `video_url`
- generate captions for the requested styles
- write `/output/results.json`
- exit with code `0`

Example input:

```json
[
  {
    "task_id": "v1",
    "video_url": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
  }
]
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

Run the harness locally:

```bash
python -m track2_captioner.harness
```

For local paths instead of `/input` and `/output`, set:

```bash
set TRACK2_INPUT=sample_input/tasks.json
set TRACK2_OUTPUT=sample_output/results.json
python -m track2_captioner.harness
```

Useful harness environment variables:

- `TRACK2_DRY_RUN=true`
- `TRACK2_MAX_FRAMES=8`
- `RUN_CHECKS=true`
- `MODEL_PROXY_URL=https://us-central1-your-project.cloudfunctions.net/fireworksChat`
- `AUTO_TRANSCRIBE=true`
- `WHISPER_MODEL=base`
- `WHISPER_LANGUAGE=en`

By default, the submission harness skips internal judge/check calls for speed. The hidden evaluation scores the final captions.

## Firebase Proxy

Track 2 does not inject API keys, so the safest cloud setup is:

```text
Public Docker image
 -> Firebase HTTPS function
 -> Fireworks API
```

The Firebase function stores `FIREWORKS_API_KEY` as a Firebase secret. The Docker image only contains `MODEL_PROXY_URL`.

Setup guide:

[firebase_proxy/README.md](firebase_proxy/README.md)

After deployment, put the function URL in `.env` for local testing or set it as an environment variable when running Docker:

```bash
set MODEL_PROXY_URL=https://us-central1-your-project.cloudfunctions.net/fireworksChat
```

Never commit `.env`, never put a Fireworks API key in the Dockerfile, and rotate the key after judging.

If Firebase blocks deployment because the project is not on Blaze, use the Cloudflare Worker fallback:

[cloudflare_proxy/README.md](cloudflare_proxy/README.md)

## Docker

Build:

```bash
docker build -t amd-track2-captioner .
```

Build for final submission with the Firebase proxy URL baked in:

```bash
docker build --build-arg MODEL_PROXY_URL=https://us-central1-your-project.cloudfunctions.net/fireworksChat -t amd-track2-captioner .
```

Build with Whisper support:

```bash
docker build --build-arg INSTALL_WHISPER=true -t amd-track2-captioner .
```

Run:

```bash
docker run --rm --env-file .env -v "%cd%/sample_input:/input:ro" -v "%cd%/sample_output:/output" amd-track2-captioner
```

On macOS/Linux, use:

```bash
docker run --rm --env-file .env -v "$PWD/sample_input:/input:ro" -v "$PWD/sample_output:/output" amd-track2-captioner
```

Run the web app from the Docker image:

```bash
docker run --rm --env-file .env -p 8501:8501 --entrypoint streamlit amd-track2-captioner run app.py --server.address=0.0.0.0 --server.port=8501
```

## Expected Output

The pipeline writes JSON shaped like this:

```json
[
  {
    "video_id": "clip_001",
    "source_path": "data/videos/clip_001.mp4",
    "observations": {
      "setting": "...",
      "subjects": ["..."],
      "actions": ["..."],
      "sequence": ["..."],
      "visible_text": [],
      "audio_or_speech": [],
      "uncertainties": []
    },
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    },
    "checks": {
      "formal": {"accuracy": "pass", "tone": "pass", "notes": ""},
      "sarcastic": {"accuracy": "pass", "tone": "pass", "notes": ""}
    }
  }
]
```

## Tuning Plan

When the official videos and required Fireworks models are available:

1. Run `--dry-run` to confirm file discovery.
2. Run on 1-2 clips and inspect `outputs/captions.json`.
3. Tune only the prompt files in `prompts/`.
4. Add transcripts as `data/transcripts/<video-name>.txt` if a clip has meaningful speech.
5. Submit the best structured output and showcase the pipeline in the demo video.

## Submission Checklist

- Public Docker image
- Linux/amd64 image manifest
- `/input/tasks.json` to `/output/results.json` harness
- README with setup and usage
- Valid JSON output with all requested styles
- Container image under 10GB compressed
- No hardcoded example answers

## Notes for the Gemma Prize

Track 2 has a separate Best Use of Gemma prize. If Gemma is available through Fireworks for the track, set `FIREWORKS_MODEL` to the official Gemma model and mention in the README/demo exactly where Gemma is used:

- visual observation extraction
- style caption generation
- self-checking and repair

Be explicit. Judges should not have to infer the Gemma contribution.
