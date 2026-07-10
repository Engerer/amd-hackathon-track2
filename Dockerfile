FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL="https://track2-fireworks-proxy.proxide-track2.workers.dev"
ARG FIREWORKS_MODEL="accounts/fireworks/models/qwen3p7-plus"
ARG FIREWORKS_CAPTION_MODEL="accounts/fireworks/models/qwen3p7-plus"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_MODEL} \
    AUTO_TRANSCRIBE=off \
    RUN_CHECKS=false \
    TRACK2_RUNTIME_TARGET_SECONDS=540 \
    TRACK2_HARD_DEADLINE_SECONDS=585 \
    TRACK2_FRAME_PROFILE=fast \
    TRACK2_MAX_FRAMES=15 \
    TRACK2_MODEL_CALL_RESERVE_SECONDS=75 \
    TRACK2_ENABLE_STYLE_RETRY=false \
    TRACK2_AUDIO_CUES=true \
    TRACK2_AUDIO_CUE_SECONDS=20 \
    TRACK2_MAX_TRANSCRIBED_CLIPS=0 \
    TRACK2_TRANSCRIBE_MAX_DURATION_SECONDS=90 \
    TRACK2_TRANSCRIBE_BEFORE_SECONDS=360 \
    FIREWORKS_CAPTION_MAX_TOKENS=700 \
    FIREWORKS_MAX_RETRIES=1 \
    FIREWORKS_REQUEST_TIMEOUT_SECONDS=60 \
    WHISPER_MODEL=tiny

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_WHISPER=false

COPY requirements.txt requirements-whisper.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_WHISPER" = "true" ]; then pip install --no-cache-dir -r requirements-whisper.txt; fi
RUN if [ "$INSTALL_WHISPER" = "true" ]; then python -c "import whisper; whisper.load_model('tiny')"; fi

COPY . .

EXPOSE 8501

ENTRYPOINT ["python", "-m", "track2_captioner.harness"]
