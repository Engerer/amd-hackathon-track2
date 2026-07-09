FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL=""
ARG FIREWORKS_MODEL="accounts/fireworks/models/qwen3p7-plus"
ARG FIREWORKS_CAPTION_MODEL="accounts/fireworks/models/qwen3p7-plus"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_MODEL} \
    AUTO_TRANSCRIBE=false \
    RUN_CHECKS=false \
    TRACK2_RUNTIME_TARGET_SECONDS=540 \
    TRACK2_HARD_DEADLINE_SECONDS=585 \
    TRACK2_FRAME_PROFILE=balanced \
    TRACK2_MAX_FRAMES=12 \
    TRACK2_ENABLE_STYLE_RETRY=true \
    FIREWORKS_CAPTION_MAX_TOKENS=900 \
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
