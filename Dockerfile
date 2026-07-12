FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL=""
ARG FIREWORKS_MODEL="accounts/fireworks/models/kimi-k2p6"
ARG FIREWORKS_CAPTION_MODEL="accounts/fireworks/models/kimi-k2p6"
ARG FIREWORKS_SELECTOR_MODEL="accounts/fireworks/models/qwen3p7-plus"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL} \
    FIREWORKS_SELECTOR_MODEL=${FIREWORKS_SELECTOR_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_REASONING_EFFORT=none \
    TRACK2_MAX_FRAMES=3 \
    RUN_CHECKS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["python", "-m", "track2_captioner.harness"]
