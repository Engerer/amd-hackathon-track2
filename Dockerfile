FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL=""
ARG FIREWORKS_MODEL="accounts/fireworks/models/kimi-k2p6"
ARG FIREWORKS_CAPTION_MODEL="accounts/fireworks/models/kimi-k2p6"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_REASONING_EFFORT=medium \
    FIREWORKS_MAX_ATTEMPTS=2 \
    TRACK2_TASK_WORKERS=2 \
    TRACK2_RUNTIME_BUDGET_SECONDS=540 \
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
