FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL=""
ARG FIREWORKS_MODEL="accounts/fireworks/models/kimi-k2p6"
ARG FIREWORKS_CAPTION_MODEL="accounts/fireworks/models/deepseek-v4-pro"
ARG FIREWORKS_CAPTION_MODELS="accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/glm-5p2"
ARG FIREWORKS_RERANK_MODEL="accounts/fireworks/models/glm-5p2"
ARG FIREWORKS_JUDGE_MODEL="accounts/fireworks/models/glm-5p2"
ARG AUTO_TRANSCRIBE="true"
ARG WHISPER_MODEL="tiny"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_CAPTION_MODEL=${FIREWORKS_CAPTION_MODEL} \
    FIREWORKS_CAPTION_MODELS=${FIREWORKS_CAPTION_MODELS} \
    FIREWORKS_RERANK_MODEL=${FIREWORKS_RERANK_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_JUDGE_MODEL} \
    AUTO_TRANSCRIBE=${AUTO_TRANSCRIBE} \
    WHISPER_MODEL=${WHISPER_MODEL}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_WHISPER=true

COPY requirements.txt requirements-whisper.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_WHISPER" = "true" ]; then pip install --no-cache-dir -r requirements-whisper.txt; fi
RUN if [ "$INSTALL_WHISPER" = "true" ]; then python -c "import whisper; whisper.load_model('tiny')"; fi

COPY . .

EXPOSE 8501

ENTRYPOINT ["python", "-m", "track2_captioner.harness"]
