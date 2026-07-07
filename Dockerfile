FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_PROXY_URL=""
ARG FIREWORKS_MODEL="accounts/fireworks/models/gemma-4-26b-a4b-it"

ENV MODEL_PROXY_URL=${MODEL_PROXY_URL} \
    FIREWORKS_MODEL=${FIREWORKS_MODEL} \
    FIREWORKS_JUDGE_MODEL=${FIREWORKS_MODEL}

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_WHISPER=false

COPY requirements.txt requirements-whisper.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN if [ "$INSTALL_WHISPER" = "true" ]; then pip install --no-cache-dir -r requirements-whisper.txt; fi

COPY . .

EXPOSE 8501

ENTRYPOINT ["python", "-m", "track2_captioner.harness"]
