FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG BREEZE_REVISION=008f769016b0a24711becd7a4925030bc93f608c
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    BREEZE_MODEL_PATH=/models/breeze-tts-2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/breezeblue-ai/breeze-tts.git \
    && cd breeze-tts && git checkout "${BREEZE_REVISION}" \
    && pip install -r requirements.txt \
    && pip install "runpod>=1.7,<2" "huggingface-hub>=0.34"

RUN --mount=type=secret,id=hf_token \
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; p=Path('/run/secrets/hf_token'); t=p.read_text().strip() if p.exists() else None; snapshot_download('BreezeBlue/Breeze-TTS-2', local_dir='/models/breeze-tts-2', token=t or None)"

COPY handler.py /opt/breeze-tts/handler.py
WORKDIR /opt/breeze-tts
CMD ["python", "handler.py"]

