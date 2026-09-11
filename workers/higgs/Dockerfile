FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HIGGS_MODEL_PATH=/models/higgs-audio-v3-tts-4b \
    SGLANG_PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip uv \
    && uv pip install --system --break-system-packages --prerelease=allow "sglang-omni==0.1.5" "runpod>=1.7,<2" "requests>=2.32"

RUN --mount=type=secret,id=hf_token \
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; p=Path('/run/secrets/hf_token'); t=p.read_text().strip() if p.exists() else None; snapshot_download('bosonai/higgs-audio-v3-tts-4b', local_dir='/models/higgs-audio-v3-tts-4b', token=t or None)"

WORKDIR /opt/worker
COPY handler.py .
CMD ["python", "handler.py"]
