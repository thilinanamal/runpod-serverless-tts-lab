FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ARG VOXCPM_VERSION=2.0.3
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/huggingface \
    VOXCPM_MODEL_PATH=/models/voxcpm2 \
    VOXCPM_OPTIMIZE=0

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install "voxcpm==${VOXCPM_VERSION}" "runpod>=1.7,<2" "huggingface-hub>=0.34"

# openbmb/VoxCPM2 is Apache-2.0 and ungated, so no build secret is needed here.
# Keeping this a plain RUN lets Runpod's console GitHub builder build the image
# directly, which the BuildKit secret mount used by the other two workers blocks.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('openbmb/VoxCPM2', local_dir='/models/voxcpm2')"

WORKDIR /opt/worker
COPY handler.py .
CMD ["python", "handler.py"]
