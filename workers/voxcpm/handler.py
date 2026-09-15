from __future__ import annotations

import base64
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import runpod
import soundfile as sf
import torch
from voxcpm import VoxCPM

MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/models/voxcpm2")
# torch.compile costs several minutes on every cold start and nothing is cached
# between scale-to-zero cycles, so optimization is opt-in.
OPTIMIZE = os.environ.get("VOXCPM_OPTIMIZE", "0").lower() in {"1", "true", "yes"}
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "12000"))
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES", str(20 * 1024 * 1024)))

_model = None

_MAGIC_SUFFIXES = (
    (b"RIFF", ".wav"),
    (b"fLaC", ".flac"),
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"\xff\xf3", ".mp3"),
)


def _decode_audio(value: str) -> bytes:
    if not value:
        raise ValueError("reference_audio is empty")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError("reference_audio exceeds the configured size limit")
    return raw


def _suffix_for(raw: bytes) -> str:
    for magic, suffix in _MAGIC_SUFFIXES:
        if raw.startswith(magic):
            return suffix
    if raw[4:8] == b"ftyp":
        return ".m4a"
    return ".wav"


def _load():
    global _model
    if _model is None:
        # The repository config pins dtype bfloat16; VoxCPM keeps it on CUDA.
        _model = VoxCPM.from_pretrained(
            MODEL_PATH, load_denoiser=False, optimize=OPTIMIZE, device="cuda"
        )
    return _model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def handler(job):
    data = job.get("input") or {}
    text = str(data.get("text", "")).strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1-{MAX_TEXT_CHARS} characters")
    ref_b64 = data.get("reference_audio")
    ref_text = str(data.get("reference_text", "")).strip()
    if ref_text and not ref_b64:
        raise ValueError("reference_text requires reference_audio")
    cfg_value = float(data.get("cfg_value", 2.0))
    if not math.isfinite(cfg_value) or cfg_value <= 0:
        raise ValueError("cfg_value must be finite and greater than zero")
    inference_timesteps = int(data.get("inference_timesteps", 10))
    if not 1 <= inference_timesteps <= 100:
        raise ValueError("inference_timesteps must be between 1 and 100")
    max_len = int(data.get("max_len", 4096))
    if not 16 <= max_len <= 8192:
        raise ValueError("max_len must be between 16 and 8192")
    normalize = bool(data.get("normalize", False))
    seed = int(data.get("seed", 42))
    model = _load()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        kwargs = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "max_len": max_len,
            "normalize": normalize,
        }
        if ref_b64:
            raw = _decode_audio(ref_b64)
            ref_path = temp / f"reference{_suffix_for(raw)}"
            ref_path.write_bytes(raw)
            # Reference audio alone is controllable cloning; adding the exact
            # transcript switches VoxCPM2 into higher-fidelity ultimate cloning.
            kwargs["reference_wav_path"] = str(ref_path)
            if ref_text:
                kwargs.update(prompt_wav_path=str(ref_path), prompt_text=ref_text)

        _seed_everything(seed)
        wav = model.generate(**kwargs)

        sample_rate = int(model.tts_model.sample_rate)
        output_path = temp / "output.wav"
        sf.write(output_path, wav, sample_rate, subtype="PCM_16")
        audio = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return {
        "audio_base64": audio,
        "format": "wav",
        "sample_rate": sample_rate,
        "model": "openbmb/VoxCPM2",
    }


runpod.serverless.start({"handler": handler})
