from __future__ import annotations

import base64
import math
import os
import tempfile
from pathlib import Path

import runpod
import soundfile as sf

from breeze_infer.runtime import load_runtime, resolve_device, set_all_seeds, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

MODEL_PATH = Path(os.environ.get("BREEZE_MODEL_PATH", "/models/breeze-tts-2"))
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "12000"))
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES", str(20 * 1024 * 1024)))

_runtime = None
_tokenizer = None
_model = None
_audio_tokenizer = None


def _decode_audio(value: str) -> bytes:
    if not value:
        raise ValueError("reference_audio is empty")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError("reference_audio exceeds the configured size limit")
    return raw


def _load():
    global _runtime, _tokenizer, _model, _audio_tokenizer
    if _runtime is None:
        _tokenizer, _model, _audio_tokenizer = load_runtime(
            MODEL_PATH, device=resolve_device(), attn_implementation="eager"
        )
        update_generation_config_for_breeze(_model)
        _runtime = FastBreezeStreamingRuntime(
            _model,
            _audio_tokenizer,
            FastStreamingConfig(
                max_new_tokens=1500,
                max_seq_len=2048,
                fast_all=False,
                fast_text_encoder=False,
                fast_backbone_prefill=False,
                fast_backbone_decode=False,
                fast_depth_decoder=False,
                fast_codec=False,
                repetition_penalty=1.1,
            ),
            tokenizer=_tokenizer,
        )
    return _runtime, _tokenizer, _model, _audio_tokenizer


def handler(job):
    data = job.get("input") or {}
    text = str(data.get("text", "")).strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1-{MAX_TEXT_CHARS} characters")
    ref_b64 = data.get("reference_audio")
    ref_text = str(data.get("reference_text", "")).strip()
    if bool(ref_b64) != bool(ref_text):
        raise ValueError("reference_audio and reference_text must be supplied together")
    cfg_scale = float(data.get("cfg_scale", 4.0 if data.get("instruction") else 1.0))
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise ValueError("cfg_scale must be finite and greater than zero")
    seed = int(data.get("seed", 42))
    instruction = str(data.get("instruction", "")).strip() or None
    runtime, tokenizer, model, audio_tokenizer = _load()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        request = {"id": job.get("id", "request"), "text": text, "speaker": "S0"}
        if instruction:
            request["instruction"] = instruction
        if ref_b64:
            ref_path = temp / "reference_audio"
            ref_path.write_bytes(_decode_audio(ref_b64))
            request.update(ref_audio_path=str(ref_path), ref_text=ref_text)

        set_all_seeds(seed)
        template = get_template(select_template_name(request))
        inputs = prepare_inputs(
            tokenizer, audio_tokenizer, model, [request], template,
            guidance_scale=cfg_scale, guidance_scale_ref=None, guidance_scale_ins=None,
        )
        output_path = temp / "output.wav"
        with sf.SoundFile(output_path, "w", samplerate=runtime.sample_rate, channels=1, subtype="PCM_16") as output:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=request["id"], seed=seed):
                output.write(chunk.audio)
        audio = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return {"audio_base64": audio, "format": "wav", "sample_rate": runtime.sample_rate, "model": "BreezeBlue/Breeze-TTS-2"}


runpod.serverless.start({"handler": handler})

