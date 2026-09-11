from __future__ import annotations

import base64
import os
import subprocess
import time

import requests
import runpod

MODEL_PATH = os.environ.get("HIGGS_MODEL_PATH", "/models/higgs-audio-v3-tts-4b")
PORT = int(os.environ.get("SGLANG_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "12000"))
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES", str(20 * 1024 * 1024)))
_process = None


def _start_server():
    global _process
    if _process and _process.poll() is None:
        return
    _process = subprocess.Popen([
        "sgl-omni", "serve", "--model-path", MODEL_PATH, "--port", str(PORT)
    ])
    deadline = time.time() + int(os.environ.get("MODEL_START_TIMEOUT", "1200"))
    while time.time() < deadline:
        if _process.poll() is not None:
            raise RuntimeError(f"SGLang-Omni exited with code {_process.returncode}")
        try:
            if requests.get(f"{BASE_URL}/health", timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("SGLang-Omni did not become healthy before timeout")


def _reference_data_url(value: str) -> str:
    if value.startswith("data:"):
        raw = base64.b64decode(value.split(",", 1)[1], validate=True)
        if len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("reference_audio exceeds the configured size limit")
        return value
    raw = base64.b64decode(value, validate=True)
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError("reference_audio exceeds the configured size limit")
    return "data:audio/wav;base64," + value


def handler(job):
    data = job.get("input") or {}
    text = str(data.get("text", "")).strip()
    if not text or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1-{MAX_TEXT_CHARS} characters")
    ref_b64 = data.get("reference_audio")
    ref_text = str(data.get("reference_text", "")).strip()
    if bool(ref_b64) != bool(ref_text):
        raise ValueError("reference_audio and reference_text must be supplied together")
    _start_server()
    payload = {
        "model": MODEL_PATH,
        "voice": "default",
        "input": text,
        "response_format": "wav",
        "stream": False,
        "temperature": float(data.get("temperature", 0.8)),
        "top_k": int(data.get("top_k", 50)),
        "max_new_tokens": int(data.get("max_new_tokens", 2048)),
        "seed": int(data.get("seed", 42)),
    }
    if ref_b64:
        payload["references"] = [{"audio_path": _reference_data_url(ref_b64), "text": ref_text}]
    response = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload, timeout=1800)
    response.raise_for_status()
    return {
        "audio_base64": base64.b64encode(response.content).decode("ascii"),
        "format": "wav",
        "sample_rate": 24000,
        "model": "bosonai/higgs-audio-v3-tts-4b",
    }


runpod.serverless.start({"handler": handler})

