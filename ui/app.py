from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
API_ROOT = "https://api.runpod.ai/v2"


def _audio_base64(path: str | None) -> str | None:
    if not path:
        return None
    raw = Path(path).read_bytes()
    if len(raw) > 20 * 1024 * 1024:
        raise gr.Error("Reference audio must be 20 MB or smaller")
    return base64.b64encode(raw).decode("ascii")


def _runpod_request(method: str, url: str, **kwargs):
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise gr.Error("RUNPOD_API_KEY is missing from .env")
    response = requests.request(method, url, headers={"Authorization": f"Bearer {key}"}, timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def generate(model, text, reference_audio, reference_text, instruction, cfg_scale,
             temperature, top_k, max_new_tokens, seed, progress=gr.Progress()):
    endpoint_var = "BREEZE_ENDPOINT_ID" if model == "Breeze TTS 2" else "HIGGS_ENDPOINT_ID"
    endpoint = os.environ.get(endpoint_var, "").strip()
    if not endpoint:
        raise gr.Error(f"{endpoint_var} is missing from .env")
    if not text.strip():
        raise gr.Error("Enter text to synthesize")
    ref = _audio_base64(reference_audio)
    if bool(ref) != bool(reference_text.strip()):
        raise gr.Error("Reference audio and its exact transcript are required together")

    payload = {"text": text.strip(), "seed": int(seed)}
    if ref:
        payload.update(reference_audio=ref, reference_text=reference_text.strip())
    if model == "Breeze TTS 2":
        payload.update(instruction=instruction.strip() or None, cfg_scale=float(cfg_scale))
    else:
        payload.update(temperature=float(temperature), top_k=int(top_k), max_new_tokens=int(max_new_tokens))

    progress(0.02, desc="Submitting job")
    submitted = _runpod_request("POST", f"{API_ROOT}/{endpoint}/run", json={"input": payload})
    job_id = submitted["id"]
    started = time.monotonic()
    timeout = 30 * 60
    while time.monotonic() - started < timeout:
        status = _runpod_request("GET", f"{API_ROOT}/{endpoint}/status/{job_id}")
        state = status.get("status", "UNKNOWN")
        elapsed = int(time.monotonic() - started)
        progress(min(0.9, 0.04 + elapsed / timeout), desc=f"{state.lower()} ({elapsed}s)")
        if state == "COMPLETED":
            output = status.get("output") or {}
            encoded = output.get("audio_base64")
            if not encoded:
                raise gr.Error(f"Worker completed without audio: {output}")
            filename = f"{model.lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}.wav"
            path = OUTPUT_DIR / filename
            path.write_bytes(base64.b64decode(encoded))
            progress(1, desc="Complete")
            return str(path), f"Completed job `{job_id}` in {elapsed}s. Saved to `{path}`"
        if state in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            raise gr.Error(f"Runpod job {state.lower()}: {status.get('error', status)}")
        time.sleep(3)
    try:
        _runpod_request("POST", f"{API_ROOT}/{endpoint}/cancel/{job_id}")
    finally:
        raise gr.Error(f"Job {job_id} exceeded the 30-minute UI timeout and was cancelled")


with gr.Blocks(title="Serverless TTS Lab") as demo:
    gr.Markdown("# Serverless TTS Lab\nCompare Breeze TTS 2 and Higgs Audio V3 on separate scale-to-zero Runpod endpoints.")
    with gr.Row():
        model = gr.Radio(["Breeze TTS 2", "Higgs Audio V3"], value="Breeze TTS 2", label="Model")
        seed = gr.Number(value=42, precision=0, label="Seed")
    text = gr.Textbox(lines=7, label="Text", placeholder="Breeze: use (sigh). Higgs: use <|sfx:sigh|>Uh ...")
    with gr.Row():
        reference_audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference audio (optional)")
        reference_text = gr.Textbox(lines=5, label="Exact reference transcript")
    instruction = gr.Textbox(label="Breeze instruction", placeholder="Speak slowly with a restrained, serious tone.")
    with gr.Accordion("Generation settings", open=False):
        cfg_scale = gr.Slider(0.1, 8, value=4, step=0.1, label="Breeze CFG scale")
        temperature = gr.Slider(0.1, 2, value=0.8, step=0.05, label="Higgs temperature")
        top_k = gr.Slider(1, 200, value=50, step=1, label="Higgs top-k")
        max_new_tokens = gr.Slider(128, 4096, value=2048, step=128, label="Higgs max new tokens")
    submit = gr.Button("Generate", variant="primary")
    output_audio = gr.Audio(type="filepath", label="Generated WAV")
    status = gr.Markdown()
    submit.click(generate, [model, text, reference_audio, reference_text, instruction, cfg_scale,
                            temperature, top_k, max_new_tokens, seed], [output_audio, status])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(server_name="127.0.0.1", server_port=7860)

