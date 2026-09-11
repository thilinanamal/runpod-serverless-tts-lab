# Runpod Serverless TTS Lab

Two independent, scale-to-zero Runpod workers and a local Gradio client for comparing:

- BreezeBlue/Breeze-TTS-2
- bosonai/higgs-audio-v3-tts-4b

Each endpoint is configured for one GPU per worker and up to two workers. Reference audio and generated WAV data travel in the job payload, so no persistent Runpod network volume is required.

> Both model-weight licenses restrict self-hosted use. Review and accept the upstream licenses before building or generating audio. Only clone voices you have permission to use.

## Build images

The GitHub Actions workflow builds both Linux/amd64 images and publishes them to GHCR. In the repository **Actions** tab, run **Build serverless workers**, or push changes under `workers/`.

Images:

```text
ghcr.io/OWNER/REPOSITORY/breeze-worker:latest
ghcr.io/OWNER/REPOSITORY/higgs-worker:latest
```

The packages include the weights in the images to avoid paying to download them during every cold start. If Hugging Face requires authentication after license acceptance, create a repository Actions secret named `HF_TOKEN` and rerun the workflow.

## Local UI

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ui/requirements.txt
cp .env.example .env
# Fill RUNPOD_API_KEY and both endpoint IDs.
python ui/app.py
```

Open http://127.0.0.1:7860. The UI submits asynchronous jobs, polls until completion, and creates a local WAV file under `outputs/`. Gradio also presents a download button.

## Worker input

Common fields are `text`, optional `reference_audio` (raw base64 or a data URL), `reference_text`, and generation settings. See `examples/` for complete payloads.
