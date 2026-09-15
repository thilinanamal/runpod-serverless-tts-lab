# Runpod Serverless TTS Lab

Three independent, scale-to-zero Runpod workers and a local Gradio client for comparing:

- BreezeBlue/Breeze-TTS-2
- bosonai/higgs-audio-v3-tts-4b
- openbmb/VoxCPM2 (bfloat16, 48 kHz output)

Each endpoint is configured for one GPU per worker and up to two workers. Reference audio and generated WAV data travel in the job payload, so no persistent Runpod network volume is required.

> The Breeze and Higgs model-weight licenses restrict self-hosted use. Review and accept those upstream licenses before building or generating audio. VoxCPM2 is Apache-2.0 and ungated. Only clone voices you have permission to use.

## Build images

The GitHub Actions workflow builds all three Linux/amd64 images and publishes them to GHCR. In the repository **Actions** tab, run **Build serverless workers**, or push changes under `workers/`.

Images:

```text
ghcr.io/OWNER/REPOSITORY/breeze-worker:latest
ghcr.io/OWNER/REPOSITORY/higgs-worker:latest
ghcr.io/OWNER/REPOSITORY/voxcpm-worker:latest
```

The packages include the weights in the images to avoid paying to download them during every cold start. If Hugging Face requires authentication after license acceptance, create a repository Actions secret named `HF_TOKEN` and rerun the workflow.

## Local UI

```bash
cp .env.example .env
# Fill RUNPOD_API_KEY and all three endpoint IDs.
chmod +x run-ui.sh
./run-ui.sh
```

Open http://127.0.0.1:7860. The UI submits asynchronous jobs, polls until completion, and creates a local WAV file under `outputs/`. Gradio also presents a download button.

## Worker input

Common fields are `text`, optional `reference_audio` (raw base64 or a data URL), `reference_text`, and generation settings. See `examples/` for complete payloads.

## Deploying a worker

Runpod's management API **cannot create a repo-linked endpoint** — the GitHub build
integration is console-only. `create-endpoint` accepts an `image` or a `templateId`
and nothing else, so the image has to exist before any API call can use it. Two ways
to get one:

**A. Let Runpod build it (no GHCR, no Actions).** In the Runpod console:
Serverless → New Endpoint → *GitHub repo* → this repository → branch → set the
Dockerfile path (e.g. `workers/voxcpm/Dockerfile`) and the build context to the
repository root. Runpod builds and pushes to
`registry.runpod.net/<user>-<repo>-<branch>-<dockerfile-path>:<short-sha>` and
rebuilds on every push to that branch. Note the `voxcpm` Dockerfile is deliberately
free of BuildKit secret mounts so this path works; `breeze` and `higgs` use
`--mount=type=secret` for their gated weights and need path B.

**B. Build in GitHub Actions, then deploy the image.** Run **Build serverless
workers**, make the GHCR package public (or add a Runpod registry credential), then
create the endpoint from `ghcr.io/OWNER/REPOSITORY/<worker>-worker:<sha>`. Pin the
SHA rather than `:latest` so a tag move cannot silently change what is served.

Endpoint settings that suit these workers: queue type, one GPU per worker, min 0
workers (free when idle), max 1–2, FlashBoot on, and a container disk large enough
for the baked weights — 60 GB for VoxCPM2, whose image carries ~5 GB of weights on
top of the CUDA PyTorch base.

## VoxCPM2 notes

- **Voice design without a reference.** Put a description in parentheses at the start of `text`, then the sentence to speak: `(A young woman, gentle and sweet voice)Hello there!`
- **Cloning modes.** Unlike the other two workers, `reference_audio` may be sent on its own — that is controllable cloning. Adding the exact `reference_text` switches the model to higher-fidelity ultimate cloning. Sending `reference_text` alone is rejected.
- **Output is 48 kHz** via AudioVAE V2 super-resolution; reference audio is resampled to 16 kHz on the way in.
- **Cold starts.** `VOXCPM_OPTIMIZE` defaults to `0`. Set it to `1` on the endpoint to enable `torch.compile`, which raises throughput but adds several minutes to every cold start because nothing is cached across scale-to-zero cycles.
- **Denoiser off.** The worker loads with `load_denoiser=False`, so the ZipEnhancer weights are not pulled at runtime and `denoise` is unavailable.
- VRAM is roughly 8 GB, so a 24 GB GPU class is ample.
