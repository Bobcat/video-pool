# video-pool

`video-pool` is a local FastAPI service for video generation backends.

The service owns model lifecycle, queueing, GPU memory reporting, and video
artifact responses. It does not own browser UI. The Workbench can consume this
service in the same style as `image-pool`, `llm-pool`, `tts-pool`, and
`asr-pool`.

## Current Scope

- Model load and unload admin endpoints.
- Public loaded-model endpoint.
- GPU memory reporting through `nvidia-smi`.
- Text-to-video and image-to-video request schemas.
- A stub backend that writes deterministic artifact manifests.

WAN, LightX2V, ComfyUI, real MP4/WebM output, progress events, and persisted job
history are not implemented yet.

## API

```http
GET /healthz
GET /v1/models
GET /v1/admin/models
GET /v1/admin/gpu-memory
POST /v1/admin/models/{model}/load
POST /v1/admin/models/{model}/unload
POST /v1/videos/generations
POST /v1/videos/image-to-video
```

Video responses return artifact URLs instead of base64 payloads.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8014
```

