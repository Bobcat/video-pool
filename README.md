# video-pool

`video-pool` is a local FastAPI service for video generation backends.

The service owns model lifecycle, request queueing, GPU memory reporting, and
video artifacts. Browser UI lives outside this repository. The Workbench can use
this service in the same style as `image-pool`, `llm-pool`, and `tts-pool`.

## Index

- [What It Does](#what-it-does)
- [Repository Role](#repository-role)
- [Related Repositories](#related-repositories)
- [Code Map](#code-map)
- [API Surface](#api-surface)
- [Runtime Model](#runtime-model)
- [Configuration](#configuration)
- [Development](#development)
- [Tests](#tests)
- [Deployment Notes](#deployment-notes)
- [License](#license)

## What It Does

- Lists configured and loaded video models.
- Loads and unloads models at runtime.
- Reports GPU memory through `nvidia-smi`.
- Runs text-to-video requests.
- Defines an image-to-video API shape for compatible backends.
- Returns generated video artifact URLs.
- Exposes model-specific generation parameters to clients.
- Exposes runtime load controls for backends that support them.

Implemented backends:

- `stub`: deterministic test backend.
- `diffusers_wan_t2v`: in-process Diffusers WAN text-to-video backend.
- `lightx2v_serve`: managed LightX2V server subprocess backend.

## Repository Role

This repository owns the video pool API and runtime adapters.

It does not own:

- browser UI
- model downloads
- model training
- shared workbench navigation
- long-term artifact management
- a multi-user job history system

Generated artifacts are written under the configured artifact root and served
from `/artifacts/...`.

## Related Repositories

- `llm-workbench`: browser UI and API proxy for the video-pool views.
- `image-pool`: similar runtime-admin pattern for image generation and edit
  models.
- `llm-pool` and `tts-pool`: related model pool services with similar lifecycle
  goals.
- LightX2V runtime checkout: required only for the `lightx2v_serve` backend.

## Code Map

- `app/main.py`: FastAPI app, routes, exception handlers, artifact mount.
- `app/config.py`: settings schema and `settings.json + local.json` merge.
- `app/schemas.py`: public request/response schemas.
- `app/engine/router.py`: model lifecycle, scheduling, runtime selection.
- `app/engine/scheduler.py`: per-model queue and inflight control.
- `app/engine/stub.py`: deterministic stub backend.
- `app/engine/wan.py`: Diffusers WAN text-to-video runtime.
- `app/engine/lightx2v_serve.py`: LightX2V subprocess runtime.
- `app/engine/common.py`: shared runtime state, GPU memory, parameter schemas.
- `config/settings.json`: default service and model configuration.
- `config/lightx2v/`: LightX2V backend config templates.
- `docs/runtime-admin-api.md`: detailed runtime admin API notes.
- `tests/`: config, API, router, and backend adapter tests.

## API Surface

Core endpoints:

```http
GET /healthz
GET /v1/models
GET /v1/admin/models
GET /v1/admin/gpu-memory
POST /v1/admin/models/{model_name}/load
POST /v1/admin/models/{model_name}/unload
POST /v1/videos/generations
POST /v1/videos/image-to-video
```

`GET /v1/models` returns loaded models only. `GET /v1/admin/models` returns all
configured models with runtime state.

Video responses return artifact URLs, not base64 payloads. The API mounts the
artifact root at `/artifacts`.

See [docs/runtime-admin-api.md](docs/runtime-admin-api.md) for the current admin
payloads and parameter schemas.

## Runtime Model

Each configured model has runtime state:

- `loaded`
- `loading`
- `last_error`
- scheduler state
- observed or configured VRAM estimate

Model load creates a backend runtime and registers it with the scheduler. Heavy
runtime creation runs outside the FastAPI event loop, so admin polling can keep
responding while a model loads.

Model unload unregisters the scheduler executor, closes the runtime, and releases
CUDA memory where the backend supports it.

The LightX2V backend starts and owns a backend server subprocess. It writes
backend logs and generated MP4 artifacts under the artifact root.

## Configuration

Default configuration lives in `config/settings.json`.

`config/local.json` is optional. When present, it is deep-merged over
`settings.json`. Use it for local model paths, enabled models, ports, and backend
runtime choices.

Important config areas:

- `service.host`
- `service.port`
- `service.artifact_root`
- `engine.models`
- per-model `generation_parameters`
- per-model `image_to_video_parameters`
- LightX2V subprocess settings for `lightx2v_serve`

The `data/` and `tmp/` directories are ignored by git.

## Development

Create a virtual environment and install the base service plus test tools:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Install WAN backend dependencies when working on the in-process Diffusers WAN
backend:

```bash
.venv/bin/pip install -e '.[dev,wan]'
```

Run the service:

```bash
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8014
```

## Tests

Run the test suite:

```bash
.venv/bin/pytest
```

Targeted checks used during development:

```bash
.venv/bin/pytest tests/test_config.py tests/test_api.py tests/test_router.py tests/test_engine_lightx2v_serve.py -q
```

## Deployment Notes

This repo does not include a production deployment stack.

Run it as a local service behind the Workbench or another trusted local client.
The service is designed for runtime model control on a single machine. It does
not implement authentication, quotas, or multi-user scheduling.

## License

No license file is currently included.
