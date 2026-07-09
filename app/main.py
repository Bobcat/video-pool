from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import load_settings, resolve_artifact_root
from app.engine.common import ModelNotLoadedError, UnknownModelError, UnsupportedBackendError
from app.engine.router import VideoRouterEngine
from app.schemas import AdminLoadRequest, ImageToVideoRequest, VideoGenerationRequest


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    artifact_root = resolve_artifact_root(settings)
    artifact_root.mkdir(parents=True, exist_ok=True)
    engine = VideoRouterEngine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await engine.load_enabled_models()
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="video-pool", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine
    app.mount("/artifacts", StaticFiles(directory=artifact_root), name="artifacts")

    @app.exception_handler(UnknownModelError)
    async def _unknown_model_handler(_request, exc: UnknownModelError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": {"message": str(exc), "type": "unknown_model"}})

    @app.exception_handler(ModelNotLoadedError)
    async def _model_not_loaded_handler(_request, exc: ModelNotLoadedError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": {"message": str(exc), "type": "model_not_loaded"}})

    @app.exception_handler(UnsupportedBackendError)
    async def _unsupported_backend_handler(_request, exc: UnsupportedBackendError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "unsupported_backend"}})

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "bad_request"}})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return engine.public_models_payload()

    @app.get("/v1/admin/models")
    async def admin_models() -> dict:
        return engine.admin_models_payload()

    @app.get("/v1/admin/gpu-memory")
    async def gpu_memory() -> dict:
        return engine.gpu_memory_payload()

    @app.post("/v1/admin/models/{model_name:path}/load")
    async def load_model(model_name: str, load_request: AdminLoadRequest | None = None) -> dict:
        return await engine.load_model(model_name, load_request)

    @app.post("/v1/admin/models/{model_name:path}/unload")
    async def unload_model(model_name: str) -> dict:
        return await engine.unload_model(model_name)

    @app.post("/v1/videos/generations")
    async def video_generations(request: VideoGenerationRequest) -> dict:
        return (await engine.generate(request)).model_dump()

    @app.post("/v1/videos/image-to-video")
    async def image_to_video(request: ImageToVideoRequest) -> dict:
        return (await engine.image_to_video(request)).model_dump()

    return app


app = create_app()
