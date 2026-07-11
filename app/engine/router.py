from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import ModelSettings, Settings, resolve_artifact_root
from app.engine.common import (
    ModelNotLoadedError,
    ModelRuntimeState,
    UnknownModelError,
    UnsupportedBackendError,
    VideoJob,
    estimate_model_artifact_size_mib,
    load_constraints_for_backend,
    load_recommendations_for_backend,
    query_gpu_memory,
    query_primary_gpu_used_mib,
    release_loaded_torch_cuda_memory,
    reset_loaded_torch_cuda_context,
)
from app.engine.scheduler import LoadedModelExecutor, RuntimeScheduler
from app.engine.lightx2v_serve import LightX2VServeRuntime
from app.engine.stub import StubVideoRuntime
from app.engine.wan import WanTextToVideoRuntime
from app.schemas import AdminLoadRequest, ImageToVideoRequest, VideoData, VideoGenerationRequest, VideoResponse


_INPROCESS_CUDA_BACKENDS = {"diffusers_wan_t2v"}
_LIGHTX2V_DEFINITION_LOAD_DEFAULT_KEYS = {
    "lightx2v_text_len": "text_len",
    "lightx2v_sample_guide_scale": "sample_guide_scale",
    "lightx2v_sample_shift": "sample_shift",
    "lightx2v_enable_cfg": "enable_cfg",
    "lightx2v_denoising_step_list": "denoising_step_list",
    "lightx2v_cpu_offload": "cpu_offload",
    "lightx2v_offload_granularity": "offload_granularity",
    "lightx2v_t5_cpu_offload": "t5_cpu_offload",
    "lightx2v_clip_cpu_offload": "clip_cpu_offload",
    "lightx2v_vae_cpu_offload": "vae_cpu_offload",
    "lightx2v_lazy_load": "lazy_load",
    "lightx2v_unload_modules": "unload_modules",
    "lightx2v_self_attn_1_type": "self_attn_1_type",
    "lightx2v_cross_attn_1_type": "cross_attn_1_type",
    "lightx2v_cross_attn_2_type": "cross_attn_2_type",
    "lightx2v_rope_type": "rope_type",
}


class VideoRouterEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact_root = resolve_artifact_root(settings)
        self._scheduler = RuntimeScheduler()
        self._states: dict[str, ModelRuntimeState] = {}
        self._runtimes: dict[str, object] = {}
        for name, model_settings in settings.engine.models.items():
            self._states[name] = ModelRuntimeState(
                name=name,
                backend=model_settings.backend,
                enabled=model_settings.enabled,
                target_inflight=model_settings.target_inflight,
            )

    async def load_enabled_models(self) -> None:
        for model_name, model_settings in self.settings.engine.models.items():
            if model_settings.enabled:
                await self.load_model(model_name)

    async def close(self) -> None:
        await self._scheduler.close()
        for runtime in self._runtimes.values():
            self._close_runtime(runtime)
        self._runtimes.clear()
        release_loaded_torch_cuda_memory()
        for state in self._states.values():
            state.loaded = False

    async def load_model(self, model_name: str, load_request: AdminLoadRequest | None = None) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        state = self._states[model_name]
        load_override = _load_override_payload(load_request)
        if state.loaded:
            if load_override:
                raise ValueError("load overrides cannot be applied while the model is loaded")
            return self._state_payload(model_name)
        state.loading = True
        state.last_error = None
        gpu_used_before_mib = query_primary_gpu_used_mib()
        try:
            runtime_settings = _apply_load_override(model_settings, load_override)
            runtime = await asyncio.to_thread(self._create_runtime, model_name, runtime_settings)
            executor = LoadedModelExecutor(
                model_name=model_name,
                complete_fn=runtime.complete,
                target_inflight=runtime_settings.target_inflight,
            )
            await self._scheduler.register(model_name, executor)
            self._runtimes[model_name] = runtime
            state.loaded = True
            state.loaded_at = time.time()
            state.load_override = dict(load_override)
            gpu_used_after_mib = query_primary_gpu_used_mib()
            observed_vram_mib = _observed_vram_delta_mib(gpu_used_before_mib, gpu_used_after_mib)
            if observed_vram_mib is not None:
                state.observed_vram_mib = observed_vram_mib
            return self._state_payload(model_name)
        except Exception as exc:
            message = str(exc)
            state.last_error = message
            exc.__traceback__ = None
            exc.__context__ = None
            exc.__cause__ = None
            release_loaded_torch_cuda_memory()
            raise RuntimeError(message) from None
        finally:
            state.loading = False

    async def unload_model(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        await self._scheduler.unregister(model_name)
        runtime = self._runtimes.pop(model_name, None)
        self._close_runtime(runtime)
        del runtime
        state = self._states[model_name]
        state.loaded = False
        state.loaded_at = None
        state.load_override = {}
        release_loaded_torch_cuda_memory()
        if _uses_inprocess_cuda(model_settings.backend) and not self._has_loaded_inprocess_cuda_model():
            reset_loaded_torch_cuda_context()
        return self._state_payload(model_name)

    async def generate(self, request: VideoGenerationRequest) -> VideoResponse:
        job = VideoJob(
            operation="text_to_video",
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            n=request.n,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            num_frames=request.num_frames,
            quality=request.quality,
            seed=request.seed,
            metadata=dict(request.metadata),
        )
        return await self._complete(job, "video.generation")

    async def image_to_video(self, request: ImageToVideoRequest) -> VideoResponse:
        model_settings = self._model_settings(request.model)
        if len(request.images) > model_settings.max_images:
            raise ValueError(f"model accepts at most {model_settings.max_images} input images")
        job = VideoJob(
            operation="image_to_video",
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            n=request.n,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
            num_frames=request.num_frames,
            quality=request.quality,
            seed=request.seed,
            metadata=dict(request.metadata),
            images=tuple(request.images),
        )
        return await self._complete(job, "video.image_to_video")

    async def _complete(self, job: VideoJob, object_name: str) -> VideoResponse:
        model_settings = self._model_settings(job.model)
        if job.n > model_settings.max_output_videos:
            raise ValueError(f"model returns at most {model_settings.max_output_videos} videos")
        started_at = time.perf_counter()
        result = await self._scheduler.complete(job.model, job)
        metrics = dict(result.metrics)
        metrics.setdefault("pool_total_wall_ms", (time.perf_counter() - started_at) * 1000)
        return VideoResponse(
            id=f"vid-{uuid.uuid4().hex}",
            object=object_name,
            created=int(time.time()),
            model=job.model,
            data=[
                VideoData(
                    url=item.url,
                    path=item.path,
                    mime_type=item.mime_type,
                    revised_prompt=item.revised_prompt,
                    width=item.width,
                    height=item.height,
                    num_frames=item.num_frames,
                    fps=item.fps,
                    duration_seconds=item.duration_seconds,
                )
                for item in result.videos
            ],
            metrics=metrics,
        )

    def public_models_payload(self) -> dict[str, Any]:
        data = []
        for model_name in sorted(self.settings.engine.models):
            state = self._states[model_name]
            if state.loaded:
                data.append(self._public_model_payload(model_name))
        return {"object": "list", "data": data}

    def admin_models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [self._state_payload(model_name) for model_name in sorted(self.settings.engine.models)],
        }

    def gpu_memory_payload(self) -> dict[str, Any]:
        gpus, error = query_gpu_memory()
        payload: dict[str, Any] = {"gpus": gpus, "models": [], "error": error}
        for model_name, model_settings in sorted(self.settings.engine.models.items()):
            payload["models"].append(self._gpu_model_payload(model_name, model_settings))
        return payload

    def _model_settings(self, model_name: str) -> ModelSettings:
        try:
            return self.settings.engine.models[model_name]
        except KeyError as exc:
            raise UnknownModelError(f"unknown model: {model_name}") from exc

    def _create_runtime(self, model_name: str, model_settings: ModelSettings) -> object:
        if model_settings.backend == "stub":
            return StubVideoRuntime(model_name, model_settings, self.artifact_root)
        if model_settings.backend == "diffusers_wan_t2v":
            return WanTextToVideoRuntime(model_name, model_settings, self.artifact_root)
        if model_settings.backend == "lightx2v_serve":
            return LightX2VServeRuntime(model_name, model_settings, self.artifact_root)
        raise UnsupportedBackendError(f"unsupported backend: {model_settings.backend}")

    def _close_runtime(self, runtime: object | None) -> None:
        close = getattr(runtime, "close", None)
        if close is not None:
            close()

    def _has_loaded_inprocess_cuda_model(self) -> bool:
        return any(state.loaded and _uses_inprocess_cuda(state.backend) for state in self._states.values())

    def _public_model_payload(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        return {
            "id": model_name,
            "object": "model",
            "owned_by": "video-pool",
            "backend": model_settings.backend,
            "capabilities": self._capabilities_payload(model_settings),
            "recommended_steps": model_settings.recommended_steps,
            "recommended_guidance": model_settings.recommended_guidance,
            "generation_parameters": dict(model_settings.generation_parameters),
            "image_to_video_parameters": dict(model_settings.image_to_video_parameters),
        }

    def _state_payload(self, model_name: str) -> dict[str, Any]:
        model_settings = self._model_settings(model_name)
        state = self._states[model_name]
        scheduler_state = self._scheduler.snapshot(model_name) or {
            "target_inflight": state.target_inflight,
            "inflight": 0,
            "queued": 0,
        }
        vram_estimate_mib, vram_estimate_source = self._vram_estimate(model_name, model_settings)
        return {
            "id": model_name,
            "backend": model_settings.backend,
            "enabled": state.enabled,
            "loaded": state.loaded,
            "loading": state.loading,
            "loaded_at": state.loaded_at,
            "last_error": state.last_error,
            "scheduler": scheduler_state,
            "capabilities": self._capabilities_payload(model_settings),
            "model_path": model_settings.model_path,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
            "recommended_steps": model_settings.recommended_steps,
            "recommended_guidance": model_settings.recommended_guidance,
            "generation_parameters": dict(model_settings.generation_parameters),
            "image_to_video_parameters": dict(model_settings.image_to_video_parameters),
            "load_constraints": load_constraints_for_backend(model_settings.backend),
            "load_recommendations": load_recommendations_for_backend(model_settings.backend),
            "load_override": dict(state.load_override),
            "definition": _definition_payload(model_settings),
        }

    def _gpu_model_payload(self, model_name: str, model_settings: ModelSettings) -> dict[str, Any]:
        vram_estimate_mib, vram_estimate_source = self._vram_estimate(model_name, model_settings)
        return {
            "name": model_name,
            "backend": model_settings.backend,
            "loaded": self._states[model_name].loaded,
            "loading": self._states[model_name].loading,
            "vram_estimate_mib": vram_estimate_mib,
            "vram_estimate_source": vram_estimate_source,
        }

    def _capabilities_payload(self, model_settings: ModelSettings) -> dict[str, Any]:
        return {
            "input_modalities": list(model_settings.modalities),
            "output_modalities": list(model_settings.output_modalities),
            "tasks": list(model_settings.tasks),
            "max_images": model_settings.max_images,
            "max_output_videos": model_settings.max_output_videos,
        }

    def _vram_estimate(self, model_name: str, model_settings: ModelSettings) -> tuple[int | None, str]:
        state = self._states[model_name]
        if state.observed_vram_mib is not None:
            return state.observed_vram_mib, "observed_load_delta"
        if model_settings.vram_estimate_mib is not None:
            return model_settings.vram_estimate_mib, "configured"
        if state.artifact_size_mib is None:
            state.artifact_size_mib = estimate_model_artifact_size_mib(model_settings.model_path)
        if state.artifact_size_mib is not None:
            return state.artifact_size_mib, "artifact_size"
        return None, "unknown"


def _observed_vram_delta_mib(before_mib: int | None, after_mib: int | None) -> int | None:
    if before_mib is None or after_mib is None:
        return None
    delta = after_mib - before_mib
    if delta <= 0:
        return None
    return delta


def _uses_inprocess_cuda(backend: str) -> bool:
    return backend in _INPROCESS_CUDA_BACKENDS


def _load_override_payload(load_request: AdminLoadRequest | None) -> dict[str, Any]:
    if load_request is None:
        return {}
    fields_set = getattr(load_request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(load_request, "__fields_set__", set())
    return {field_name: getattr(load_request, field_name) for field_name in fields_set}


def _apply_load_override(model_settings: ModelSettings, load_override: dict[str, Any]) -> ModelSettings:
    if not load_override:
        return model_settings.model_copy(update={"load_override": {}})
    supported_keys = set(load_constraints_for_backend(model_settings.backend))
    unsupported = sorted(key for key in load_override if key not in supported_keys)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"unsupported load override for {model_settings.backend} backend: {names}")
    return model_settings.model_copy(update={"load_override": dict(load_override)})


def _definition_payload(model_settings: ModelSettings) -> dict[str, Any]:
    payload = {
        "model_path": model_settings.model_path,
        "backend": model_settings.backend,
        "enabled": model_settings.enabled,
        "target_inflight": model_settings.target_inflight,
        "recommended_steps": model_settings.recommended_steps,
        "recommended_guidance": model_settings.recommended_guidance,
        "generation_parameters": dict(model_settings.generation_parameters),
        "image_to_video_parameters": dict(model_settings.image_to_video_parameters),
    }
    if model_settings.backend == "lightx2v_serve":
        payload.update(_lightx2v_definition_load_defaults(model_settings))
    return payload


def _lightx2v_definition_load_defaults(model_settings: ModelSettings) -> dict[str, Any]:
    config_json = model_settings.lightx2v_config_json
    if not config_json:
        return {}
    try:
        config_payload = json.loads(Path(config_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(config_payload, dict):
        return {}
    return {
        load_key: config_payload[config_key]
        for load_key, config_key in _LIGHTX2V_DEFINITION_LOAD_DEFAULT_KEYS.items()
        if config_key in config_payload
    }
