from __future__ import annotations

import asyncio
from pathlib import Path
import re
import time
from typing import Any

from app.config import ModelSettings
from app.engine.common import GeneratedVideoPayload, VideoJob, VideoResult, release_torch_cuda_memory

_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class WanTextToVideoRuntime:
    def __init__(self, model_name: str, settings: ModelSettings, artifact_root: Path) -> None:
        self.model_name = model_name
        self.settings = settings
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            from diffusers import AutoencoderKLWan, WanPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers Wan backend dependencies are not installed; install video-pool with the 'wan' extra"
            ) from exc

        model_path = str(settings.model_path or _DEFAULT_MODEL_ID)
        transformer_dtype = _torch_dtype(
            torch,
            _load_override_str(settings, "wan_transformer_dtype", "bfloat16"),
        )
        vae_dtype = _torch_dtype(torch, _load_override_str(settings, "wan_vae_dtype", "float32"))
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=vae_dtype)
        self._pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=transformer_dtype)
        if _load_override_bool(settings, "wan_sequential_cpu_offload", True):
            self._pipe.enable_sequential_cpu_offload()
        elif torch.cuda.is_available():
            self._pipe.to("cuda")
        if _load_override_bool(settings, "wan_vae_tiling", True) and hasattr(self._pipe.vae, "enable_tiling"):
            self._pipe.vae.enable_tiling()
        self._torch = torch

    async def complete(self, job: VideoJob) -> VideoResult:
        return await asyncio.to_thread(self._complete_sync, job)

    def close(self) -> None:
        maybe_free_model_hooks = getattr(self._pipe, "maybe_free_model_hooks", None)
        if maybe_free_model_hooks is not None:
            maybe_free_model_hooks()
        remove_all_hooks = getattr(self._pipe, "remove_all_hooks", None)
        if remove_all_hooks is not None:
            remove_all_hooks()
        del self._pipe
        release_torch_cuda_memory(self._torch)

    def _complete_sync(self, job: VideoJob) -> VideoResult:
        if job.operation != "text_to_video":
            raise ValueError("Wan text-to-video backend does not support image-to-video requests")

        started_at = time.perf_counter()
        width, height = _parse_size(job.size)
        num_frames = _num_frames_for_job(job)
        steps = _metadata_int(job.metadata, "steps", self.settings.recommended_steps or 30)
        guidance = _metadata_float(job.metadata, "guidance", self.settings.recommended_guidance or 5.0)
        guidance_2 = _metadata_optional_float(job.metadata, "guidance_2")
        max_sequence_length = _metadata_int(job.metadata, "max_sequence_length", 512)
        negative_prompt = _metadata_str(job.metadata, "negative_prompt", "")
        generator = None
        if job.seed is not None:
            generator = self._torch.Generator(device="cpu").manual_seed(job.seed)

        videos: list[GeneratedVideoPayload] = []
        for index in range(job.n):
            pipe_kwargs: dict[str, Any] = {
                "prompt": job.prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
                "max_sequence_length": max_sequence_length,
            }
            if negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
            if guidance_2 is not None:
                pipe_kwargs["guidance_scale_2"] = guidance_2
            output = self._pipe(**pipe_kwargs)
            frames = output.frames[0]
            artifact_name = f"{int(time.time())}-{job.request_id}-{index + 1}.mp4"
            artifact_path = self.artifact_root / artifact_name
            _export_to_video(frames, artifact_path, job.fps)
            videos.append(
                GeneratedVideoPayload(
                    url=f"/artifacts/{artifact_name}",
                    path=str(artifact_path),
                    mime_type="video/mp4",
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=job.fps,
                    duration_seconds=num_frames / job.fps,
                    revised_prompt=job.prompt,
                )
            )

        return VideoResult(
            videos=videos,
            metrics={
                "backend": "diffusers_wan_t2v",
                "backend_inference_wall_ms": (time.perf_counter() - started_at) * 1000,
                "operation": job.operation,
                "video_count": len(videos),
                "input_image_count": 0,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": job.fps,
                "steps": steps,
                "guidance": guidance,
                "guidance_2": guidance_2,
                "max_sequence_length": max_sequence_length,
            },
        )


def _parse_size(size: str) -> tuple[int, int]:
    if size == "auto":
        return (832, 480)
    match = _SIZE_RE.match(size)
    if match is None:
        raise ValueError("size must be 'auto' or '<width>x<height>'")
    width = int(match.group(1))
    height = int(match.group(2))
    if width < 64 or height < 64 or width > 1920 or height > 1920:
        raise ValueError("size must be between 64x64 and 1920x1920")
    return (width, height)


def _num_frames_for_job(job: VideoJob) -> int:
    if job.num_frames is not None:
        num_frames = job.num_frames
        if (num_frames - 1) % 4 != 0:
            raise ValueError("Wan requires num_frames to be 4k + 1")
        return num_frames

    raw_frames = max(1, round(job.duration_seconds * job.fps))
    frame_group = max(0, round((raw_frames - 1) / 4))
    return (frame_group * 4) + 1


def _metadata_int(metadata: dict[str, Any], key: str, fallback: int) -> int:
    try:
        value = int(metadata.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(1, value)


def _metadata_float(metadata: dict[str, Any], key: str, fallback: float) -> float:
    try:
        value = float(metadata.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(0.0, value)


def _metadata_optional_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _metadata_str(metadata: dict[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def _load_override_str(settings: ModelSettings, key: str, fallback: str) -> str:
    value = settings.load_override.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def _load_override_bool(settings: ModelSettings, key: str, fallback: bool) -> bool:
    value = settings.load_override.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _torch_dtype(torch_module: Any, name: str) -> Any:
    normalized = str(name or "").strip().lower()
    if normalized == "float16":
        return torch_module.float16
    if normalized == "bfloat16":
        return torch_module.bfloat16
    if normalized == "float32":
        return torch_module.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def _export_to_video(frames: list[Any], artifact_path: Path, fps: int) -> None:
    from diffusers.utils import export_to_video

    export_to_video(frames, str(artifact_path), fps=fps)
