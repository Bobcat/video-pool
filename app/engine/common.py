from __future__ import annotations

import gc
import ctypes
import ctypes.util
from pathlib import Path
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas import VideoInput


class UnknownModelError(RuntimeError):
    pass


class ModelNotLoadedError(RuntimeError):
    pass


class UnsupportedBackendError(RuntimeError):
    pass


def release_torch_cuda_memory(torch_module: Any) -> None:
    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    cuda.empty_cache()
    ipc_collect = getattr(cuda, "ipc_collect", None)
    if ipc_collect is not None:
        ipc_collect()


def release_loaded_torch_cuda_memory() -> None:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        gc.collect()
        return
    release_torch_cuda_memory(torch_module)


def reset_loaded_torch_cuda_context() -> bool:
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return False
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available() or not cuda.is_initialized():
        return False
    release_torch_cuda_memory(torch_module)
    cudart_path = ctypes.util.find_library("cudart") or "libcudart.so"
    try:
        cudart = ctypes.CDLL(cudart_path)
    except OSError:
        return False
    cuda_device_reset = getattr(cudart, "cudaDeviceReset", None)
    if cuda_device_reset is None:
        return False
    cuda_device_reset.restype = ctypes.c_int
    return cuda_device_reset() == 0


def query_gpu_memory() -> tuple[list[dict[str, object]], str | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return [], "nvidia-smi not found"
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        return [], message or "nvidia-smi failed"
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out"

    gpus: list[dict[str, object]] = []
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            used_mib = int(parts[2])
            total_mib = int(parts[3])
            free_mib = int(parts[4]) if len(parts) > 4 else max(0, total_mib - used_mib)
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "used_mib": used_mib,
                "total_mib": total_mib,
                "free_mib": free_mib,
                "used_over_total": f"{used_mib}MiB / {total_mib}MiB",
            }
        )
    return gpus, None


def query_primary_gpu_used_mib() -> int | None:
    gpus, _ = query_gpu_memory()
    if not gpus:
        return None
    used = gpus[0].get("used_mib")
    if isinstance(used, int):
        return used
    return None


def estimate_model_artifact_size_mib(model_path: str | None) -> int | None:
    path_value = str(model_path or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    try:
        if path.is_file():
            total_bytes = path.stat().st_size
        elif path.is_dir():
            total_bytes = sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
        else:
            return None
    except OSError:
        return None
    if total_bytes <= 0:
        return None
    return max(1, int(total_bytes / (1024 * 1024)))


@dataclass(slots=True)
class GeneratedVideoPayload:
    url: str
    path: str
    mime_type: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    revised_prompt: str | None = None


@dataclass(slots=True)
class VideoResult:
    videos: list[GeneratedVideoPayload]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VideoJob:
    operation: Literal["text_to_video", "image_to_video"]
    model: str
    prompt: str
    size: str
    n: int
    duration_seconds: float
    fps: int
    num_frames: int | None
    quality: str
    seed: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    images: tuple[VideoInput, ...] = ()
    request_id: str = field(default_factory=lambda: f"vidreq-{uuid.uuid4().hex}")


@dataclass(slots=True)
class ModelRuntimeState:
    name: str
    backend: str
    enabled: bool
    loaded: bool = False
    loading: bool = False
    target_inflight: int = 1
    loaded_at: float | None = None
    last_error: str | None = None
    observed_vram_mib: int | None = None
    artifact_size_mib: int | None = None
    load_override: dict[str, Any] = field(default_factory=dict)


def load_constraints_for_backend(backend: str) -> dict[str, Any]:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "diffusers_wan_t2v":
        return {
            "wan_transformer_dtype": {
                "kind": "enum",
                "label": "transformer dtype",
                "default": "bfloat16",
                "allowed_values": ["bfloat16", "float16"],
            },
            "wan_vae_dtype": {
                "kind": "enum",
                "label": "VAE dtype",
                "default": "float32",
                "allowed_values": ["float32", "bfloat16", "float16"],
            },
            "wan_sequential_cpu_offload": {
                "kind": "boolean",
                "label": "sequential CPU offload",
                "default": True,
            },
            "wan_vae_tiling": {
                "kind": "boolean",
                "label": "VAE tiling",
                "default": True,
            },
        }
    if normalized_backend == "lightx2v_serve":
        return {
            "lightx2v_text_len": {
                "kind": "integer",
                "label": "text_len",
                "default": 512,
                "minimum": 64,
                "maximum": 1024,
                "step": 64,
            },
            "lightx2v_sample_guide_scale": {
                "kind": "number",
                "label": "sample_guide_scale",
                "default": 1.0,
                "minimum": 0.0,
                "maximum": 20.0,
                "step": 0.1,
            },
            "lightx2v_sample_shift": {
                "kind": "number",
                "label": "sample_shift",
                "default": 5.0,
                "minimum": 0.0,
                "maximum": 20.0,
                "step": 0.1,
            },
            "lightx2v_enable_cfg": {
                "kind": "boolean",
                "label": "enable_cfg",
                "default": False,
            },
            "lightx2v_denoising_step_list": {
                "kind": "integer_list",
                "label": "denoising_step_list",
                "default": [1000, 750, 500, 250],
            },
            "lightx2v_cpu_offload": {
                "kind": "boolean",
                "label": "cpu_offload",
                "default": True,
            },
            "lightx2v_offload_granularity": {
                "kind": "enum",
                "label": "offload_granularity",
                "default": "block",
                "allowed_values": ["block", "model"],
            },
            "lightx2v_t5_cpu_offload": {
                "kind": "boolean",
                "label": "t5_cpu_offload",
                "default": True,
            },
            "lightx2v_vae_cpu_offload": {
                "kind": "boolean",
                "label": "vae_cpu_offload",
                "default": False,
            },
            "lightx2v_self_attn_1_type": {
                "kind": "enum",
                "label": "self_attn_1_type",
                "default": "torch_sdpa",
                "allowed_values": ["torch_sdpa"],
            },
            "lightx2v_cross_attn_1_type": {
                "kind": "enum",
                "label": "cross_attn_1_type",
                "default": "torch_sdpa",
                "allowed_values": ["torch_sdpa"],
            },
            "lightx2v_cross_attn_2_type": {
                "kind": "enum",
                "label": "cross_attn_2_type",
                "default": "torch_sdpa",
                "allowed_values": ["torch_sdpa"],
            },
            "lightx2v_rope_type": {
                "kind": "enum",
                "label": "rope_type",
                "default": "torch",
                "allowed_values": ["torch"],
            },
        }
    return {}


def load_recommendations_for_backend(_backend: str) -> dict[str, Any]:
    return {}
