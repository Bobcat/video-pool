from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import ModelSettings
from app.engine.common import GeneratedVideoPayload, VideoJob, VideoResult


_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_LIGHTX2V_CONFIG_OVERRIDE_KEYS = {
    "lightx2v_text_len": "text_len",
    "lightx2v_sample_guide_scale": "sample_guide_scale",
    "lightx2v_sample_shift": "sample_shift",
    "lightx2v_enable_cfg": "enable_cfg",
    "lightx2v_denoising_step_list": "denoising_step_list",
    "lightx2v_cpu_offload": "cpu_offload",
    "lightx2v_offload_granularity": "offload_granularity",
    "lightx2v_t5_cpu_offload": "t5_cpu_offload",
    "lightx2v_vae_cpu_offload": "vae_cpu_offload",
    "lightx2v_self_attn_1_type": "self_attn_1_type",
    "lightx2v_cross_attn_1_type": "cross_attn_1_type",
    "lightx2v_cross_attn_2_type": "cross_attn_2_type",
    "lightx2v_rope_type": "rope_type",
}


class LightX2VServeRuntime:
    def __init__(self, model_name: str, settings: ModelSettings, artifact_root: Path) -> None:
        self.model_name = model_name
        self.settings = settings
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        self.host = settings.lightx2v_host
        self.port = settings.lightx2v_port or _pick_free_port(self.host)
        self.base_url = f"http://{self.host}:{self.port}"
        self.log_path = self.artifact_root / f"{self.model_name}.lightx2v.log"
        self._log_file = None
        self.process = self._start_process(self._command())
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    async def complete(self, job: VideoJob) -> VideoResult:
        return await asyncio.to_thread(self._complete_sync, job)

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=self.settings.lightx2v_stop_timeout_s)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
        finally:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None

    def _complete_sync(self, job: VideoJob) -> VideoResult:
        if job.operation == "image_to_video" and not job.images:
            raise ValueError("LightX2V image-to-video requests require an input image")
        if job.n != 1:
            raise ValueError("LightX2V serve backend currently supports n=1")

        started_at = time.perf_counter()
        width, height = _parse_size(job.size)
        num_frames = _num_frames_for_job(job)
        steps = _metadata_int(job.metadata, "steps", self.settings.recommended_steps or 4)
        negative_prompt = _metadata_str(job.metadata, "negative_prompt", "")
        use_prompt_enhancer = _metadata_bool(job.metadata, "use_prompt_enhancer", False)
        lora_name = _metadata_optional_str(job.metadata, "lora_name")
        lora_strength = _metadata_optional_float(job.metadata, "lora_strength")
        payload: dict[str, Any] = {
            "prompt": job.prompt,
            "negative_prompt": negative_prompt,
            "use_prompt_enhancer": use_prompt_enhancer,
            "infer_steps": steps,
            "target_shape": [height, width],
            "target_video_length": num_frames,
            "target_fps": job.fps,
        }
        if lora_name is not None:
            payload["lora_name"] = lora_name
        if lora_strength is not None:
            payload["lora_strength"] = lora_strength
        if job.seed is not None:
            payload["seed"] = job.seed
        if job.operation == "image_to_video":
            payload["image_path"] = _image_data_url_to_base64(job.images[0].data_url)

        task_payload = self._post_json("/v1/tasks/video/", payload)
        task_id = _required_str(task_payload, "task_id")
        self._wait_for_task(task_id)
        video_bytes = self._get_bytes(f"/v1/tasks/{task_id}/result")

        artifact_name = f"{int(time.time())}-{job.request_id}-1.mp4"
        artifact_path = self.artifact_root / artifact_name
        artifact_path.write_bytes(video_bytes)

        return VideoResult(
            videos=[
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
            ],
            metrics={
                "backend": "lightx2v_serve",
                "backend_inference_wall_ms": (time.perf_counter() - started_at) * 1000,
                "operation": job.operation,
                "video_count": 1,
                "input_image_count": len(job.images),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": job.fps,
                "steps": steps,
                "use_prompt_enhancer": use_prompt_enhancer,
                "lightx2v_task_id": task_id,
            },
        )

    def _command(self) -> list[str]:
        model_path = _required_setting(self.settings.model_path, "model_path")
        model_cls = _required_setting(self.settings.lightx2v_model_cls, "lightx2v_model_cls")
        task = _required_setting(self.settings.lightx2v_task, "lightx2v_task")
        config_json = self._effective_config_json_path(
            _required_setting(self.settings.lightx2v_config_json, "lightx2v_config_json")
        )
        command = [
            self.settings.lightx2v_python,
            "-m",
            "lightx2v.server",
            "--model_cls",
            model_cls,
            "--task",
            task,
            "--model_path",
            model_path,
            "--config_json",
            config_json,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--max_queue_size",
            str(self.settings.lightx2v_max_queue_size),
        ]
        if self.settings.lightx2v_lora_dir is not None:
            command.extend(["--lora_dir", self.settings.lightx2v_lora_dir])
        if self.settings.lightx2v_metric_port is not None:
            command.extend(["--metric_port", str(self.settings.lightx2v_metric_port)])
        command.extend(self.settings.lightx2v_extra_args)
        return command

    def _effective_config_json_path(self, config_json: str) -> str:
        overrides = _lightx2v_config_overrides(self.settings.load_override)
        if not overrides:
            return config_json
        source_path = Path(config_json)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"could not read LightX2V config: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LightX2V config is not valid JSON: {source_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"LightX2V config must contain a JSON object: {source_path}")
        payload.update(overrides)
        effective_path = self.artifact_root / f"{_safe_filename(self.model_name)}.lightx2v.effective.json"
        effective_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return str(effective_path)

    def _start_process(self, command: list[str]) -> subprocess.Popen:
        try:
            self._log_file = self.log_path.open("a", encoding="utf-8")
            self._log_file.write(f"\n--- LightX2V start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ---\n")
            self._log_file.write(" ".join(command) + "\n")
            self._log_file.flush()
            return subprocess.Popen(
                command,
                env=self._subprocess_env(),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as exc:
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            raise RuntimeError(f"LightX2V python executable not found: {command[0]}") from exc

    def _subprocess_env(self) -> dict[str, str] | None:
        if not self.settings.lightx2v_library_path and not self.settings.lightx2v_env and self.settings.lightx2v_cache_dir is None:
            return None
        env = os.environ.copy()
        if self.settings.lightx2v_library_path:
            existing_library_path = env.get("LD_LIBRARY_PATH", "")
            path_items = [*self.settings.lightx2v_library_path]
            if existing_library_path:
                path_items.append(existing_library_path)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(path_items)
        if self.settings.lightx2v_cache_dir is not None:
            env["LIGHTX2V_CACHE_DIR"] = self.settings.lightx2v_cache_dir
        env.update(self.settings.lightx2v_env)
        return env

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.lightx2v_start_timeout_s
        last_error = "service status endpoint did not respond"
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(f"LightX2V server exited during startup with code {return_code}")
            try:
                self._get_json("/v1/service/status", timeout_s=1.0)
                return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise RuntimeError(
            f"LightX2V server did not become ready within {self.settings.lightx2v_start_timeout_s:g}s: {last_error}"
        )

    def _wait_for_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.lightx2v_timeout_s
        while time.monotonic() < deadline:
            payload = self._get_json(f"/v1/tasks/{task_id}/status")
            status = str(payload.get("status") or "")
            if status == "completed":
                return payload
            if status in {"failed", "cancelled"}:
                detail = payload.get("error") or f"task ended with status={status}"
                raise RuntimeError(str(detail))
            time.sleep(self.settings.lightx2v_poll_interval_s)
        self._delete(f"/v1/tasks/{task_id}")
        raise TimeoutError(f"LightX2V task {task_id} timed out after {self.settings.lightx2v_timeout_s:g}s")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _read_json_response(request, timeout_s=self.settings.lightx2v_timeout_s)

    def _get_json(self, path: str, *, timeout_s: float | None = None) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", method="GET")
        return _read_json_response(request, timeout_s=timeout_s or self.settings.lightx2v_timeout_s)

    def _get_bytes(self, path: str) -> bytes:
        request = Request(f"{self.base_url}{path}", method="GET")
        with urlopen(request, timeout=self.settings.lightx2v_timeout_s) as response:
            return response.read()

    def _delete(self, path: str) -> None:
        request = Request(f"{self.base_url}{path}", method="DELETE")
        try:
            with urlopen(request, timeout=5.0) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError):
            return


def _read_json_response(request: Request, *, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except HTTPError as exc:
        raw = exc.read()
        message = raw.decode("utf-8", errors="replace").strip()
        raise RuntimeError(message or f"LightX2V HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("LightX2V returned a non-object JSON response")
    return payload


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
        return job.num_frames
    return max(1, round(job.duration_seconds * job.fps))


def _metadata_int(metadata: dict[str, Any], key: str, fallback: int) -> int:
    try:
        value = int(metadata.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(1, value)


def _metadata_str(metadata: dict[str, Any], key: str, fallback: str) -> str:
    value = metadata.get(key, fallback)
    if value is None:
        return fallback
    return str(value)


def _metadata_optional_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _metadata_optional_float(metadata: dict[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _metadata_bool(metadata: dict[str, Any], key: str, fallback: bool) -> bool:
    value = metadata.get(key, fallback)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _lightx2v_config_overrides(load_override: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for load_key, config_key in _LIGHTX2V_CONFIG_OVERRIDE_KEYS.items():
        if load_key in load_override:
            overrides[config_key] = load_override[load_key]
    return overrides


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def _image_data_url_to_base64(data_url: str) -> str:
    if not data_url.startswith("data:image/"):
        raise ValueError("input image must be a data URL with an image media type")
    marker = ";base64,"
    if marker not in data_url:
        raise ValueError("input image data URL is missing base64 payload")
    payload = data_url.split(marker, 1)[1]
    try:
        base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError("input image data URL payload is not valid base64") from exc
    return payload


def _required_setting(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required for lightx2v_serve models")
    return str(value)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"LightX2V response did not include {key}")
    return value


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
