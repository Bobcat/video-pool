from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from app.config import ModelSettings
from app.engine.common import GeneratedVideoPayload, VideoJob, VideoResult


_SIZE_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")


class StubVideoRuntime:
    def __init__(self, model_name: str, settings: ModelSettings, artifact_root: Path) -> None:
        self.model_name = model_name
        self.settings = settings
        self.artifact_root = artifact_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    async def complete(self, job: VideoJob) -> VideoResult:
        return await asyncio.to_thread(self._complete_sync, job)

    def _complete_sync(self, job: VideoJob) -> VideoResult:
        started_at = time.perf_counter()
        width, height = _parse_size(job.size)
        num_frames = job.num_frames or max(1, round(job.duration_seconds * job.fps))
        if job.operation == "image_to_video":
            _validate_image_to_video_inputs(job)

        videos: list[GeneratedVideoPayload] = []
        for index in range(job.n):
            artifact_name = f"{int(time.time())}-{job.request_id}-{index + 1}.json"
            artifact_path = self.artifact_root / artifact_name
            artifact = _artifact_payload(job, index, width, height, num_frames)
            artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            videos.append(
                GeneratedVideoPayload(
                    url=f"/artifacts/{artifact_name}",
                    path=str(artifact_path),
                    mime_type="application/json",
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    fps=job.fps,
                    duration_seconds=num_frames / job.fps,
                    revised_prompt=f"{job.prompt} [stub video {index + 1}]",
                )
            )

        return VideoResult(
            videos=videos,
            metrics={
                "backend": "stub",
                "backend_inference_wall_ms": (time.perf_counter() - started_at) * 1000,
                "operation": job.operation,
                "video_count": len(videos),
                "input_image_count": len(job.images),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": job.fps,
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


def _validate_image_to_video_inputs(job: VideoJob) -> None:
    if not job.images:
        raise ValueError("image-to-video requests require at least one input image")
    for image in job.images:
        if not image.data_url.startswith("data:image/"):
            raise ValueError("input images must be data URLs with an image media type")
        _header, separator, payload = image.data_url.partition(",")
        if not separator:
            raise ValueError("input image data URL is missing base64 payload")
        try:
            base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("input image data URL payload is not valid base64") from exc


def _artifact_payload(job: VideoJob, index: int, width: int, height: int, num_frames: int) -> dict[str, Any]:
    digest = hashlib.sha256(f"{job.model}:{job.operation}:{job.prompt}:{job.seed}:{index}".encode("utf-8")).hexdigest()
    return {
        "stub": True,
        "model": job.model,
        "operation": job.operation,
        "prompt": job.prompt,
        "seed": job.seed,
        "index": index,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": job.fps,
        "duration_seconds": num_frames / job.fps,
        "input_image_count": len(job.images),
        "content_hash": digest,
    }

