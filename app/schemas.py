from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    data_url: str


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "832x480"
    duration_seconds: float = Field(default=5.0, gt=0, le=60)
    fps: int = Field(default=16, ge=1, le=60)
    num_frames: int | None = Field(default=None, ge=1, le=600)
    quality: Literal["auto", "low", "medium", "high"] = "auto"
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageToVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(min_length=1)
    images: list[VideoInput] = Field(default_factory=list)
    n: int = Field(default=1, ge=1, le=4)
    size: str = "832x480"
    duration_seconds: float = Field(default=5.0, gt=0, le=60)
    fps: int = Field(default=16, ge=1, le=60)
    num_frames: int | None = Field(default=None, ge=1, le=600)
    quality: Literal["auto", "low", "medium", "high"] = "auto"
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    path: str
    mime_type: str
    revised_prompt: str | None = None
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float


class VideoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["video.generation", "video.image_to_video"]
    created: int
    model: str
    status: Literal["completed"] = "completed"
    data: list[VideoData]
    metrics: dict[str, Any] = Field(default_factory=dict)

