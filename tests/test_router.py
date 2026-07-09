import unittest
from unittest import mock

from app.config import EngineSettings, ModelSettings, ServiceSettings, Settings
from app.engine.common import GeneratedVideoPayload, VideoJob, VideoResult
from app.engine import router as router_module
from app.engine.router import VideoRouterEngine
from app.schemas import AdminLoadRequest


class ClosableRuntime:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def complete(self, job: VideoJob) -> VideoResult:
        return VideoResult(
            videos=[
                GeneratedVideoPayload(
                    url="/artifacts/test.json",
                    path="/tmp/test.json",
                    mime_type="application/json",
                    width=320,
                    height=192,
                    num_frames=1,
                    fps=1,
                    duration_seconds=1.0,
                )
            ]
        )


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_unload_closes_runtime(self):
        settings = Settings(
            service=ServiceSettings(artifact_root="/tmp/video-pool-test"),
            engine=EngineSettings(
                models={
                    "test-video": ModelSettings(
                        backend="stub",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                    )
                }
            ),
        )
        engine = VideoRouterEngine(settings)
        runtime = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", return_value=runtime),
            mock.patch("app.engine.router.release_loaded_torch_cuda_memory") as release_memory,
            mock.patch("app.engine.router.reset_loaded_torch_cuda_context") as reset_cuda,
        ):
            await engine.load_model("test-video")
            payload = await engine.unload_model("test-video")

        self.assertTrue(runtime.closed)
        self.assertFalse(payload["loaded"])
        release_memory.assert_called_once()
        reset_cuda.assert_not_called()

    async def test_unload_resets_cuda_after_last_inprocess_cuda_model(self):
        settings = Settings(
            service=ServiceSettings(artifact_root="/tmp/video-pool-test"),
            engine=EngineSettings(
                models={
                    "wan-video": ModelSettings(
                        backend="diffusers_wan_t2v",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                    )
                }
            ),
        )
        engine = VideoRouterEngine(settings)
        runtime = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", return_value=runtime),
            mock.patch("app.engine.router.release_loaded_torch_cuda_memory"),
            mock.patch("app.engine.router.reset_loaded_torch_cuda_context") as reset_cuda,
        ):
            await engine.load_model("wan-video")
            payload = await engine.unload_model("wan-video")

        self.assertTrue(runtime.closed)
        self.assertFalse(payload["loaded"])
        reset_cuda.assert_called_once()

    async def test_unload_keeps_cuda_context_when_another_inprocess_cuda_model_is_loaded(self):
        settings = Settings(
            service=ServiceSettings(artifact_root="/tmp/video-pool-test"),
            engine=EngineSettings(
                models={
                    "wan-a": ModelSettings(
                        backend="diffusers_wan_t2v",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                    ),
                    "wan-b": ModelSettings(
                        backend="diffusers_wan_t2v",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                    ),
                }
            ),
        )
        engine = VideoRouterEngine(settings)
        runtime_a = ClosableRuntime()
        runtime_b = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", side_effect=[runtime_a, runtime_b]),
            mock.patch("app.engine.router.release_loaded_torch_cuda_memory"),
            mock.patch("app.engine.router.reset_loaded_torch_cuda_context") as reset_cuda,
        ):
            await engine.load_model("wan-a")
            await engine.load_model("wan-b")
            payload = await engine.unload_model("wan-a")

        self.assertTrue(runtime_a.closed)
        self.assertFalse(payload["loaded"])
        self.assertFalse(runtime_b.closed)
        reset_cuda.assert_not_called()

    async def test_load_records_observed_vram_delta(self):
        settings = Settings(
            service=ServiceSettings(artifact_root="/tmp/video-pool-test"),
            engine=EngineSettings(
                models={
                    "test-video": ModelSettings(
                        backend="stub",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                        vram_estimate_mib=9999,
                    )
                }
            ),
        )
        engine = VideoRouterEngine(settings)
        runtime = ClosableRuntime()

        with (
            mock.patch.object(engine, "_create_runtime", return_value=runtime),
            mock.patch("app.engine.router.query_primary_gpu_used_mib", side_effect=[1000, 1245]),
        ):
            payload = await engine.load_model("test-video")

        self.assertTrue(payload["loaded"])
        self.assertEqual(payload["vram_estimate_mib"], 245)
        self.assertEqual(payload["vram_estimate_source"], "observed_load_delta")

    async def test_load_model_records_runtime_load_override(self):
        settings = Settings(
            service=ServiceSettings(artifact_root="/tmp/video-pool-test"),
            engine=EngineSettings(
                models={
                    "wan-video": ModelSettings(
                        backend="diffusers_wan_t2v",
                        enabled=False,
                        modalities=("text",),
                        tasks=("text_to_video",),
                    )
                }
            ),
        )
        engine = VideoRouterEngine(settings)
        runtime = ClosableRuntime()

        with mock.patch.object(engine, "_create_runtime", return_value=runtime):
            payload = await engine.load_model(
                "wan-video",
                AdminLoadRequest(wan_vae_tiling=False),
            )

        self.assertEqual(payload["load_override"], {"wan_vae_tiling": False})

    def test_observed_vram_delta_ignores_negative_or_empty_samples(self):
        self.assertIsNone(router_module._observed_vram_delta_mib(None, 1200))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, None))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1100))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1200))
        self.assertEqual(router_module._observed_vram_delta_mib(1200, 1456), 256)


if __name__ == "__main__":
    unittest.main()
