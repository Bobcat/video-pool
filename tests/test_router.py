import unittest
from unittest import mock

from app.config import EngineSettings, ModelSettings, ServiceSettings, Settings
from app.engine.common import GeneratedVideoPayload, VideoJob, VideoResult
from app.engine import router as router_module
from app.engine.router import VideoRouterEngine


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
        ):
            await engine.load_model("test-video")
            payload = await engine.unload_model("test-video")

        self.assertTrue(runtime.closed)
        self.assertFalse(payload["loaded"])
        release_memory.assert_called_once()

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

    def test_observed_vram_delta_ignores_negative_or_empty_samples(self):
        self.assertIsNone(router_module._observed_vram_delta_mib(None, 1200))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, None))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1100))
        self.assertIsNone(router_module._observed_vram_delta_mib(1200, 1200))
        self.assertEqual(router_module._observed_vram_delta_mib(1200, 1456), 256)


if __name__ == "__main__":
    unittest.main()

