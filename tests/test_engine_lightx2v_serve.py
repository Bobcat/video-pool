import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.config import ModelSettings
from app.engine.common import VideoJob
from app.engine.lightx2v_serve import LightX2VServeRuntime


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def _settings() -> ModelSettings:
    return ModelSettings(
        backend="lightx2v_serve",
        model_path="/models/wan",
        lightx2v_python="/opt/lightx2v/bin/python",
        lightx2v_host="127.0.0.1",
        lightx2v_port=8123,
        lightx2v_model_cls="wan2.1_distill",
        lightx2v_task="i2v",
        lightx2v_config_json="/configs/wan_i2v.json",
        lightx2v_lora_dir="/loras",
        lightx2v_metric_port=8124,
        lightx2v_extra_args=("--dit_quantized", "true"),
    )


class LightX2VServeRuntimeTests(unittest.TestCase):
    def test_starts_lightx2v_server_command(self):
        process = FakeProcess()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(LightX2VServeRuntime, "_start_process", return_value=process) as start_process,
            mock.patch.object(LightX2VServeRuntime, "_wait_until_ready"),
        ):
            runtime = LightX2VServeRuntime("wan-i2v", _settings(), Path(tmpdir))
            runtime.close()

        command = start_process.call_args.args[0]
        self.assertEqual(command[:3], ["/opt/lightx2v/bin/python", "-m", "lightx2v.server"])
        self.assertIn("--model_cls", command)
        self.assertIn("wan2.1_distill", command)
        self.assertIn("--task", command)
        self.assertIn("i2v", command)
        self.assertIn("--config_json", command)
        self.assertIn("/configs/wan_i2v.json", command)
        self.assertIn("--lora_dir", command)
        self.assertIn("/loras", command)
        self.assertIn("--dit_quantized", command)
        self.assertTrue(process.terminated)

    def test_image_to_video_submits_task_and_saves_artifact(self):
        process = FakeProcess()
        encoded = base64.b64encode(b"image bytes").decode("ascii")
        job = VideoJob(
            operation="image_to_video",
            model="wan-i2v",
            prompt="animate the image",
            size="832x480",
            n=1,
            duration_seconds=5.0,
            fps=16,
            num_frames=81,
            quality="auto",
            seed=42,
            metadata={"steps": 4, "negative_prompt": "blur"},
            images=(mock.Mock(data_url=f"data:image/png;base64,{encoded}"),),
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(LightX2VServeRuntime, "_start_process", return_value=process),
            mock.patch.object(LightX2VServeRuntime, "_wait_until_ready"),
        ):
            runtime = LightX2VServeRuntime("wan-i2v", _settings(), Path(tmpdir))
            with (
                mock.patch.object(runtime, "_post_json", return_value={"task_id": "task-1"}) as post_json,
                mock.patch.object(runtime, "_wait_for_task", return_value={"status": "completed"}),
                mock.patch.object(runtime, "_get_bytes", return_value=b"mp4 bytes"),
            ):
                result = runtime._complete_sync(job)

            payload = post_json.call_args.args[1]
            self.assertEqual(payload["prompt"], "animate the image")
            self.assertEqual(payload["negative_prompt"], "blur")
            self.assertEqual(payload["infer_steps"], 4)
            self.assertEqual(payload["target_shape"], [480, 832])
            self.assertEqual(payload["target_video_length"], 81)
            self.assertEqual(payload["image_path"], encoded)
            self.assertEqual(result.videos[0].mime_type, "video/mp4")
            self.assertEqual(Path(result.videos[0].path).read_bytes(), b"mp4 bytes")


if __name__ == "__main__":
    unittest.main()
