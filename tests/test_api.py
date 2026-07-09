import base64
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def _settings_path(root: Path) -> Path:
    settings_path = root / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "service": {
                    "artifact_root": str(root / "artifacts"),
                },
                "engine": {
                    "models": {
                        "stub-video": {
                            "backend": "stub",
                            "enabled": True,
                            "modalities": ["text", "image"],
                            "output_modalities": ["video"],
                            "tasks": ["text_to_video", "image_to_video"],
                            "max_images": 1,
                            "max_output_videos": 1,
                            "vram_estimate_mib": 0,
                        },
                        "wan-video": {
                            "backend": "diffusers_wan_t2v",
                            "enabled": False,
                            "modalities": ["text"],
                            "output_modalities": ["video"],
                            "tasks": ["text_to_video"],
                            "max_images": 0,
                            "max_output_videos": 1,
                            "vram_estimate_mib": 11100,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return settings_path


class ApiTests(unittest.TestCase):
    def test_healthz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_public_models_returns_loaded_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["id"], "stub-video")
        self.assertEqual(payload["data"][0]["owned_by"], "video-pool")
        self.assertIn("image_to_video", payload["data"][0]["capabilities"]["tasks"])

    def test_admin_models_returns_load_constraints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                response = client.get("/v1/admin/models")

        self.assertEqual(response.status_code, 200)
        models = {model["id"]: model for model in response.json()["data"]}
        self.assertIn("wan_vae_tiling", models["wan-video"]["load_constraints"])
        self.assertEqual(models["wan-video"]["load_constraints"]["wan_vae_tiling"]["default"], True)
        self.assertEqual(models["wan-video"]["load_override"], {})

    def test_video_generation_returns_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with TestClient(create_app(_settings_path(root))) as client:
                response = client.post(
                    "/v1/videos/generations",
                    json={
                        "model": "stub-video",
                        "prompt": "a calm test clip",
                        "size": "320x192",
                        "duration_seconds": 2.0,
                        "fps": 8,
                    },
                )
                artifact_response = client.get(response.json()["data"][0]["url"])
                artifact_file_exists = Path(response.json()["data"][0]["path"]).is_file()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "video.generation")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["data"][0]["width"], 320)
        self.assertEqual(payload["data"][0]["height"], 192)
        self.assertEqual(payload["data"][0]["num_frames"], 16)
        self.assertEqual(payload["data"][0]["mime_type"], "application/json")
        self.assertTrue(artifact_file_exists)
        self.assertEqual(artifact_response.status_code, 200)
        self.assertTrue(artifact_response.json()["stub"])

    def test_image_to_video_requires_input_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                response = client.post(
                    "/v1/videos/image-to-video",
                    json={"model": "stub-video", "prompt": "animate it", "images": []},
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "bad_request")

    def test_image_to_video_accepts_data_url(self):
        encoded = base64.b64encode(b"not really an image yet").decode("ascii")
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                response = client.post(
                    "/v1/videos/image-to-video",
                    json={
                        "model": "stub-video",
                        "prompt": "animate it",
                        "images": [{"name": "test.png", "data_url": f"data:image/png;base64,{encoded}"}],
                        "num_frames": 3,
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "video.image_to_video")
        self.assertEqual(payload["data"][0]["num_frames"], 3)
        self.assertEqual(payload["metrics"]["input_image_count"], 1)

    def test_unloaded_model_rejects_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with TestClient(create_app(_settings_path(Path(tmpdir)))) as client:
                unload_response = client.post("/v1/admin/models/stub-video/unload")
                response = client.post(
                    "/v1/videos/generations",
                    json={"model": "stub-video", "prompt": "test"},
                )

        self.assertEqual(unload_response.status_code, 200)
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
