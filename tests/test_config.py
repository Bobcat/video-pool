import tempfile
import unittest
from pathlib import Path

from app.config import load_settings, resolve_artifact_root


class ConfigTests(unittest.TestCase):
    def test_default_settings_have_stub_video_model(self):
        settings = load_settings()

        self.assertEqual(settings.service.port, 8014)
        self.assertIn("stub-video", settings.engine.models)
        model = settings.engine.models["stub-video"]
        self.assertEqual(model.backend, "stub")
        self.assertTrue(model.enabled)
        self.assertEqual(model.tasks, ("text_to_video", "image_to_video"))
        self.assertEqual(model.output_modalities, ("video",))
        self.assertEqual(model.max_images, 1)
        self.assertEqual(model.max_output_videos, 1)
        self.assertEqual(model.generation_parameters["size"]["default"], "832x480")
        self.assertEqual(model.image_to_video_parameters["fps"]["default"], 16)

    def test_local_settings_override_base_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.json"
            local_path = root / "local.json"
            settings_path.write_text('{"service": {"port": 1}, "engine": {"models": {}}}', encoding="utf-8")
            local_path.write_text('{"service": {"port": 2, "artifact_root": "out"}}', encoding="utf-8")

            settings = load_settings(settings_path)

        self.assertEqual(settings.service.port, 2)
        self.assertEqual(settings.service.artifact_root, "out")

    def test_relative_artifact_root_resolves_under_project_root(self):
        settings = load_settings()

        root = resolve_artifact_root(settings)

        self.assertTrue(root.is_absolute())
        self.assertTrue(str(root).endswith("/video-pool/data/videos"))


if __name__ == "__main__":
    unittest.main()

