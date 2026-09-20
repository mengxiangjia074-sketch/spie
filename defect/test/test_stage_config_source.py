import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_ROOT = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_ROOT))

from GUI.core.jsonnet_store import load_effective  # noqa: E402


STAGE_CONFIG = (
    PROJECT_ROOT
    / "detection"
    / "calibration"
    / "configs"
    / "stage_to_pixel"
    / "stage2pixel.jsonnet"
)
CAPTURE_CONFIGS = (
    PROJECT_ROOT / "config" / "pcb_capture" / "capture.jsonnet",
    PROJECT_ROOT / "config" / "pcb_two_zoom_capture" / "capture.jsonnet",
)


class StageConfigSourceTests(unittest.TestCase):
    def test_stage_config_defines_board_shared_values_directly(self):
        source = STAGE_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("board2pixel", source)

        settings = load_effective(STAGE_CONFIG)
        self.assertEqual(settings["camera"], None)
        self.assertFalse(settings["keep_camera_autofocus"])
        self.assertEqual(settings["port"], "/dev/lensdetect-stage")
        self.assertEqual(settings["baudrate"], 38400)
        self.assertEqual(settings["x_lead_mm_per_rev"], 1.0)
        self.assertEqual(settings["y_lead_mm_per_rev"], 1.0)
        self.assertEqual(settings["x_slave_address"], 1)
        self.assertEqual(settings["y_slave_address"], 2)

    def test_capture_configs_inherit_shared_values_from_stage_config(self):
        stage = load_effective(STAGE_CONFIG)
        shared_keys = (
            "camera",
            "camera_name",
            "frame_width",
            "frame_height",
            "camera_startup_discard_frames",
            "capture_discard_frames",
            "keep_camera_autofocus",
            "port",
            "baudrate",
            "x_lead_mm_per_rev",
            "y_lead_mm_per_rev",
            "x_slave_address",
            "y_slave_address",
            "speed_rpm",
            "timeout_seconds",
            "position_tolerance_mm",
        )

        for path in CAPTURE_CONFIGS:
            with self.subTest(config=path):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("board2pixel", source)
                settings = load_effective(path)
                for key in shared_keys:
                    self.assertEqual(settings[key], stage[key], key)
                self.assertEqual(settings["capture_delay_seconds"], stage["settle_seconds"])


if __name__ == "__main__":
    unittest.main()
