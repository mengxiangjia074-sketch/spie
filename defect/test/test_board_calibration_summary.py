import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PROJECT_ROOT / "detection" / "calibration"
sys.path.insert(0, str(CALIBRATION_DIR))

import calibrate_board_to_pixel
import calibration_summary


class BoardCalibrationSummaryTests(unittest.TestCase):
    def test_current_lens_position_reads_zoom_and_focus_on_one_connection(self):
        calls = []
        fake_lens_camera = types.ModuleType("LensCamera")
        fake_lens_camera.connect = lambda device: calls.append(("connect", device)) or {}
        fake_lens_camera.motor_supported = lambda name, capabilities: True
        fake_lens_camera.ensure_ready = (
            lambda name, init_if_needed: calls.append(("ready", name, init_if_needed))
        )
        fake_lens_camera.read_current = lambda name: {"zoom": 1501, "focus": 4321}[name]
        fake_lens_camera.close = lambda: calls.append(("close",))

        with mock.patch.dict(sys.modules, {"LensCamera": fake_lens_camera}):
            position = calibration_summary.current_lens_position(device_number=2)

        self.assertEqual(position, {"zoom": 1501, "focus": 4321})
        self.assertEqual(calls.count(("connect", 2)), 1)
        self.assertEqual(calls.count(("close",)), 1)

    def test_board_summary_saves_current_focus_with_zoom(self):
        core_result = {
            "type": "calibration_board_to_pixel_affine_core",
            "created_at_utc": "2026-08-25T00:00:00Z",
            "board_to_pixel": {"matrix_2x3": [[1, 0, 0], [0, 1, 0]]},
            "pixel_to_board": {"matrix_2x3": [[1, 0, 0], [0, 1, 0]]},
            "image": {"width_px": 3840, "height_px": 2160},
        }
        stage_reference = {
            "recorded": True,
            "position_mm": {"x": 1.0, "y": 2.0},
            "coordinate_source": "test",
            "configuration": {"port": "COM_TEST"},
        }

        with mock.patch.object(
            calibrate_board_to_pixel.calibration_summary,
            "current_lens_position",
            return_value={"zoom": 6353, "focus": 4567},
        ), mock.patch.object(
            calibrate_board_to_pixel.calibration_summary,
            "append_entry",
            return_value=Path("calibrate.json"),
        ) as append_entry:
            result = calibrate_board_to_pixel.append_calibration_summary(
                core_result,
                stage_reference,
                SimpleNamespace(lens_device=1),
            )

        entry = append_entry.call_args.args[0]
        self.assertEqual(result, Path("calibrate.json"))
        self.assertEqual(entry["lens_focus"], 4567)
        self.assertNotIn("coordinate_source", entry["stage_reference"])
        self.assertNotIn("configuration", entry["stage_reference"])
        self.assertEqual(append_entry.call_args.kwargs, {"zoom": 6353})


if __name__ == "__main__":
    unittest.main()
