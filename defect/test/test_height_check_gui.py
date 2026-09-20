import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DETECTION_DIR))

import height_check  # noqa: E402
from GUI.core.jsonnet_store import load_effective  # noqa: E402
from GUI.core.paths import CALIBRATION_DIR, CALIBRATION_RESULT_NAMES  # noqa: E402
from GUI.pages.registry import GUI_HIDDEN_TASK_IDS, TASKS, task_by_id  # noqa: E402


class HeightCheckGuiTests(unittest.TestCase):
    def test_height_check_is_first_visible_calibration_task(self):
        visible = [
            task
            for task in TASKS
            if task.id not in GUI_HIDDEN_TASK_IDS
            and task.script.is_relative_to(CALIBRATION_DIR)
        ]

        self.assertEqual(visible[0].id, "height_check")
        self.assertEqual(visible[0].nav_label, "高度检测")
        self.assertTrue(visible[0].script.is_file())
        self.assertIn("height_check", CALIBRATION_RESULT_NAMES)

    def test_height_check_fields_exist_in_its_config(self):
        task = task_by_id("height_check")
        self.assertIsNotNone(task)
        values = load_effective(task.config)
        fields = {field.key for field in task.fields}

        self.assertTrue(fields.issubset(values))
        self.assertEqual(values["small_zoom"], 2800)
        self.assertGreaterEqual(values["margin"], 0.0)
        self.assertEqual(values["output_dir"], "working_data/height_check")

    def test_white_frame_inside_margin_passes(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (80, 60), (520, 340), (255, 255, 255), 18)

        result = height_check.check_height(image, margin=0.05)

        self.assertTrue(result["passed"])

    def test_white_frame_touching_edge_fails(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (0, 30), (560, 370), (255, 255, 255), 18)

        result = height_check.check_height(image, margin=0.02)

        self.assertFalse(result["passed"])

    def test_warm_gray_projection_does_not_pull_the_frame_box(self):
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(image, (80, 60), (520, 340), (255, 255, 255), 18)
        cv2.line(image, (80, 340), (80, 395), (144, 168, 187), 9)

        frame = height_check.detect_white_frame(image)

        self.assertLess(frame["bbox_corners_px"]["y_max"], 360)
        self.assertEqual(frame["bbox_method"], "connected_component_edge_support")
        self.assertEqual(frame["white_mask_hsv"]["minimum_value"], 190)

    def test_live_run_centres_the_first_frame_then_checks_the_second(self):
        before = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(before, (40, 60), (440, 340), (255, 255, 255), 18)
        final = np.zeros_like(before)
        cv2.rectangle(final, (100, 60), (500, 340), (255, 255, 255), 18)
        args = SimpleNamespace(
            image=None,
            calibration=None,
            small_zoom=2800,
            output_dir="working_data/height_check",
            align_frame_center=True,
            margin=0.02,
            save_raw=False,
        )
        camera = MagicMock()
        run_dir = Path("working_data/height_check/mock_run")
        calibration = {
            "stage": {
                "pixel_to_stage": np.eye(2),
                "stage_to_pixel": np.eye(2),
            },
            "intrinsics": {
                "camera_matrix": np.eye(3),
                "distortion_coefficients": np.zeros(5),
                "image_width_px": 600,
                "image_height_px": 400,
                "file": "calibrate.json",
                "position_name": "zoom_02800",
            },
        }

        with (
            patch.object(
                height_check.capture_pcb,
                "load_calibration_summary_document",
                return_value={},
            ),
            patch.object(
                height_check.two_zoom,
                "autofocus_position_for_zoom",
                return_value={"focus": 4936, "iris": 400},
            ),
            patch.object(
                height_check.two_zoom,
                "load_visual_phase_calibrations",
                return_value=calibration,
            ),
            patch.object(
                height_check.capture_pcb,
                "create_run_directory",
                return_value=run_dir,
            ),
            patch.object(height_check.capture_pcb, "write_json"),
            patch.object(height_check.capture_pcb, "write_png"),
            patch.object(height_check.two_zoom, "move_lens_to_phase"),
            patch.object(
                height_check.capture_pcb,
                "open_camera",
                return_value=(camera, {"index": 0}),
            ),
            patch.object(
                height_check.capture_pcb,
                "capture_frame",
                side_effect=[before, final],
            ) as capture,
            patch.object(
                height_check.capture_pcb,
                "build_undistort_maps",
                return_value=(None, None),
            ),
            patch.object(height_check.cv2, "remap", side_effect=lambda image, *_: image),
            patch.object(
                height_check.two_zoom,
                "execute_relative_stage_move",
                return_value={"actual_position_mm": [0.0, 0.0]},
            ) as move,
            patch.object(Path, "mkdir"),
        ):
            with redirect_stdout(StringIO()):
                code = height_check.run(args)

        self.assertEqual(code, 0)
        self.assertEqual(capture.call_count, 2)
        move.assert_called_once()
        requested_delta = np.asarray(move.call_args.args[2], dtype=np.float64)
        np.testing.assert_allclose(requested_delta, [60.0, 0.0], atol=1.0)
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
