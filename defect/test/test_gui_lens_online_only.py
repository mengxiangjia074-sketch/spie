import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
CALIBRATION_DIR = DETECTION_DIR / "calibration"
sys.path.insert(0, str(DETECTION_DIR))
sys.path.insert(0, str(CALIBRATION_DIR))

import calibrate_lens as calibration
from GUI.core.jsonnet_store import load_effective
from GUI.pages.registry import task_by_id


REMOVED_FIELDS = {
    "image_dir",
    "skip_lens",
    "full_workflow",
    "device",
    "no_init",
    "settle_seconds",
    "zoom_positions",
    "zoom_steps",
    "iris_position",
    "focus_metric",
    "focus_min",
    "focus_max",
    "focus_golden_final_steps",
    "focus_score_max_side",
    "focus_score_frames",
    "focus_score_aggregate",
    "focus_discard_frames",
    "focus_startup_discard_frames",
    "focus_settle",
    "focus_roi",
    "focus_roi_scale",
    "focus_blur_ksize",
    "focus_golden_tolerance",
    "focus_golden_bracket_steps",
    "focus_golden_tie_relative_margin",
    "focus_golden_highres_auto",
    "focus_golden_max_iter",
}


class LensOnlineOnlyTests(unittest.TestCase):
    def test_reprojection_error_is_printed_for_the_calibration_console(self):
        output = {
            "positions": [
                {
                    "name": "online_capture",
                    "calibration": {"rms_reprojection_error": 0.489356},
                }
            ]
        }

        stream = StringIO()
        with redirect_stdout(stream):
            calibration.print_calibration_errors(output)

        self.assertEqual(
            stream.getvalue().strip(),
            "标定误差: online_capture RMS 重投影误差 0.4894 px",
        )

    def test_gui_exposes_only_online_camera_calibration_fields(self):
        task = task_by_id("lens")
        self.assertIsNotNone(task)
        field_keys = {field.key for field in task.fields}
        sections = {field.section for field in task.fields}

        self.assertTrue(REMOVED_FIELDS.isdisjoint(field_keys))
        self.assertTrue({"标定板", "采集", "相机", "输出"}.issuperset(sections))
        self.assertIn("frames", field_keys)
        self.assertIn("camera_name", field_keys)
        self.assertIn("capture_dir", field_keys)
        self.assertNotIn("离线", task.subtitle)
        self.assertNotIn("电动镜头", task.subtitle)

    def test_lens_config_contains_no_removed_workflow_fields(self):
        task = task_by_id("lens")
        values = load_effective(task.config)
        self.assertTrue(REMOVED_FIELDS.isdisjoint(values))
        self.assertEqual(values["camera_name"], "CamSPC")

    def test_runtime_forces_online_capture_and_never_calls_removed_workflows(self):
        args = SimpleNamespace(
            image_dir="offline-images",
            skip_lens=False,
            full_workflow=True,
        )
        output = {"positions": []}
        frames = [{"index": 1}]

        with (
            patch.object(calibration, "load_image_tools", return_value=("cv2", "np")),
            patch.object(calibration, "build_output_header", return_value=output),
            patch.object(
                calibration,
                "capture_from_camera",
                return_value=("objects", "images", (1920, 1080), frames),
            ) as capture,
            patch.object(calibration, "calibrate_camera", return_value={"ok": True}),
            patch.object(
                calibration,
                "write_calibration_outputs",
                return_value=(Path("calibration.json"), Path("calibration.log")),
            ) as write_outputs,
            patch.object(calibration, "run_full_workflow") as full_workflow,
            patch.object(calibration, "collect_from_images") as offline_capture,
            patch.object(calibration, "move_lens_position") as move_lens,
        ):
            calibration.run_calibration(args)

        self.assertIsNone(args.image_dir)
        self.assertTrue(args.skip_lens)
        self.assertFalse(args.full_workflow)
        capture.assert_called_once_with("cv2", "np", args, "online_capture")
        full_workflow.assert_not_called()
        offline_capture.assert_not_called()
        move_lens.assert_not_called()
        saved_output = write_outputs.call_args.args[1]
        self.assertEqual(saved_output["positions"][0]["name"], "online_capture")
        self.assertNotIn("lens", saved_output["positions"][0])


if __name__ == "__main__":
    unittest.main()
