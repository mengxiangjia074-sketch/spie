import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_DIR))

from GUI.core.paths import (  # noqa: E402
    CALIBRATION_RESULT_NAMES,
    HIDDEN_CALIBRATION_RESULT_NAMES,
    calibration_result_roots,
    capture_result_roots,
)
from GUI.pages.registry import GUI_HIDDEN_TASK_IDS, TASKS  # noqa: E402
from GUI.widgets.resultview import (  # noqa: E402
    _entry_headline,
    visible_summary_document,
)


class GuiCalibrationVisibilityTests(unittest.TestCase):
    def test_only_board_task_is_hidden(self):
        self.assertEqual(GUI_HIDDEN_TASK_IDS, {"board"})
        self.assertNotIn("stage", GUI_HIDDEN_TASK_IDS)

    def test_stage_results_are_calibration_results_and_board_stays_hidden(self):
        hidden = set(HIDDEN_CALIBRATION_RESULT_NAMES)
        self.assertEqual(hidden, {"board_pixel_affine"})
        self.assertIn("stage_command_to_pixel", CALIBRATION_RESULT_NAMES)
        self.assertTrue(
            hidden.isdisjoint(path.name for path in calibration_result_roots())
        )
        self.assertTrue(hidden.isdisjoint(path.name for path in capture_result_roots()))

    def test_stage_calibration_precedes_zoom_calibration(self):
        task_ids = [task.id for task in TASKS]

        self.assertLess(task_ids.index("stage"), task_ids.index("zoom_transform"))

    def test_stage_calibration_exposes_two_independent_zoom_ranges(self):
        stage = next(task for task in TASKS if task.id == "stage")
        field_names = [field.key for field in stage.fields]

        for name in (
            "zoom_1",
            "zoom_1_x_step_mm",
            "zoom_1_y_step_mm",
            "zoom_2",
            "zoom_2_x_step_mm",
            "zoom_2_y_step_mm",
            "lens_device",
            "calibration",
        ):
            self.assertIn(name, field_names)
        self.assertNotIn("x_step_mm", field_names)
        self.assertNotIn("y_step_mm", field_names)

    def test_only_board_summary_entries_are_filtered_without_mutation(self):
        document = {
            "schema_version": 1,
            "entries": [
                {"type": "lens_checkerboard_camera_calibration", "name": "lens"},
                {"type": "calibration_board_to_pixel_affine_core", "name": "board"},
                {"type": "stage_command_to_pixel_calibration", "name": "stage"},
                {"type": "zoom_pixel_homography_calibration", "name": "zoom"},
            ],
        }

        visible = visible_summary_document(document)

        self.assertEqual(
            [entry["name"] for entry in visible["entries"]],
            ["lens", "stage", "zoom"],
        )
        self.assertEqual(len(document["entries"]), 4)

    def test_zoom_summary_without_fit_quality_still_shows_zoom_pair(self):
        entry = {
            "type": "zoom_pixel_homography_calibration",
            "source_zoom": 2800,
            "target_zoom": 6353,
        }

        self.assertEqual(_entry_headline(entry), "2800 → 6353")

    def test_zoom_summary_shows_both_pixel_lengths(self):
        entry = {
            "type": "zoom_pixel_homography_calibration",
            "source_zoom": 2800,
            "target_zoom": 6353,
            "source_pixel_scale": {"mm_per_pixel": 0.07},
            "target_pixel_scale": {"mm_per_pixel": 0.03},
        }

        self.assertEqual(
            _entry_headline(entry),
            "2800 → 6353, 像素长度 2800: 0.070000, 6353: 0.030000 mm/px",
        )


if __name__ == "__main__":
    unittest.main()
