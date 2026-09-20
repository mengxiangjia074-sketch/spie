import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PROJECT_ROOT / "detection" / "calibration"
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(DETECTION_DIR))

import calibrate_stage_to_pixel as calibration


class StageCalibrationSummaryTests(unittest.TestCase):
    def test_two_zoom_profiles_use_independent_ranges_and_autofocus(self):
        document = {
            "entries": [
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_02800",
                    "lens_zoom": 2800,
                    "lens_focus": 4936,
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_06353",
                    "lens_zoom": 6353,
                    "lens_focus": 5881,
                },
            ]
        }
        args = SimpleNamespace(
            zoom_1=2800,
            zoom_1_x_step_mm=[2, 4, 6],
            zoom_1_y_step_mm=[2, 4],
            zoom_2=6353,
            zoom_2_x_step_mm=[0.5, 1.0],
            zoom_2_y_step_mm=0.75,
        )

        profiles = calibration.build_zoom_profiles(
            args, document, Path("calibrate.json")
        )

        self.assertEqual(
            [(item["zoom"], item["focus"]) for item in profiles],
            [(2800, 4936), (6353, 5881)],
        )
        self.assertEqual(profiles[0]["x_step_mm"], [2.0, 4.0, 6.0])
        self.assertEqual(profiles[0]["y_step_mm"], [2.0, 4.0])
        self.assertEqual(profiles[1]["x_step_mm"], [0.5, 1.0])
        self.assertEqual(profiles[1]["y_step_mm"], [0.75])

    def test_missing_exact_autofocus_zoom_is_rejected(self):
        document = {
            "entries": [
                {
                    "type": "pcb_autofocus_lens_position",
                    "lens_zoom": 2801,
                    "lens_focus": 4936,
                }
            ]
        }

        with self.assertRaisesRegex(
            calibration.StagePixelCalibrationError,
            "focus for zoom 2800",
        ):
            calibration.load_autofocus_focus(document, Path("calibrate.json"), 2800)

    def test_fit_errors_are_printed_for_the_calibration_console(self):
        diagnostics = {
            "rmse_px": 0.650268,
            "mean_error_px": 0.485558,
            "max_error_px": 2.257488,
        }

        stream = StringIO()
        with redirect_stdout(stream):
            calibration.print_calibration_errors(diagnostics)

        output = stream.getvalue().strip()
        self.assertTrue(output.startswith("标定误差:"))
        self.assertIn("RMSE 0.6503 px", output)
        self.assertIn("平均误差 0.4856 px", output)
        self.assertIn("最大误差 2.2575 px", output)

    def test_two_zoom_errors_are_printed_together_in_zoom_order(self):
        completed = [
            {
                "zoom": 2800,
                "fit_quality": {
                    "rmse_px": 0.25,
                    "mean_error_px": 0.20,
                    "max_error_px": 0.40,
                },
            },
            {
                "zoom": 6353,
                "fit_quality": {
                    "rmse_px": 0.35,
                    "mean_error_px": 0.30,
                    "max_error_px": 0.50,
                },
            },
        ]

        stream = StringIO()
        with redirect_stdout(stream):
            calibration.print_completed_calibration_errors(completed)

        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("标定误差: Zoom 2800:"))
        self.assertTrue(lines[1].startswith("标定误差: Zoom 6353:"))

    def test_summary_omits_detailed_fit_and_coordinate_fields(self):
        section = {
            "equation": "pixel = matrix * stage",
            "matrix_2x2": [[42.0, 0.1], [0.0, 42.1]],
            "input_unit": "commanded_mm",
            "output_unit": "pixel",
            "coordinate_convention": {"pixel_u_positive": "image right"},
            "calibration_start_stage_position_mm": [8.0, -3.0],
            "fit_quality": {
                "rmse_px": 0.5,
                "condition_number": 1.01,
                "minimum_registration_response": 0.6,
                "mean_registration_response": 0.8,
            },
            "pixel_to_stage_command": {
                "equation": "stage = inverse * pixel",
                "matrix_2x2": [[0.02, 0.0], [0.0, 0.02]],
                "input_unit": "pixel",
                "output_unit": "commanded_mm",
            },
        }
        document = {
            "type": "stage_command_to_pixel_calibration",
            "calibrated_at_utc": "2026-08-29T14:00:00Z",
            "stage_command_to_pixel": section,
        }
        args = SimpleNamespace(lens_device=0)

        with mock.patch.object(
            calibration.calibration_summary,
            "append_entry",
            return_value=Path("calibrate.json"),
        ) as append_entry:
            result = calibration.append_calibration_summary(document, args)

        self.assertEqual(result, Path("calibrate.json"))
        saved = append_entry.call_args.args[0]
        serialized = repr(saved)
        for field in (
            "fit_quality",
            "coordinate_convention",
            "calibration_start_stage_position_mm",
            "condition_number",
            "minimum_registration_response",
            "mean_registration_response",
        ):
            self.assertNotIn(field, serialized)
        self.assertEqual(
            saved["stage_command_to_pixel"]["matrix_2x2"], section["matrix_2x2"]
        )
        self.assertEqual(
            saved["pixel_to_stage_command"], section["pixel_to_stage_command"]
        )
        self.assertEqual(append_entry.call_args.kwargs, {"device_number": 0})
        self.assertIn("fit_quality", section)

    def test_summary_is_saved_under_explicit_zoom_with_focus(self):
        section = {
            "equation": "pixel = matrix * stage",
            "matrix_2x2": [[42.0, 0.1], [0.0, 42.1]],
            "input_unit": "commanded_mm",
            "output_unit": "pixel",
            "pixel_to_stage_command": {
                "equation": "stage = inverse * pixel",
                "matrix_2x2": [[0.02, 0.0], [0.0, 0.02]],
                "input_unit": "pixel",
                "output_unit": "commanded_mm",
            },
        }
        document = {
            "type": "stage_command_to_pixel_calibration",
            "calibrated_at_utc": "2026-08-29T14:00:00Z",
            "lens_zoom": 6353,
            "lens_focus": 5881,
            "stage_command_to_pixel": section,
        }
        args = SimpleNamespace(lens_device=0)

        with mock.patch.object(
            calibration.calibration_summary,
            "append_entry",
            return_value=Path("calibrate.json"),
        ) as append_entry:
            calibration.append_calibration_summary(document, args, zoom=6353)

        saved = append_entry.call_args.args[0]
        self.assertEqual(saved["lens_focus"], 5881)
        self.assertEqual(
            append_entry.call_args.kwargs,
            {"device_number": 0, "zoom": 6353},
        )


if __name__ == "__main__":
    unittest.main()
