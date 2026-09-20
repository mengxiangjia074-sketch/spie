import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PROJECT_ROOT / "detection" / "calibration"
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(CALIBRATION_DIR))
sys.path.insert(0, str(DETECTION_DIR))

import calibrate_zoom_pixel_transform as calibration  # noqa: E402


class ZoomPixelTransformMathTests(unittest.TestCase):
    def test_bidirectional_errors_are_printed_for_the_calibration_console(self):
        quality = {
            "forward_inliers": {
                "rmse_px": 0.15392,
                "mean_error_px": 0.13475,
                "max_error_px": 0.36496,
            },
            "reverse_inliers": {
                "rmse_px": 0.07341,
                "mean_error_px": 0.06427,
                "max_error_px": 0.17408,
            },
        }

        stream = StringIO()
        with redirect_stdout(stream):
            calibration.print_calibration_errors(quality)

        output = stream.getvalue().strip()
        self.assertTrue(output.startswith("标定误差:"))
        self.assertIn("源到目标内点 RMSE 0.1539 px", output)
        self.assertIn("目标到源内点 RMSE 0.0734 px", output)

    def test_preview_window_uses_two_thirds_of_screen_and_is_centered(self):
        fake_cv2 = mock.Mock(WINDOW_NORMAL=0, WINDOW_KEEPRATIO=0)

        with mock.patch.object(
            calibration, "get_screen_size", return_value=(1920, 1080)
        ):
            calibration.configure_preview_window(fake_cv2, "zoom preview")

        fake_cv2.namedWindow.assert_called_once_with("zoom preview", 0)
        fake_cv2.resizeWindow.assert_called_once_with("zoom preview", 1280, 720)
        fake_cv2.moveWindow.assert_called_once_with("zoom preview", 320, 180)

    def test_estimates_forward_and_inverse_homography_with_outliers(self):
        source = np.asarray(
            [
                (100.0 + 90.0 * column, 80.0 + 75.0 * row)
                for row in range(8)
                for column in range(10)
            ],
            dtype=np.float64,
        )
        expected = np.asarray(
            [
                [2.68, 0.018, -1530.0],
                [-0.012, 2.71, -640.0],
                [0.000011, -0.000007, 1.0],
            ],
            dtype=np.float64,
        )
        target = calibration.project_points(np, source, expected)
        generator = np.random.default_rng(20260828)
        target += generator.normal(0.0, 0.035, target.shape)
        target[[5, 39, 67]] += np.asarray([[25.0, -18.0], [-31.0, 20.0], [16.0, 27.0]])

        fitted, inverse, inliers, quality, predicted = calibration.estimate_homography(
            cv2,
            np,
            source,
            target,
            threshold_px=0.5,
            max_iterations=5000,
            confidence=0.999,
            min_inlier_ratio=0.8,
            max_inlier_rmse_px=0.2,
        )

        self.assertEqual(int(inliers.sum()), len(source) - 3)
        self.assertLess(quality["forward_inliers"]["rmse_px"], 0.1)
        expected_points = calibration.project_points(np, source[inliers], expected)
        self.assertLess(
            float(np.max(np.linalg.norm(predicted[inliers] - expected_points, axis=1))),
            0.08,
        )
        round_trip = calibration.project_points(
            np, calibration.project_points(np, source, fitted), inverse
        )
        self.assertLess(float(np.max(np.abs(round_trip - source))), 1e-8)

    def test_zero_distortion_preserves_pixels_and_scales_intrinsics(self):
        intrinsics = {
            "camera_matrix": np.asarray(
                [[1000.0, 0.0, 500.0], [0.0, 900.0, 400.0], [0.0, 0.0, 1.0]]
            ),
            "distortion_coefficients": np.zeros(5),
            "image_width_px": 1000,
            "image_height_px": 800,
        }
        points = np.asarray([[20.0, 30.0], [1000.0, 700.0]], dtype=np.float64)

        corrected, matrix = calibration.undistort_points(
            cv2, np, points, intrinsics, 2000, 1600
        )

        np.testing.assert_allclose(corrected, points, atol=1e-9)
        np.testing.assert_allclose(
            matrix,
            [[2000.0, 0.0, 1000.0], [0.0, 1800.0, 800.0], [0.0, 0.0, 1.0]],
        )

    def test_calculates_mm_per_pixel_from_checkerboard_spacing(self):
        board_size = (4, 3)
        points = np.asarray(
            [
                (10.0 + 40.0 * column, 20.0 + 50.0 * row)
                for row in range(board_size[1])
                for column in range(board_size[0])
            ],
            dtype=np.float64,
        )

        scale = calibration.calculate_checkerboard_pixel_scale(
            np, points, board_size, square_size_mm=2.0
        )

        self.assertEqual(scale["square_size_mm"], 2.0)
        self.assertAlmostEqual(scale["horizontal"]["mm_per_pixel"], 0.05)
        self.assertAlmostEqual(scale["vertical"]["mm_per_pixel"], 0.04)
        self.assertAlmostEqual(scale["mm_per_pixel"], 0.045)
        self.assertAlmostEqual(scale["pixels_per_mm"], 1.0 / 0.045)
        self.assertEqual(scale["horizontal"]["sample_count"], 9)
        self.assertEqual(scale["vertical"]["sample_count"], 8)
        omitted = {
            "mean_adjacent_corner_spacing_px",
            "median_adjacent_corner_spacing_px",
            "minimum_adjacent_corner_spacing_px",
            "maximum_adjacent_corner_spacing_px",
        }
        self.assertTrue(omitted.isdisjoint(scale["horizontal"]))
        self.assertTrue(omitted.isdisjoint(scale["vertical"]))


class ZoomPixelTransformSummaryTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "entries": [
                {
                    "type": "lens_checkerboard_camera_calibration",
                    "name": "zoom_01501",
                    "lens_zoom": 1501,
                    "positions": [
                        {
                            "name": "zoom_01501",
                            "image_width": 3840,
                            "image_height": 2160,
                            "camera_matrix": [
                                [6800.0, 0.0, 1900.0],
                                [0.0, 6800.0, 1080.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "distortion_coefficients": [0, 0, 0, 0, 0],
                        }
                    ],
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_01501",
                    "lens_zoom": 1501,
                    "lens_focus": 4552,
                },
            ]
        }

    def test_loads_matching_intrinsics_and_autofocus_focus(self):
        path = Path("calibrate.json")
        intrinsics = calibration.load_intrinsics(np, self.document, path, 1501)
        focus = calibration.load_focus(self.document, path, 1501)

        self.assertEqual(intrinsics["position_name"], "zoom_01501")
        self.assertEqual(intrinsics["image_width_px"], 3840)
        self.assertEqual(focus, 4552)

    def test_focus_requires_matching_pcb_autofocus_entry(self):
        document = {
            "entries": [
                {
                    "type": "calibration_board_to_pixel_affine_core",
                    "name": "zoom_01501",
                    "lens_zoom": 1501,
                    "lens_focus": 4999,
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_06353",
                    "lens_zoom": 6353,
                    "lens_focus": 5864,
                },
            ]
        }

        with self.assertRaisesRegex(
            calibration.ZoomTransformCalibrationError,
            "pcb_autofocus_lens_position focus for zoom 1501",
        ):
            calibration.load_focus(document, Path("calibrate.json"), 1501)

    def test_pair_summary_name_does_not_claim_single_zoom(self):
        entry = {
            "type": calibration.RESULT_TYPE,
            "source_zoom": 1501,
            "target_zoom": 6353,
            "coordinate_space": "undistorted_pixel",
            "coordinate_convention": "OpenCV pixels: u right, v down",
            "source_image": {"width_px": 640, "height_px": 480},
            "target_image": {"width_px": 640, "height_px": 480},
            "source_to_target": {"matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "target_to_source": {"matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "source_pixel_scale": {"mm_per_pixel": 0.05},
            "target_pixel_scale": {"mm_per_pixel": 0.02},
            "fit_quality": {"inlier_ratio": 1.0},
            "detail_report": "working_data/report.json",
            "saved_at_utc": "old timestamp",
        }
        with mock.patch.object(
            calibration.calibration_summary,
            "append_entry",
            return_value=Path("calibrate.json"),
        ) as append_entry:
            result = calibration.append_calibration_summary(entry, 1501, 6353)

        self.assertEqual(result, Path("calibrate.json"))
        self.assertEqual(
            append_entry.call_args.kwargs,
            {
                "name": "zoom_01501_to_zoom_06353",
                "include_saved_at_utc": False,
            },
        )
        saved_entry = append_entry.call_args.args[0]
        self.assertTrue(calibration.SUMMARY_OMITTED_FIELDS.isdisjoint(saved_entry))
        self.assertIn("source_to_target", saved_entry)
        self.assertIn("target_to_source", saved_entry)
        self.assertIn("source_pixel_scale", saved_entry)
        self.assertIn("target_pixel_scale", saved_entry)
        self.assertIn("fit_quality", entry)
        self.assertNotIn("zoom", append_entry.call_args.kwargs)

    def test_gui_registry_exposes_zoom_transform_task(self):
        from GUI.pages.registry import task_by_id

        task = task_by_id("zoom_transform")
        keys = {field.key for field in task.fields}
        self.assertEqual(task.script.name, "calibrate_zoom_pixel_transform.py")
        self.assertTrue(
            {
                "source_zoom",
                "target_zoom",
                "square_size_mm",
                "ransac_threshold_px",
                "output_dir",
            }.issubset(keys)
        )
        self.assertTrue({"source_focus", "target_focus"}.isdisjoint(keys))
        settings = calibration.load_settings()
        self.assertTrue({"source_focus", "target_focus"}.isdisjoint(settings))

    def test_offline_run_writes_artifacts_and_requests_summary_update(self):
        source = np.asarray(
            [
                (80.0 + 45.0 * column, 70.0 + 40.0 * row)
                for row in range(8)
                for column in range(10)
            ],
            dtype=np.float64,
        )
        expected = np.asarray(
            [[1.08, 0.01, 18.0], [-0.006, 1.1, 12.0], [0.00001, 0.00002, 1.0]]
        )
        target = calibration.project_points(np, source, expected)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        document = {
            "entries": [
                *self.document["entries"],
                {
                    "type": "lens_checkerboard_camera_calibration",
                    "name": "zoom_06353",
                    "lens_zoom": 6353,
                    "positions": [
                        {
                            "name": "zoom_06353",
                            "image_width": 640,
                            "image_height": 480,
                            "camera_matrix": [
                                [500.0, 0.0, 320.0],
                                [0.0, 500.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "distortion_coefficients": [0, 0, 0, 0, 0],
                        }
                    ],
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_06353",
                    "lens_zoom": 6353,
                    "lens_focus": 5864,
                },
            ]
        }
        document["entries"][0]["positions"][0].update(
            {
                "image_width": 640,
                "image_height": 480,
                "camera_matrix": [
                    [500.0, 0.0, 320.0],
                    [0.0, 500.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
        args = SimpleNamespace(
            source_zoom=1501,
            target_zoom=6353,
            lens_device=None,
            lens_no_init=False,
            lens_settle_seconds=0.0,
            board_cols=10,
            board_rows=8,
            square_size_mm=3.0,
            camera_name="unused",
            camera=None,
            frame_width=640,
            frame_height=480,
            camera_startup_discard_frames=0,
            capture_discard_frames=0,
            keep_camera_autofocus=False,
            capture_now=True,
            source_image="source.png",
            target_image="target.png",
            calibration=None,
            ransac_threshold_px=0.5,
            ransac_iterations=1000,
            ransac_confidence=0.995,
            min_inlier_ratio=0.95,
            max_inlier_rmse_px=0.1,
            output_dir="working_data/zoom_pixel_transform",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            with (
                mock.patch.object(
                    calibration, "load_summary_document", return_value=document
                ),
                mock.patch.object(
                    calibration, "allocate_run_directory", return_value=run_dir
                ),
                mock.patch.object(
                    calibration, "read_image", side_effect=[frame.copy(), frame.copy()]
                ),
                mock.patch.object(
                    calibration,
                    "detect_checkerboard",
                    side_effect=[
                        (source.copy(), "synthetic"),
                        (target.copy(), "synthetic"),
                    ],
                ),
                mock.patch.object(
                    calibration,
                    "append_calibration_summary",
                    return_value=Path("calibrate.json"),
                ) as append_summary,
            ):
                result = calibration.run_calibration(args)

            self.assertEqual(result, 0)
            self.assertTrue((run_dir / "zoom_pixel_transform.json").is_file())
            self.assertEqual(len(list(run_dir.glob("*.png"))), 7)
            saved_entry = append_summary.call_args.args[0]
            self.assertEqual(saved_entry["coordinate_space"], "undistorted_pixel")
            self.assertEqual(
                np.asarray(saved_entry["source_to_target"]["matrix_3x3"]).shape,
                (3, 3),
            )
            self.assertAlmostEqual(
                saved_entry["source_pixel_scale"]["horizontal"]["mm_per_pixel"],
                3.0 / 45.0,
            )
            self.assertAlmostEqual(
                saved_entry["source_pixel_scale"]["vertical"]["mm_per_pixel"],
                3.0 / 40.0,
            )
            self.assertIn("target_pixel_scale", saved_entry)
            self.assertEqual(append_summary.call_args.args[1:], (1501, 6353))


if __name__ == "__main__":
    unittest.main()
