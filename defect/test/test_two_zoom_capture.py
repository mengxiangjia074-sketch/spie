import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "capture"))

import capture_pcb_two_zooms  # noqa: E402


def detection_args():
    return SimpleNamespace(
        white_block_threshold=220,
        white_block_min_area_px=100.0,
        white_block_max_area_fraction=0.08,
        white_block_min_square_ratio=0.7,
        white_block_min_fill_ratio=0.7,
        white_block_morph_kernel=3,
        global_crop_padding_px=0,
    )


class TwoZoomCaptureTests(unittest.TestCase):
    def setUp(self):
        self.summary_path = Path("calibrate.json")
        self.document = {
            "schema_version": 1,
            "entries": [
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_02800",
                    "lens_zoom": 2800,
                    "lens_focus": 4552,
                    "lens_iris": 410,
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "name": "zoom_06353",
                    "lens_zoom": 6353,
                    "lens_focus": 5858,
                    "lens_iris": 420,
                },
                {
                    "type": "zoom_pixel_homography_calibration",
                    "name": "zoom_02800_to_zoom_06353",
                    "source_zoom": 2800,
                    "target_zoom": 6353,
                    "source_pixel_scale": {
                        "mm_per_pixel": 0.045,
                        "horizontal": {"mm_per_pixel": 0.05},
                        "vertical": {"mm_per_pixel": 0.04},
                    },
                    "target_pixel_scale": {
                        "mm_per_pixel": 0.0225,
                        "horizontal": {"mm_per_pixel": 0.025},
                        "vertical": {"mm_per_pixel": 0.02},
                    },
                    "source_to_target": {
                        "matrix_3x3": [[2, 0, 10], [0, 2, 20], [0, 0, 1]]
                    },
                    "target_to_source": {
                        "matrix_3x3": [
                            [0.5, 0, -5],
                            [0, 0.5, -10],
                            [0, 0, 1],
                        ]
                    },
                },
            ],
        }

    def test_capture_frame_drains_the_reported_camera_buffer(self):
        class FakeCamera:
            def __init__(self):
                self.read_count = 0

            def read(self):
                self.read_count += 1
                return True, np.full((2, 2, 3), self.read_count, dtype=np.uint8)

        camera = FakeCamera()
        args = SimpleNamespace(
            capture_delay_seconds=0.0,
            capture_discard_frames=3,
            _capture_buffer_size_actual=4.0,
        )

        frame = capture_pcb_two_zooms.capture_pcb.capture_frame(
            camera, args, "test"
        )

        self.assertEqual(camera.read_count, 5)
        self.assertTrue(np.all(frame == 5))

    def build_args(self):
        values = vars(detection_args())
        return SimpleNamespace(
            **values,
            small_zoom=2800,
            large_zoom=6353,
            large_target_pixel=[200.0, 200.0],
        )

    def test_build_capture_args_uses_autofocus_position_for_each_zoom(self):
        runtime = capture_pcb_two_zooms.build_capture_args(
            self.build_args(), self.document, self.summary_path
        )

        self.assertEqual(runtime.small_focus, 4552)
        self.assertEqual(runtime.small_iris, 410)
        self.assertEqual(runtime.large_focus, 5858)
        self.assertEqual(runtime.large_iris, 420)
        self.assertTrue(runtime.undistort)
        self.assertEqual(runtime.lens_zoom, 6353)
        self.assertFalse(hasattr(runtime, "small_physical"))
        self.assertFalse(hasattr(runtime, "pcb_width_mm"))
        self.assertFalse(hasattr(runtime, "pcb_height_mm"))

    def test_missing_autofocus_focus_is_rejected(self):
        del self.document["entries"][0]["lens_focus"]

        with self.assertRaisesRegex(
            capture_pcb_two_zooms.TwoZoomCaptureError,
            "has no lens_focus",
        ):
            capture_pcb_two_zooms.autofocus_position_for_zoom(
                self.document, self.summary_path, 2800
            )

    def test_zoom_transform_supports_forward_and_reverse_directions(self):
        forward = capture_pcb_two_zooms.load_zoom_transform(
            np, self.document, self.summary_path, 2800, 6353
        )
        reverse = capture_pcb_two_zooms.load_zoom_transform(
            np, self.document, self.summary_path, 6353, 2800
        )

        self.assertEqual(
            capture_pcb_two_zooms.transform_zoom_pixel(np, forward, [3, 4]),
            [16.0, 28.0],
        )
        self.assertEqual(
            capture_pcb_two_zooms.transform_zoom_pixel(np, reverse, [16, 28]),
            [3.0, 4.0],
        )
        self.assertEqual(reverse["stored_direction"], "target_to_source")
        self.assertEqual(forward["source_pixel_scale"]["mm_per_pixel"], 0.045)
        self.assertEqual(forward["target_pixel_scale"]["mm_per_pixel"], 0.0225)
        self.assertEqual(reverse["source_pixel_scale"]["mm_per_pixel"], 0.0225)
        self.assertEqual(reverse["target_pixel_scale"]["mm_per_pixel"], 0.045)

    def test_calculates_pcb_size_from_four_inward_corners(self):
        transform = capture_pcb_two_zooms.load_zoom_transform(
            np, self.document, self.summary_path, 2800, 6353
        )
        geometry = capture_pcb_two_zooms.calculate_pcb_geometry_from_white_blocks(
            np,
            {
                "top_left": [100.0, 100.0],
                "top_right": [1100.0, 100.0],
                "bottom_right": [1100.0, 600.0],
                "bottom_left": [100.0, 600.0],
            },
            transform,
        )

        self.assertEqual(geometry["small_size_px"], {"width": 1000.0, "height": 500.0})
        self.assertEqual(geometry["physical_size_mm"], {"width": 50.0, "height": 20.0})
        self.assertEqual(geometry["large_size_px"], {"width": 2000.0, "height": 1000.0})
        np.testing.assert_allclose(
            geometry["large_mapped_inner_corners_px"]["top_left"], [210, 220]
        )
        np.testing.assert_allclose(geometry["large_width_direction"], [1, 0])
        np.testing.assert_allclose(geometry["large_height_direction"], [0, 1])

    def test_automatic_grid_covers_inferred_large_zoom_size_serpentine(self):
        transform = capture_pcb_two_zooms.load_zoom_transform(
            np, self.document, self.summary_path, 2800, 6353
        )
        geometry = capture_pcb_two_zooms.calculate_pcb_geometry_from_white_blocks(
            np,
            {
                "top_left": [100.0, 100.0],
                "top_right": [1100.0, 100.0],
                "bottom_right": [1100.0, 600.0],
                "bottom_left": [100.0, 600.0],
            },
            transform,
        )
        args = SimpleNamespace(
            large_target_pixel=[100.0, 100.0],
            capture_margin_mm=0.0,
            overlap_fraction=0.2,
        )
        stage = {"pixel_to_stage": np.asarray([[0.01, 0.0], [0.0, 0.01]])}

        cells, grid = capture_pcb_two_zooms.build_automatic_capture_grid(
            np, args, geometry, stage, frame_width=1200, frame_height=800
        )

        self.assertEqual((grid["columns"], grid["rows"]), (2, 2))
        self.assertEqual(
            [(cell["row"], cell["col"]) for cell in cells],
            [(0, 0), (0, 1), (1, 1), (1, 0)],
        )
        np.testing.assert_allclose(
            [cell["pixel_offset_px"] for cell in cells],
            [[0, 0], [900, 0], [900, 300], [0, 300]],
        )
        np.testing.assert_allclose(
            [cell["stage_offset_mm"] for cell in cells],
            [[0, 0], [-9, 0], [-9, -3], [0, -3]],
        )

    def test_pixel_alignment_uses_current_zoom_stage_matrix(self):
        stage = {
            "stage_to_pixel": np.asarray([[10.0, 0.0], [0.0, 20.0]]),
            "pixel_to_stage": np.asarray([[0.1, 0.0], [0.0, 0.05]]),
        }

        plan = capture_pcb_two_zooms.calculate_pixel_alignment_plan(
            np, [100.0, 200.0], [150.0, 140.0], stage
        )

        np.testing.assert_allclose(plan["requested_pixel_delta_px"], [50, -60])
        np.testing.assert_allclose(plan["requested_relative_stage_delta_mm"], [5, -3])
        self.assertAlmostEqual(plan["conversion_error_px"], 0.0)

    def test_absolute_stage_restore_uses_saved_small_zoom_photo_position(self):
        controllers = []

        class FakeController:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.closed = False
                controllers.append(self)

            def close(self):
                self.closed = True

        motion_module = SimpleNamespace(
            MODE_ABSOLUTE="absolute",
            XYMotionController=FakeController,
        )
        args = SimpleNamespace(
            port="COM3",
            baudrate=38400,
            x_lead_mm_per_rev=1.0,
            y_lead_mm_per_rev=1.0,
            x_slave_address=1,
            y_slave_address=2,
        )
        expected = {
            "actual_position_mm": [24.5, -16.0],
            "error_distance_mm": 0.002,
        }
        with (
            mock.patch.dict(sys.modules, {"motion_controller": motion_module}),
            mock.patch.object(
                capture_pcb_two_zooms.capture_pcb,
                "read_stage_position",
                return_value=np.asarray([18.0, -17.0]),
            ),
            mock.patch.object(
                capture_pcb_two_zooms.capture_pcb,
                "move_stage_absolute",
                return_value=expected,
            ) as move,
        ):
            result = capture_pcb_two_zooms.execute_absolute_stage_move(
                np,
                args,
                [24.5, -16.0],
                "restore small zoom photo position",
                200.0,
            )

        self.assertIs(result, expected)
        np.testing.assert_allclose(move.call_args.args[2], [24.5, -16.0])
        self.assertEqual(move.call_args.args[5], "restore small zoom photo position")
        self.assertEqual(move.call_args.args[6], 200.0)
        self.assertEqual(controllers[0].args, ("COM3",))
        self.assertTrue(controllers[0].closed)

    def test_detects_four_blocks_center_anchor_and_crop(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        # A closed white fixture border encloses all four markers. RETR_EXTERNAL
        # cannot see the markers in this layout.
        cv2.rectangle(image, (15, 15), (785, 585), (255, 255, 255), 10)
        for first, second in (
            ((50, 50), (100, 100)),
            ((700, 50), (750, 100)),
            ((700, 500), (750, 550)),
            ((50, 500), (100, 550)),
        ):
            cv2.rectangle(image, first, second, (255, 255, 255), -1)

        result, mask = capture_pcb_two_zooms.detect_four_white_blocks(
            cv2, np, image, detection_args()
        )
        annotated = capture_pcb_two_zooms.draw_white_block_detection(
            cv2, np, image, result
        )
        cropped, bounds = capture_pcb_two_zooms.crop_global_image(np, image, result, 0)

        np.testing.assert_allclose(result["blocks_center_px"], [400, 300])
        np.testing.assert_allclose(result["inner_corners_center_px"], [400, 300])
        np.testing.assert_allclose(result["alignment_center_px"], [400, 300])
        np.testing.assert_allclose(result["image_center_px"], [400, 300])
        np.testing.assert_allclose(result["top_left_block_bottom_right_px"], [100, 100])
        self.assertEqual(result["accepted_candidate_count"], 4)
        self.assertEqual(result["standalone_candidate_count"], 4)
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertEqual(annotated.shape, image.shape)
        self.assertEqual(bounds, [100, 100, 701, 501])
        self.assertEqual(cropped.shape[:2], (401, 601))

    def test_negative_global_crop_padding_contracts_inward(self):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        detection = {
            "inner_corners_px": {
                "top_left": [50.0, 40.0],
                "top_right": [350.0, 40.0],
                "bottom_right": [350.0, 260.0],
                "bottom_left": [50.0, 260.0],
            }
        }

        cropped, bounds = capture_pcb_two_zooms.crop_global_image(
            np, image, detection, -20
        )

        self.assertEqual(bounds, [70, 60, 331, 241])
        self.assertEqual(cropped.shape[:2], (181, 261))

    def test_excessive_negative_global_crop_padding_is_rejected(self):
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        detection = {
            "inner_corners_px": {
                "top_left": [100.0, 100.0],
                "top_right": [200.0, 100.0],
                "bottom_right": [200.0, 200.0],
                "bottom_left": [100.0, 200.0],
            }
        }

        with self.assertRaisesRegex(
            capture_pcb_two_zooms.TwoZoomCaptureError, "valid crop"
        ):
            capture_pcb_two_zooms.crop_global_image(np, image, detection, -60)

    def test_falls_back_threshold_and_rejects_hollow_pcb_features(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        for first, second in (
            ((50, 50), (100, 100)),
            ((700, 50), (750, 100)),
            ((700, 500), (750, 550)),
            ((50, 500), (100, 550)),
        ):
            cv2.rectangle(image, first, second, (215, 215, 215), -1)
        for center in ((260, 200), (540, 200), (540, 400), (260, 400)):
            cv2.circle(image, center, 35, (255, 255, 255), -1)
            cv2.circle(image, center, 20, (0, 0, 0), -1)

        result, _ = capture_pcb_two_zooms.detect_four_white_blocks(
            cv2, np, image, detection_args()
        )

        self.assertEqual(result["requested_threshold"], 220)
        self.assertEqual(result["threshold"], 210)
        self.assertTrue(result["threshold_fallback_used"])
        self.assertTrue(result["layout_quality"]["plausible"])
        self.assertGreaterEqual(result["rejected_hollow_candidate_count"], 4)
        np.testing.assert_allclose(
            [
                result["blocks"][name]["center_px"]
                for name in capture_pcb_two_zooms.BLOCK_NAMES
            ],
            [[75, 75], [725, 75], [725, 525], [75, 525]],
        )

    def test_block_layout_can_be_offset_from_image_quadrants(self):
        image = np.zeros((600, 1000, 3), dtype=np.uint8)
        for first, second in (
            ((50, 100), (100, 150)),
            ((350, 100), (400, 150)),
            ((350, 400), (400, 450)),
            ((50, 400), (100, 450)),
        ):
            cv2.rectangle(image, first, second, (255, 255, 255), -1)

        result, _ = capture_pcb_two_zooms.detect_four_white_blocks(
            cv2, np, image, detection_args()
        )

        np.testing.assert_allclose(result["blocks_center_px"], [225, 275])
        np.testing.assert_allclose(result["alignment_center_px"], [225, 275])
        np.testing.assert_allclose(result["image_center_px"], [500, 300])
        self.assertGreater(result["center_to_image_distance_px"], 270)

    def test_alignment_uses_inward_corners_instead_of_block_centers(self):
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        for first, second in (
            ((50, 50), (130, 130)),
            ((680, 60), (750, 130)),
            ((690, 480), (750, 550)),
            ((50, 490), (120, 550)),
        ):
            cv2.rectangle(image, first, second, (255, 255, 255), -1)

        result, _ = capture_pcb_two_zooms.detect_four_white_blocks(
            cv2, np, image, detection_args()
        )

        np.testing.assert_allclose(result["blocks_center_px"], [402.5, 305.0])
        np.testing.assert_allclose(result["inner_corners_center_px"], [405.0, 307.5])
        np.testing.assert_allclose(result["alignment_center_px"], [405.0, 307.5])
        np.testing.assert_allclose(result["center_to_image_delta_px"], [-5.0, -7.5])


if __name__ == "__main__":
    unittest.main()
