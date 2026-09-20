import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "test"))

try:
    import cv2
    import numpy as np
    import capture_optical_center as target
except ModuleNotFoundError:
    cv2 = None
    np = None
    target = None


@unittest.skipIf(target is None, "OpenCV/NumPy is not installed")
class CaptureOpticalCenterTests(unittest.TestCase):
    def test_loads_matching_zoom_and_scales_principal_point(self):
        document = {
            "entries": [{
                "type": "lens_checkerboard_camera_calibration",
                "lens_zoom": 1234,
                "positions": [{
                    "name": "zoom_01234",
                    "image_width": 1000,
                    "image_height": 500,
                    "camera_matrix": [
                        [900.0, 0.0, 400.0],
                        [0.0, 901.0, 200.0],
                        [0.0, 0.0, 1.0],
                    ],
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrate.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            calibration = target.load_optical_center_calibration(path, 1234)
            center = target.scaled_optical_center(calibration, 2000, 1000)

        self.assertEqual(center, (800.0, 400.0))
        self.assertEqual(calibration.position_name, "zoom_01234")

    def test_reports_available_zooms_when_requested_calibration_is_missing(self):
        document = {
            "entries": [{
                "lens_zoom": 1501,
                "positions": [{
                    "image_width": 100,
                    "image_height": 80,
                    "camera_matrix": [
                        [90.0, 0.0, 50.0],
                        [0.0, 91.0, 40.0],
                        [0.0, 0.0, 1.0],
                    ],
                }],
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrate.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                target.OpticalCenterCaptureError, "available zooms: 1501"
            ):
                target.load_optical_center_calibration(path, 6353)

    def test_annotation_marks_center_without_modifying_source_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        annotated = target.annotate_optical_center(cv2, frame, 160.0, 120.0)

        self.assertFalse(np.any(frame))
        self.assertTrue(np.any(annotated))
        self.assertEqual(tuple(annotated[120, 160]), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
