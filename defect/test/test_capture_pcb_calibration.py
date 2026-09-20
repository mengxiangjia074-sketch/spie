import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "capture"))

import capture_pcb


class CapturePcbCalibrationTests(unittest.TestCase):
    def test_loads_calibration_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrate.json"
            path.write_text(json.dumps({"entries": [{"type": "example"}]}))

            document = capture_pcb.load_calibration_summary_document(path)

        self.assertEqual(document["entries"][0]["type"], "example")

    def test_rejects_document_without_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrate.json"
            path.write_text(json.dumps({"entries": {}}))

            with self.assertRaisesRegex(
                capture_pcb.PcbMosaicError, "entries"
            ):
                capture_pcb.load_calibration_summary_document(path)

    def test_parses_board_matrix_and_stage_configuration(self):
        result = capture_pcb.parse_board_calibration(
            np,
            {
                "board_to_pixel": {
                    "matrix_2x3": [[100.0, 0.0, 20.0], [0.0, 80.0, 30.0]]
                },
                "stage_reference": {
                    "configuration": {"port": "COM3", "baudrate": 38400}
                },
            },
        )

        np.testing.assert_allclose(
            result["board_to_pixel"],
            [[100.0, 0.0, 20.0], [0.0, 80.0, 30.0]],
        )
        self.assertEqual(result["stage_reference_configuration"]["port"], "COM3")

    def test_parses_stage_matrix_and_calculates_missing_inverse(self):
        result = capture_pcb.parse_stage_calibration(
            np,
            {
                "stage_command_to_pixel": {
                    "matrix_2x2": [[10.0, 2.0], [1.0, -5.0]]
                },
                "source_board_calibration_json": "board.json",
            },
        )

        np.testing.assert_allclose(
            result["pixel_to_stage"] @ result["stage_to_pixel"],
            np.eye(2),
            atol=1e-12,
        )
        self.assertEqual(result["source_board_calibration_json"], "board.json")

    def test_rejects_inconsistent_stage_inverse(self):
        with self.assertRaisesRegex(
            capture_pcb.PcbMosaicError, "inconsistent"
        ):
            capture_pcb.parse_stage_calibration(
                np,
                {
                    "stage_command_to_pixel": {
                        "matrix_2x2": [[10.0, 0.0], [0.0, 5.0]],
                        "pixel_to_stage_command": {
                            "matrix_2x2": [[1.0, 0.0], [0.0, 1.0]]
                        },
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
