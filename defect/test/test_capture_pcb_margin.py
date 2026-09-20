import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "capture"))

import capture_pcb


class CapturePcbMarginTests(unittest.TestCase):
    def test_grid_axis_expands_only_far_edge(self):
        count, spacing, start = capture_pcb.grid_axis(
            129.0, 87.0, 0.2, 5.0
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(spacing, 47.0)
        self.assertAlmostEqual(start + (count - 1) * spacing + 87.0, 134.0)

    def test_single_frame_keeps_zero_origin(self):
        count, spacing, start = capture_pcb.grid_axis(
            20.0, 40.0, 0.2, 5.0
        )

        self.assertEqual(count, 1)
        self.assertEqual(spacing, 0.0)
        self.assertEqual(start, 0.0)

    def test_capture_cells_keep_top_left_and_expand_bottom_right(self):
        np = capture_pcb.load_numpy()
        args = SimpleNamespace(
            pcb_width_mm=129.0,
            pcb_height_mm=66.0,
            frame_width=87,
            frame_height=48,
            overlap_fraction=0.2,
            capture_margin_mm=5.0,
        )
        identity = np.eye(2, dtype=np.float64)
        cells, grid = capture_pcb.build_capture_cells(
            np,
            {"board_to_pixel": identity},
            {"pixel_to_stage": identity},
            args,
        )

        self.assertEqual((grid["columns"], grid["rows"]), (2, 2))
        self.assertEqual(cells[0]["board_offset_mm"], [0.0, 0.0])
        self.assertEqual(cells[-1]["board_offset_mm"], [0.0, 23.0])
        self.assertEqual(
            grid["coverage_bounds_board_mm"],
            {"x_min": 0.0, "x_max": 134.0, "y_min": 0.0, "y_max": 71.0},
        )


if __name__ == "__main__":
    unittest.main()
