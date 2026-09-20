import ast
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_ROOT = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_ROOT))

from GUI.core.jsonnet_store import load_effective  # noqa: E402
from GUI.core.paths import CONFIGS, SCRIPTS  # noqa: E402
from GUI.pages.registry import TASKS  # noqa: E402


class GuiTwoZoomCaptureTests(unittest.TestCase):
    def _mosaic_field_names(self) -> set[str]:
        registry_path = DETECTION_ROOT / "GUI" / "pages" / "registry.py"
        tree = ast.parse(registry_path.read_text(encoding="utf-8"))

        tasks = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "TASKS"
                for target in node.targets
            )
        )
        mosaic = next(
            task
            for task in tasks.elts
            if isinstance(task, ast.Call)
            and any(
                keyword.arg == "id"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "mosaic"
                for keyword in task.keywords
            )
        )
        fields = next(
            keyword.value for keyword in mosaic.keywords if keyword.arg == "fields"
        )

        names: set[str] = set()
        helper_fields = {
            "camera_name_field": {"camera_name"},
            "camera_index_field": {"camera"},
            "frame_size_fields": {"frame_width", "frame_height"},
            "stage_serial_fields": {
                "port",
                "baudrate",
                "x_lead_mm_per_rev",
                "y_lead_mm_per_rev",
                "x_slave_address",
                "y_slave_address",
            },
        }
        for item in fields.elts:
            call = item.value if isinstance(item, ast.Starred) else item
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id == "F":
                names.add(call.args[0].value)
            else:
                names.update(helper_fields.get(call.func.id, set()))
        return names

    def test_gui_launches_two_zoom_capture_with_dedicated_config(self):
        self.assertEqual(SCRIPTS["mosaic"].name, "capture_pcb_two_zooms.py")
        self.assertEqual(CONFIGS["capture"].parent.name, "pcb_two_zoom_capture")

    def test_gui_fields_match_two_zoom_config(self):
        fields = self._mosaic_field_names()
        config_keys = set(load_effective(CONFIGS["capture"]))

        self.assertTrue(
            {
                "small_zoom",
                "white_block_threshold",
                "white_block_min_area_px",
                "white_block_max_area_fraction",
                "white_block_min_square_ratio",
                "white_block_min_fill_ratio",
                "white_block_morph_kernel",
                "global_crop_padding_px",
                "large_zoom",
                "large_target_pixel",
            }.issubset(fields)
        )
        self.assertTrue(fields.issubset(config_keys), fields - config_keys)
        self.assertTrue({"pcb_width_mm", "pcb_height_mm"}.isdisjoint(config_keys))
        self.assertTrue(
            {
                "lens_zoom",
                "small_focus",
                "large_focus",
                "undistort",
                "capture_autofocus",
                "small_physical",
                "small_target_pixel",
                "large_physical",
                "pcb_width_mm",
                "pcb_height_mm",
            }.isdisjoint(fields)
        )

    def test_local_image_section_precedes_white_block_detection(self):
        mosaic = next(task for task in TASKS if task.id == "mosaic")
        section_order = list(dict.fromkeys(field.section for field in mosaic.fields))

        self.assertLess(
            section_order.index("局部图"), section_order.index("白块识别")
        )


if __name__ == "__main__":
    unittest.main()
