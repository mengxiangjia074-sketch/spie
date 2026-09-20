import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMS_PATH = PROJECT_ROOT / "detection" / "GUI" / "widgets" / "forms.py"


class GuiNumericWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(FORMS_PATH.read_text(encoding="utf-8"))

    def test_numeric_spin_boxes_ignore_wheel_events(self):
        classes = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
        }

        for name in ("NoWheelSpinBox", "NoWheelDoubleSpinBox"):
            wheel_event = next(
                node
                for node in classes[name].body
                if isinstance(node, ast.FunctionDef) and node.name == "wheelEvent"
            )
            calls = [
                node
                for node in ast.walk(wheel_event)
                if isinstance(node, ast.Call)
            ]
            self.assertTrue(
                any(
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "ignore"
                    for call in calls
                ),
                name,
            )

    def test_all_numeric_editors_use_no_wheel_spin_boxes(self):
        constructors = {
            node.func.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("NoWheelSpinBox", constructors)
        self.assertIn("NoWheelDoubleSpinBox", constructors)
        self.assertNotIn("QSpinBox", constructors)
        self.assertNotIn("QDoubleSpinBox", constructors)


if __name__ == "__main__":
    unittest.main()
