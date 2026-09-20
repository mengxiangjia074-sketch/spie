import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

try:
    from PySide6.QtWidgets import QApplication
    from GUI.pages.motion_control import AxisPanel, STATUS_DEFINITIONS
    from motion_controller import ADDR_X_AXIS
except ModuleNotFoundError:
    QApplication = None
    AxisPanel = None
    STATUS_DEFINITIONS = ()
    ADDR_X_AXIS = 1


@unittest.skipIf(AxisPanel is None, "PySide6 is not installed")
class MotionControlStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = AxisPanel("X轴", ADDR_X_AXIS)

    def tearDown(self):
        self.panel.deleteLater()

    def test_status_bits_have_independent_indicators_and_severity(self):
        expected_keys = {definition[0] for definition in STATUS_DEFINITIONS}
        self.assertEqual(set(self.panel.status_indicators), expected_keys)
        self.assertTrue(
            all(
                indicator.property("active") is False
                for indicator in self.panel.status_indicators.values()
            )
        )

        self.panel.update_status(
            {
                "fault": True,
                "enabled": True,
                "running": False,
                "invalid": True,
                "cmd_done": True,
                "path_done": False,
                "home_done": False,
            }
        )

        self.assertTrue(self.panel.status_indicators["fault"].property("active"))
        self.assertEqual(
            self.panel.status_indicators["fault"].property("severity"), "alert"
        )
        self.assertTrue(self.panel.status_indicators["enabled"].property("active"))
        self.assertEqual(
            self.panel.status_indicators["enabled"].property("severity"), "normal"
        )
        self.assertFalse(self.panel.status_indicators["running"].property("active"))
        self.assertTrue(self.panel.status_indicators["invalid"].property("active"))

    def test_reset_clears_all_status_indicators(self):
        self.panel.update_status(
            {key: True for key, *_rest in STATUS_DEFINITIONS}
        )
        self.panel.reset_display()

        self.assertTrue(
            all(
                indicator.property("active") is False
                for indicator in self.panel.status_indicators.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
