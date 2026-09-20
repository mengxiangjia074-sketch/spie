import os
import sys
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from GUI.app import (
        MainWindow,
        MODE_CALIBRATION,
        MODE_CAMERA,
        MODE_CAPTURE,
        MODE_MOTION,
        MODE_RFID,
    )
except ImportError:
    QApplication = None
    MainWindow = None


@unittest.skipIf(MainWindow is None, "PySide6 is not installed")
class GuiConsoleSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_device_and_task_logs_stay_in_their_own_console(self):
        panels = self.window.console_panels
        self.window.camera_page.console_output.emit("comm-info", "camera-only")
        self.window.motion_page.console_output.emit("comm-info", "motion-only")
        self.window.rfid_page.console_output.emit("comm-info", "rfid-only")
        self.window.manager.task_output.emit("mosaic", "stdout", "capture-only")
        self.window.manager.task_output.emit("lens", "stdout", "calibration-only")
        self.app.processEvents()

        self.assertEqual(panels[MODE_CAMERA].view.toPlainText(), "camera-only")
        self.assertEqual(panels[MODE_MOTION].view.toPlainText(), "motion-only")
        self.assertEqual(panels[MODE_RFID].view.toPlainText(), "rfid-only")
        self.assertEqual(panels[MODE_CAPTURE].view.toPlainText(), "capture-only")
        self.assertEqual(
            panels[MODE_CALIBRATION].view.toPlainText(), "calibration-only"
        )

        panels[MODE_CAMERA]._clear()
        self.assertEqual(panels[MODE_CAMERA].view.toPlainText(), "")
        self.assertEqual(panels[MODE_MOTION].view.toPlainText(), "motion-only")

    def test_calibration_error_lines_use_blue_accent_text(self):
        panel = self.window.console_panels[MODE_CALIBRATION]

        with mock.patch.object(panel, "_append_html") as append_html:
            panel._on_output("lens", "stdout", "标定误差: RMS 重投影误差 0.4894 px")

        fragment = append_html.call_args.args[0]
        self.assertIn(f'color:{panel._colors()["accent"]}', fragment)
        self.assertIn("标定误差: RMS 重投影误差 0.4894 px", fragment)

    def test_console_selection_and_visibility_are_independent_per_mode(self):
        self.window.resize(1280, 820)
        self.window.show()
        self.window.enter_mode(MODE_CAMERA)
        self.window._set_console_visible(MODE_CAMERA, True)
        self.app.processEvents()
        self.assertIs(
            self.window.console_stack.currentWidget(),
            self.window.console_panels[MODE_CAMERA],
        )
        self.assertTrue(self.window._console_visible_by_mode[MODE_CAMERA])

        self.window.enter_mode(MODE_MOTION)
        self.assertIs(
            self.window.console_stack.currentWidget(),
            self.window.console_panels[MODE_MOTION],
        )
        self.assertFalse(self.window._console_visible_by_mode[MODE_MOTION])
        self.assertTrue(self.window._console_visible_by_mode[MODE_CAMERA])

        self.window._set_console_visible(MODE_MOTION, True)
        self.window.resizeDocks(
            [self.window.console_dock], [500], Qt.Vertical
        )
        self.app.processEvents()
        self.assertGreaterEqual(self.window.console_dock.height(), 480)


if __name__ == "__main__":
    unittest.main()
