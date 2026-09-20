import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from GUI.pages import component_inspection_page as page_module
    from GUI.pages.component_inspection_page import ComponentInspectionPage
except ImportError:
    QApplication = None
    ComponentInspectionPage = None


if QApplication is not None:

    class FakeTaskManager(QObject):
        busy_changed = Signal(bool)
        task_started = Signal(object)
        task_finished = Signal(str, str, int)

        def __init__(self):
            super().__init__()
            self._busy = False

        def busy(self):
            return self._busy

        def start_capture(self):
            self._busy = True
            self.busy_changed.emit(True)
            self.task_started.emit(SimpleNamespace(id="mosaic"))

        def finish_capture(self, result="成功", exit_code=0):
            self._busy = False
            self.busy_changed.emit(False)
            self.task_finished.emit("mosaic", result, exit_code)


@unittest.skipIf(ComponentInspectionPage is None, "PySide6 is not installed")
class ComponentInspectionCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def _page(self, capture_directory):
        manager = FakeTaskManager()
        patches = (
            mock.patch.object(page_module, "latest_ground_truth", return_value=None),
            mock.patch.object(
                page_module, "latest_capture_source", return_value=capture_directory
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        page = ComponentInspectionPage(manager)
        self.addCleanup(page.deleteLater)
        return page, manager

    def test_capture_button_starts_mosaic_and_loads_finished_capture(self):
        capture_directory = PROJECT_ROOT / "working_data" / "capture_test"
        page, manager = self._page(capture_directory)
        page.inspection_edit.clear()
        page.capture_requested.connect(manager.start_capture)

        page.capture_button.click()

        self.assertTrue(page._capture_pending)
        self.assertTrue(manager.busy())
        self.assertFalse(page.capture_button.isEnabled())
        self.assertIn("正在执行两倍率 PCB 拍摄", page.status_label.text())

        manager.finish_capture()

        self.assertFalse(page._capture_pending)
        self.assertEqual(page.inspection_edit.text(), str(capture_directory))
        self.assertTrue(page.capture_button.isEnabled())
        self.assertIn("可以开始完整性检测", page.status_label.text())

    def test_failed_capture_keeps_existing_inspection_input(self):
        capture_directory = PROJECT_ROOT / "working_data" / "capture_test"
        page, manager = self._page(capture_directory)
        existing = PROJECT_ROOT / "existing_images"
        page.inspection_edit.setText(str(existing))
        page.capture_requested.connect(manager.start_capture)

        page.capture_button.click()
        manager.finish_capture("相机打开失败", 1)

        self.assertFalse(page._capture_pending)
        self.assertEqual(page.inspection_edit.text(), str(existing))
        self.assertIn("拍摄未完成", page.status_label.text())


if __name__ == "__main__":
    unittest.main()
