import hashlib
import os
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QLabel
    from GUI.pages.home import HomePage
except ModuleNotFoundError:
    QApplication = None
    HomePage = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class HomeBackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = HomePage()
        self.page.resize(1200, 760)

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _digest(page):
        image = page.grab().toImage()
        return hashlib.sha256(image.bits().tobytes()).digest()

    def test_animation_runs_only_while_home_page_is_visible(self):
        self.assertFalse(self.page._background_timer.isActive())

        self.page.show()
        self.app.processEvents()
        self.assertTrue(self.page._background_timer.isActive())

        self.page.hide()
        self.app.processEvents()
        self.assertFalse(self.page._background_timer.isActive())

    def test_different_animation_frames_render_different_backgrounds(self):
        self.page.show()
        self.app.processEvents()
        self.page._background_timer.stop()

        self.page._background_frame = 0
        self.page.repaint()
        first = self._digest(self.page)
        self.page._background_frame = 40
        self.page.repaint()
        second = self._digest(self.page)

        self.assertNotEqual(first, second)

    def test_function_cards_follow_requested_two_row_order(self):
        expected = (
            ("标定", "图像采集", "RFID检测"),
            ("镜头控制", "位移台控制", "相机控制"),
        )

        actual = tuple(
            tuple(
                self.page.cards_grid.itemAtPosition(row, column)
                .widget()
                .findChild(QLabel, "EntryTitle")
                .text()
                for column in range(3)
            )
            for row in range(2)
        )

        self.assertEqual(actual, expected)

    def test_home_header_has_branded_visual_hierarchy(self):
        eyebrow = self.page.findChild(QLabel, "HomeEyebrow")
        title = self.page.findChild(QLabel, "HomeTitle")

        self.assertEqual(eyebrow.text(), "DefectDetection")
        self.assertEqual(title.text(), "主界面")
        self.assertIsNone(self.page.findChild(QLabel, "HomeSubtitle"))


if __name__ == "__main__":
    unittest.main()
