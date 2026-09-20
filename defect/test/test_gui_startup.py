import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_ROOT = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_ROOT))

from GUI import main as gui_main  # noqa: E402


class GuiStartupTests(unittest.TestCase):
    def test_main_window_opens_maximized(self):
        events = []

        class FakeApplication:
            def __init__(self, arguments):
                events.append(("application", arguments))

            def setApplicationName(self, name):
                events.append(("application_name", name))

            def setOrganizationName(self, name):
                events.append(("organization_name", name))

            def setStyle(self, style):
                events.append(("style", style))

            def exec(self):
                events.append(("exec",))
                return 0

        class FakeMainWindow:
            def showMaximized(self):
                events.append(("show_maximized",))

        widgets_module = ModuleType("PySide6.QtWidgets")
        widgets_module.QApplication = FakeApplication
        app_module = ModuleType("GUI.app")
        app_module.MainWindow = FakeMainWindow

        with mock.patch.dict(
            sys.modules,
            {
                "PySide6.QtWidgets": widgets_module,
                "GUI.app": app_module,
            },
        ):
            result = gui_main.main()

        self.assertEqual(result, 0)
        self.assertIn(("show_maximized",), events)
        self.assertLess(events.index(("show_maximized",)), events.index(("exec",)))


if __name__ == "__main__":
    unittest.main()
