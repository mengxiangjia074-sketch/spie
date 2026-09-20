import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from defect_sim import (  # noqa: E402
    CODEX_AGENT_MODEL,
    DefectSimConfig,
    _codex_image_failure_message,
    check_codex_image_capability,
    simulate_defect,
)

try:
    from PIL import Image, ImageDraw
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication, QScrollArea

    from GUI.app import DEFECT_SIM_KEY, MODE_CAPTURE, MainWindow
    from GUI.pages.defect_sim_page import DefectSimPage, latest_local_image
    from GUI.widgets.resultview import ImageView
except ImportError:
    QApplication = None
    MainWindow = None


class DefectSimulationBackendTests(unittest.TestCase):
    def test_codex_image_403_is_not_reported_as_logged_out(self):
        message = _codex_image_failure_message(
            'image generation failed: http 403 Forbidden: {"detail":"Forbidden"}'
        )

        self.assertIn("Codex 已登录 ChatGPT", message)
        self.assertIn("图像生成权限", message)
        self.assertNotIn("请先运行 `codex login`", message)

    @mock.patch("defect_sim.subprocess.run")
    @mock.patch("defect_sim.shutil.which", return_value="/usr/bin/codex")
    def test_codex_capability_probe_checks_session_tools(self, _which, run):
        run.side_effect = [
            mock.Mock(returncode=0, stdout="Logged in using ChatGPT", stderr=""),
            mock.Mock(returncode=0, stdout="IMAGE_GEN_AVAILABLE", stderr=""),
        ]

        available, message = check_codex_image_capability()

        self.assertTrue(available)
        self.assertIn("image_gen", message)
        probe_command = run.call_args_list[1].args[0]
        self.assertNotIn("gpt-5.4", probe_command)
        self.assertIn(CODEX_AGENT_MODEL, probe_command)

    @mock.patch("defect_sim.subprocess.run")
    @mock.patch("defect_sim.shutil.which", return_value="/usr/bin/codex")
    def test_codex_capability_probe_does_not_treat_login_as_permission(self, _which, run):
        run.side_effect = [
            mock.Mock(returncode=0, stdout="Logged in using ChatGPT", stderr=""),
            mock.Mock(returncode=0, stdout="IMAGE_GEN_UNAVAILABLE", stderr=""),
        ]

        available, message = check_codex_image_capability()

        self.assertFalse(available)
        self.assertIn("未提供 image_gen", message)

    def test_local_inpaint_saves_image_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            image = Image.new("RGB", (80, 60), (25, 105, 45))
            ImageDraw.Draw(image).rectangle((30, 22, 50, 38), fill=(30, 30, 30))
            image.save(source)

            outputs = simulate_defect(
                source,
                (28, 20, 52, 40),
                root / "output",
                DefectSimConfig(wire_api="inpaint"),
            )

            self.assertTrue(outputs["edited"].is_file())
            self.assertTrue(outputs["metadata"].is_file())

    def test_latest_local_image_uses_latest_capture_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "20260830_100000" / "images"
            newer = root / "20260830_110000" / "images"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "r00_c00.png").write_bytes(b"older")
            expected = newer / "r00_c01.png"
            expected.write_bytes(b"newer")
            os.utime(older.parent, (1, 1))
            os.utime(newer.parent, (2, 2))

            self.assertEqual(latest_local_image(root), expected)


@unittest.skipIf(MainWindow is None, "PySide6 or Pillow is not installed")
class DefectSimulationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def test_page_is_immediately_after_pcb_capture(self):
        with mock.patch.object(DefectSimPage, "_load_latest_image"):
            window = MainWindow()
        try:
            sidebar, _ = window._workspaces[MODE_CAPTURE]
            keys = [sidebar.item(row).data(Qt.UserRole) for row in range(sidebar.count())]
            mosaic_index = keys.index("mosaic")
            self.assertEqual(keys[mosaic_index + 1], DEFECT_SIM_KEY)
            self.assertIsInstance(window.pages[DEFECT_SIM_KEY], DefectSimPage)
        finally:
            window.close()
            self.app.processEvents()

    def test_advanced_settings_use_a_scrollable_dialog(self):
        with mock.patch.object(DefectSimPage, "_load_latest_image"):
            page = DefectSimPage()
        try:
            self.assertIsNotNone(page.advanced_dialog.findChild(QScrollArea))
            self.assertGreaterEqual(page.advanced_dialog.minimumWidth(), 560)
            self.assertGreaterEqual(page.prompt_edit.minimumHeight(), 120)
            self.assertEqual(page.capability_button.text(), "检测生图权限")
            self.assertEqual(page.capability_label.text(), "尚未检测")
            self.assertFalse(page.advanced_dialog.isVisible())
        finally:
            page.deleteLater()
            self.app.processEvents()

    def test_result_image_manual_zoom_is_not_reset_by_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "result.png"
            Image.new("RGB", (1200, 800), (25, 105, 45)).save(image_path)
            view = ImageView()
            view.resize(480, 320)
            view.show()
            self.app.processEvents()
            try:
                self.assertTrue(view.load(image_path))
                fitted_scale = view.transform().m11()
                wheel = mock.Mock()
                wheel.angleDelta.return_value = QPoint(0, 120)

                view.wheelEvent(wheel)
                zoomed_scale = view.transform().m11()
                view.resize(500, 340)
                self.app.processEvents()

                self.assertGreater(zoomed_scale, fitted_scale)
                self.assertAlmostEqual(view.transform().m11(), zoomed_scale)
                wheel.accept.assert_called_once_with()
            finally:
                view.close()
                view.deleteLater()
                self.app.processEvents()

    def test_errors_are_written_only_to_capture_console(self):
        with mock.patch.object(DefectSimPage, "_load_latest_image"):
            window = MainWindow()
        try:
            window.enter_mode(MODE_CAPTURE)
            previous_status = window.defect_sim_page.status_label.text()
            window.defect_sim_page._set_status("模拟错误", error=True)
            self.app.processEvents()

            self.assertIn(
                "[缺陷模拟] 模拟错误",
                window.console_panels[MODE_CAPTURE].view.toPlainText(),
            )
            self.assertNotIn(
                "模拟错误", window.console_panels["calibration"].view.toPlainText()
            )
            self.assertEqual(
                window.defect_sim_page.status_label.text(), previous_status
            )
            self.assertTrue(window._console_visible_by_mode[MODE_CAPTURE])
            self.assertFalse(window.console_dock.isHidden())
        finally:
            window.close()
            self.app.processEvents()

    def test_missing_input_reports_without_a_dialog(self):
        with mock.patch.object(DefectSimPage, "_load_latest_image"):
            window = MainWindow()
        try:
            window.enter_mode(MODE_CAPTURE)
            window.defect_sim_page.image_path = None

            window.defect_sim_page._generate()
            self.app.processEvents()

            self.assertIn(
                "请先加载局部图",
                window.console_panels[MODE_CAPTURE].view.toPlainText(),
            )
        finally:
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
