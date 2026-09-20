import math
import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_DIR))

try:
    from PySide6.QtCore import QPoint, Qt, QObject, Signal
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from GUI.core.camera_worker import (
        CameraWorker,
        exposure_backend_value,
        exposure_milliseconds,
        property_is_supported,
    )
    from GUI.core.runner import TaskManager
    from GUI.pages.camera_control import (
        CameraControlPage,
        ExposureTimeSlider,
        PreviewWidget,
        PropertySlider,
    )
    from GUI.widgets.forms import NoWheelDoubleSpinBox
    from GUI.pages.home import EntryCard, HomePage
except ImportError:
    QApplication = None
    CameraControlPage = None


class FakeCameraWorker(QObject):
    devices_ready = Signal(bool, list, str)
    connected = Signal(dict)
    disconnected = Signal(str)
    failed = Signal(str)
    frame_ready = Signal(object)
    stats_updated = Signal(float, int, int)
    properties_updated = Signal(dict)
    property_updated = Signal(str, float, bool, str)
    resolution_updated = Signal(int, int)
    photo_saved = Signal(str)
    recording_changed = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.calls = []

    def _call(self, name, *args):
        self.calls.append((name, *args))

    def scan(self):
        self._call("scan")

    def connect_device(self, *args):
        self._call("connect", *args)

    def disconnect_device(self):
        self._call("disconnect")

    def set_resolution(self, *args):
        self._call("resolution", *args)

    def set_property(self, *args):
        self._call("property", *args)

    def reset_properties(self, *args):
        self._call("reset", *args)

    def white_balance_once(self):
        self._call("white_balance_once")

    def take_photo(self, *args):
        self._call("photo", *args)

    def start_recording(self, *args):
        self._call("start_recording", *args)

    def stop_recording(self):
        self._call("stop_recording")


@unittest.skipIf(
    QApplication is None or CameraControlPage is None,
    "PySide6 is not installed",
)
class CameraControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_page(self):
        worker = FakeCameraWorker()
        page = CameraControlPage(
            TaskManager(), worker=worker, auto_scan=False
        )
        return page, worker

    def test_camera_worker_preserves_linux_device_path(self):
        worker = CameraWorker()
        worker.connect_device("/dev/video0", 1920, 1080, "MJPG")

        request = worker._queue.get_nowait()

        self.assertEqual(request["index"], "/dev/video0")
        self.assertEqual(request["width"], 1920)

    def test_camera_worker_normalizes_numeric_camera_source(self):
        worker = CameraWorker()
        worker.connect_device("2", 640, 480, "MJPG")

        request = worker._queue.get_nowait()

        self.assertEqual(request["index"], 2)

    @staticmethod
    def properties():
        values = {
            "auto_exposure": (True, 1.0),
            "exposure": (True, -6.0),
            "exposure_target": (True, 50.0),
            "auto_white_balance": (True, 1.0),
            "white_balance": (True, 4500.0),
            "white_balance_red": (False, -1.0),
            "hue": (True, 37.0),
            "saturation": (True, 46.0),
            "contrast": (True, 50.0),
            "gamma": (True, 5.0),
        }
        return {
            name: {"supported": supported, "value": value}
            for name, (supported, value) in values.items()
        }

    def test_home_has_six_function_cards(self):
        home = HomePage()
        self.assertEqual(len(home.findChildren(EntryCard)), 6)

    def test_preview_click_maps_to_native_pixels_and_supports_history(self):
        preview = PreviewWidget()
        preview.resize(800, 600)
        preview.set_native_size(1920, 1080)
        frame = QImage(1600, 900, QImage.Format_RGB888)
        frame.fill(QColor("black"))
        preview.set_frame(frame)
        preview.show()
        self.app.processEvents()

        QTest.mouseClick(preview, Qt.LeftButton, Qt.NoModifier, QPoint(400, 20))
        self.assertEqual(preview.points, ())

        QTest.mouseClick(preview, Qt.LeftButton, Qt.NoModifier, QPoint(400, 300))
        self.assertEqual(preview.points, ((960, 540),))
        QTest.mouseClick(preview, Qt.LeftButton, Qt.NoModifier, QPoint(200, 300))
        self.assertEqual(len(preview.points), 2)

        preview.undo_point()
        self.assertEqual(preview.points, ((960, 540),))
        preview.clear_points()
        self.assertEqual(preview.points, ())
        preview.close()

    def test_preview_zoom_preserves_native_pixel_mapping(self):
        preview = PreviewWidget()
        preview.resize(800, 600)
        preview.set_native_size(1920, 1080)
        frame = QImage(1600, 900, QImage.Format_RGB888)
        frame.fill(QColor("black"))
        preview.set_frame(frame)
        preview.show()
        self.app.processEvents()

        zoom_values = []
        preview.zoom_changed.connect(zoom_values.append)
        preview.set_zoom(2.0)
        self.assertEqual(preview.zoom_factor, 2.0)
        self.assertAlmostEqual(preview._display_rect().width(), 1600.0)

        QTest.mouseClick(preview, Qt.LeftButton, Qt.NoModifier, QPoint(400, 300))
        QTest.mouseClick(preview, Qt.LeftButton, Qt.NoModifier, QPoint(200, 300))
        self.assertEqual(preview.points, ((960, 540), (720, 540)))

        preview.reset_zoom()
        self.assertEqual(preview.zoom_factor, 1.0)
        self.assertEqual(zoom_values, [2.0, 1.0])
        preview.close()

    def test_preview_zoom_controls_follow_connection_and_zoom_state(self):
        page, worker = self.make_page()
        self.assertFalse(page.zoom_out_button.isEnabled())
        self.assertFalse(page.zoom_in_button.isEnabled())
        self.assertFalse(page.fit_zoom_button.isEnabled())

        worker.connected.emit(
            {
                "index": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "backend": "DSHOW",
                "pixel_format": "MJPG",
                "properties": self.properties(),
            }
        )
        self.assertTrue(page.zoom_out_button.isEnabled())
        self.assertTrue(page.zoom_in_button.isEnabled())
        self.assertFalse(page.fit_zoom_button.isEnabled())

        page.zoom_in_button.click()
        self.assertEqual(page.preview.zoom_factor, 1.25)
        self.assertEqual(page.zoom_value_label.text(), "125%")
        self.assertTrue(page.fit_zoom_button.isEnabled())

        page.fit_zoom_button.click()
        self.assertEqual(page.preview.zoom_factor, 1.0)
        self.assertEqual(page.zoom_value_label.text(), "100%")
        self.assertFalse(page.fit_zoom_button.isEnabled())

        worker.disconnected.emit("已断开相机")
        self.assertFalse(page.zoom_out_button.isEnabled())
        self.assertFalse(page.zoom_in_button.isEnabled())
        self.assertFalse(page.fit_zoom_button.isEnabled())
        page.close()

    def test_coordinate_buttons_follow_preview_points(self):
        page, _worker = self.make_page()
        page.preview.points_changed.emit([(123, 456), (300, 200)])
        self.assertEqual(page.coordinate_label.text(), "像素坐标: (300, 200)  共 2 点")
        self.assertTrue(page.undo_point_button.isEnabled())
        self.assertTrue(page.clear_points_button.isEnabled())

        page.preview.points_changed.emit([])
        self.assertEqual(page.coordinate_label.text(), "像素坐标: --")
        self.assertFalse(page.undo_point_button.isEnabled())
        self.assertFalse(page.clear_points_button.isEnabled())

    def test_slider_updates_number_only_after_drag_is_released(self):
        control = PropertySlider("brightness", "亮度", 0, 100)
        self.assertIsInstance(control.value_spin, NoWheelDoubleSpinBox)
        control.set_value(20)
        committed = []
        control.value_committed.connect(
            lambda name, value: committed.append((name, value))
        )

        control.slider.setSliderDown(True)
        control.slider.setValue(75)
        self.assertEqual(control.value_spin.value(), 20)
        self.assertEqual(committed, [])

        control.slider.setSliderDown(False)
        self.app.processEvents()
        self.assertEqual(control.value_spin.value(), 75)
        self.assertEqual(committed, [("brightness", 75.0)])

        class WheelEvent:
            ignored = False

            def ignore(self):
                self.ignored = True

        wheel_event = WheelEvent()
        control.value_spin.wheelEvent(wheel_event)
        self.assertTrue(wheel_event.ignored)
        self.assertEqual(control.value_spin.value(), 75)

    def test_exposure_slider_uses_only_supported_camspc_steps(self):
        control = ExposureTimeSlider()
        control.set_value(62.5)
        committed = []
        control.value_committed.connect(
            lambda name, value: committed.append((name, value))
        )

        control.slider.setSliderDown(True)
        control.slider.setValue(2)
        self.assertEqual(control.value_spin.value(), 62.5)
        self.assertEqual(committed, [])

        control.slider.setSliderDown(False)
        self.app.processEvents()
        self.assertEqual(control.value_spin.value(), 250.0)
        self.assertEqual(committed, [("exposure", 250.0)])

        control.value_spin.setValue(200.0)
        control._spin_committed()
        self.assertEqual(control.value_spin.value(), 250.0)
        self.assertEqual(committed[-1], ("exposure", 250.0))

    def test_capture_paths_are_logged_and_buttons_remain_equal_width(self):
        page, worker = self.make_page()
        messages = []
        page.console_output.connect(
            lambda channel, text: messages.append((channel, text))
        )
        self.assertFalse(hasattr(page, "capture_result_label"))

        worker.photo_saved.emit(r"D:\photos\photo.png")
        worker.recording_changed.emit(True, r"D:\photos\record.avi")
        worker.recording_changed.emit(False, r"D:\photos\record.avi")
        self.assertIn("照片已保存: D:\\photos\\photo.png", messages[0][1])
        self.assertIn("开始录像，保存位置: D:\\photos\\record.avi", messages[1][1])
        self.assertIn("录像已保存: D:\\photos\\record.avi", messages[2][1])

        page.resize(1200, 800)
        page.show()
        page.content_splitter.setSizes((350, 800))
        self.app.processEvents()
        narrow_widths = [button.width() for button in page.capture_buttons]
        self.assertEqual(len(set(narrow_widths)), 1)

        page.record_button.setText("停止录像")
        page.content_splitter.setSizes((430, 720))
        self.app.processEvents()
        wide_widths = [button.width() for button in page.capture_buttons]
        self.assertEqual(len(set(wide_widths)), 1)
        self.assertGreater(wide_widths[0], narrow_widths[0])
        page.close()

    def test_page_enables_only_supported_camera_properties(self):
        page, worker = self.make_page()
        worker.devices_ready.emit(
            True,
            [{"index": 0, "name": "CamSPC", "source": "DirectShow"}],
            "检测到 1 台相机",
        )
        self.assertTrue(page.connect_button.isEnabled())

        worker.connected.emit(
            {
                "index": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "backend": "DSHOW",
                "pixel_format": "MJPG",
                "properties": self.properties(),
            }
        )

        self.assertTrue(page.is_connected())
        self.assertTrue(page.auto_exposure_check.isChecked())
        self.assertFalse(page.property_controls["exposure"].isEnabled())
        self.assertTrue(page.property_controls["exposure_target"].isEnabled())
        self.assertEqual(
            page.property_controls["exposure_target"].value_spin.value(), 50.0
        )
        self.assertFalse(page.property_controls["white_balance"].isEnabled())
        self.assertFalse(page.property_controls["white_balance_red"].isEnabled())
        self.assertNotIn("brightness", page.property_controls)
        self.assertEqual(page.property_controls["hue"].value_spin.value(), 37.0)
        self.assertEqual(page.preview_info.text(), "1920 × 1080")

        page.auto_exposure_check.setChecked(False)
        page.auto_wb_check.setChecked(False)
        self.assertTrue(page.property_controls["exposure"].isEnabled())
        self.assertTrue(page.property_controls["white_balance"].isEnabled())
        self.assertTrue(page.one_shot_wb_button.isEnabled())
        self.assertIn(("property", "auto_exposure", 0.0), worker.calls)
        self.assertIn(("property", "auto_white_balance", 0.0), worker.calls)

        worker.disconnected.emit("已断开相机")
        self.assertFalse(page.is_connected())
        self.assertFalse(page.photo_button.isEnabled())

    def test_linux_camera_device_path_is_used_for_connection(self):
        page, worker = self.make_page()
        worker.devices_ready.emit(
            True,
            [
                {
                    "index": 2,
                    "name": "CamSPC",
                    "source": "V4L2",
                    "device": "/dev/video2",
                }
            ],
            "检测到 1 台相机",
        )

        self.assertEqual(page.device_combo.currentData(), "/dev/video2")
        page.connect_button.click()
        self.assertTrue(any(call[:2] == ("connect", "/dev/video2") for call in worker.calls))

    def test_exposure_conversion_and_property_support(self):
        self.assertAlmostEqual(exposure_milliseconds(-5), 31.25)
        self.assertAlmostEqual(exposure_backend_value(31.25), -5.0)
        self.assertTrue(math.isnan(exposure_backend_value(0)))
        self.assertTrue(property_is_supported("exposure", -6.0))
        self.assertTrue(property_is_supported("auto_exposure", -1.0))
        self.assertTrue(property_is_supported("exposure_target", 50.0))
        self.assertFalse(property_is_supported("gamma", math.nan))

    def test_exposure_time_and_target_are_applied_and_read_back(self):
        page, worker = self.make_page()
        properties = self.properties()
        properties["auto_exposure"]["value"] = 0.0
        worker.connected.emit(
            {
                "index": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "backend": "DSHOW",
                "pixel_format": "MJPG",
                "properties": properties,
            }
        )

        exposure = page.property_controls["exposure"]
        target = page.property_controls["exposure_target"]
        self.assertEqual(exposure.label.text(), "曝光时间")
        self.assertEqual(target.label.text(), "曝光目标值")
        self.assertFalse(hasattr(page, "exposure_time_label"))

        exposure.value_spin.setValue(250.0)
        exposure._spin_committed()
        target.value_spin.setValue(65)
        target._spin_committed()
        self.assertIn(("property", "exposure", -2.0), worker.calls)
        self.assertIn(("property", "exposure_target", 65.0), worker.calls)

        worker.property_updated.emit("exposure", -2.0, True, "已应用")
        worker.property_updated.emit("exposure_target", 63.0, True, "已应用")
        self.assertEqual(exposure.value_spin.value(), 250.0)
        self.assertEqual(exposure.slider.value(), 2)
        self.assertIn("250 ms", exposure.toolTip())
        self.assertEqual(target.value_spin.value(), 63.0)

    def test_rejected_color_value_restores_readback_without_disabling_control(self):
        page, worker = self.make_page()
        worker.connected.emit(
            {
                "index": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "backend": "DSHOW",
                "pixel_format": "MJPG",
                "properties": self.properties(),
            }
        )
        hue = page.property_controls["hue"]
        gamma = page.property_controls["gamma"]

        hue.value_spin.setValue(80)
        hue._spin_committed()
        worker.property_updated.emit(
            "hue", 37.0, False, "相机驱动拒绝了该值，请选择其他值"
        )
        self.assertEqual(hue.value_spin.value(), 37.0)
        self.assertTrue(hue.isEnabled())
        self.assertTrue(page._properties["hue"]["supported"])

        gamma.value_spin.setValue(20)
        gamma._spin_committed()
        worker.property_updated.emit(
            "gamma", 5.0, False, "相机驱动拒绝了该值，请选择其他值"
        )
        self.assertEqual(gamma.value_spin.value(), 5.0)
        self.assertTrue(gamma.isEnabled())
        self.assertTrue(page._properties["gamma"]["supported"])
        self.assertEqual(
            page.status_label.text(),
            "伽马值: 相机驱动拒绝了该值，请选择其他值",
        )

        hue.value_spin.setValue(40)
        hue._spin_committed()
        self.assertEqual(worker.calls[-1], ("property", "hue", 40.0))
        self.assertFalse(page.property_controls["white_balance_red"].isEnabled())
        page.close()

    def test_worker_returns_actual_value_after_driver_rejects_color_value(self):
        class FakeCamera:
            def set(self, property_id, value):
                return False

            def get(self, property_id):
                return 37.0

        worker = CameraWorker()
        worker._camera = FakeCamera()
        worker._property_ids = {"hue": 13}

        actual, ok, message = worker._set_property_value("hue", 80.0)

        self.assertFalse(ok)
        self.assertEqual(actual, 37.0)
        self.assertIn("请选择其他值", message)

    def test_worker_selects_white_balance_fallback(self):
        class FakeCv2:
            CAP_PROP_WB_TEMPERATURE = 45
            CAP_PROP_WHITE_BALANCE_BLUE_U = 17

        class FakeCamera:
            def get(self, property_id):
                return {45: -1.0, 17: 4500.0}.get(property_id, -1.0)

        worker = CameraWorker()
        worker._cv2 = FakeCv2()
        worker._camera = FakeCamera()
        properties = worker._read_properties()

        self.assertTrue(properties["white_balance"]["supported"])
        self.assertEqual(properties["white_balance"]["value"], 4500.0)
        self.assertEqual(
            properties["white_balance"]["backend_property"],
            "CAP_PROP_WHITE_BALANCE_BLUE_U",
        )

    def test_worker_maps_camspc_brightness_to_exposure_target(self):
        class FakeCv2:
            CAP_PROP_BRIGHTNESS = 10

        class FakeCamera:
            def get(self, property_id):
                return 50.0 if property_id == 10 else -1.0

        worker = CameraWorker()
        worker._cv2 = FakeCv2()
        worker._camera = FakeCamera()
        properties = worker._read_properties()

        self.assertTrue(properties["exposure_target"]["supported"])
        self.assertEqual(properties["exposure_target"]["value"], 50.0)
        self.assertEqual(
            properties["exposure_target"]["backend_property"],
            "CAP_PROP_BRIGHTNESS",
        )


if __name__ == "__main__":
    unittest.main()
