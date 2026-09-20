import os
import sys
import time
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_DIR))

try:
    from PySide6.QtWidgets import QApplication

    from GUI.core.runner import TaskManager
    import GUI.core.motion_worker as motion_worker_module
    import GUI.pages.motion_control as motion_control_module
    from GUI.pages.motion_control import MotionControlPage
    from motion_controller import MODE_RELATIVE, MotionController
except ImportError:
    QApplication = None
    MotionControlPage = None
    MotionController = None
    MODE_RELATIVE = 0x41


class FakeMotionController:
    instances = []

    def __init__(self, port, slave_address, baudrate, logger):
        self.port = port
        self.baudrate = baudrate
        self.instrument = SimpleNamespace(address=slave_address)
        self.slave_address = slave_address
        self.closed = False
        self.alarm_reads = 0
        self.quiet_alarm_reads = 0
        self.quiet_communication_entries = 0
        self.__class__.instances.append(self)

    def read_register(self, _address):
        return 51200

    def read_status(self):
        return {
            "fault": False,
            "enabled": True,
            "running": False,
            "invalid": False,
            "cmd_done": True,
            "path_done": True,
            "home_done": False,
            "raw": 0,
        }

    def read_motor_position_pulses(self):
        return 0

    def read_alarm(self):
        self.alarm_reads += 1
        return 0

    def read_alarm_quiet(self):
        self.quiet_alarm_reads += 1
        return 0

    @contextmanager
    def quiet_communication(self):
        self.quiet_communication_entries += 1
        yield

    def close(self):
        self.closed = True


@unittest.skipIf(MotionController is None, "motion controller dependencies are missing")
class MotionControllerLogRoutingTests(unittest.TestCase):
    def test_log_callback_replaces_terminal_output(self):
        messages = []
        controller = MotionController.__new__(MotionController)
        controller.slave_address = 1
        controller._log_callback = messages.append
        controller.write_register = mock.Mock()

        terminal = StringIO()
        with redirect_stdout(terminal):
            controller.set_motion_mode(MODE_RELATIVE)

        self.assertEqual(terminal.getvalue(), "")
        self.assertEqual(messages, ["[轴1] 设置运动模式: 相对"])

    def test_automatic_poll_context_suppresses_axis_communication_logs(self):
        emitted = mock.Mock()
        controller = motion_worker_module.LoggedMotionController.__new__(
            motion_worker_module.LoggedMotionController
        )
        controller._logger = SimpleNamespace(
            log_signal=SimpleNamespace(emit=emitted)
        )
        controller._communication_logging_enabled = True

        controller._emit_communication_log("send", "[轴1] before")
        with controller.quiet_communication():
            controller._emit_communication_log("send", "[轴1] polling")
            controller._emit_communication_log("recv", "[轴2] polling")
        controller._emit_communication_log("recv", "[轴2] after")

        self.assertEqual(
            emitted.call_args_list,
            [
                mock.call("send", "[轴1] before"),
                mock.call("recv", "[轴2] after"),
            ],
        )


@unittest.skipIf(
    QApplication is None or MotionControlPage is None,
    "PySide6 is not installed",
)
class MotionControlConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_until(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return predicate()

    def test_page_starts_worker_and_connects_and_disconnects(self):
        FakeMotionController.instances.clear()
        with mock.patch.object(
            motion_worker_module,
            "LoggedMotionController",
            FakeMotionController,
        ):
            page = MotionControlPage(TaskManager())
            page.port_combo.addItem("Test motion stage", "COM_TEST")
            page.port_combo.setCurrentIndex(page.port_combo.count() - 1)

            self.assertTrue(self.wait_until(page.worker.isRunning))
            page._on_connect()
            self.assertTrue(page._is_connecting)
            self.assertTrue(page.disconnect_btn.isEnabled())

            self.assertTrue(self.wait_until(page.is_connected))
            self.assertFalse(page._is_connecting)
            self.assertTrue(page.disconnect_btn.isEnabled())
            self.assertTrue(page.auto_refresh_cb.isChecked())
            self.assertTrue(page._refresh_timer.isActive())
            self.assertTrue(page.manual_refresh_btn.isEnabled())

            page._on_disconnect()
            self.assertTrue(self.wait_until(lambda: not page.is_connected()))
            self.assertFalse(page.disconnect_btn.isEnabled())
            self.assertFalse(page.auto_refresh_cb.isChecked())
            self.assertFalse(page._refresh_timer.isActive())
            self.assertFalse(page.manual_refresh_btn.isEnabled())
            self.assertTrue(all(item.closed for item in FakeMotionController.instances))

            page.shutdown_worker()
            self.assertFalse(page.worker.isRunning())
            page.close()

    def test_usb_port_is_preferred_and_refresh_preserves_selection(self):
        ports = [
            SimpleNamespace(device="COM1", description="Virtual Port", vid=None),
            SimpleNamespace(device="COM3", description="USB Serial Port", vid=0x0403),
        ]
        with mock.patch.object(
            motion_control_module.serial.tools.list_ports,
            "comports",
            return_value=ports,
        ):
            page = MotionControlPage(TaskManager())
            self.assertEqual(page.port_combo.currentData(), "COM3")

            page.port_combo.setCurrentIndex(page.port_combo.findData("COM1"))
            page._refresh_ports()
            self.assertEqual(page.port_combo.currentData(), "COM1")

            page.shutdown_worker()
            page.close()

    def test_successful_worker_results_are_forwarded_to_motion_console(self):
        page = MotionControlPage(TaskManager())
        try:
            with mock.patch.object(page, "_append_log") as append_log:
                page._on_result("[轴1] 正向点动已启动", False)

            append_log.assert_called_once_with("info", "[轴1] 正向点动已启动")
            self.assertEqual(page.status_label.text(), "[轴1] 正向点动已启动")
        finally:
            page.shutdown_worker()
            page.close()

    def test_automatic_refresh_silences_alarm_poll_but_manual_refresh_logs_it(self):
        worker = motion_worker_module.MotionWorker()
        controller = FakeMotionController("COM_TEST", 1, 38400, worker.logger)
        worker._controller = controller
        queued = []
        worker.submit = queued.append

        worker.refresh_all()
        worker.refresh_all()
        self.assertEqual(len(queued), 1)
        queued.pop()()
        self.assertEqual(controller.quiet_alarm_reads, 2)
        self.assertEqual(controller.alarm_reads, 0)
        self.assertEqual(controller.quiet_communication_entries, 1)

        worker.refresh_all(log_alarm=True)
        self.assertEqual(len(queued), 1)
        queued.pop()()
        self.assertEqual(controller.quiet_alarm_reads, 2)
        self.assertEqual(controller.alarm_reads, 2)
        self.assertEqual(controller.quiet_communication_entries, 1)

        worker._controller = None
        worker.deleteLater()

    def test_auto_refresh_reports_only_consecutive_failures_and_recovery(self):
        worker = motion_worker_module.MotionWorker()
        results = []
        worker.result_ready.connect(
            lambda message, is_error: results.append((message, is_error))
        )
        timeout = RuntimeError("No communication with the instrument (no answer)")

        worker._handle_refresh_result(timeout, manual=False)
        worker._handle_refresh_result(timeout, manual=False)
        self.assertEqual(results, [])

        worker._handle_refresh_result(timeout, manual=False)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0][1])
        self.assertIn("连续 3 次无响应", results[0][0])

        worker._handle_refresh_result(timeout, manual=False)
        self.assertEqual(len(results), 1)

        worker._handle_refresh_result(None, manual=False)
        self.assertEqual(results[-1], ("位移台自动刷新通讯已恢复", False))

        worker.deleteLater()

    def test_manual_refresh_error_is_reported_immediately(self):
        worker = motion_worker_module.MotionWorker()
        results = []
        worker.result_ready.connect(
            lambda message, is_error: results.append((message, is_error))
        )
        timeout = RuntimeError("No communication with the instrument (no answer)")

        worker._handle_refresh_result(timeout, manual=True)

        self.assertEqual(results, [(str(timeout), True)])
        worker.deleteLater()


if __name__ == "__main__":
    unittest.main()
