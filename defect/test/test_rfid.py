import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_DIR))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import E720_RFID_Tool as E720
except ModuleNotFoundError:
    E720 = None

try:
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QApplication, QMessageBox
    from GUI.core.rfid_worker import RfidWorker, detect_memory_capacity
    from GUI.pages.rfid_control import CONTINUOUS_MISS_LIMIT, RfidControlPage
except ModuleNotFoundError:
    QApplication = None
    QMessageBox = None
    QSignalSpy = None
    RfidWorker = None
    detect_memory_capacity = None
    RfidControlPage = None
    CONTINUOUS_MISS_LIMIT = 2


def response_frame(frame_type, command, payload=b""):
    middle = bytes([
        frame_type,
        command,
        (len(payload) >> 8) & 0xFF,
        len(payload) & 0xFF,
    ]) + payload
    return bytes([E720.FRAME_HEADER]) + middle + bytes([
        E720.calc_checksum(middle),
        E720.FRAME_END,
    ])


@unittest.skipIf(E720 is None, "pyserial is not installed")
class RfidProtocolTests(unittest.TestCase):
    def test_transmit_power_range_matches_module_limit(self):
        self.assertEqual(sorted(E720.POWER_TABLE), list(range(15, 27)))

    def test_parse_frame_uses_the_complete_frame_length(self):
        frame = response_frame(E720.TYPE_RESPONSE, E720.CMD_GET_REGION, b"\x01")

        parsed = E720.parse_frame(frame)

        self.assertIsNotNone(parsed)
        self.assertEqual(len(frame), 8)
        self.assertEqual(parsed["total_len"], len(frame))
        self.assertTrue(parsed["checksum_ok"])

    def test_parse_frame_handles_prefix_and_truncation(self):
        frame = response_frame(E720.TYPE_RESPONSE, E720.CMD_GET_CHANNEL)

        parsed = E720.parse_frame(b"\x00\x01" + frame)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["start_index"], 2)
        self.assertEqual(parsed["total_len"], len(frame))
        self.assertIsNone(E720.parse_frame(frame[:-1]))

    def test_send_and_parse_handles_multiple_frames_and_rejects_bad_checksum(self):
        first = response_frame(E720.TYPE_RESPONSE, E720.CMD_GET_REGION, b"\x01")
        bad = bytearray(response_frame(
            E720.TYPE_RESPONSE, E720.CMD_GET_CHANNEL, b"\x02"))
        bad[-2] ^= 0xFF
        last = response_frame(E720.TYPE_RESPONSE, E720.CMD_GET_POWER, b"\x0A\x28")
        module = E720.E720Module()
        module.send_command = lambda *_args, **_kwargs: first + bytes(bad) + last

        frames = module.send_and_parse(E720.CMD_GET_REGION)

        self.assertEqual(
            [frame["command"] for frame in frames],
            [E720.CMD_GET_REGION, E720.CMD_GET_POWER],
        )

    def test_single_inventory_parses_a_tag_notification(self):
        epc = bytes.fromhex("112233445566778899AABBCC")
        payload = bytes([0xB0]) + bytes.fromhex("3000") + epc + bytes.fromhex("1234")
        module = E720.E720Module()
        module.send_command = lambda *_args, **_kwargs: response_frame(
            E720.TYPE_NOTIFICATION, E720.CMD_SINGLE_INVENTORY, payload)

        tags = module.single_inventory()

        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["epc_hex"], epc.hex().upper())
        self.assertEqual(tags[0]["rssi"], -80)

    def test_notification_parser_supports_non_96_bit_epc(self):
        epc = bytes.fromhex("1122334455667788")
        payload = bytes([0xC9]) + bytes.fromhex("2000") + epc + bytes.fromhex("ABCD")
        frame = E720.parse_frame(response_frame(
            E720.TYPE_NOTIFICATION, E720.CMD_SINGLE_INVENTORY, payload))

        tag = E720.parse_notification_frame(frame)

        self.assertEqual(tag["epc"], epc)
        self.assertEqual(tag["crc"], bytes.fromhex("ABCD"))
        self.assertEqual(tag["rssi"], -55)

    def test_single_inventory_deduplicates_epc_and_keeps_strongest_rssi(self):
        epc = bytes.fromhex("112233445566778899AABBCC")
        weak = bytes([0xBA]) + bytes.fromhex("3000") + epc + bytes.fromhex("1234")
        strong = bytes([0xD8]) + bytes.fromhex("3000") + epc + bytes.fromhex("1234")
        module = E720.E720Module()
        module.send_command = lambda *_args, **_kwargs: (
            response_frame(E720.TYPE_NOTIFICATION, E720.CMD_SINGLE_INVENTORY, weak)
            + response_frame(E720.TYPE_NOTIFICATION, E720.CMD_SINGLE_INVENTORY, strong)
        )

        tags = module.single_inventory()

        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["rssi"], -40)

    def test_selected_tag_read_preserves_success_and_out_of_range_details(self):
        epc = bytes.fromhex("112233445566778899AABBCC")
        success_payload = bytes([14]) + bytes.fromhex("3000") + epc + b"\x12\x34"
        module = E720.E720Module()
        module.send_and_parse = lambda *_args, **_kwargs: [
            {
                "type": E720.TYPE_RESPONSE,
                "command": E720.CMD_READ_DATA,
                "payload": success_payload,
            }
        ]

        success = module.read_selected_tag_result(E720.MEMBANK_USER, 0, 1)

        self.assertTrue(success["ok"])
        self.assertEqual(success["data"], b"\x12\x34")
        module.send_and_parse = lambda *_args, **_kwargs: [
            {
                "type": E720.TYPE_RESPONSE,
                "command": E720.CMD_ERROR,
                "payload": bytes([E720.ERROR_READ_OUT_OF_RANGE]),
            }
        ]

        failure = module.read_selected_tag_result(E720.MEMBANK_USER, 32, 1)

        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error_code"], E720.ERROR_READ_OUT_OF_RANGE)


@unittest.skipIf(detect_memory_capacity is None, "PySide6/pyserial is not installed")
class RfidCapacityTests(unittest.TestCase):
    def test_probe_detects_absent_and_exact_contiguous_capacity(self):
        absent = detect_memory_capacity(lambda _address: ("out_of_range", "边界"))
        calls = []

        def probe(address):
            calls.append(address)
            return ("ok", "") if address < 37 else ("out_of_range", "边界")

        present = detect_memory_capacity(probe)

        self.assertFalse(absent["present"])
        self.assertEqual(absent["words"], 0)
        self.assertTrue(present["present"])
        self.assertEqual(present["words"], 37)
        self.assertEqual(present["bytes"], 74)
        self.assertLess(len(calls), 16)

    def test_probe_does_not_report_partial_capacity_after_read_error(self):
        def probe(address):
            if address < 4:
                return "ok", ""
            return "error", "标签无响应"

        result = detect_memory_capacity(probe)

        self.assertTrue(result["present"])
        self.assertIsNone(result["words"])
        self.assertIn("标签无响应", result["message"])


@unittest.skipIf(
    QApplication is None or RfidControlPage is None,
    "PySide6/pyserial is not installed",
)
class RfidGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = RfidControlPage()
        self.page._on_connected(True, "test")
        self.page.current_epc = "00112233445566778899AABB"
        self.page.selected_epc_label.setText(self.page.current_epc)

    def tearDown(self):
        self.page.shutdown_worker()
        self.page.deleteLater()
        self.app.processEvents()

    def test_invalid_hex_is_rejected_before_worker_submission(self):
        self.page.write_data_edit.setText("GGGG")
        self.page.worker.write_user = mock.Mock()

        self.page._on_write_clicked()

        self.page.worker.write_user.assert_not_called()
        self.assertIn("有效的十六进制", self.page.feedback_label.text())

    def test_power_options_start_at_15_dbm_and_default_to_15(self):
        values = [
            self.page.power_combo.itemData(index)
            for index in range(self.page.power_combo.count())
        ]

        self.assertEqual(values, list(range(15, 27)))
        self.assertEqual(self.page.power_combo.currentData(), 15)

    def test_recognition_settings_use_requested_defaults(self):
        self.assertEqual(self.page.scan_interval.value(), 1.0)
        self.assertTrue(self.page.rssi_filter_check.isChecked())
        self.assertEqual(self.page.rssi_threshold_spin.value(), -40)
        self.assertEqual(self.page.miss_limit_spin.value(), 2)
        self.assertFalse(hasattr(self.page, "set_miss_limit_btn"))

    def test_scan_timing_controls_are_disabled_until_connected(self):
        self.assertTrue(self.page.scan_interval.isEnabled())
        self.assertTrue(self.page.miss_limit_spin.isEnabled())

        self.page._on_connected(False, "disconnected")

        self.assertFalse(self.page.scan_interval.isEnabled())
        self.assertFalse(self.page.miss_limit_spin.isEnabled())
        self.assertTrue(self.page.rssi_filter_check.isChecked())
        self.assertFalse(self.page.rssi_threshold_spin.isEnabled())

        self.page._on_connected(True, "connected")

        self.assertTrue(self.page.scan_interval.isEnabled())
        self.assertTrue(self.page.miss_limit_spin.isEnabled())
        self.assertTrue(self.page.rssi_threshold_spin.isEnabled())

    def test_rssi_filter_updates_existing_results_immediately(self):
        strong = {"epc_hex": "111122223333444455556666", "rssi": -35}
        weak = {"epc_hex": "AAAABBBBCCCCDDDDEEEEFFFF", "rssi": -75}
        self.page._inventory_request_mode = "single"
        self.page._on_inventory_done([strong, weak])

        self.assertTrue(self.page.rssi_filter_check.isChecked())
        self.assertTrue(self.page.rssi_threshold_spin.isEnabled())
        self.assertEqual(self.page.rssi_threshold_spin.value(), -40)
        self.assertEqual(self.page.tag_table.rowCount(), 1)
        self.assertEqual(self.page.tag_table.item(0, 0).text(), strong["epc_hex"])

        self.page.rssi_threshold_spin.setValue(-80)
        self.assertEqual(self.page.tag_table.rowCount(), 2)

        self.page.rssi_filter_check.setChecked(False)
        self.assertFalse(self.page.rssi_threshold_spin.isEnabled())
        self.assertEqual(self.page.tag_table.rowCount(), 2)

    def test_valid_user_data_is_normalized_before_submission(self):
        self.page.write_data_edit.setText("0a 0B 0c 0D")
        self.page.worker.write_user = mock.Mock()

        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.Yes
        ):
            self.page._on_write_clicked()

        self.page.worker.write_user.assert_called_once_with(
            self.page.current_epc,
            "0A0B0C0D",
            self.page.write_addr_spin.value(),
        )

    def test_capacity_detection_stops_scanning_and_submits_selected_tag(self):
        self.page._periodic = True
        self.page.worker.detect_capacity = mock.Mock()

        self.page._on_detect_capacity_clicked()

        self.assertFalse(self.page._periodic)
        self.assertFalse(self.page.detect_capacity_btn.isEnabled())
        self.page.worker.detect_capacity.assert_called_once_with(
            self.page.current_epc
        )
        self.assertIn("正在检测", self.page.capacity_result_label.text())

    def test_capacity_result_shows_region_presence_and_epc_data_space(self):
        results = {
            "user": {
                "present": False,
                "words": 0,
                "bytes": 0,
                "message": "读超出存储区范围",
            },
            "epc": {
                "present": True,
                "words": 8,
                "bytes": 16,
                "message": "",
            },
        }

        self.page._on_capacity_done(self.page.current_epc, results)

        text = self.page.capacity_result_label.text()
        self.assertIn("User 区：不存在", text)
        self.assertIn("EPC 卡号区：存在，总容量 8 Word / 16 字节", text)
        self.assertIn("卡号数据 6 Word / 12 字节", text)

    def test_inventory_requests_do_not_queue_while_one_is_running(self):
        self.page.worker.single_inventory = mock.Mock()

        self.page._on_single_clicked()
        self.page._on_single_clicked()

        self.page.worker.single_inventory.assert_called_once_with()
        self.page._on_inventory_done([])
        self.assertFalse(self.page._round_busy)
        self.assertEqual(self.page.current_epc, "")
        self.assertEqual(self.page.selected_epc_label.text(), "(未选择)")

    def test_continuous_inventory_uses_default_two_miss_limit(self):
        first = {"epc_hex": "111122223333444455556666", "rssi": -30}
        second = {"epc_hex": "AAAABBBBCCCCDDDDEEEEFFFF", "rssi": -25}

        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([first])
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([second])
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([])

        self.assertEqual(self.page.tag_table.rowCount(), 1)
        self.assertEqual(self.page.tag_table.item(0, 0).text(), second["epc_hex"])
        self.assertIn("当前保留 1 张", self.page.feedback_label.text())

    def test_continuous_inventory_removes_tag_after_repeated_misses(self):
        tag = {"epc_hex": "111122223333444455556666", "rssi": -30}
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([tag])

        for _ in range(CONTINUOUS_MISS_LIMIT):
            self.page._inventory_request_mode = "periodic"
            self.page._on_inventory_done([])

        self.assertEqual(self.page.tag_table.rowCount(), 0)

    def test_continuous_miss_limit_can_be_set_from_recognition_controls(self):
        tag = {"epc_hex": "111122223333444455556666", "rssi": -30}
        self.assertEqual(self.page.miss_limit_spin.value(), CONTINUOUS_MISS_LIMIT)
        self.assertEqual(
            self.page._continuous_miss_limit, CONTINUOUS_MISS_LIMIT
        )

        self.assertFalse(hasattr(self.page, "set_miss_limit_btn"))
        self.page.miss_limit_spin.setValue(3)
        self.assertEqual(self.page._continuous_miss_limit, 3)

        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([tag])
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([])
        self.assertEqual(self.page.tag_table.rowCount(), 1)
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([])
        self.assertEqual(self.page.tag_table.rowCount(), 1)
        self.page._inventory_request_mode = "periodic"
        self.page._on_inventory_done([])
        self.assertEqual(self.page.tag_table.rowCount(), 0)

    def test_lower_miss_limit_removes_already_stale_tags_immediately(self):
        tag = {"epc_hex": "111122223333444455556666", "rssi": -30}
        self.page._continuous_tags[tag["epc_hex"]] = {
            "tag": tag,
            "misses": 2,
        }
        self.page._last_inventory_tags = [tag]
        self.page._fill_tags([tag])

        self.page.miss_limit_spin.setValue(3)
        self.assertEqual(self.page.tag_table.rowCount(), 1)
        self.page.miss_limit_spin.setValue(2)

        self.assertEqual(self.page.tag_table.rowCount(), 0)
        self.assertEqual(self.page._continuous_tags, {})
        self.assertIn("已移除 1 张标签", self.page.feedback_label.text())

    def test_clear_inventory_resets_results_without_stopping_periodic_scan(self):
        tag = {"epc_hex": self.page.current_epc, "rssi": -45}
        self.page._fill_tags([tag])
        self.page._continuous_tags[tag["epc_hex"]] = {
            "tag": tag,
            "misses": 0,
        }
        self.page._periodic = True

        self.page.clear_tags_btn.click()

        self.assertEqual(self.page.tag_table.rowCount(), 0)
        self.assertEqual(self.page._continuous_tags, {})
        self.assertEqual(self.page._last_inventory_tags, [])
        self.assertEqual(self.page.current_epc, "")
        self.assertEqual(self.page.selected_epc_label.text(), "(未选择)")
        self.assertTrue(self.page._periodic)
        self.assertIn("连续扫描仍在运行", self.page.feedback_label.text())

    def test_verified_epc_write_updates_current_selection(self):
        old_epc = self.page.current_epc
        new_epc = "AABBCCDDEEFF001122334455"
        self.page._fill_tags([{"epc_hex": old_epc, "rssi": -45}])

        self.page._on_epc_write_done(True, "ok", new_epc)

        self.assertEqual(self.page.current_epc, new_epc)
        self.assertEqual(self.page.selected_epc_label.text(), new_epc)
        self.assertEqual(self.page.tag_table.item(0, 0).text(), new_epc)

    def test_unverified_epc_write_clears_stale_selection(self):
        self.page._fill_tags([{"epc_hex": self.page.current_epc, "rssi": -45}])

        self.page._on_epc_write_done(True, "需要重扫", "")

        self.assertEqual(self.page.current_epc, "")
        self.assertEqual(self.page.tag_table.rowCount(), 0)
        self.assertIn("重新扫描", self.page.selected_epc_label.text())


@unittest.skipIf(
    QApplication is None or RfidWorker is None,
    "PySide6/pyserial is not installed",
)
class RfidWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_command_exception_keeps_operation_context(self):
        worker = RfidWorker()
        spy = QSignalSpy(worker.operation_failed)
        worker.start()
        worker.submit("inventory", lambda: 1 / 0)

        try:
            if spy.count() == 0:
                spy.wait(1000)
            self.assertGreater(spy.count(), 0)
            self.assertEqual(spy.at(0)[0], "inventory")
            self.assertIn("zero", spy.at(0)[1])
        finally:
            worker.stop()
            self.assertTrue(worker.wait(1000))

    def test_capacity_worker_checks_user_and_epc_banks_without_writing(self):
        class FakeModule:
            connected = True

            def __init__(self):
                self.reads = []

            def select_clear(self):
                return True

            def select_epc(self, _epc):
                return True

            def read_selected_tag_result(self, bank, address, word_count):
                self.reads.append((bank, address, word_count))
                limit = {
                    E720.MEMBANK_USER: 4,
                    E720.MEMBANK_EPC: 8,
                }[bank]
                if address < limit:
                    return {
                        "ok": True,
                        "data": b"\x00\x00",
                        "error_code": None,
                        "message": "",
                    }
                return {
                    "ok": False,
                    "data": None,
                    "error_code": E720.ERROR_READ_OUT_OF_RANGE,
                    "message": "读超出存储区范围",
                }

        module = FakeModule()
        worker = RfidWorker()
        worker._module = module
        worker.submit = lambda _operation, function: function() is None
        spy = QSignalSpy(worker.capacity_done)

        worker.detect_capacity("00112233445566778899AABB")

        self.assertEqual(spy.count(), 1)
        results = spy.at(0)[1]
        self.assertEqual(results["user"]["words"], 4)
        self.assertEqual(results["epc"]["words"], 8)
        self.assertTrue(module.reads)
        worker.deleteLater()

    def test_stop_releases_connected_module(self):
        class FakeModule:
            connected = True

            def __init__(self):
                self.disconnect_count = 0

            def disconnect(self):
                self.disconnect_count += 1
                self.connected = False

        module = FakeModule()
        worker = RfidWorker()
        worker._module = module
        worker.start()

        worker.stop()

        self.assertTrue(worker.wait(1000))
        self.assertEqual(module.disconnect_count, 1)
        self.assertIsNone(worker._module)


if __name__ == "__main__":
    unittest.main()
