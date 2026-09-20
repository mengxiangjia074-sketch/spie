"""E720 RFID 读写模块控制工作线程。

复用 detection/E720_RFID_Tool.py 的协议实现 (E720Module/帧工具/常量),
不改动该文件; LoggedE720Module 在每次指令收发时发射通讯日志信号。
所有串口操作 (含 scan 的数秒循环) 都在本线程串行执行, GUI 只入队请求。
"""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QObject, QThread, Signal

import E720_RFID_Tool as E720
from E720_RFID_Tool import (
    MEMBANK_EPC,
    MEMBANK_USER,
    E720Module,
    format_hex,
    hex_to_bytes,
    parse_frame,
    parse_notification_frame,
    REGION_NAMES,
    TYPE_NOTIFICATION,
)


MAX_MEMORY_CAPACITY_WORDS = 0x10000
PROBE_OK = "ok"
PROBE_OUT_OF_RANGE = "out_of_range"
PROBE_ERROR = "error"


def detect_memory_capacity(probe, max_words: int = MAX_MEMORY_CAPACITY_WORDS) -> dict:
    """Return the size of a contiguous RFID bank using read-only probes."""
    status, message = probe(0)
    if status == PROBE_OUT_OF_RANGE:
        return {"present": False, "words": 0, "bytes": 0, "message": message}
    if status != PROBE_OK:
        return {
            "present": None,
            "words": None,
            "bytes": None,
            "message": message or "无法读取存储区起始位置",
        }

    last_valid = 0
    first_invalid = None
    address = 1
    while address < max_words:
        status, message = probe(address)
        if status == PROBE_OK:
            last_valid = address
            address *= 2
            continue
        if status == PROBE_OUT_OF_RANGE:
            first_invalid = address
            break
        return {
            "present": True,
            "words": None,
            "bytes": None,
            "message": message or f"读取 Word {address} 失败",
        }

    if first_invalid is None:
        final_address = max_words - 1
        if last_valid != final_address:
            status, message = probe(final_address)
            if status == PROBE_OK:
                return {
                    "present": True,
                    "words": max_words,
                    "bytes": max_words * 2,
                    "message": "",
                }
            if status == PROBE_OUT_OF_RANGE:
                first_invalid = final_address
            else:
                return {
                    "present": True,
                    "words": None,
                    "bytes": None,
                    "message": message or f"读取 Word {final_address} 失败",
                }
        else:
            return {
                "present": True,
                "words": max_words,
                "bytes": max_words * 2,
                "message": "",
            }

    low = last_valid + 1
    high = first_invalid
    while low < high:
        address = (low + high) // 2
        status, message = probe(address)
        if status == PROBE_OK:
            low = address + 1
        elif status == PROBE_OUT_OF_RANGE:
            high = address
        else:
            return {
                "present": True,
                "words": None,
                "bytes": None,
                "message": message or f"读取 Word {address} 失败",
            }

    return {"present": True, "words": low, "bytes": low * 2, "message": ""}


class CommLogger(QObject):
    """通讯日志信号发射器 (direction, message)。"""

    log_signal = Signal(str, str)


class LoggedE720Module(E720Module):
    """重写 send_command, 在每次指令收发时发射日志信号。"""

    def __init__(self, logger: CommLogger):
        super().__init__()
        self._logger = logger

    def send_command(self, command, params=b"", verbose=False):
        self._logger.log_signal.emit(
            "send", f"cmd=0x{command:02X} pl={params.hex().upper()}")
        try:
            raw = super().send_command(command, params, verbose)
        except Exception as exc:
            self._logger.log_signal.emit("error", f"发送异常: {exc}")
            raise
        if raw:
            self._logger.log_signal.emit("recv", format_hex(raw))
        else:
            self._logger.log_signal.emit("recv", "(无响应)")
        return raw


class RfidWorker(QThread):
    """串行执行 E720 串口命令的后台线程。"""

    connected_signal = Signal(bool, str)     # success, message
    info_ready = Signal(dict)                # 模块信息摘要
    inventory_done = Signal(list)            # [{epc_hex, rssi, ...}]
    read_done = Signal(bool, str, str, str)  # ok, epc_hex, data_hex, ascii
    capacity_done = Signal(str, object)       # epc_hex, {user: ..., epc: ...}
    write_done = Signal(bool, str)           # ok, message
    epc_write_done = Signal(bool, str, str)  # ok, message, effective EPC (or "")
    settings_done = Signal(bool, str)        # ok, message
    operation_failed = Signal(str, str)      # operation, error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._module: LoggedE720Module | None = None
        self._accepting = True
        self._logger = CommLogger()

    @property
    def logger(self) -> CommLogger:
        return self._logger

    # ---- 线程主循环 ---------------------------------------------------------

    def run(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                operation, cmd = item
                if cmd is None:
                    break
                try:
                    cmd()
                except Exception as exc:
                    message = str(exc) or type(exc).__name__
                    self._logger.log_signal.emit(
                        "error", f"{operation} 异常: {message}")
                    self.operation_failed.emit(operation, message)
        finally:
            try:
                self._disconnect_module()
            except Exception as exc:
                self._logger.log_signal.emit("error", f"关闭串口异常: {exc}")

    def stop(self):
        self._accepting = False
        self._discard_pending()
        self._queue.put(None)

    def submit(self, operation: str, func) -> bool:
        if not self._accepting:
            self.operation_failed.emit(operation, "RFID 工作线程已停止")
            return False
        self._queue.put((operation, func))
        return True

    def _discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _disconnect_module(self) -> None:
        module = self._module
        self._module = None
        if module is not None:
            module.disconnect()

    def _module_ok(self) -> LoggedE720Module:
        if self._module is None or not self._module.connected:
            raise RuntimeError("模块未连接")
        return self._module

    # ---- 连接 ---------------------------------------------------------------

    def connect_device(self, port: str, baudrate: int):
        def _connect():
            self._disconnect_module()
            module = LoggedE720Module(self._logger)
            if not module.connect(port, baudrate):
                self._module = None
                self.connected_signal.emit(False, f"连接 {port} 失败")
                return
            self._module = module
            self.connected_signal.emit(True, f"已连接 {port} @ {baudrate}")
            try:
                self._emit_info()
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                self._logger.log_signal.emit(
                    "error", f"读取模块信息异常: {message}")
                self.operation_failed.emit("info", message)
        self.submit("connect", _connect)

    def disconnect_device(self):
        # 断开命令应紧跟当前正在执行的命令，丢弃尚未开始的读写/扫描请求。
        self._discard_pending()

        def _disconnect():
            self._disconnect_module()
            self.connected_signal.emit(False, "已断开连接")
        self.submit("disconnect", _disconnect)

    def _emit_info(self):
        info = self._module_ok().get_module_summary()
        info["region_name"] = REGION_NAMES.get(info["region"] or 0, "?")
        self.info_ready.emit(info)

    def refresh_info(self):
        self.submit("info", self._emit_info)

    # ---- 寻卡 ---------------------------------------------------------------

    def single_inventory(self):
        """单次寻卡 (一轮); 连续扫描由页面定时器按间隔反复调用本命令。"""
        def _inventory():
            tags = self._module_ok().single_inventory()
            self.inventory_done.emit(tags)
        self.submit("inventory", _inventory)

    # ---- 读写 ---------------------------------------------------------------

    def read_tag(self, epc_hex: str, bank: int, addr: int, words: int):
        def _read():
            module = self._module_ok()
            epc = hex_to_bytes(epc_hex)
            data = module.read_tag(epc, bank, addr, words)
            if data is None:
                self.read_done.emit(False, epc_hex, "", "")
                return
            try:
                text = data.decode("ascii")
                if not all(32 <= ord(c) < 127 for c in text):
                    text = ""
            except (UnicodeDecodeError, ValueError):
                text = ""
            self.read_done.emit(True, epc_hex, format_hex(data), text)
        self.submit("read", _read)

    def write_user(self, epc_hex: str, data_hex: str, addr: int):
        def _write():
            module = self._module_ok()
            try:
                epc = hex_to_bytes(epc_hex)
                data = hex_to_bytes(data_hex)
            except ValueError:
                self.write_done.emit(False, "EPC 或写入数据不是有效的十六进制")
                return
            if not data:
                self.write_done.emit(False, "写入数据不能为空")
                return
            if len(data) % 2 != 0:
                self.write_done.emit(False, "数据长度必须是偶数字节 (1 Word = 2 字节)")
                return
            if not module.write_tag(epc, data, MEMBANK_USER, addr):
                self.write_done.emit(False, "写入失败 (详见控制台日志)")
                return
            # 回读验证
            verify = module.read_tag(epc, MEMBANK_USER, addr, len(data) // 2)
            if verify == data:
                self.write_done.emit(True, "写入成功, 回读验证一致 ✓")
            elif verify is None:
                self.write_done.emit(
                    False, "写入指令已被接受, 但无法回读验证; 请手动读取确认")
            else:
                self.write_done.emit(
                    False, f"写入指令已被接受, 但回读不一致: {format_hex(verify)}")
        self.submit("write-user", _write)

    def detect_capacity(self, epc_hex: str):
        """只读检测 User 与 EPC 存储区是否存在及其容量。"""

        def _detect():
            module = self._module_ok()
            try:
                epc = hex_to_bytes(epc_hex)
            except ValueError as exc:
                raise RuntimeError("EPC 不是有效的十六进制") from exc
            if len(epc) != 12:
                raise RuntimeError("当前 EPC 必须是 12 字节")
            if not module.select_clear():
                raise RuntimeError("清除标签筛选失败")
            time.sleep(0.08)
            if not module.select_epc(epc):
                raise RuntimeError("选择目标标签失败，请确认标签仍在读取范围内")
            time.sleep(0.08)

            results = {}
            for key, bank in (("user", MEMBANK_USER), ("epc", MEMBANK_EPC)):
                def probe(address: int, current_bank=bank):
                    last_message = ""
                    for _attempt in range(2):
                        result = module.read_selected_tag_result(
                            current_bank, address, 1
                        )
                        if result["ok"]:
                            return PROBE_OK, ""
                        if result["error_code"] == E720.ERROR_READ_OUT_OF_RANGE:
                            return PROBE_OUT_OF_RANGE, result["message"]
                        last_message = result["message"]
                    return PROBE_ERROR, last_message

                results[key] = detect_memory_capacity(probe)

            self.capacity_done.emit(epc_hex.upper(), results)

        self.submit("capacity", _detect)

    def write_epc(self, old_epc_hex: str, new_epc_hex: str):
        def _write():
            module = self._module_ok()
            try:
                old_epc = hex_to_bytes(old_epc_hex)
                new_epc = hex_to_bytes(new_epc_hex)
            except ValueError:
                self.epc_write_done.emit(
                    False, "EPC 不是有效的十六进制", old_epc_hex.upper())
                return
            if len(old_epc) != 12:
                self.epc_write_done.emit(
                    False, "当前 EPC 必须是 12 字节", old_epc_hex.upper())
                return
            if len(new_epc) != 12:
                self.epc_write_done.emit(
                    False, "新 EPC 必须是 12 字节 (24 位十六进制)",
                    old_epc_hex.upper())
                return
            # EPC 数据在 EPC 存储区 Word 偏移 2 (跳过 CRC-16 与 PC)
            if not module.write_tag(old_epc, new_epc, MEMBANK_EPC, 2):
                self.epc_write_done.emit(
                    False, "写入失败 (详见控制台日志)", old_epc_hex.upper())
                return
            # 用新 EPC 验证
            tags = module.single_inventory()
            found_new = any(t["epc_hex"] == new_epc_hex.upper() for t in tags)
            found_old = any(t["epc_hex"] == old_epc_hex.upper() for t in tags)
            if found_new:
                self.epc_write_done.emit(
                    True, "EPC 修改成功, 新卡号已生效 ✓", new_epc_hex.upper())
            elif found_old:
                self.epc_write_done.emit(
                    False, "旧 EPC 仍存在, 新 EPC 未生效", old_epc_hex.upper())
            else:
                self.epc_write_done.emit(
                    True, "写入指令成功, 但未扫描到新卡号; 请重新扫描确认", "")
        self.submit("write-epc", _write)

    # ---- 设置 ---------------------------------------------------------------

    def set_power(self, dbm: int):
        def _set():
            ok = self._module_ok().set_power(dbm)
            self.settings_done.emit(ok, f"功率 {dbm} dBm "
                                   + ("已设置" if ok else "设置失败"))
            if ok:
                self._emit_info()
        self.submit("settings", _set)

    def set_region(self, code: int, name: str):
        def _set():
            ok = self._module_ok().set_region(code)
            self.settings_done.emit(ok, f"地区 {name} "
                                   + ("已设置" if ok else "设置失败"))
            if ok:
                self._emit_info()
        self.submit("settings", _set)

    def set_channel(self, channel: int):
        def _set():
            ok = self._module_ok().set_channel(channel)
            self.settings_done.emit(ok, f"信道 {channel} "
                                   + ("已设置" if ok else "设置失败"))
            if ok:
                self._emit_info()
        self.submit("settings", _set)

    # ---- 调试 ---------------------------------------------------------------

    def debug_send(self, hex_text: str):
        """发送原始指令 (不含帧头 BB 与帧尾 7E), 结果发到通讯日志。"""

        def _debug():
            module = self._module_ok()
            try:
                data = hex_to_bytes(hex_text)
            except ValueError:
                self.settings_done.emit(False, "无效的十六进制指令")
                return
            if len(data) < 2:
                self.settings_done.emit(False, "指令太短")
                return
            frame = (bytes([E720.FRAME_HEADER]) + data
                     + bytes([E720.calc_checksum(data), E720.FRAME_END]))
            module.ser.reset_input_buffer()
            module.ser.write(frame)
            module.ser.flush()
            self._logger.log_signal.emit("send", f"[调试] {format_hex(frame)}")
            import time as _time
            _time.sleep(0.3)
            resp = module._read_raw()
            if resp:
                self._logger.log_signal.emit("recv", f"[调试] {format_hex(resp)}")
                offset = 0
                while offset < len(resp):
                    pos = resp.find(E720.FRAME_HEADER_BYTES, offset)
                    if pos < 0:
                        break
                    f = parse_frame(resp[pos:])
                    if not f:
                        offset = pos + 1
                        continue
                    offset = pos + f["total_len"]
                    self._logger.log_signal.emit(
                        "info",
                        f"type=0x{f['type']:02X} cmd=0x{f['command']:02X} "
                        f"pl={format_hex(f['payload'])} "
                        f"csum={'OK' if f['checksum_ok'] else 'ERR'}")
                    if f["type"] == TYPE_NOTIFICATION:
                        tag = parse_notification_frame(f)
                        if tag:
                            self._logger.log_signal.emit(
                                "info",
                                f"EPC:{tag['epc_hex']} RSSI:{tag['rssi']}dBm")
            else:
                self._logger.log_signal.emit("recv", "[调试] (无响应)")
        self.submit("debug", _debug)
