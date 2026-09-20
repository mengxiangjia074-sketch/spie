"""LensConnect 镜头控制工作线程。

LensCtrl SDK 的所有调用都是阻塞式的 (每次 USB 读取 sleep 0.1 s, 移动/初始化
会轮询等待到位), 直接在 GUI 线程调用会卡死界面, 因此本线程串行执行全部硬件
交互: GUI 线程只入队请求, 通过信号接收结果。请求队列同时天然避免了移动过程
中的并发读写。

对齐厂商 GUI 程序 LensConnect_Windows_GUI_x86_2.2.0 的功能:
扫描/连接、zoom/focus/iris/滤镜手动控制、速度与间隙补偿、镜头信息/温度/
用户标识读写、预设位顺序执行。

注意: LensCtrl.OptFilterMove 向 DeviceMove 少传一个 mask 参数 (SDK 潜在
TypeError), 这里直接组合 LensCtrl.DeviceMove, 不改动 detection 代码。
"""

from __future__ import annotations

import importlib
import queue
import time

from PySide6.QtCore import QThread, Signal

from LensCamera.lens_ports import LensControlError
from LensCamera.lensconnect_adapter import LensConnectController

MOTORS = ("zoom", "focus", "iris")
OPT = "opt"
KELVIN_OFFSET = 273.15
POLL_INTERVAL = 1.0   # s, 已连接且空闲时的轮询周期
IDLE_INTERVAL = 0.5   # s, 未连接时等待请求的超时
WAIT_SLICE = 0.1      # s, 预设位等待的可中断时间切片
USER_AREA_LENGTH = 32

# 深层参数寄存器 (详细信息对话框, 对应厂商 GUI 的 "More" 页)
_DETAIL_REGS = {
    "zoom": {
        "pos_min": "ZOOM_POSITION_MIN", "pos_max": "ZOOM_POSITION_MAX",
        "mech_min": "ZOOM_MECH_STEP_MIN", "mech_max": "ZOOM_MECH_STEP_MAX",
        "init_pos": "ZOOM_INIT_POSITION",
    },
    "focus": {
        "pos_min": "FOCUS_POSITION_MIN", "pos_max": "FOCUS_POSITION_MAX",
        "mech_min": "FOCUS_MECH_STEP_MIN", "mech_max": "FOCUS_MECH_STEP_MAX",
        "init_pos": "FOCUS_INIT_POSITION",
    },
    "iris": {
        "pos_min": "IRIS_POSITION_MIN", "pos_max": "IRIS_POSITION_MAX",
        "mech_min": "IRIS_MECH_STEP_MIN", "mech_max": "IRIS_MECH_STEP_MAX",
        "init_pos": "IRIS_INIT_POSITION",
    },
}


class LensWorker(QThread):
    """串行执行镜头 USB 操作的后台线程。"""

    scan_done = Signal(bool, list, str)      # ok, ["0: SN", ...], message
    connected = Signal(dict)                 # 连接后的完整快照
    disconnected = Signal()
    failed = Signal(str)                     # 操作级错误 (连接已自动善后)
    polled = Signal(dict)                    # 周期/操作后的状态刷新
    feedback = Signal(str, str)              # motor ("zoom"/"user"/...), 结果文本
    details_ready = Signal(dict)             # 详细信息 (深层寄存器)
    preset_progress = Signal(int, str)       # 预设序号 (0 基), 阶段文本
    preset_finished = Signal(bool, str)
    calibrated_position_finished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = LensConnectController()
        self._queue: queue.Queue = queue.Queue()
        self._connected = False
        self._capabilities = 0
        self._preset_stop = False
        self._quitting = False
        self._LC = None  # LensCtrl, 连接后可用

    # ---- GUI 线程调用的请求接口 --------------------------------------------

    def scan(self):
        self._put("scan")

    def connect_device(self, index):
        self._put("connect", index=index)

    def disconnect_device(self):
        self._put("disconnect")

    def move(self, motor, target):
        self._put("move", motor=motor, target=int(target))

    def step(self, motor, delta):
        self._put("step", motor=motor, delta=int(delta))

    def initialize(self, motor):
        self._put("init", motor=motor)

    def move_calibrated_position(self, position):
        self._put("calibrated_position", position=dict(position))

    def set_speed(self, motor, pps):
        self._put("speed", motor=motor, pps=int(pps))

    def set_backlash(self, motor, enabled):
        self._put("backlash", motor=motor, enabled=bool(enabled))

    def refresh_temperature(self):
        self._put("temperature")

    def refresh_details(self):
        self._put("details")

    def write_user_area(self, text):
        self._put("user_area", text=str(text))

    def run_preset(self, positions):
        self._put("preset", positions=positions)

    def stop_preset(self):
        self._preset_stop = True

    def shutdown(self):
        self._quitting = True
        self._put("noop")

    def _put(self, kind, **payload):
        self._queue.put({"kind": kind, **payload})

    # ---- 线程主循环 ---------------------------------------------------------

    def run(self):  # noqa: C901 - dispatch 循环本质上是平铺的
        handlers = {
            "noop": lambda p: None,
            "scan": self._do_scan,
            "connect": self._do_connect,
            "disconnect": self._do_disconnect,
            "move": self._do_move,
            "step": self._do_step,
            "init": self._do_init,
            "calibrated_position": self._do_calibrated_position,
            "speed": self._do_speed,
            "backlash": self._do_backlash,
            "temperature": self._do_temperature,
            "details": self._do_details,
            "user_area": self._do_user_area,
            "preset": self._do_preset,
        }
        while not self._quitting:
            try:
                item = self._queue.get(
                    timeout=POLL_INTERVAL if self._connected else IDLE_INTERVAL
                )
                kind = item["kind"]
                if kind == "noop":
                    continue
                self._safe(lambda: handlers[kind](item), kind)
                if self._quitting and self._queue.empty():
                    break
            except queue.Empty:
                if self._connected:
                    self._safe(self._do_poll, "poll")
        if self._connected:
            self._close_quietly()

    def _safe(self, fn, action):
        try:
            fn()
        except LensControlError as exc:
            self._report_failure(action, f"失败: {exc}")
        except Exception as exc:  # USB 栈抛出的各类异常
            self._report_failure(action, f"异常: {exc}")

    def _report_failure(self, action, detail):
        self.failed.emit(f"{action} {detail}")
        if action == "calibrated_position":
            self.calibrated_position_finished.emit(False, detail)
        if action == "poll" and self._connected:
            # 空闲轮询失败按连接丢失处理, 避免 USB 已拔出后无限报错。
            self._close_quietly()
            self.disconnected.emit()

    # ---- SDK 访问 ----------------------------------------------------------

    def _load_sdk(self):
        """加载 Controller 模块并取得 LensCtrl / 常量 (连接前调用)。"""
        self.controller.modules.load()
        self._LC = self.controller.LensCtrl
        return {
            "CV": self.controller.CV,
            "DV": self.controller.DV,
            "DA": importlib.import_module("DevAddr"),
            "UsbCtrl": self.controller.UsbCtrl,
        }

    def _sdk(self):
        if self._LC is None:
            raise LensControlError("LensConnect SDK 未加载")
        return {
            "CV": self.controller.CV,
            "DV": self.controller.DV,
            "DA": importlib.import_module("DevAddr"),
            "UsbCtrl": self.controller.UsbCtrl,
        }

    def _check(self, retval, action):
        self.controller.check_success(retval, action)

    def _supported(self, motor):
        return self.controller.motor_supported(motor, self._capabilities)

    def _initialized(self, motor):
        mask = self.controller.CV.OPT_FILTER_MASK if motor == OPT \
            else self.controller.motors[motor]["mask"]
        return (self._LC.status2 & mask) == self.controller.DV.INIT_COMPLETED

    # ---- 各操作 ------------------------------------------------------------

    def _do_scan(self, payload=None):
        devices = []
        try:
            self._load_sdk()
            usb = self.controller.UsbCtrl
            num = self.controller.count_devices()
            for index in range(num):
                retval, serial = usb.UsbGetSnDevice(index)
                self._check(retval, "read device serial")
                devices.append(f"{index}: {serial}")
            self.scan_done.emit(True, devices, f"检测到 {num} 台设备")
        except Exception as exc:
            self.scan_done.emit(False, [], f"扫描失败: {exc}")

    def _do_connect(self, payload):
        index = payload["index"]
        try:
            sdk = self._load_sdk()
            self._capabilities = self.controller.connect(index)
            self._connected = True
            for motor in MOTORS:
                if self._supported(motor) and not self._initialized(motor):
                    retval = self.controller.motors[motor]["init"]()
                    self._check(retval, f"initialize {motor}")
            snapshot = self._read_snapshot(sdk)
            self.connected.emit(snapshot)
        except Exception as exc:
            self._connected = False
            self._close_quietly()
            self.failed.emit(f"连接失败: {exc}")

    def _do_disconnect(self, payload=None):
        if self._connected:
            self._close_quietly()
        self.disconnected.emit()

    def _close_quietly(self):
        self._connected = False
        try:
            self.controller.close()
        except Exception:
            pass

    def _read_snapshot(self, sdk):
        LC, CV = self._LC, sdk["CV"]
        retval, status1 = LC.Status1Read()
        self._check(retval, "read status1")
        LC.Status2ReadSet()

        snapshot = {
            "device": self.controller.get_last_connected_device_number(),
            "capabilities": self._capabilities,
            "status1": status1,
            "status2": LC.status2,
            "motors": {name: self._read_motor_block(sdk, name) for name in MOTORS},
        }

        retval, model = LC.ModelName()
        snapshot["model"] = model.strip() if retval == 0 else "?"
        retval, version = LC.FWVersion()
        snapshot["fw_version"] = version if retval == 0 else "?"
        retval, version = LC.ProtocolVersion()
        snapshot["protocol_version"] = version if retval == 0 else "?"
        retval, revision = LC.LensRevision()
        snapshot["revision"] = revision if retval == 0 else None
        retval, address = LC.LensAddress()
        snapshot["lens_address"] = (
            f"0x{address:02X}({address})" if retval == 0 else "?"
        )
        retval, user_area = LC.UserAreaRead()
        snapshot["user_area"] = user_area.strip() if retval == 0 else ""
        snapshot.update(self._read_temperature())

        if self._capabilities & CV.OPT_FILTER_MASK:
            snapshot["opt"] = self._read_opt_block()
        else:
            snapshot["opt"] = {"supported": False}
        return snapshot

    def _read_motor_block(self, sdk, name):
        """一个电机的完整参数块 (含深层寄存器, 对应厂商 GUI 的 More 页)。"""
        if not self._supported(name):
            return {"supported": False}
        LC, DA, usb = self._LC, sdk["DA"], sdk["UsbCtrl"]
        LC_attr = name.capitalize()

        block = {
            "supported": True,
            "initialized": self._initialized(name),
            "min": getattr(LC, f"{name}MinAddr"),
            "max": getattr(LC, f"{name}MaxAddr"),
            "speed": getattr(LC, f"{name}SpeedPPS"),
            "current": None,
        }
        if block["initialized"]:
            block["current"] = self.controller.read_current(name)

        for key, reg_name in _DETAIL_REGS[name].items():
            retval, value = usb.UsbRead2BytesInt(getattr(DA, reg_name))
            self._check(retval, f"read {name} {reg_name}")
            block[key] = value

        retval, block["speed_min"] = getattr(LC, f"{LC_attr}SpeedMinRead")()
        self._check(retval, f"read {name} speed min")
        retval, block["speed_max"] = getattr(LC, f"{LC_attr}SpeedMaxRead")()
        self._check(retval, f"read {name} speed max")
        retval, block["backlash"] = getattr(LC, f"{LC_attr}BacklashRead")()
        self._check(retval, f"read {name} backlash")
        retval, block["count"] = getattr(LC, f"{LC_attr}CountValRead")()
        self._check(retval, f"read {name} count")
        retval, block["count_max"] = getattr(LC, f"{LC_attr}CountMaxRead")()
        self._check(retval, f"read {name} count max")
        return block

    def _read_opt_block(self):
        LC = self._LC
        block = {
            "supported": True,
            "initialized": self._initialized(OPT),
            "current": None,
        }
        retval = LC.OptFilterParameterReadSet()  # 读最大值 + 当前位置
        self._check(retval, "read optical filter parameters")
        block["max"] = LC.optFilMaxAddr
        if block["initialized"]:
            block["current"] = LC.optCurrentAddr
        retval, block["count_max"] = LC.OptFilterCountMaxRead()
        self._check(retval, "read optical filter count max")
        return block

    def _read_temperature(self):
        retval, kelvin = self._LC.TempKelvinVal()
        self._check(retval, "read temperature")
        celsius = kelvin - KELVIN_OFFSET
        return {
            "temperature_c": round(celsius, 1),
            "temperature_f": round(celsius * 9.0 / 5.0 + 32.0, 1),
        }

    def _collect_status(self, sdk):
        """poll/操作后的轻量状态: status1/2 + 各电机当前位置 + 温度。"""
        LC, CV = self._LC, sdk["CV"]
        retval, status1 = LC.Status1Read()
        self._check(retval, "read status1")
        LC.Status2ReadSet()
        positions = {}
        for name in MOTORS:
            if self._supported(name) and self._initialized(name):
                positions[name] = self.controller.read_current(name)
            else:
                positions[name] = None
        status = {
            "status1": status1,
            "status2": LC.status2,
            "positions": positions,
            "opt_current": None,
        }
        if self._capabilities & CV.OPT_FILTER_MASK and self._initialized(OPT):
            LC.OptFilterCurrentAddrReadSet()
            status["opt_current"] = LC.optCurrentAddr
        status.update(self._read_temperature())
        return status

    def _do_poll(self, payload=None):
        self.polled.emit(self._collect_status(self._sdk()))

    def _move_motor(self, name, target):
        return self.controller.move_connected_motor(
            name, target, self._capabilities,
            init_if_needed=True, clamp_target=True,
        )

    def _move_opt(self, sdk, target):
        """绕开 LensCtrl.OptFilterMove 缺 mask 参数的问题, 直接组合底层调用。"""
        LC, CV, DA = self._LC, sdk["CV"], sdk["DA"]
        target = int(max(0, min(LC.optFilMaxAddr, target)))
        wait = LC.WaitCalc(abs(target - LC.optCurrentAddr), CV.OPT_FILTER_SPEED)
        retval, actual = LC.DeviceMove(
            DA.OPT_FILTER_POSITION_VAL, target, CV.OPT_FILTER_MASK, wait
        )
        self._check(retval, f"move optical filter to {target}")
        return {"target": target, "actual": actual}

    def _do_move(self, payload):
        sdk = self._sdk()
        motor, target = payload["motor"], payload["target"]
        if motor == OPT:
            result = self._move_opt(sdk, target)
            self.feedback.emit(motor, f"滤镜 → {result['actual']}")
        else:
            result = self._move_motor(motor, target)
            if not result["moved"]:
                text = f"已在目标位置 {target}"
            elif result["error"] != 0:
                text = (f"{result['before']} → {result['actual']} "
                        f"(偏差 {result['error']:+d})")
            else:
                text = f"{result['before']} → {result['actual']}"
            self.feedback.emit(motor, text)
        self.polled.emit(self._collect_status(sdk))

    def _do_step(self, payload):
        motor, delta = payload["motor"], payload["delta"]
        self.controller.ensure_ready(motor, True)
        motor_range = self.controller.get_motor_range(motor)
        current = self.controller.read_current(motor)
        target = int(max(motor_range["min"], min(motor_range["max"],
                                                 current + delta)))
        self._do_move({"motor": motor, "target": target})

    def _do_init(self, payload):
        sdk = self._sdk()
        motor = payload["motor"]
        if motor == OPT:
            retval = self._LC.OptFilterInit()
            self._check(retval, "initialize optical filter")
            self._LC.OptFilterParameterReadSet()
            self.feedback.emit(motor, f"初始化完成, 当前 {self._LC.optCurrentAddr}")
        else:
            retval = self.controller.motors[motor]["init"]()
            self._check(retval, f"initialize {motor}")
            current = self.controller.read_current(motor)
            self.feedback.emit(motor, f"初始化完成, 当前 {current}")
        self.polled.emit(self._collect_status(sdk))

    def _do_calibrated_position(self, payload):
        sdk = self._sdk()
        position = {motor: int(payload["position"][motor]) for motor in MOTORS}
        ranges = {}
        for motor in MOTORS:
            supported = self._supported(motor)
            ranges[motor] = {
                "supported": supported,
                **(self.controller.get_motor_range(motor) if supported else {}),
            }
        self.controller.validate_targets(position, ranges)
        actual = {}
        for motor in MOTORS:
            result = self._move_motor(motor, position[motor])
            actual[motor] = int(result["actual"])
            self.feedback.emit(motor, f"标定位置 → {actual[motor]}")
        self.polled.emit(self._collect_status(sdk))
        message = "已转到 Zoom {} / Focus {} / Iris {}".format(
            actual["zoom"], actual["focus"], actual["iris"]
        )
        self.calibrated_position_finished.emit(True, message)

    def _do_speed(self, payload):
        sdk = self._sdk()
        motor, pps = payload["motor"], payload["pps"]
        write_fn = getattr(self._LC, f"{motor.capitalize()}SpeedWrite")
        retval = write_fn(pps)
        self._check(retval, f"write {motor} speed")
        self.feedback.emit(motor, f"速度已写入 {pps} PPS")
        self.polled.emit(self._collect_status(sdk))

    def _do_backlash(self, payload):
        sdk = self._sdk()
        motor, enabled = payload["motor"], payload["enabled"]
        write_fn = getattr(self._LC, f"{motor.capitalize()}BacklashWrite")
        retval = write_fn(1 if enabled else 0)
        self._check(retval, f"write {motor} backlash")
        self.feedback.emit(motor, f"间隙补偿 {'开启' if enabled else '关闭'}")
        self.polled.emit(self._collect_status(sdk))

    def _do_temperature(self, payload=None):
        sdk = self._sdk()
        temps = self._read_temperature()
        self.feedback.emit("temp", f"温度 {temps['temperature_c']} °C "
                                   f"/ {temps['temperature_f']} °F")
        self.polled.emit(self._collect_status(sdk))

    def _do_details(self, payload=None):
        self.details_ready.emit(self._read_snapshot(self._sdk()))

    def _do_user_area(self, payload):
        self._sdk()
        text = payload["text"][:USER_AREA_LENGTH]
        retval = self._LC.UserAreaWrite(text)
        self._check(retval, "write user area")
        self.feedback.emit("user", f"用户标识已写入: {text!r}")

    # ---- 预设位 ------------------------------------------------------------

    def _do_preset(self, payload):
        sdk = self._sdk()
        positions = payload["positions"]
        self._preset_stop = False
        for index, pos in enumerate(positions):
            if self._preset_stop:
                self.preset_finished.emit(False, "已停止")
                return
            self.preset_progress.emit(index, "移动镜头")
            try:
                for motor in MOTORS:
                    value = pos.get(motor)
                    if value is not None:
                        self._move_motor(motor, value)
                if self._preset_stop:
                    self.preset_finished.emit(False, "已停止")
                    return
                filt = pos.get("filter")
                if filt is not None and (self._capabilities & sdk["CV"].OPT_FILTER_MASK):
                    self._move_opt(sdk, int(filt))
            except Exception as exc:
                self.preset_finished.emit(False, f"位置 {index + 1} 失败: {exc}")
                return
            wait = float(pos.get("wait_seconds", 0.0))
            self.preset_progress.emit(index, f"等待 {wait:g} s")
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if self._preset_stop:
                    self.preset_finished.emit(False, "已停止")
                    return
                time.sleep(WAIT_SLICE)
        self.preset_finished.emit(True, f"完成 {len(positions)} 个预设位")
