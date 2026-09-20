"""位移台控制工作线程 (参照 detection/motion_gui.py 的 CommandWorker 移植)。

所有串口 (Modbus RTU) 通讯都在本线程串行执行, 避免阻塞 GUI; 每次读写通过
CommLogger 发射日志信号供通讯日志面板显示。与 motion_gui.py 的差异:

- PySide6 信号;
- 适配新版 motion_controller.py: 位置读取用 read_motor_position_pulses()
  (新 API 的 read_motor_position 返回 mm 且每次读硬件细分数),
  home(wait=False) 不阻塞队列, 绝对定位先换算出硬件坐标 mm 再交给 SDK;
- 零点偏移以脉冲保存, 与参照实现一致 (纯软件偏移, 不改动硬件位置)。
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager, nullcontext

from PySide6.QtCore import QObject, QThread, Signal

from motion_controller import (
    ADDR_X_AXIS,
    ADDR_Y_AXIS,
    MODE_ABSOLUTE,
    MODE_RELATIVE,
    REG_ALARM,
    REG_PULSE_PER_REV,
    MotionController,
)

AXES = (ADDR_X_AXIS, ADDR_Y_AXIS)
DEFAULT_PPR = 51200
AUTO_REFRESH_ERROR_THRESHOLD = 3

ALARM_NAMES = {
    0x01: "过流", 0x02: "过压", 0x40: "电流采样回路故障",
    0x80: "锁轴(缺相)故障", 0x200: "EEPROM故障",
    0x100: "参数自整定故障", 0x020: "超差报警",
    0x008: "编码器断线报警", 0x009: "输入IO重复配置",
    0x00A: "过温报警",
}


class CommLogger(QObject):
    """通讯日志信号发射器 (direction, message)。"""

    log_signal = Signal(str, str)


class LoggedMotionController(MotionController):
    """重写读写方法, 在每次通讯时发射日志信号。"""

    def __init__(self, port, slave_address, baudrate, logger):
        self._logger = logger
        self._communication_logging_enabled = True
        super().__init__(
            port,
            slave_address,
            baudrate,
            log_callback=self._emit_controller_log,
        )

    def _emit_controller_log(self, message: str) -> None:
        self._logger.log_signal.emit("info", message)

    def _emit_communication_log(self, direction: str, message: str) -> None:
        if self._communication_logging_enabled:
            self._logger.log_signal.emit(direction, message)

    @contextmanager
    def quiet_communication(self):
        """Temporarily suppress raw Modbus logs during automatic polling."""
        previous = self._communication_logging_enabled
        self._communication_logging_enabled = False
        try:
            yield
        finally:
            self._communication_logging_enabled = previous

    def read_register(self, register_address):
        self._emit_communication_log(
            "send", f"[轴{self.slave_address}] 读寄存器 0x{register_address:04X}")
        try:
            result = super().read_register(register_address)
            self._emit_communication_log(
                "recv", f"[轴{self.slave_address}] <- 0x{result:04X} ({result})")
            return result
        except Exception as exc:
            self._emit_communication_log(
                "error", f"[轴{self.slave_address}] 读错误: {exc}")
            raise

    def read_alarm_quiet(self) -> int:
        """读取报警寄存器，但不输出自动轮询的发送/接收和“无报警”日志。"""
        try:
            return MotionController.read_register(self, REG_ALARM)
        except Exception as exc:
            self._emit_communication_log(
                "error", f"[轴{self.slave_address}] 读取报警失败: {exc}"
            )
            raise

    def write_register(self, register_address, value):
        self._emit_communication_log(
            "send", f"[轴{self.slave_address}] 写寄存器 0x{register_address:04X}"
                    f" = 0x{value:04X} ({value})")
        try:
            super().write_register(register_address, value)
            self._emit_communication_log(
                "recv", f"[轴{self.slave_address}] <- OK"
            )
        except Exception as exc:
            self._emit_communication_log(
                "error", f"[轴{self.slave_address}] 写错误: {exc}")
            raise

    def read_registers(self, start_address, count):
        self._emit_communication_log(
            "send", f"[轴{self.slave_address}] 读寄存器 0x{start_address:04X} x{count}")
        try:
            result = super().read_registers(start_address, count)
            values = ", ".join(f"0x{v:04X}" for v in result)
            self._emit_communication_log(
                "recv", f"[轴{self.slave_address}] <- [{values}]")
            return result
        except Exception as exc:
            self._emit_communication_log(
                "error", f"[轴{self.slave_address}] 读错误: {exc}")
            raise

    def write_registers(self, start_address, values):
        values_text = ", ".join(f"0x{v:04X}" for v in values)
        self._emit_communication_log(
            "send", f"[轴{self.slave_address}] 写寄存器 0x{start_address:04X}"
                    f" = [{values_text}]")
        try:
            super().write_registers(start_address, values)
            self._emit_communication_log(
                "recv", f"[轴{self.slave_address}] <- OK"
            )
        except Exception as exc:
            self._emit_communication_log(
                "error", f"[轴{self.slave_address}] 写错误: {exc}")
            raise


class MotionWorker(QThread):
    """串行执行位移台串口命令的后台线程。"""

    result_ready = Signal(str, bool)            # message, is_error
    position_updated = Signal(int, int, float)  # axis, display pulses, mm
    status_updated = Signal(int, dict)          # axis, status dict
    alarm_updated = Signal(int, int, str)       # axis, code, desc
    connected_signal = Signal(bool, str)        # success, message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: queue.Queue = queue.Queue()
        self._controller: LoggedMotionController | None = None
        self._running = True
        self._cancel_connect = threading.Event()
        self._refresh_lock = threading.Lock()
        self._refresh_pending = False
        self._auto_refresh_failure_count = 0
        self._auto_refresh_failure_reported = False
        self._logger = CommLogger()
        self._leads = {ADDR_X_AXIS: 1.0, ADDR_Y_AXIS: 1.0}
        self._pulses_per_rev = DEFAULT_PPR
        self._zero_offsets = {ADDR_X_AXIS: 0, ADDR_Y_AXIS: 0}

    @property
    def logger(self) -> CommLogger:
        return self._logger

    # ---- 线程主循环 ---------------------------------------------------------

    def run(self):
        try:
            while True:
                try:
                    cmd = self._queue.get(timeout=0.1)
                    if cmd is None:
                        break
                    try:
                        cmd()
                    except Exception as exc:
                        self.result_ready.emit(str(exc), True)
                except queue.Empty:
                    continue
        finally:
            self._close_controller()
            self._running = False

    def stop(self):
        self._cancel_connect.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)

    def submit(self, func):
        self._queue.put(func)

    def _require_controller(self) -> LoggedMotionController:
        if self._controller is None:
            raise RuntimeError("设备未连接")
        return self._controller

    def _set_axis(self, axis: int) -> LoggedMotionController:
        """切换当前通讯的轴地址 (X/Y 共用一个串口, 从站地址不同)。"""
        ctrl = self._require_controller()
        ctrl.instrument.address = axis
        ctrl.slave_address = axis
        return ctrl

    def _pulses_to_mm(self, pulses: int, axis: int) -> float:
        return pulses / float(self._pulses_per_rev) * self._leads[axis]

    def _close_controller(self) -> None:
        controller = self._controller
        self._controller = None
        self._auto_refresh_failure_count = 0
        self._auto_refresh_failure_reported = False
        if controller is not None:
            try:
                controller.close()
            except Exception:
                pass

    # ---- 连接管理 -----------------------------------------------------------

    def connect_device(self, port, baudrate, lead_x, lead_y, pulses_per_rev):
        self._cancel_connect.clear()

        def _connect():
            try:
                self._close_controller()
                self._controller = LoggedMotionController(
                    port, ADDR_X_AXIS, baudrate, self._logger)
                self._leads = {ADDR_X_AXIS: lead_x, ADDR_Y_AXIS: lead_y}
                self._pulses_per_rev = pulses_per_rev
                self._zero_offsets = {ADDR_X_AXIS: 0, ADDR_Y_AXIS: 0}
                responding_axes: set[int] = set()
                # 从硬件读取实际细分数, 确保软件与硬件一致
                try:
                    ctrl = self._set_axis(ADDR_X_AXIS)
                    hw_ppr = ctrl.read_register(REG_PULSE_PER_REV)
                    responding_axes.add(ADDR_X_AXIS)
                    if hw_ppr > 0:
                        self._pulses_per_rev = hw_ppr
                        self._logger.log_signal.emit(
                            "info", f"硬件细分数: {hw_ppr} 脉冲/转")
                except Exception:
                    self._logger.log_signal.emit(
                        "info",
                        f"读取硬件细分数失败, 使用设定值: {pulses_per_rev}")
                if self._cancel_connect.is_set():
                    self._close_controller()
                    return
                # 读取两个轴的初始状态
                for axis in AXES:
                    if self._cancel_connect.is_set():
                        self._close_controller()
                        return
                    try:
                        ctrl = self._set_axis(axis)
                        self.status_updated.emit(axis, ctrl.read_status())
                        responding_axes.add(axis)
                    except Exception:
                        continue
                    try:
                        self._emit_position(axis)
                    except Exception:
                        pass
                    try:
                        self._emit_alarm(axis, ctrl.read_alarm())
                    except Exception:
                        pass
                if self._cancel_connect.is_set():
                    self._close_controller()
                    return
                if not responding_axes:
                    raise RuntimeError(
                        f"已打开 {port}，但 X/Y 轴均未响应。请检查串口、波特率、"
                        "从站地址和设备电源。"
                    )
                missing_axes = [str(axis) for axis in AXES if axis not in responding_axes]
                if missing_axes:
                    self._logger.log_signal.emit(
                        "info", f"警告: 轴 {', '.join(missing_axes)} 未响应"
                    )
                self._set_axis(ADDR_X_AXIS)
                self.connected_signal.emit(
                    True, f"已连接 {port} ({baudrate}bps, "
                          f"细分数={self._pulses_per_rev})")
            except Exception as exc:
                self._close_controller()
                self.connected_signal.emit(False, str(exc))
        self.submit(_connect)

    def disconnect_device(self):
        self._cancel_connect.set()

        def _disconnect():
            self._close_controller()
            self.connected_signal.emit(False, "已断开连接")
        self.submit(_disconnect)

    # ---- 运动控制 -------------------------------------------------------------

    def move_to_position_mm(self, axis, distance_mm, speed_rpm, mode, lead):
        def _move():
            ctrl = self._set_axis(axis)
            self._leads[axis] = lead
            position = int(distance_mm / lead * self._pulses_per_rev)
            mode_name = "绝对" if mode == MODE_ABSOLUTE else "相对"

            if mode == MODE_ABSOLUTE:
                # 用户坐标 + 零点偏移 = 硬件坐标 (换回 mm 交给 SDK)
                target_pulses = position + self._zero_offsets[axis]
                target_mm = self._pulses_to_mm(target_pulses, axis)
                self._logger.log_signal.emit(
                    "info",
                    f"[轴{axis}] {mode_name}定位 {distance_mm}mm "
                    f"(脉冲={position}, 零偏={self._zero_offsets[axis]}, "
                    f"目标={target_pulses}脉冲), {speed_rpm}RPM, 导程{lead}mm")
                ctrl.move_to_position(target_mm, speed_rpm, MODE_ABSOLUTE, lead)
            else:
                self._logger.log_signal.emit(
                    "info",
                    f"[轴{axis}] {mode_name}定位 {distance_mm}mm "
                    f"({position}脉冲), {speed_rpm}RPM, 导程{lead}mm")
                ctrl.move_to_position(distance_mm, speed_rpm, MODE_RELATIVE, lead)

            self.result_ready.emit(
                f"[轴{axis}] 定位指令已发送: {mode_name} "
                f"{distance_mm}mm @ {speed_rpm}RPM", False)
        self.submit(_move)

    def jog_forward(self, axis):
        def _jog():
            self._set_axis(axis).jog_forward()
            self.result_ready.emit(f"[轴{axis}] 正向点动已启动", False)
        self.submit(_jog)

    def jog_reverse(self, axis):
        def _jog():
            self._set_axis(axis).jog_reverse()
            self.result_ready.emit(f"[轴{axis}] 反向点动已启动", False)
        self.submit(_jog)

    def jog_stop(self, axis):
        def _stop():
            self._set_axis(axis).stop()
            self.result_ready.emit(f"[轴{axis}] 点动已停止", False)
        self.submit(_stop)

    def home(self, axis):
        def _home():
            ctrl = self._set_axis(axis)
            offset = self._zero_offsets[axis]
            if offset != 0:
                # 回到用户定义的相对零点
                target_mm = self._pulses_to_mm(offset, axis)
                ctrl.move_to_position(target_mm, 100, MODE_ABSOLUTE,
                                      self._leads[axis])
                self.result_ready.emit(
                    f"[轴{axis}] 回零指令已发送 (回到相对零点)", False)
            else:
                # 无自定义零点时使用硬件回零 (不阻塞队列)
                ctrl.home(wait=False)
                self.result_ready.emit(
                    f"[轴{axis}] 回零指令已发送 (硬件回零)", False)
        self.submit(_home)

    def set_zero(self, axis):
        def _set():
            ctrl = self._set_axis(axis)
            # 读取当前硬件位置作为零点偏移 (纯软件偏移, 不修改硬件位置)
            self._zero_offsets[axis] = ctrl.read_motor_position_pulses()
            self.position_updated.emit(axis, 0, 0.0)
            self.result_ready.emit(f"[轴{axis}] 当前位置已设为零点", False)
        self.submit(_set)

    def emergency_stop(self, axis):
        def _stop():
            self._set_axis(axis).stop()
            self.result_ready.emit(f"[轴{axis}] 急停已执行", False)
        self.submit(_stop)

    def emergency_stop_all(self):
        def _stop_all():
            for axis in AXES:
                try:
                    self._set_axis(axis).stop()
                except Exception:
                    pass
            self.result_ready.emit("全局急停已执行", False)
        self.submit(_stop_all)

    def clear_alarm(self, axis, clear_history=False):
        def _clear():
            self._set_axis(axis).clear_alarm(clear_history)
            label = "历史" if clear_history else "当前"
            self.result_ready.emit(f"[轴{axis}] 已清除{label}报警", False)
        self.submit(_clear)

    # ---- 状态读取 -------------------------------------------------------------

    def _emit_position(self, axis):
        ctrl = self._set_axis(axis)
        pulses = ctrl.read_motor_position_pulses()
        display = pulses - self._zero_offsets[axis]
        self.position_updated.emit(axis, display, self._pulses_to_mm(display, axis))

    def read_position(self, axis):
        self.submit(lambda: self._emit_position(axis))

    def read_status(self, axis):
        def _read():
            self.status_updated.emit(axis, self._set_axis(axis).read_status())
        self.submit(_read)

    def read_alarm(self, axis):
        def _read():
            self._emit_alarm(axis, self._set_axis(axis).read_alarm())
        self.submit(_read)

    def _emit_alarm(self, axis, alarm):
        if alarm != 0:
            names = [name for code, name in ALARM_NAMES.items() if alarm & code]
            desc = f"{', '.join(names)} (0x{alarm:04X})"
        else:
            desc = "无报警"
        self.alarm_updated.emit(axis, alarm, desc)

    def _handle_refresh_result(self, error, *, manual: bool) -> None:
        if manual:
            if error is not None:
                self.result_ready.emit(str(error), True)
            return

        if error is None:
            recovered = self._auto_refresh_failure_reported
            self._auto_refresh_failure_count = 0
            self._auto_refresh_failure_reported = False
            if recovered:
                self.result_ready.emit("位移台自动刷新通讯已恢复", False)
            return

        self._auto_refresh_failure_count += 1
        if self._auto_refresh_failure_count < AUTO_REFRESH_ERROR_THRESHOLD:
            return
        if not self._auto_refresh_failure_reported:
            self._auto_refresh_failure_reported = True
            self.result_ready.emit(
                "位移台自动刷新连续 {} 次无响应: {}".format(
                    AUTO_REFRESH_ERROR_THRESHOLD, error
                ),
                True,
            )

    def refresh_all(self, *, log_alarm=False):
        """刷新所有轴；自动轮询容忍瞬时超时，手动刷新立即报告。"""
        with self._refresh_lock:
            if self._refresh_pending:
                return
            self._refresh_pending = True

        def _refresh():
            first_error = None
            try:
                controller = self._require_controller()
                log_context = (
                    nullcontext()
                    if log_alarm
                    else controller.quiet_communication()
                )
                with log_context:
                    stop_after_error = False
                    for axis in AXES:
                        for reader in (
                            lambda axis=axis: self._emit_position(axis),
                            lambda axis=axis: self.status_updated.emit(
                                axis, self._set_axis(axis).read_status()
                            ),
                            lambda axis=axis: self._emit_alarm(
                                axis,
                                (
                                    self._set_axis(axis).read_alarm()
                                    if log_alarm
                                    else self._set_axis(axis).read_alarm_quiet()
                                ),
                            ),
                        ):
                            try:
                                reader()
                            except Exception as exc:
                                if first_error is None:
                                    first_error = exc
                                if not log_alarm:
                                    stop_after_error = True
                                    break
                        if stop_after_error:
                            break
                self._handle_refresh_result(first_error, manual=log_alarm)
            finally:
                with self._refresh_lock:
                    self._refresh_pending = False

        self.submit(_refresh)
