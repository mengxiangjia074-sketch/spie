"""位移台控制页: detection/motion_gui.py 的移植 (PySide6 + 本 GUI 主题)。

交互式双轴 Modbus RTU 位移台控制: 连接栏 (串口/波特率/导程/细分数/全局急停),
X/Y 轴面板 (位置/状态/报警显示, 绝对/相对定位, 点动, 回零, 设零, 清报警)。
通讯日志不在本页显示, 通过 console_output 信号合并到主窗口的运行控制台
(comm-send/recv/info/error 通道着色); 本页只留刷新控制与状态行。

位移台串口与检测任务互斥: 任务运行中禁止连接, 连接后禁止启动任务
(后者由 app.py 的 _start_task 检查)。镜头控制为 USB 设备, 可同时连接。
"""

from __future__ import annotations

from datetime import datetime

import serial.tools.list_ports
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from GUI.core.motion_worker import MotionWorker
from GUI.core.runner import TaskManager
from motion_controller import (
    MODE_ABSOLUTE,
    MODE_RELATIVE,
    ADDR_X_AXIS,
    ADDR_Y_AXIS,
)

REFRESH_INTERVAL_MS = 500  # 自动刷新间隔，避免长期高频轮询造成偶发超时

# 通讯日志方向 -> 控制台通道前缀/行前缀 (着色由 console.py 的 comm-* 通道决定)
LOG_PREFIX = {"send": ">>", "recv": "<<", "info": "--", "error": "!!"}

AXIS_LEADS = {ADDR_X_AXIS: "x", ADDR_Y_AXIS: "y"}  # axis -> 导程输入框前缀

STATUS_DEFINITIONS = (
    ("fault", "故障", "alert", "驱动器检测到故障，请查看下方报警信息"),
    ("enabled", "使能", "normal", "驱动输出已使能，可以接受运动命令"),
    ("running", "运行", "normal", "当前轴正在执行定位、点动或回零"),
    ("invalid", "无效", "alert", "当前指令或驱动器状态无效"),
    ("cmd_done", "指令完成", "normal", "最近一条控制指令已经执行完成"),
    ("path_done", "路径完成", "normal", "本次定位路径已经运行到终点"),
    ("home_done", "回零完成", "normal", "当前轴已经完成原点搜索和回零"),
)


class MotionStatusBadge(QFrame):
    """Compact themed indicator for one motion-controller status bit."""

    def __init__(self, label: str, severity: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MotionStatusBadge")
        self.setProperty("active", False)
        self.setProperty("severity", severity)
        self.setToolTip(tooltip)
        self.setMinimumHeight(29)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        self.dot = QLabel("●")
        self.dot.setObjectName("MotionStatusDot")
        self.dot.setToolTip(tooltip)
        self.text = QLabel(label)
        self.text.setObjectName("MotionStatusText")
        self.text.setToolTip(tooltip)
        layout.addWidget(self.dot)
        layout.addWidget(self.text)
        layout.addStretch(1)

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if self.property("active") == active:
            return
        self.setProperty("active", active)
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        for label in (self.dot, self.text):
            style.unpolish(label)
            style.polish(label)
        self.update()


class AxisPanel(QGroupBox):
    """一个轴的控制面板 (状态显示 + 运动控制), 对应 motion_gui 的 AxisPanel。"""

    def __init__(self, axis_name: str, axis_addr: int, parent=None):
        super().__init__(f"{axis_name} (地址 {axis_addr})", parent)
        self.axis_addr = axis_addr
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(6)

        # ---- 位置: 脉冲 = mm ----
        pos_layout = QHBoxLayout()
        self.pos_pulse_label = QLabel("--")
        self.pos_pulse_label.setObjectName("StatValue")
        self.pos_pulse_label.setFont(QFont("Consolas", 12, QFont.Bold))
        pos_layout.addWidget(self.pos_pulse_label)
        pos_layout.addWidget(QLabel("脉冲  ="))
        self.pos_mm_label = QLabel("--")
        self.pos_mm_label.setObjectName("StatValue")
        self.pos_mm_label.setFont(QFont("Consolas", 12, QFont.Bold))
        pos_layout.addWidget(self.pos_mm_label)
        pos_layout.addWidget(QLabel("mm"))
        pos_layout.addStretch(1)
        layout.addLayout(pos_layout)

        # ---- 状态位 / 报警 ----
        status_grid = QGridLayout()
        status_grid.setContentsMargins(0, 0, 0, 0)
        status_grid.setHorizontalSpacing(6)
        status_grid.setVerticalSpacing(6)
        self.status_indicators: dict[str, MotionStatusBadge] = {}
        for index, (key, label, severity, tooltip) in enumerate(STATUS_DEFINITIONS):
            indicator = MotionStatusBadge(label, severity, tooltip)
            self.status_indicators[key] = indicator
            status_grid.addWidget(indicator, index // 4, index % 4)
        for column in range(4):
            status_grid.setColumnStretch(column, 1)
        layout.addLayout(status_grid)
        self.alarm_label = QLabel("--")
        self.alarm_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.alarm_label)

        # ---- 运动模式 ----
        mode_layout = QHBoxLayout()
        self.mode_absolute = QRadioButton("绝对")
        self.mode_relative = QRadioButton("相对")
        self.mode_absolute.setChecked(True)
        mode_layout.addWidget(self.mode_absolute)
        mode_layout.addWidget(self.mode_relative)
        mode_layout.addStretch(1)
        layout.addLayout(mode_layout)

        # ---- 距离 + 速度 ----
        param_layout = QHBoxLayout()
        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(-99999, 99999)
        self.distance_input.setValue(10.0)
        self.distance_input.setDecimals(3)
        self.distance_input.setSuffix(" mm")
        param_layout.addWidget(QLabel("距离"))
        param_layout.addWidget(self.distance_input)
        self.speed_input = QSpinBox()
        self.speed_input.setRange(1, 6000)
        self.speed_input.setValue(600)
        self.speed_input.setSuffix(" RPM")
        param_layout.addWidget(QLabel("速度"))
        param_layout.addWidget(self.speed_input)
        param_layout.addStretch(1)
        layout.addLayout(param_layout)

        # ---- 定位运行 + 急停 ----
        btn_layout = QHBoxLayout()
        self.move_btn = QPushButton("定位运行")
        self.move_btn.setProperty("class", "primary")
        self.move_btn.setMinimumHeight(30)
        btn_layout.addWidget(self.move_btn)
        self.estop_btn = QPushButton("急停")
        self.estop_btn.setProperty("class", "danger")
        self.estop_btn.setMinimumHeight(30)
        btn_layout.addWidget(self.estop_btn)
        layout.addLayout(btn_layout)

        # ---- 点动 ----
        jog_layout = QHBoxLayout()
        self.jog_fwd_btn = QPushButton("正点动 >")
        jog_layout.addWidget(self.jog_fwd_btn)
        self.jog_rev_btn = QPushButton("< 反点动")
        jog_layout.addWidget(self.jog_rev_btn)
        self.jog_stop_btn = QPushButton("停止点动")
        jog_layout.addWidget(self.jog_stop_btn)
        layout.addLayout(jog_layout)

        # ---- 回零 / 设零 / 清报警 ----
        other_layout = QHBoxLayout()
        self.home_btn = QPushButton("回零")
        other_layout.addWidget(self.home_btn)
        self.set_zero_btn = QPushButton("设为零点")
        other_layout.addWidget(self.set_zero_btn)
        self.clear_alarm_btn = QPushButton("清除报警")
        other_layout.addWidget(self.clear_alarm_btn)
        layout.addLayout(other_layout)

    # ---- 外部更新 ----------------------------------------------------------

    def update_position(self, pulses, mm):
        self.pos_pulse_label.setText(f"{pulses:,}")
        self.pos_mm_label.setText(f"{mm:.4f}")

    def update_status(self, status):
        for key, indicator in self.status_indicators.items():
            indicator.set_active(status.get(key, False))

    def update_alarm(self, code, desc):
        if code != 0:
            self.alarm_label.setText(
                f"<span style='color:red;font-weight:bold'>报警: {desc}</span>")
        else:
            self.alarm_label.setText(
                f"<span style='color:green'>报警: 无</span>")

    def set_controls_enabled(self, enabled):
        for widget in (
            self.move_btn, self.estop_btn,
            self.jog_fwd_btn, self.jog_rev_btn, self.jog_stop_btn,
            self.home_btn, self.set_zero_btn, self.clear_alarm_btn,
        ):
            widget.setEnabled(enabled)

    def reset_display(self):
        self.pos_pulse_label.setText("--")
        self.pos_mm_label.setText("--")
        for indicator in self.status_indicators.values():
            indicator.set_active(False)
        self.alarm_label.setText("--")

    def get_mode(self):
        return MODE_ABSOLUTE if self.mode_absolute.isChecked() else MODE_RELATIVE


class MotionControlPage(QWidget):
    """位移台控制页 (motion_gui.py 的功能移植)。"""

    console_output = Signal(str, str)  # channel ("comm-send"/...), text

    def __init__(self, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._is_connected = False
        self._is_connecting = False

        self.worker = MotionWorker(self)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._on_refresh_timer)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_connection_group())
        outer.addWidget(self._build_axes_group(), 1)
        outer.addWidget(self._build_status_row())

        self._wire_worker()
        self.worker.start()
        self._set_controls_enabled(False)
        self._refresh_ports()
        self.manager.busy_changed.connect(self._on_task_busy)

    # ---- UI ----------------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        title = QLabel("位移台控制 (Modbus RTU)")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "双轴位移台串口控制: 定位/点动/回零/设零/报警清除。"
            "通讯日志输出到底部运行控制台。位移台串口与检测任务互斥; "
            "镜头控制为 USB 设备, 可同时连接。"
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("连接设置")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(QLabel("串口"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(180)
        layout.addWidget(self.port_combo)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("波特率"))
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(
            ["2400", "4800", "9600", "19200", "38400", "57600", "115200"])
        self.baudrate_combo.setCurrentText("38400")
        layout.addWidget(self.baudrate_combo)

        layout.addWidget(QLabel("X 导程"))
        self.lead_x_input = QDoubleSpinBox()
        self.lead_x_input.setRange(0.01, 100)
        self.lead_x_input.setValue(1.0)
        self.lead_x_input.setDecimals(2)
        self.lead_x_input.setSuffix(" mm/转")
        layout.addWidget(self.lead_x_input)

        layout.addWidget(QLabel("Y 导程"))
        self.lead_y_input = QDoubleSpinBox()
        self.lead_y_input.setRange(0.01, 100)
        self.lead_y_input.setValue(1.0)
        self.lead_y_input.setDecimals(2)
        self.lead_y_input.setSuffix(" mm/转")
        layout.addWidget(self.lead_y_input)

        layout.addWidget(QLabel("细分数"))
        self.ppr_input = QSpinBox()
        self.ppr_input.setRange(100, 999999)
        self.ppr_input.setValue(51200)
        self.ppr_input.setSuffix(" 脉冲/转")
        layout.addWidget(self.ppr_input)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.setProperty("class", "primary")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        layout.addWidget(self.disconnect_btn)

        self.global_estop_btn = QPushButton("全局急停")
        self.global_estop_btn.setProperty("class", "danger")
        self.global_estop_btn.setMinimumHeight(32)
        self.global_estop_btn.clicked.connect(
            self.worker.emergency_stop_all)
        layout.addWidget(self.global_estop_btn)

        layout.addStretch(1)
        return group

    def _build_axes_group(self) -> QWidget:
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.x_panel = AxisPanel("X轴", ADDR_X_AXIS)
        self.y_panel = AxisPanel("Y轴", ADDR_Y_AXIS)
        self._connect_axis_panel(self.x_panel, ADDR_X_AXIS)
        self._connect_axis_panel(self.y_panel, ADDR_Y_AXIS)
        layout.addWidget(self.x_panel, 1)
        layout.addWidget(self.y_panel, 1)
        return holder

    def _connect_axis_panel(self, panel: AxisPanel, axis: int):
        lead_input = (self.lead_x_input if axis == ADDR_X_AXIS
                      else self.lead_y_input)
        worker = self.worker
        panel.move_btn.clicked.connect(lambda: worker.move_to_position_mm(
            axis, panel.distance_input.value(),
            panel.speed_input.value(), panel.get_mode(),
            lead_input.value()))
        panel.estop_btn.clicked.connect(lambda: worker.emergency_stop(axis))
        panel.jog_fwd_btn.clicked.connect(lambda: worker.jog_forward(axis))
        panel.jog_rev_btn.clicked.connect(lambda: worker.jog_reverse(axis))
        panel.jog_stop_btn.clicked.connect(lambda: worker.jog_stop(axis))
        panel.home_btn.clicked.connect(lambda: worker.home(axis))
        panel.set_zero_btn.clicked.connect(lambda: worker.set_zero(axis))
        panel.clear_alarm_btn.clicked.connect(
            lambda: worker.clear_alarm(axis, False))

    def _build_status_row(self) -> QWidget:
        """刷新控制 + 状态行; 通讯日志输出到主窗口的运行控制台。"""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.manual_refresh_btn = QPushButton("刷新状态")
        self.manual_refresh_btn.clicked.connect(
            lambda: self.worker.refresh_all(log_alarm=True)
        )
        layout.addWidget(self.manual_refresh_btn)
        self.auto_refresh_cb = QCheckBox(f"自动刷新 ({REFRESH_INTERVAL_MS}ms)")
        self.auto_refresh_cb.toggled.connect(self._on_auto_refresh_toggled)
        layout.addWidget(self.auto_refresh_cb)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("HintLabel")
        layout.addWidget(self.status_label, 1)

        hint = QLabel("通讯日志显示在底部“运行控制台”。")
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)
        return row

    # ---- worker 信号 -------------------------------------------------------

    def _wire_worker(self):
        worker = self.worker
        worker.result_ready.connect(self._on_result)
        worker.position_updated.connect(self._on_position_updated)
        worker.status_updated.connect(self._on_status_updated)
        worker.alarm_updated.connect(self._on_alarm_updated)
        worker.connected_signal.connect(self._on_connected)
        worker.logger.log_signal.connect(self._append_log)

    def _panel_for_axis(self, axis):
        return self.x_panel if axis == ADDR_X_AXIS else self.y_panel

    def _on_result(self, message, is_error):
        if is_error:
            self.status_label.setText(f"错误: {message}")
            self._append_log("error", f"错误: {message}")
        else:
            self.status_label.setText(message)
            self._append_log("info", message)

    def _on_position_updated(self, axis, pulses, mm):
        self._panel_for_axis(axis).update_position(pulses, mm)

    def _on_status_updated(self, axis, status):
        self._panel_for_axis(axis).update_status(status)

    def _on_alarm_updated(self, axis, alarm_code, description):
        self._panel_for_axis(axis).update_alarm(alarm_code, description)

    def _on_connected(self, success, message):
        self._is_connecting = False
        self._is_connected = success
        self.connect_btn.setEnabled(not success and not self.manager.busy())
        self.disconnect_btn.setEnabled(success)
        for widget in (self.port_combo, self.baudrate_combo,
                       self.lead_x_input, self.lead_y_input, self.ppr_input):
            widget.setEnabled(not success)
        self._set_controls_enabled(success)
        self.status_label.setText(message)
        if success:
            # 状态指示依赖实时读取；连接成功后默认持续刷新。
            self.auto_refresh_cb.setChecked(True)
            self._append_log("info", message)
        else:
            tag = "info" if message == "已断开连接" else "error"
            self._append_log(tag, message)
            self.x_panel.reset_display()
            self.y_panel.reset_display()
            self.auto_refresh_cb.setChecked(False)

    # ---- 连接 -----------------------------------------------------------------

    def _refresh_ports(self):
        previous_port = self.port_combo.currentData()
        self.port_combo.clear()
        try:
            ports = sorted(
                serial.tools.list_ports.comports(), key=lambda item: item.device
            )
        except Exception as exc:
            self.status_label.setText(f"枚举串口失败: {exc}")
            return
        previous_index = -1
        usb_index = -1
        for index, port in enumerate(ports):
            self.port_combo.addItem(f"{port.device} - {port.description}",
                                    port.device)
            if port.device == previous_port:
                previous_index = index
            if usb_index < 0 and getattr(port, "vid", None) is not None:
                usb_index = index
        selected_index = previous_index if previous_index >= 0 else usb_index
        if selected_index >= 0:
            self.port_combo.setCurrentIndex(selected_index)

    def _on_connect(self):
        if self.manager.busy():
            QMessageBox.warning(
                self,
                "有任务正在运行",
                "位移台串口与检测任务互斥。\n请等待当前任务结束或先停止它, "
                "再连接位移台。",
            )
            return
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "错误", "请选择串口")
            return
        self._is_connecting = True
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        for widget in (self.port_combo, self.baudrate_combo,
                       self.lead_x_input, self.lead_y_input, self.ppr_input):
            widget.setEnabled(False)
        self.status_label.setText("正在连接…")
        self.worker.connect_device(
            port, int(self.baudrate_combo.currentText()),
            self.lead_x_input.value(), self.lead_y_input.value(),
            self.ppr_input.value(),
        )

    def _on_disconnect(self):
        self.auto_refresh_cb.setChecked(False)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText(
            "正在取消连接…" if self._is_connecting else "正在断开…"
        )
        self.worker.disconnect_device()

    def _set_controls_enabled(self, enabled):
        self.x_panel.set_controls_enabled(enabled)
        self.y_panel.set_controls_enabled(enabled)
        self.global_estop_btn.setEnabled(enabled)
        self.manual_refresh_btn.setEnabled(enabled)
        self.auto_refresh_cb.setEnabled(enabled)

    # ---- 通讯日志 -------------------------------------------------------------

    def _append_log(self, direction, message):
        """通讯日志转发到运行控制台 (comm-* 通道按 发送/接收/信息/错误 着色)。"""
        now = datetime.now()
        ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        channel = f"comm-{direction}" if direction in LOG_PREFIX else "comm-info"
        self.console_output.emit(
            channel, f"[{ts}] {LOG_PREFIX.get(direction, '--')} {message}")

    # ---- 自动刷新 -------------------------------------------------------------

    def _on_auto_refresh_toggled(self, checked):
        if checked and self._is_connected:
            self._refresh_timer.start(REFRESH_INTERVAL_MS)
        else:
            self._refresh_timer.stop()
            if not self._is_connected:
                self.auto_refresh_cb.setChecked(False)

    def _on_refresh_timer(self):
        if self._is_connected:
            self.worker.refresh_all()

    # ---- 生命周期 -------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._is_connected

    def disconnect_if_connected(self) -> None:
        """已连接则请求断开 (排队执行, 进行中的命令会先完成再断开)。"""
        if self._is_connected or self._is_connecting:
            self.auto_refresh_cb.setChecked(False)
            self.worker.disconnect_device()

    def _on_task_busy(self, busy: bool):
        """检测任务运行时禁用连接 (已连接则保持)。"""
        if busy and not self._is_connected:
            self.connect_btn.setEnabled(False)
        elif not self._is_connected and not self._is_connecting:
            self.connect_btn.setEnabled(True)

    def shutdown_worker(self):
        """主窗口关闭时调用: 断开串口并结束线程。"""
        self._refresh_timer.stop()
        self.worker.stop()
        self.worker.wait(5000)
