"""镜头控制页: 厂商程序 LensConnect_Windows_GUI_x86_2.2.0 的移植。

与 TaskPage 的“编辑配置 + 子进程运行”模式不同, 本页是交互式硬件控制:
LensWorker 线程持有 USB 连接, 页面把按钮/滑条动作转成请求, 通过信号刷新。
镜头与任务脚本共用独占硬件 —— 任务运行中禁止连接镜头 (本页检查),
连接镜头后禁止启动任务 (app.py 的 _start_task 检查)。
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GUI.core.lens_worker import MOTORS, LensWorker
from GUI.core.paths import CALIBRATION_SUMMARY
from GUI.core.runner import TaskManager

# 三个电机的界面文案: 步进减小/增大方向的语义标签 (地址小 → 大)
MOTOR_LABELS = {
    "zoom": {"title": "Zoom 变焦", "minus": "广角", "plus": "远摄"},
    "focus": {"title": "Focus 对焦", "minus": "近", "plus": "远"},
    "iris": {"title": "Iris 光圈", "minus": "开", "plus": "闭"},
}
FILTER_LABELS = ("No IRCF", "IRCF In", "滤镜 2", "滤镜 3")
PRESET_COUNT = 4
USER_AREA_LENGTH = 32
ADDR_SPIN_RANGE = (0, 65535)
STEP_SPIN_RANGE = (1, 65535)
DEFAULT_STEP_LENGTH = 100
AUTOFOCUS_ENTRY_TYPE = "pcb_autofocus_lens_position"
CALIBRATED_ZOOM_BUTTON_COUNT = 2


def _lens_address(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} 不是有效的镜头地址")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip():
        result = int(value.strip(), 0)
    else:
        raise ValueError(f"{field} 不是有效的镜头地址")
    if result < 0:
        raise ValueError(f"{field} 不能小于 0")
    return result


def load_pcb_autofocus_positions(path=CALIBRATION_SUMMARY) -> list[dict]:
    """Load at most two complete Zoom/Focus/Iris autofocus positions."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取 {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 不是有效 JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        raise ValueError(f"{path} 缺少 entries 列表")

    positions_by_zoom = {}
    for entry in document["entries"]:
        if not isinstance(entry, dict) or entry.get("type") != AUTOFOCUS_ENTRY_TYPE:
            continue
        nested = entry.get("lens_position")
        nested = nested if isinstance(nested, dict) else {}
        zoom_value = entry.get("lens_zoom")
        focus_value = entry.get("lens_focus")
        iris_value = entry.get("lens_iris")
        try:
            device = entry.get("device")
            position = {
                "zoom": _lens_address(
                    nested.get("zoom") if zoom_value is None else zoom_value,
                    "lens_zoom",
                ),
                "focus": _lens_address(
                    nested.get("focus") if focus_value is None else focus_value,
                    "lens_focus",
                ),
                "iris": _lens_address(
                    nested.get("iris") if iris_value is None else iris_value,
                    "lens_iris",
                ),
                "device": (
                    None if device is None else _lens_address(device, "device")
                ),
            }
        except (TypeError, ValueError):
            continue
        positions_by_zoom[position["zoom"]] = position
    return sorted(positions_by_zoom.values(), key=lambda item: item["zoom"])[
        :CALIBRATED_ZOOM_BUTTON_COUNT
    ]


def format_bits(value: int) -> str:
    """16 位状态/能力值按 4 位分组显示, 如 0000 0000 0000 1111。"""
    text = f"{value & 0xFFFF:016b}"
    return " ".join(text[i:i + 4] for i in range(0, 16, 4))


class MotorCard(QGroupBox):
    """单个电机 (zoom/focus/iris) 的手动控制组, 对应厂商 GUI 的 Address 区。"""

    move_requested = Signal(str, int)   # motor, 绝对地址
    step_requested = Signal(str, int)   # motor, 地址增量
    init_requested = Signal(str)
    calibrated_position_requested = Signal(int)

    def __init__(self, motor: str, parent=None):
        super().__init__(MOTOR_LABELS[motor]["title"], parent)
        self.motor = motor
        self._range = ADDR_SPIN_RANGE
        self._has_position = False
        self._connected = False

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(*ADDR_SPIN_RANGE)
        self.slider.setToolTip("拖动松开后移动到对应地址")
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.current_label = QLabel("----")
        self.current_label.setObjectName("StatValue")
        layout.addWidget(self.slider, 0, 0, 1, 2)
        layout.addWidget(self.current_label, 0, 2)

        self.min_label = QLabel(str(ADDR_SPIN_RANGE[0]))
        self.min_label.setObjectName("HintLabel")
        self.max_label = QLabel(str(ADDR_SPIN_RANGE[1]))
        self.max_label.setObjectName("HintLabel")
        self.max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.min_label, 1, 0, 1, 2)
        layout.addWidget(self.max_label, 1, 2)

        layout.addWidget(QLabel("步进"), 2, 0)
        step_bar = QHBoxLayout()
        step_bar.setSpacing(6)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(*STEP_SPIN_RANGE)
        self.step_spin.setValue(DEFAULT_STEP_LENGTH)
        self.step_spin.setToolTip("步进地址增量，可直接输入 1–65535")
        self.minus_btn = QPushButton(MOTOR_LABELS[motor]["minus"])
        self.minus_btn.setToolTip("地址减小方向")
        self.plus_btn = QPushButton(MOTOR_LABELS[motor]["plus"])
        self.plus_btn.setToolTip("地址增大方向")
        self.minus_btn.clicked.connect(
            lambda: self.step_requested.emit(self.motor, -self.step_spin.value())
        )
        self.plus_btn.clicked.connect(
            lambda: self.step_requested.emit(self.motor, self.step_spin.value())
        )
        step_bar.addWidget(self.step_spin)
        step_bar.addWidget(self.minus_btn)
        step_bar.addWidget(self.plus_btn)
        layout.addLayout(step_bar, 2, 1, 1, 2)

        layout.addWidget(QLabel("转到"), 3, 0)
        goto_bar = QHBoxLayout()
        goto_bar.setSpacing(6)
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(*ADDR_SPIN_RANGE)
        self.goto_btn = QPushButton("Go to")
        self.goto_btn.setProperty("class", "primary")
        self.goto_btn.clicked.connect(
            lambda: self.move_requested.emit(self.motor, self.goto_spin.value())
        )
        goto_bar.addWidget(self.goto_spin)
        goto_bar.addWidget(self.goto_btn)
        goto_bar.addStretch(1)
        layout.addLayout(goto_bar, 3, 1, 1, 2)

        next_row = 4
        self.calibrated_zoom_buttons: list[QPushButton] = []
        self._calibrated_positions: list[dict] = []
        self._calibrated_positions_enabled = True
        if motor == "zoom":
            calibrated_bar = QHBoxLayout()
            calibrated_bar.setSpacing(6)
            calibrated_bar.addWidget(QLabel("PCB 对焦位置"))
            for index in range(CALIBRATED_ZOOM_BUTTON_COUNT):
                button = QPushButton("未配置")
                button.setEnabled(False)
                button.clicked.connect(
                    lambda _=False, value=index:
                    self.calibrated_position_requested.emit(value)
                )
                self.calibrated_zoom_buttons.append(button)
                calibrated_bar.addWidget(button, 1)
            layout.addLayout(calibrated_bar, next_row, 0, 1, 3)
            next_row += 1

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.init_button = QPushButton("初始化")
        self.init_button.clicked.connect(
            lambda: self.init_requested.emit(self.motor)
        )
        self.state_label = QLabel("未连接")
        self.state_label.setObjectName("HintLabel")
        self.feedback_label = QLabel("")
        self.feedback_label.setObjectName("HintLabel")
        bottom.addWidget(self.init_button)
        bottom.addWidget(self.state_label)
        bottom.addWidget(self.feedback_label, 1)
        layout.addLayout(bottom, next_row, 0, 1, 3)

        self.set_enabled_connected(False)

    # ---- 状态刷新 ----------------------------------------------------------

    def set_enabled_connected(self, connected: bool) -> None:
        """连接状态只控制硬件操作，步进长度始终允许编辑。"""
        self._connected = connected
        self.setEnabled(True)
        self.setTitle(MOTOR_LABELS[self.motor]["title"])
        self.step_spin.setEnabled(True)
        self.init_button.setEnabled(connected)
        self._has_position = False
        if not connected:
            self.current_label.setText("----")
            self.slider.blockSignals(True)
            self.slider.setRange(*ADDR_SPIN_RANGE)
            self.slider.blockSignals(False)
            for editor in (
                self.goto_spin,
                self.slider,
                self.minus_btn,
                self.plus_btn,
                self.goto_btn,
            ):
                editor.setEnabled(False)
        self._update_calibrated_zoom_buttons()

    def set_calibrated_positions(self, positions: list[dict]) -> None:
        self._calibrated_positions = [dict(item) for item in positions]
        for index, button in enumerate(self.calibrated_zoom_buttons):
            if index >= len(self._calibrated_positions):
                button.setText("未配置")
                button.setToolTip("calibrate.json 中没有完整的自动对焦位置")
                continue
            position = self._calibrated_positions[index]
            button.setText(str(position["zoom"]))
            button.setToolTip(
                "Zoom {} / Focus {} / Iris {}".format(
                    position["zoom"], position["focus"], position["iris"]
                )
            )
        self._update_calibrated_zoom_buttons()

    def set_calibrated_positions_enabled(self, enabled: bool) -> None:
        self._calibrated_positions_enabled = enabled
        self._update_calibrated_zoom_buttons()

    def _update_calibrated_zoom_buttons(self) -> None:
        for index, button in enumerate(self.calibrated_zoom_buttons):
            button.setEnabled(
                self._connected
                and self._calibrated_positions_enabled
                and index < len(self._calibrated_positions)
            )

    def apply_block(self, block: dict) -> None:
        """连接后按电机参数块设置范围与初始值。"""
        if not block.get("supported"):
            self._connected = True
            self.setEnabled(False)
            self.setTitle(MOTOR_LABELS[self.motor]["title"] + " (不支持)")
            return
        self._connected = True
        low, high = int(block["min"]), int(block["max"])
        self._range = (low, high)
        for spin in (self.goto_spin, self.slider):
            spin.setRange(low, high)
        self.goto_spin.setValue(
            block["current"] if block["current"] is not None else low
        )
        self.min_label.setText(str(low))
        self.max_label.setText(str(high))
        self.setEnabled(True)
        self.setTitle(MOTOR_LABELS[self.motor]["title"])
        self.step_spin.setEnabled(True)
        self.init_button.setEnabled(True)
        self._apply_position(block["current"])
        self._apply_bits(operating=False, initialized=bool(block["initialized"]))

    def apply_position(self, position) -> None:
        self._apply_position(position)

    def apply_bits(self, operating: bool, initialized: bool) -> None:
        self._apply_bits(operating, initialized)

    def set_feedback(self, text: str) -> None:
        self.feedback_label.setText(text)

    # ---- 内部 --------------------------------------------------------------

    def _apply_position(self, position) -> None:
        if position is None:
            self._has_position = False
            self.current_label.setText("----")
            return
        self._has_position = True
        self.current_label.setText(str(position))
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(int(position))
            self.slider.blockSignals(False)

    def _apply_bits(self, operating: bool, initialized: bool) -> None:
        if not initialized:
            self.state_label.setText("状态: 未初始化, 请先初始化")
        elif operating:
            self.state_label.setText("状态: 运动中")
        else:
            self.state_label.setText("状态: 待机")
        # 未初始化时禁止地址操作 (初始化按钮始终可用)
        ready = initialized and self._connected and self.isEnabled()
        self.step_spin.setEnabled(self.isEnabled())
        for editor in (self.goto_spin, self.slider,
                       self.minus_btn, self.plus_btn, self.goto_btn):
            editor.setEnabled(ready)
        self._update_calibrated_zoom_buttons()

    def _on_slider_moved(self, value: int) -> None:
        self.goto_spin.blockSignals(True)
        self.goto_spin.setValue(value)
        self.goto_spin.blockSignals(False)

    def _on_slider_released(self) -> None:
        if self._has_position:
            self.move_requested.emit(self.motor, self.slider.value())


class FilterCard(QGroupBox):
    """光学滤镜 (IRCF) 控制组。"""

    move_requested = Signal(int)
    init_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("光学滤镜", parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.count_label = QLabel("--")
        self.current_label = QLabel("--")
        self.current_label.setObjectName("StatValue")
        layout.addWidget(QLabel("滤镜数量"), 0, 0)
        layout.addWidget(self.count_label, 0, 1)
        layout.addWidget(QLabel("当前滤镜"), 0, 2)
        layout.addWidget(self.current_label, 0, 3)

        self.filter_buttons = []
        button_bar = QHBoxLayout()
        button_bar.setSpacing(6)
        for index, label in enumerate(FILTER_LABELS):
            button = QPushButton(f"{index} ({label})")
            button.setToolTip(f"移动到滤镜 {index}")
            button.clicked.connect(
                lambda _=False, value=index: self.move_requested.emit(value)
            )
            self.filter_buttons.append(button)
            button_bar.addWidget(button)
        button_bar.addStretch(1)
        layout.addLayout(button_bar, 1, 0, 1, 4)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.init_button = QPushButton("初始化")
        self.init_button.clicked.connect(self.init_requested.emit)
        self.state_label = QLabel("未连接")
        self.state_label.setObjectName("HintLabel")
        bottom.addWidget(self.init_button)
        bottom.addWidget(self.state_label, 1)
        layout.addLayout(bottom, 2, 0, 1, 4)

        self.setEnabled(False)
        self._count = 0

    def set_enabled_connected(self, connected: bool) -> None:
        self.setEnabled(connected)
        self.state_label.setText("未连接" if not connected else "状态: 待机")

    def apply_block(self, block: dict) -> None:
        if not block.get("supported"):
            self.setEnabled(False)
            self.setTitle("光学滤镜 (不支持)")
            return
        self.setEnabled(True)
        self.setTitle("光学滤镜")
        self._count = block.get("count_max", block.get("max", 0))
        self.count_label.setText(str(self._count))
        self._apply_position(block.get("current"))
        self._apply_bits(False, bool(block.get("initialized")))

    def apply_position(self, position) -> None:
        self._apply_position(position)

    def apply_bits(self, operating: bool, initialized: bool) -> None:
        self._apply_bits(operating, initialized)

    def _apply_position(self, position) -> None:
        self.current_label.setText("--" if position is None else str(position))

    def _apply_bits(self, operating: bool, initialized: bool) -> None:
        if not initialized:
            self.state_label.setText("状态: 未初始化, 请先初始化")
        elif operating:
            self.state_label.setText("状态: 运动中")
        else:
            self.state_label.setText("状态: 待机")
        for index, button in enumerate(self.filter_buttons):
            button.setEnabled(initialized and self.isEnabled()
                              and index <= self._count)


class SetupCard(QGroupBox):
    """一个电机的速度/间隙补偿设置, 对应厂商 GUI 的 Parameter setup 页。"""

    apply_requested = Signal(str, int, bool)  # motor, speed pps, backlash

    def __init__(self, motor: str, parent=None):
        super().__init__(MOTOR_LABELS[motor]["title"], parent)
        self.motor = motor

        layout = QFormLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)

        range_bar = QHBoxLayout()
        range_bar.setSpacing(6)
        self.range_label = QLabel("-- .. --")
        self.range_label.setObjectName("HintLabel")
        range_bar.addWidget(self.range_label)
        range_bar.addStretch(1)
        layout.addRow("速度范围 (PPS)", range_bar)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 65535)
        layout.addRow("速度 (PPS)", self.speed_spin)

        self.backlash_combo = QComboBox()
        self.backlash_combo.addItem("关闭", False)
        self.backlash_combo.addItem("开启", True)
        layout.addRow("间隙补偿", self.backlash_combo)

        button_bar = QHBoxLayout()
        apply_button = QPushButton("写入")
        apply_button.setProperty("class", "primary")
        apply_button.clicked.connect(self._emit_apply)
        self.feedback_label = QLabel("")
        self.feedback_label.setObjectName("HintLabel")
        button_bar.addWidget(apply_button)
        button_bar.addWidget(self.feedback_label, 1)
        layout.addRow(button_bar)

        self.setEnabled(False)

    def _emit_apply(self):
        self.apply_requested.emit(
            self.motor, self.speed_spin.value(), self.backlash_combo.currentData()
        )

    def apply_block(self, block: dict) -> None:
        if not block.get("supported"):
            self.setEnabled(False)
            return
        self.setEnabled(True)
        low = int(block.get("speed_min", 1))
        high = int(block.get("speed_max", 65535))
        self.speed_spin.setRange(low, high)
        self.speed_spin.setValue(int(block.get("speed", low)))
        self.range_label.setText(f"{low} .. {high}")
        self.backlash_combo.setCurrentIndex(
            1 if int(block.get("backlash", 0)) else 0
        )

    def set_feedback(self, text: str) -> None:
        self.feedback_label.setText(text)


class PresetCard(QGroupBox):
    """一组预设位, 对应厂商 GUI 的 Position1-4。"""

    capture_requested = Signal(int)
    changed = Signal()

    def __init__(self, index: int, parent=None):
        super().__init__(f"位置 {index + 1}", parent)
        self.index = index

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.enable_check = QCheckBox("启用")
        self.enable_check.setChecked(index == 0)
        layout.addWidget(self.enable_check, 0, 3)

        self.spins: dict[str, QSpinBox] = {}
        for column, motor in enumerate(MOTORS):
            spin = QSpinBox()
            spin.setRange(*ADDR_SPIN_RANGE)
            spin.setToolTip(f"{MOTOR_LABELS[motor]['title']} 目标地址")
            self.spins[motor] = spin
            layout.addWidget(QLabel(motor.capitalize()), 1, column * 2)
            layout.addWidget(spin, 1, column * 2 + 1)

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("不动作", None)
        for value, label in enumerate(FILTER_LABELS):
            self.filter_combo.addItem(f"{value} ({label})", value)
        layout.addWidget(QLabel("滤镜"), 2, 0)
        layout.addWidget(self.filter_combo, 2, 1)

        self.wait_spin = QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 300.0)
        self.wait_spin.setSingleStep(0.5)
        self.wait_spin.setDecimals(1)
        self.wait_spin.setSuffix(" s")
        layout.addWidget(QLabel("等待"), 2, 2)
        layout.addWidget(self.wait_spin, 2, 3)

        button_bar = QHBoxLayout()
        current_button = QPushButton("读取当前位")
        current_button.clicked.connect(
            lambda: self.capture_requested.emit(self.index)
        )
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("HintLabel")
        button_bar.addWidget(current_button)
        button_bar.addWidget(self.progress_label, 1)
        layout.addLayout(button_bar, 3, 0, 1, 4)

        for spin in self.spins.values():
            spin.valueChanged.connect(self.changed)
        self.wait_spin.valueChanged.connect(self.changed)
        self.filter_combo.currentIndexChanged.connect(self.changed)
        self.enable_check.toggled.connect(self.changed)

    def values(self) -> dict:
        return {
            "enabled": self.enable_check.isChecked(),
            **{motor: spin.value() for motor, spin in self.spins.items()},
            "filter": self.filter_combo.currentData(),
            "wait_seconds": self.wait_spin.value(),
        }

    def set_values(self, values: dict) -> None:
        self.enable_check.setChecked(bool(values.get("enabled", False)))
        for motor, spin in self.spins.items():
            if values.get(motor) is not None:
                spin.setValue(int(values[motor]))
        index = self.filter_combo.findData(values.get("filter", None))
        self.filter_combo.setCurrentIndex(max(0, index))
        self.wait_spin.setValue(float(values.get("wait_seconds", 0.0)))

    def apply_ranges(self, blocks: dict) -> None:
        for motor, spin in self.spins.items():
            block = blocks.get(motor) or {}
            if block.get("supported"):
                spin.setRange(int(block["min"]), int(block["max"]))

    def capture(self, positions: dict) -> None:
        for motor, spin in self.spins.items():
            position = positions.get(motor)
            if position is not None:
                spin.setValue(int(position))

    def set_progress(self, text: str) -> None:
        self.progress_label.setText(text)


class LensDetailsDialog(QDialog):
    """镜头详细信息对话框, 对应厂商 GUI 的 "More" 信息页。"""

    def __init__(self, snapshot: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("镜头详细信息")
        self.resize(760, 560)

        layout = QVBoxLayout(self)
        general = QGroupBox("常规")
        general_form = QFormLayout(general)
        general_form.setContentsMargins(8, 4, 8, 8)
        for label, key in (
            ("镜头型号", "model"),
            ("固件版本", "fw_version"),
            ("协议版本", "protocol_version"),
            ("镜头修订号", "revision"),
            ("镜头地址", "lens_address"),
            ("能力位", "capabilities"),
            ("状态1", "status1"),
            ("状态2", "status2"),
            ("用户标识", "user_area"),
            ("温度", "temperature_c"),
        ):
            value = snapshot.get(key)
            if isinstance(value, int) and key in ("capabilities", "status1", "status2"):
                value = format_bits(value)
            if key == "temperature_c":
                value = (f"{snapshot.get('temperature_c')} °C / "
                         f"{snapshot.get('temperature_f')} °F")
            general_form.addRow(label, QLabel(str(value)))

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(("参数", "Zoom", "Focus", "Iris"))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        def row(label: str, getter):
            cells = [QTableWidgetItem(label)]
            for motor in MOTORS:
                block = snapshot["motors"].get(motor) or {}
                cells.append(QTableWidgetItem(str(getter(block))))
            table_row = table.rowCount()
            for column, cell in enumerate(cells):
                table.setItem(table_row, column, cell)

        row("支持", lambda b: "是" if b.get("supported") else "否")
        row("已初始化", lambda b: "是" if b.get("initialized") else "否")
        row("当前位置", lambda b: b.get("current", "--"))
        row("位置范围", lambda b: f"{b.get('min')} .. {b.get('max')}")
        row("光学位置范围", lambda b: f"{b.get('pos_min')} .. {b.get('pos_max')}")
        row("机械范围", lambda b: f"{b.get('mech_min')} .. {b.get('mech_max')}")
        row("初始化位置", lambda b: b.get("init_pos"))
        row("速度 (PPS)", lambda b: b.get("speed"))
        row("速度范围", lambda b: f"{b.get('speed_min')} .. {b.get('speed_max')}")
        row("间隙补偿", lambda b: "开启" if b.get("backlash") else "关闭")
        row("计数", lambda b: f"{b.get('count')} / {b.get('count_max')}")

        opt = snapshot.get("opt") or {}
        general_form.addRow(
            "光学滤镜",
            QLabel("不支持" if not opt.get("supported") else
                   f"数量 {opt.get('count_max', opt.get('max'))}, "
                   f"当前 {opt.get('current', '--')}"),
        )

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)

        layout.addWidget(general)
        layout.addWidget(table, 1)
        layout.addWidget(close_button, 0, Qt.AlignRight)


class LensControlPage(QWidget):
    """交互式镜头控制页 (厂商 GUI 的功能移植)。"""

    def __init__(self, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.connected = False
        self.snapshot: dict | None = None
        self.positions: dict = {}
        self.preset_running = False
        self._running_cards: list[PresetCard] = []
        self.calibrated_positions: list[dict] = []
        self._calibrated_positions_error = ""
        self._reload_calibrated_positions()

        self.worker = LensWorker(self)
        self._wire_worker()
        self.worker.start()

        self.settings = QSettings("LensDetect", "GUI")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_connection_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_manual_tab(), "手动控制")
        self.tabs.addTab(self._build_setup_tab(), "参数设置")
        self.tabs.addTab(self._build_preset_tab(), "预设位")
        outer.addWidget(self.tabs, 1)

        initial_message = "未连接镜头。先扫描设备, 再连接。"
        if self._calibrated_positions_error:
            initial_message += " " + self._calibrated_positions_error
        self.feedback_label = QLabel(initial_message)
        self.feedback_label.setObjectName("HintLabel")
        outer.addWidget(self.feedback_label)

        self.manager.busy_changed.connect(self._on_task_busy)
        self._load_presets()

    # ---- UI ----------------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        title = QLabel("镜头控制 (LensConnect)")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "厂商 LensConnect Controller 程序的功能移植: 手动控制 zoom / focus / "
            "iris / 滤镜, 读写速度与间隙补偿, 查看镜头信息, 顺序执行预设位。"
            "镜头为独占硬件, 连接期间不能启动检测任务。"
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_connection_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Card")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        scan_button = QPushButton("扫描设备")
        scan_button.clicked.connect(lambda: self.worker.scan())
        self.sn_combo = QComboBox()
        self.sn_combo.setMinimumWidth(220)
        self.sn_combo.setToolTip("扫描后选择要连接的设备 (序号: 序列号)")
        self.connect_button = QPushButton("连接")
        self.connect_button.setProperty("class", "primary")
        self.connect_button.clicked.connect(self._on_connect_clicked)
        self.conn_label = QLabel("未连接")
        self.conn_label.setObjectName("HintLabel")

        layout.addWidget(QLabel("设备"))
        layout.addWidget(self.sn_combo, 1)
        layout.addWidget(scan_button)
        layout.addWidget(self.connect_button)
        layout.addWidget(self.conn_label)
        return bar

    def _build_manual_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)

        # 第一行: zoom / focus / iris 三个电机控制卡
        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.motor_cards: dict[str, MotorCard] = {}
        for motor in MOTORS:
            card = MotorCard(motor)
            card.move_requested.connect(self._request_move)
            card.step_requested.connect(self._request_step)
            card.init_requested.connect(self._request_init)
            self.motor_cards[motor] = card
            cards.addWidget(card, 1)
        zoom_card = self.motor_cards["zoom"]
        zoom_card.set_calibrated_positions(self.calibrated_positions)
        zoom_card.calibrated_position_requested.connect(
            self._request_calibrated_position
        )
        layout.addLayout(cards)

        # 第二行: 镜头信息 / 温度与用户标识 / 光学滤镜
        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        bottom.addWidget(self._build_info_group(), 1)
        bottom.addWidget(self._build_temp_user_group(), 1)
        self.filter_card = FilterCard()
        self.filter_card.move_requested.connect(
            lambda value: self.worker.move("opt", value)
        )
        self.filter_card.init_requested.connect(lambda: self._request_init("opt"))
        bottom.addWidget(self.filter_card, 1)
        layout.addLayout(bottom)
        layout.addStretch(1)

        scroll.setWidget(page)
        return scroll

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("镜头信息")
        form = QFormLayout(group)
        form.setContentsMargins(8, 4, 8, 8)
        self.info_labels: dict[str, QLabel] = {}
        for key, label in (
            ("model", "镜头型号"),
            ("fw_version", "固件版本"),
            ("protocol_version", "协议版本"),
            ("lens_address", "镜头地址"),
            ("capabilities", "能力位"),
            ("status1", "状态1"),
            ("status2", "状态2"),
        ):
            value_label = QLabel("--")
            value_label.setObjectName("PathLabel")
            self.info_labels[key] = value_label
            form.addRow(label, value_label)
        more_button = QPushButton("详细信息")
        more_button.clicked.connect(lambda: self.worker.refresh_details())
        form.addRow(more_button)
        return group

    def _build_temp_user_group(self) -> QGroupBox:
        group = QGroupBox("温度 / 用户标识")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        self.temp_label = QLabel("--")
        self.temp_label.setObjectName("StatValue")
        temp_button = QPushButton("更新温度")
        temp_button.clicked.connect(lambda: self.worker.refresh_temperature())
        layout.addWidget(QLabel("温度"), 0, 0)
        layout.addWidget(self.temp_label, 0, 1)
        layout.addWidget(temp_button, 0, 2)

        self.user_edit = QLineEdit()
        self.user_edit.setMaxLength(USER_AREA_LENGTH)
        self.user_edit.setToolTip(f"最多 {USER_AREA_LENGTH} 个字符")
        user_button = QPushButton("写入")
        user_button.clicked.connect(
            lambda: self.worker.write_user_area(self.user_edit.text())
        )
        layout.addWidget(QLabel("用户标识"), 1, 0)
        layout.addWidget(self.user_edit, 1, 1)
        layout.addWidget(user_button, 1, 2)
        return group

    def _build_setup_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)
        self.setup_cards: dict[str, SetupCard] = {}
        for motor in MOTORS:
            card = SetupCard(motor)
            card.apply_requested.connect(self._on_setup_apply)
            self.setup_cards[motor] = card
            layout.addWidget(card, 1)
        scroll.setWidget(page)
        return scroll

    def _build_preset_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)

        # 预设卡按两行排布, 非全屏窗口下也能完整显示
        cards = QGridLayout()
        cards.setSpacing(10)
        self.preset_cards: list[PresetCard] = []
        for index in range(PRESET_COUNT):
            card = PresetCard(index)
            card.capture_requested.connect(self._capture_preset)
            card.changed.connect(self._save_presets)
            self.preset_cards.append(card)
            cards.addWidget(card, index // 2, index % 2)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        layout.addLayout(cards)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.preset_start_button = QPushButton("开始")
        self.preset_start_button.setProperty("class", "primary")
        self.preset_start_button.clicked.connect(self._on_preset_start)
        self.preset_stop_button = QPushButton("停止")
        self.preset_stop_button.setProperty("class", "danger")
        self.preset_stop_button.setEnabled(False)
        self.preset_stop_button.clicked.connect(self.worker.stop_preset)
        export_button = QPushButton("导出")
        export_button.clicked.connect(self._export_presets)
        import_button = QPushButton("导入")
        import_button.clicked.connect(self._import_presets)
        self.preset_status = QLabel("勾选要执行的位置, 点击开始按 1→4 顺序运行。")
        self.preset_status.setObjectName("HintLabel")
        for widget in (self.preset_start_button, self.preset_stop_button,
                       export_button, import_button):
            bar.addWidget(widget)
        bar.addWidget(self.preset_status, 1)
        layout.addLayout(bar)
        layout.addStretch(1)
        return page

    # ---- worker 信号 -------------------------------------------------------

    def _wire_worker(self):
        self.worker.scan_done.connect(self._on_scan_done)
        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.failed.connect(self._on_failed)
        self.worker.polled.connect(self._on_polled)
        self.worker.feedback.connect(self._on_feedback)
        self.worker.details_ready.connect(self._on_details)
        self.worker.preset_progress.connect(self._on_preset_progress)
        self.worker.preset_finished.connect(self._on_preset_finished)
        self.worker.calibrated_position_finished.connect(
            self._on_calibrated_position_finished
        )

    def _on_scan_done(self, ok: bool, devices: list, message: str):
        self.sn_combo.clear()
        if ok:
            self.sn_combo.addItems(devices)
        self._set_message(message)

    def _on_connected(self, snapshot: dict):
        self._reload_calibrated_positions()
        self.connected = True
        self.snapshot = snapshot
        self.positions = {
            name: (snapshot["motors"].get(name) or {}).get("current")
            for name in MOTORS
        }
        self.connect_button.setText("断开")
        self.conn_label.setText(
            f"已连接 设备 {snapshot.get('device')} ({snapshot.get('model')})"
        )

        for key, label in self.info_labels.items():
            value = snapshot.get(key)
            if isinstance(value, int) and key in ("capabilities", "status1",
                                                  "status2"):
                value = format_bits(value)
            label.setText("--" if value is None else str(value))
        self.temp_label.setText(
            f"{snapshot.get('temperature_c')} °C / "
            f"{snapshot.get('temperature_f')} °F"
        )
        self.user_edit.setText(snapshot.get("user_area", ""))

        for motor, card in self.motor_cards.items():
            card.apply_block(snapshot["motors"].get(motor) or {})
        self.motor_cards["zoom"].set_calibrated_positions(
            self.calibrated_positions
        )
        self.motor_cards["zoom"].set_calibrated_positions_enabled(True)
        for motor, card in self.setup_cards.items():
            card.apply_block(snapshot["motors"].get(motor) or {})
        self.filter_card.apply_block(snapshot.get("opt") or {})

        ranges = {motor: snapshot["motors"].get(motor) for motor in MOTORS}
        for card in self.preset_cards:
            card.apply_ranges(ranges)
        message = "已连接，Zoom / Focus / Iris 已自动初始化。"
        if self._calibrated_positions_error:
            message += " " + self._calibrated_positions_error
        self._set_message(message)

    def _on_disconnected(self):
        self.connected = False
        self.snapshot = None
        self.positions = {}
        self.connect_button.setText("连接")
        self.conn_label.setText("未连接")
        for card in self.motor_cards.values():
            card.set_enabled_connected(False)
            card.set_feedback("")
        for card in self.setup_cards.values():
            card.setEnabled(False)
            card.set_feedback("")
        self.filter_card.set_enabled_connected(False)
        for key in self.info_labels:
            self.info_labels[key].setText("--")
        self.temp_label.setText("--")
        self._set_message("已断开。")

    def _on_failed(self, message: str):
        self._set_message(message)

    def _on_polled(self, status: dict):
        if not self.connected:
            return
        status1, status2 = status["status1"], status["status2"]
        masks = {"zoom": 0x0002, "focus": 0x0004, "iris": 0x0008, "opt": 0x0010}
        self.positions = status["positions"]
        for motor, card in self.motor_cards.items():
            card.apply_position(status["positions"].get(motor))
            card.apply_bits(
                operating=bool(status1 & masks[motor]),
                initialized=not (status2 & masks[motor]),
            )
        self.filter_card.apply_position(status.get("opt_current"))
        self.filter_card.apply_bits(
            operating=bool(status1 & masks["opt"]),
            initialized=not (status2 & masks["opt"]),
        )
        self.info_labels["status1"].setText(format_bits(status1))
        self.info_labels["status2"].setText(format_bits(status2))
        self.temp_label.setText(
            f"{status.get('temperature_c')} °C / "
            f"{status.get('temperature_f')} °F"
        )

    def _on_feedback(self, motor: str, text: str):
        if motor in self.motor_cards:
            self.motor_cards[motor].set_feedback(text)
        elif motor in self.setup_cards:
            self.setup_cards[motor].set_feedback(text)
        self._set_message(f"{motor}: {text}" if motor != "user" else text)

    def _on_details(self, snapshot: dict):
        if not self.connected:
            return
        LensDetailsDialog(snapshot, self).exec()

    def _on_preset_progress(self, index: int, text: str):
        for card in self.preset_cards:
            card.set_progress("")
        if index < len(self._running_cards):
            self._running_cards[index].set_progress(text)
        self.preset_status.setText(f"位置 {index + 1}: {text}")

    def _on_preset_finished(self, ok: bool, message: str):
        self.preset_running = False
        self._running_cards = []
        self.preset_stop_button.setEnabled(False)
        self.preset_start_button.setEnabled(True)
        self._set_preset_enabled_cards(True)
        self.preset_status.setText(("完成: " if ok else "停止/失败: ") + message)
        for card in self.preset_cards:
            card.set_progress("")

    def _on_calibrated_position_finished(self, ok: bool, message: str):
        self.motor_cards["zoom"].set_calibrated_positions_enabled(
            self.connected
        )
        self._set_message(("完成: " if ok else "失败: ") + message)

    # ---- 操作 ---------------------------------------------------------------

    def is_connected(self) -> bool:
        return self.connected

    def disconnect_if_connected(self) -> None:
        """已连接则请求断开 (排队执行, 进行中的移动会先完成再断开)。"""
        if self.connected:
            self.worker.disconnect_device()

    def shutdown_worker(self):
        """主窗口关闭时调用: 断开连接并结束线程。"""
        if self.connected:
            self.worker.disconnect_device()
        self.worker.shutdown()
        self.worker.wait(3000)

    def _set_message(self, text: str):
        self.feedback_label.setText(text)

    def _reload_calibrated_positions(self) -> None:
        try:
            self.calibrated_positions = load_pcb_autofocus_positions()
            self._calibrated_positions_error = ""
            if len(self.calibrated_positions) < CALIBRATED_ZOOM_BUTTON_COUNT:
                self._calibrated_positions_error = (
                    "calibrate.json 中完整的 PCB 自动对焦位置不足 2 个。"
                )
        except ValueError as exc:
            self.calibrated_positions = []
            self._calibrated_positions_error = f"自动对焦位置读取失败: {exc}"
        cards = getattr(self, "motor_cards", None)
        if cards and "zoom" in cards:
            cards["zoom"].set_calibrated_positions(self.calibrated_positions)

    def _request_move(self, motor: str, target: int):
        self.worker.move(motor, target)

    def _request_step(self, motor: str, delta: int):
        self.worker.step(motor, delta)

    def _request_init(self, motor: str):
        if motor == "opt":
            self.worker.initialize("opt")
        else:
            self.worker.initialize(motor)

    def _request_calibrated_position(self, index: int):
        if not self.connected:
            self._set_message("请先连接镜头。")
            return
        if index < 0 or index >= len(self.calibrated_positions):
            self._set_message("对应的 PCB 自动对焦位置不存在或参数不完整。")
            return
        position = self.calibrated_positions[index]
        calibrated_device = position.get("device")
        connected_device = (self.snapshot or {}).get("device")
        if (
            calibrated_device is not None
            and connected_device is not None
            and int(calibrated_device) != int(connected_device)
        ):
            self._set_message(
                f"该位置属于设备 {calibrated_device}，当前连接设备为 "
                f"{connected_device}。"
            )
            return
        self.motor_cards["zoom"].set_calibrated_positions_enabled(False)
        self._set_message(
            "正在转到 Zoom {} / Focus {} / Iris {}…".format(
                position["zoom"], position["focus"], position["iris"]
            )
        )
        self.worker.move_calibrated_position(position)

    def _on_setup_apply(self, motor: str, pps: int, backlash: bool):
        self.worker.set_speed(motor, pps)
        self.worker.set_backlash(motor, backlash)

    def _on_connect_clicked(self):
        if self.connected:
            self.worker.disconnect_device()
            return
        if self.manager.busy():
            QMessageBox.warning(
                self,
                "有任务正在运行",
                "相机 / 位移台 / 电动镜头为独占硬件。\n"
                "请等待当前任务结束或先停止它, 再连接镜头。",
            )
            return
        if self.sn_combo.count() == 0:
            self._set_message("请先扫描设备。")
            return
        self._reload_calibrated_positions()
        device_index = int(self.sn_combo.currentText().split(":", 1)[0])
        self._set_message(
            "连接中… (将自动初始化 Zoom / Focus / Iris，可能需要数秒)"
        )
        self.worker.connect_device(device_index)

    def _on_task_busy(self, busy: bool):
        """检测任务运行时禁用连接 (已连接则保持, 断开后才可重连)。"""
        if busy and not self.connected:
            self.connect_button.setEnabled(False)
        else:
            self.connect_button.setEnabled(True)

    # ---- 预设位 -------------------------------------------------------------

    def _capture_preset(self, index: int):
        if not self.connected:
            self._set_message("未连接镜头, 无法读取当前位。")
            return
        if any(self.positions.get(motor) is None for motor in MOTORS):
            self._set_message("有电机未初始化, 无法读取完整当前位。")
            return
        self.preset_cards[index].capture(self.positions)

    def _on_preset_start(self):
        if not self.connected:
            self._set_message("请先连接镜头。")
            return
        enabled_cards = [card for card in self.preset_cards
                         if card.values()["enabled"]]
        positions = [card.values() for card in enabled_cards]
        if not positions:
            self._set_message("请至少勾选一个预设位。")
            return
        self._running_cards = enabled_cards
        self.preset_running = True
        self.preset_start_button.setEnabled(False)
        self.preset_stop_button.setEnabled(True)
        self._set_preset_enabled_cards(False)
        self.preset_status.setText("运行中…")
        self.worker.run_preset(positions)

    def _set_preset_enabled_cards(self, enabled: bool):
        for card in self.preset_cards:
            card.enable_check.setEnabled(enabled)
            for spin in card.spins.values():
                spin.setEnabled(enabled)
            card.wait_spin.setEnabled(enabled)
            card.filter_combo.setEnabled(enabled)

    def _presets_payload(self) -> list[dict]:
        return [card.values() for card in self.preset_cards]

    def _save_presets(self):
        self.settings.setValue("lens_presets",
                               json.dumps(self._presets_payload()))

    def _load_presets(self):
        raw = self.settings.value("lens_presets", "")
        if not raw:
            return
        try:
            positions = json.loads(raw)
            for card, values in zip(self.preset_cards, positions):
                card.set_values(values)
        except (ValueError, TypeError):
            pass

    def _export_presets(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出预设位", "lens_presets.txt", "TEXT File (*.txt);;All files (*.*)"
        )
        if not path:
            return
        payload = {"schema": "lensdetect.lens_presets",
                   "positions": self._presets_payload()}
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            self._set_message(f"预设位已导出到 {path}")
        except OSError as exc:
            self._set_message(f"导出失败: {exc}")

    def _import_presets(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "导入预设位", "", "TEXT File (*.txt);;All files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            positions = payload.get("positions", payload)
            if not isinstance(positions, list):
                raise ValueError("positions 必须是列表")
            for card, values in zip(self.preset_cards, positions):
                card.set_values(values)
            self._save_presets()
            self._set_message(f"预设位已从 {path} 导入")
        except (OSError, ValueError) as exc:
            self._set_message(f"导入失败: {exc}")
