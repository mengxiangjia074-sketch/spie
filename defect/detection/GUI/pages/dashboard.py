"""Overview page: environment, cameras, serial ports, calibration status."""

from __future__ import annotations

import importlib.util
import platform
import sys

from PySide6.QtCore import Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GUI.core.paths import (
    CALIBRATION_SUMMARY,
    GUI_LOG_DIR,
    PROJECT_ROOT,
    WORKING_DATA,
)
from GUI.core.runner import TaskManager, open_in_explorer
from GUI.widgets.resultview import SimpleCard, SummaryTable

CHECK_MODULES = [
    ("PySide6", "界面"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("_jsonnet", "Jsonnet 配置"),
    ("minimalmodbus", "位移台通讯"),
    ("serial", "串口"),
]
if platform.system() == "Windows":
    CHECK_MODULES.append(("pygrabber", "相机枚举"))


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _enumerate_cameras():
    try:
        from LensCamera.camera import enumerate_cameras

        return enumerate_cameras(RuntimeError)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def _enumerate_ports():
    try:
        import serial.tools.list_ports

        ports = list(serial.tools.list_ports.comports())
        if platform.system() == "Linux":
            ports = [port for port in ports if port.vid is not None]
        return ports
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


class DashboardPage(QWidget):
    navigate_requested = Signal(str)  # page key

    def __init__(self, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        manager.task_finished.connect(self._refresh_recent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        scroll_content = QWidget()
        grid = QGridLayout(scroll_content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        grid.addWidget(self._build_env_card(), 0, 0)
        grid.addWidget(self._build_summary_card(), 0, 1)
        grid.addWidget(self._build_camera_card(), 1, 0)
        grid.addWidget(self._build_port_card(), 1, 1)
        grid.addWidget(self._build_shortcuts(), 2, 0)
        grid.addWidget(self._build_recent_card(), 2, 1)
        grid.setRowStretch(3, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll)

        self.refresh_hardware()
        self.summary_table.load(CALIBRATION_SUMMARY)

    # ---- cards ------------------------------------------------------------

    def _build_env_card(self) -> SimpleCard:
        card = SimpleCard("运行环境")
        python_label = QLabel(sys.executable)
        python_label.setObjectName("PathLabel")
        python_label.setWordWrap(False)
        card.body.addWidget(python_label)
        root_label = QLabel(f"项目根目录  {PROJECT_ROOT}")
        root_label.setObjectName("PathLabel")
        card.body.addWidget(root_label)
        self._module_label = QLabel()
        self._module_label.setObjectName("HintLabel")
        self._module_label.setWordWrap(True)
        card.body.addWidget(self._module_label)
        self._render_modules()
        return card

    def _render_modules(self) -> None:
        parts = []
        for name, label in CHECK_MODULES:
            ok = _module_present(name)
            mark = "✔" if ok else "✖"
            parts.append(f"{mark} {label}")
        self._module_label.setText("  ".join(parts))
        missing = [
            label
            for name, label in CHECK_MODULES
            if not _module_present(name)
        ]
        if missing:
            self._module_label.setText(
                self._module_label.text()
                + f"   (缺少: {', '.join(missing)}; 相关功能将不可用)"
            )

    def _build_summary_card(self) -> SimpleCard:
        card = SimpleCard("标定汇总 (calibrate.json)")
        self.summary_table = SummaryTable()
        self.summary_table.setMinimumHeight(150)
        card.body.addWidget(self.summary_table)
        bar = QHBoxLayout()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(
            lambda: self.summary_table.load(CALIBRATION_SUMMARY)
        )
        view = QPushButton("查看结果")
        view.clicked.connect(lambda: self.navigate_requested.emit("results"))
        bar.addWidget(refresh)
        bar.addWidget(view)
        bar.addStretch(1)
        card.body.addLayout(bar)
        return card

    def _build_camera_card(self) -> SimpleCard:
        card = SimpleCard("相机设备 (DirectShow)")
        self._camera_label = QLabel("点击“刷新”枚举…")
        self._camera_label.setWordWrap(True)
        card.body.addWidget(self._camera_label)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_hardware)
        bar = QHBoxLayout()
        bar.addWidget(refresh)
        bar.addStretch(1)
        card.body.addLayout(bar)
        return card

    def _build_port_card(self) -> SimpleCard:
        card = SimpleCard("串口设备")
        self._port_label = QLabel("点击“刷新”枚举…")
        self._port_label.setWordWrap(True)
        card.body.addWidget(self._port_label)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_hardware)
        bar = QHBoxLayout()
        bar.addWidget(refresh)
        bar.addStretch(1)
        card.body.addLayout(bar)
        return card

    def _build_shortcuts(self) -> SimpleCard:
        card = SimpleCard("快捷入口")
        bar = QHBoxLayout()
        bar.setSpacing(8)
        for text, key in (
            ("日志目录", "logdir"),
            ("working_data", "workdir"),
            ("配置备份", "backupdir"),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, k=key: self._shortcut(k))
            bar.addWidget(button)
        bar.addStretch(1)
        card.body.addLayout(bar)
        hint = QLabel("GUI 修改配置时自动备份到 working_data/gui_backups/。")
        hint.setObjectName("HintLabel")
        card.body.addWidget(hint)
        return card

    def _shortcut(self, key: str) -> None:
        from GUI.core.paths import GUI_BACKUP_DIR

        targets = {
            "logdir": GUI_LOG_DIR,
            "workdir": WORKING_DATA,
            "backupdir": GUI_BACKUP_DIR,
        }
        path = targets[key]
        path.mkdir(parents=True, exist_ok=True)
        open_in_explorer(path)

    def _build_recent_card(self) -> SimpleCard:
        card = SimpleCard("本次会话运行记录")
        self._recent_table = QTableWidget(0, 4)
        self._recent_table.setHorizontalHeaderLabels(("时间", "任务", "结果", "退出码"))
        self._recent_table.verticalHeader().setVisible(False)
        self._recent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._recent_table.setAlternatingRowColors(True)
        self._recent_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self._recent_table.setMinimumHeight(150)
        card.body.addWidget(self._recent_table)
        hint = QLabel("双击一行打开对应日志文件。")
        hint.setObjectName("HintLabel")
        card.body.addWidget(hint)
        self._recent_table.cellDoubleClicked.connect(self._open_recent_log)
        return card

    # ---- behavior ------------------------------------------------------------

    def refresh_hardware(self) -> None:
        try:
            cameras = _enumerate_cameras()
            if cameras:
                text = "\n".join(
                    f"[{item['index']}] {item['name']}" for item in cameras
                )
            else:
                text = "未发现相机设备"
        except Exception as exc:
            text = f"枚举失败: {exc}"
        self._camera_label.setText(text)

        try:
            ports = _enumerate_ports()
            if ports:
                text = "\n".join(
                    f"{port.device}  {port.description}" for port in ports
                )
            else:
                text = "未发现串口设备"
        except Exception as exc:
            text = f"枚举失败: {exc}"
        self._port_label.setText(text)

    def _refresh_recent(self, *_args) -> None:
        table = self._recent_table
        table.setRowCount(0)
        for record in reversed(self.manager.history[-20:]):
            started = record.get("started")
            when = started.strftime("%H:%M:%S") if started else "—"
            row = table.rowCount()
            table.insertRow(row)
            values = (
                when,
                record.get("title", ""),
                record.get("result") or "…",
                str(record.get("exit_code") if record.get("exit_code") is not None else ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    result = value
                    color = (
                        "#3ecf8e" if result == "成功"
                        else "#98a2b8" if result in ("…", "已停止")
                        else "#ef5f5f"
                    )
                    item.setForeground(QBrush(QColor(color)))
                table.setItem(row, column, item)

    def _open_recent_log(self, row: int, _column: int) -> None:
        # rows are newest-first over history[-20:]
        index = len(self.manager.history[-20:]) - 1 - row
        if 0 <= index < len(self.manager.history):
            log = self.manager.history[index].get("log")
            if log is not None:
                open_in_explorer(log)
