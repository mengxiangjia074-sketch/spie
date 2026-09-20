"""RFID 检测页: detection/E720_RFID_Tool.py 的 GUI 移植。

E720 超高频 RFID 读写模块 (USB 转 TTL 串口, 默认 115200bps):
标签识别 (单次寻卡/连续扫描)、标签数据读写 (User 区/EPC 卡号, 写入自动
验证)、功率/地区/信道设置、十六进制调试。协议实现复用原文件的 E720Module,
串口操作全部在 RfidWorker 线程执行; 通讯日志合并到运行控制台。

RFID 模块为独立串口设备, 与检测任务/位移台/镜头互不占用; 返回主界面时
自动断开。
"""

from __future__ import annotations

import serial.tools.list_ports
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GUI.core.rfid_worker import RfidWorker
from E720_RFID_Tool import (
    MEMBANK_EPC,
    MEMBANK_TID,
    MEMBANK_USER,
    POWER_TABLE,
    REGIONS,
)

# 存储区下拉 (与 E720 协议 MemBank 对应)
BANKS = (
    ("User 用户区", MEMBANK_USER),
    ("EPC 卡号区", MEMBANK_EPC),
    ("TID 唯一ID", MEMBANK_TID),
)

CONTINUOUS_MISS_LIMIT = 2


class RfidControlPage(QWidget):
    """交互式 E720 RFID 控制/检测页。"""

    console_output = Signal(str, str)  # channel ("comm-send"/...), text

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = RfidWorker(self)
        self.connected = False
        self._connection_pending = False
        self.current_epc = ""  # 当前选中的标签 EPC
        self._periodic = False        # 连续扫描激活状态
        self._round_busy = False      # 一轮寻卡是否仍在执行 (防队列堆积)
        self._capacity_busy = False
        self._inventory_request_mode = ""
        self._continuous_tags: dict[str, dict] = {}
        self._last_inventory_tags: list[dict] = []
        self._continuous_miss_limit = CONTINUOUS_MISS_LIMIT

        self._periodic_timer = QTimer(self)
        self._periodic_timer.timeout.connect(self._periodic_tick)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_connection_group())

        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(self._build_scan_group(), 1)
        row1.addWidget(self._build_data_group(), 1)
        outer.addLayout(row1, 1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(self._build_write_group(), 1)
        row2.addWidget(self._build_settings_group(), 1)
        row2.addWidget(self._build_debug_group(), 1)
        outer.addLayout(row2)

        self.feedback_label = QLabel("未连接模块。选择串口后点击“连接”。")
        self.feedback_label.setObjectName("HintLabel")
        outer.addWidget(self.feedback_label)

        self._wire_worker()
        self._set_controls_enabled(False)
        self._refresh_ports()

    # ---- UI ----------------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        title = QLabel("RFID 检测 (E720 超高频读写模块)")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "标签识别 · 数据读写 (User 区 / EPC 卡号, 写入自动验证) · "
            "功率/地区/信道设置 · 十六进制调试。通讯日志显示在底部运行控制台。"
            "RFID 为独立串口设备, 返回主界面时自动断开。"
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
        self.port_combo.setMinimumWidth(200)
        layout.addWidget(self.port_combo)
        self.refresh_ports_btn = QPushButton("刷新")
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(self.refresh_ports_btn)

        layout.addWidget(QLabel("波特率"))
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(
            ["9600", "19200", "38400", "57600", "115200", "230400"])
        self.baudrate_combo.setCurrentText("115200")
        layout.addWidget(self.baudrate_combo)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.setProperty("class", "primary")
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        layout.addWidget(self.connect_btn)
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        layout.addWidget(self.disconnect_btn)

        self.status_label = QLabel("未连接")
        self.status_label.setObjectName("HintLabel")
        layout.addWidget(self.status_label, 1)
        return group

    def _build_scan_group(self) -> QGroupBox:
        group = QGroupBox("标签识别")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        bar = QHBoxLayout()
        self.single_btn = QPushButton("单次寻卡")
        self.single_btn.clicked.connect(self._on_single_clicked)
        bar.addWidget(self.single_btn)
        bar.addWidget(QLabel("间隔"))
        self.scan_interval = QDoubleSpinBox()
        self.scan_interval.setRange(0.5, 60.0)
        self.scan_interval.setDecimals(1)
        self.scan_interval.setSingleStep(0.5)
        self.scan_interval.setValue(1.0)
        self.scan_interval.setSuffix(" s")
        self.scan_interval.setToolTip("连续扫描时每隔该时间自动寻卡一次")
        self.scan_interval.valueChanged.connect(self._on_interval_changed)
        bar.addWidget(self.scan_interval)
        self.scan_btn = QPushButton("连续扫描")
        self.scan_btn.setCheckable(True)
        self.scan_btn.toggled.connect(self._on_scan_toggled)
        self.scan_btn.setToolTip("开启后按间隔反复寻卡; 再点一次或点“单次寻卡”停止")
        bar.addWidget(self.scan_btn)
        self.clear_tags_btn = QPushButton("清空")
        self.clear_tags_btn.setToolTip("清空识别结果和当前选中的标签")
        self.clear_tags_btn.clicked.connect(self._clear_inventory_results)
        bar.addWidget(self.clear_tags_btn)
        bar.addStretch(1)
        layout.addLayout(bar)

        settings_bar = QHBoxLayout()
        self.rssi_filter_check = QCheckBox("RSSI 过滤")
        self.rssi_filter_check.setChecked(True)
        self.rssi_filter_check.setToolTip("只显示信号强度不低于设定值的标签")
        self.rssi_filter_check.toggled.connect(self._on_rssi_filter_toggled)
        settings_bar.addWidget(self.rssi_filter_check)
        self.rssi_threshold_spin = QSpinBox()
        self.rssi_threshold_spin.setRange(-120, -1)
        self.rssi_threshold_spin.setValue(-40)
        self.rssi_threshold_spin.setSuffix(" dBm")
        self.rssi_threshold_spin.setToolTip("显示 RSSI 大于或等于该值的标签")
        self.rssi_threshold_spin.setEnabled(False)
        self.rssi_threshold_spin.valueChanged.connect(
            self._on_rssi_threshold_changed)
        settings_bar.addWidget(self.rssi_threshold_spin)
        settings_bar.addWidget(QLabel("连续漏读移除"))
        self.miss_limit_spin = QSpinBox()
        self.miss_limit_spin.setRange(1, 100)
        self.miss_limit_spin.setValue(CONTINUOUS_MISS_LIMIT)
        self.miss_limit_spin.setSuffix(" 次")
        self.miss_limit_spin.setToolTip(
            "连续扫描中，同一标签连续多少轮未识别到后从表格移除")
        self.miss_limit_spin.valueChanged.connect(
            self._on_continuous_miss_limit_changed)
        settings_bar.addWidget(self.miss_limit_spin)
        settings_bar.addStretch(1)
        layout.addLayout(settings_bar)

        self.tag_table = QTableWidget(0, 2)
        self.tag_table.setHorizontalHeaderLabels(("EPC", "RSSI (dBm)"))
        self.tag_table.verticalHeader().setVisible(False)
        self.tag_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tag_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tag_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.tag_table.cellClicked.connect(self._on_tag_clicked)
        layout.addWidget(self.tag_table, 1)

        hint = QLabel("点击行选中标签, 供读取/写入使用。连续扫描会按 EPC 去重并稳定更新结果。")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _build_data_group(self) -> QGroupBox:
        group = QGroupBox("标签数据读取")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)

        self.selected_epc_label = QLabel("(未选择)")
        self.selected_epc_label.setObjectName("PathLabel")
        self.selected_epc_label.setWordWrap(True)
        layout.addRow("目标标签", self.selected_epc_label)

        self.bank_combo = QComboBox()
        for name, code in BANKS:
            self.bank_combo.addItem(name, code)
        layout.addRow("存储区", self.bank_combo)

        addr_bar = QHBoxLayout()
        self.addr_spin = QSpinBox()
        self.addr_spin.setRange(0, 255)
        addr_bar.addWidget(self.addr_spin)
        self.words_spin = QSpinBox()
        self.words_spin.setRange(1, 64)
        self.words_spin.setValue(4)
        addr_bar.addWidget(QLabel("长度(Word)"))
        addr_bar.addWidget(self.words_spin)
        addr_bar.addStretch(1)
        read_btn = QPushButton("读取数据")
        read_btn.setProperty("class", "primary")
        read_btn.clicked.connect(self._on_read_clicked)
        addr_bar.addWidget(read_btn)
        layout.addRow("起始地址(Word)", addr_bar)

        self.read_hex_label = QLabel("--")
        self.read_hex_label.setObjectName("PathLabel")
        self.read_hex_label.setWordWrap(True)
        self.read_hex_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        layout.addRow("HEX", self.read_hex_label)
        self.read_ascii_label = QLabel("--")
        self.read_ascii_label.setObjectName("PathLabel")
        layout.addRow("ASCII", self.read_ascii_label)

        self.detect_capacity_btn = QPushButton("检测 User / EPC 容量")
        self.detect_capacity_btn.setProperty("class", "primary")
        self.detect_capacity_btn.clicked.connect(self._on_detect_capacity_clicked)
        layout.addRow("容量检测", self.detect_capacity_btn)

        self.capacity_result_label = QLabel("User 区：--\nEPC 卡号区：--")
        self.capacity_result_label.setObjectName("PathLabel")
        self.capacity_result_label.setWordWrap(True)
        self.capacity_result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addRow("检测结果", self.capacity_result_label)
        return group

    def _build_write_group(self) -> QGroupBox:
        group = QGroupBox("写入标签")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)

        self.write_mode_combo = QComboBox()
        self.write_mode_combo.addItem("User 数据区", "user")
        self.write_mode_combo.addItem("EPC 卡号 (永久改变标签 ID!)", "epc")
        self.write_mode_combo.currentIndexChanged.connect(
            self._on_write_mode_changed)
        layout.addRow("写入模式", self.write_mode_combo)

        self.write_data_edit = QLineEdit()
        self.write_data_edit.setPlaceholderText(
            "偶数字节的十六进制, 如 01 02 03 04")
        layout.addRow("数据(HEX)", self.write_data_edit)

        self.write_addr_spin = QSpinBox()
        self.write_addr_spin.setRange(0, 255)
        layout.addRow("起始地址(Word)", self.write_addr_spin)

        write_btn = QPushButton("写入…")
        write_btn.setProperty("class", "danger")
        write_btn.clicked.connect(self._on_write_clicked)
        layout.addRow(write_btn)

        self.write_result_label = QLabel("--")
        self.write_result_label.setObjectName("HintLabel")
        self.write_result_label.setWordWrap(True)
        layout.addRow("结果", self.write_result_label)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("模块信息与设置")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)

        self.info_label = QLabel("--")
        self.info_label.setObjectName("PathLabel")
        self.info_label.setWordWrap(True)
        layout.addRow("模块", self.info_label)

        power_bar = QHBoxLayout()
        self.power_combo = QComboBox()
        for dbm in sorted(POWER_TABLE):
            self.power_combo.addItem(f"{dbm} dBm", dbm)
        self.power_combo.setCurrentText("15 dBm")
        power_bar.addWidget(self.power_combo)
        power_btn = QPushButton("设置")
        power_btn.clicked.connect(
            lambda: self.worker.set_power(self.power_combo.currentData()))
        power_bar.addWidget(power_btn)
        power_bar.addStretch(1)
        layout.addRow("发射功率", power_bar)

        region_bar = QHBoxLayout()
        self.region_combo = QComboBox()
        for name, code in REGIONS.items():
            self.region_combo.addItem(name, code)
        region_bar.addWidget(self.region_combo)
        region_btn = QPushButton("设置")
        region_btn.clicked.connect(
            lambda: self.worker.set_region(
                self.region_combo.currentData(),
                self.region_combo.currentText()))
        region_bar.addWidget(region_btn)
        region_bar.addStretch(1)
        layout.addRow("工作地区", region_bar)

        channel_bar = QHBoxLayout()
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 63)
        channel_bar.addWidget(self.channel_spin)
        channel_btn = QPushButton("设置")
        channel_btn.clicked.connect(
            lambda: self.worker.set_channel(self.channel_spin.value()))
        channel_bar.addWidget(channel_btn)
        channel_bar.addStretch(1)
        layout.addRow("工作信道", channel_bar)

        refresh_info_btn = QPushButton("刷新模块信息")
        refresh_info_btn.clicked.connect(lambda: self.worker.refresh_info())
        layout.addRow(refresh_info_btn)
        return group

    def _build_debug_group(self) -> QGroupBox:
        group = QGroupBox("调试")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        self.debug_edit = QLineEdit()
        self.debug_edit.setPlaceholderText("如 00 22 00 00 (不含帧头BB/帧尾7E)")
        layout.addWidget(self.debug_edit)
        send_btn = QPushButton("发送指令")
        send_btn.clicked.connect(
            lambda: self.worker.debug_send(self.debug_edit.text()))
        layout.addWidget(send_btn)
        hint = QLabel("收发结果输出到运行控制台。")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return group

    # ---- worker 信号 -------------------------------------------------------

    def _wire_worker(self):
        worker = self.worker
        worker.connected_signal.connect(self._on_connected)
        worker.info_ready.connect(self._on_info_ready)
        worker.inventory_done.connect(self._on_inventory_done)
        worker.read_done.connect(self._on_read_done)
        worker.capacity_done.connect(self._on_capacity_done)
        worker.write_done.connect(self._on_write_done)
        worker.epc_write_done.connect(self._on_epc_write_done)
        worker.settings_done.connect(
            lambda ok, msg: self._set_message(msg))
        worker.operation_failed.connect(self._on_operation_failed)
        worker.logger.log_signal.connect(self._append_log)
        worker.start()

    def _on_connected(self, ok: bool, message: str):
        self._connection_pending = False
        self.connected = ok
        self.connect_btn.setEnabled(not ok)
        self.disconnect_btn.setEnabled(ok)
        for widget in (self.port_combo, self.baudrate_combo,
                       self.refresh_ports_btn):
            widget.setEnabled(not ok)
        self.status_label.setText(message)
        self._set_controls_enabled(ok)
        if not ok:
            self._set_periodic(False)  # 断开时停止连续扫描并复位按钮
            self._round_busy = False
            self._continuous_tags.clear()
            self._last_inventory_tags.clear()
            self.tag_table.setRowCount(0)
            self.current_epc = ""
            self.selected_epc_label.setText("(未选择)")
            self._capacity_busy = False
            self._reset_capacity_result()
            self.info_label.setText("--")

    def _on_info_ready(self, info: dict):
        self.info_label.setText(
            f"HW:{info.get('hw', '?')} SW:{info.get('sw', '?')} "
            f"厂商:{info.get('mfr', '?')} | 地区:{info.get('region_name', '?')} "
            f"功率:{info.get('power', '?')}dBm 信道:{info.get('channel', '?')}")

    @staticmethod
    def _deduplicate_tags(tags: list) -> list:
        unique = {}
        for tag in tags:
            epc = tag.get("epc_hex", "")
            if not epc:
                continue
            previous = unique.get(epc)
            if (previous is None
                    or tag.get("rssi", -256) > previous.get("rssi", -256)):
                unique[epc] = dict(tag)
        return sorted(
            unique.values(), key=lambda item: item.get("rssi", -256), reverse=True)

    def _fill_tags(self, tags: list, message: str | None = None):
        tags = self._deduplicate_tags(tags)
        table = self.tag_table
        table.setRowCount(0)
        selected_row = -1
        for tag in tags:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(tag["epc_hex"]))
            table.setItem(row, 1, QTableWidgetItem(str(tag["rssi"])))
            if tag["epc_hex"] == self.current_epc:
                selected_row = row
        if selected_row >= 0:
            table.selectRow(selected_row)
        elif self.current_epc:
            self.current_epc = ""
            self.selected_epc_label.setText("(未选择)")
            self._reset_capacity_result()
        self._set_message(message or f"本次识别 {len(tags)} 张标签")

    def _filter_tags(self, tags: list) -> list:
        tags = self._deduplicate_tags(tags)
        if not self.rssi_filter_check.isChecked():
            return tags
        threshold = self.rssi_threshold_spin.value()
        return [tag for tag in tags if tag.get("rssi", -256) >= threshold]

    def _refresh_rssi_filter(self):
        visible = self._filter_tags(self._last_inventory_tags)
        total = len(self._last_inventory_tags)
        if self.rssi_filter_check.isChecked():
            message = (f"RSSI 下限 {self.rssi_threshold_spin.value()} dBm: "
                       f"显示 {len(visible)}/{total} 张标签")
        else:
            message = f"RSSI 过滤已关闭: 显示 {len(visible)} 张标签"
        self._fill_tags(visible, message)

    def _on_rssi_filter_toggled(self, checked: bool):
        self.rssi_threshold_spin.setEnabled(self.connected and checked)
        self._refresh_rssi_filter()

    def _on_rssi_threshold_changed(self, _value: int):
        if self.rssi_filter_check.isChecked():
            self._refresh_rssi_filter()

    def _on_continuous_miss_limit_changed(self, limit: int):
        self._continuous_miss_limit = limit
        removed = []
        for epc, entry in list(self._continuous_tags.items()):
            if entry["misses"] >= limit:
                removed.append(epc)
                del self._continuous_tags[epc]

        if removed:
            merged = self._deduplicate_tags([
                entry["tag"] for entry in self._continuous_tags.values()
            ])
            self._last_inventory_tags = merged
            visible = self._filter_tags(merged)
            self._fill_tags(
                visible,
                f"连续漏读移除次数已设为 {limit} 次，已移除 {len(removed)} 张标签",
            )
            return
        self._set_message(f"连续漏读移除次数已设为 {limit} 次")

    def _merge_continuous_tags(self, tags: list) -> list:
        current = self._deduplicate_tags(tags)
        seen = {tag["epc_hex"] for tag in current}
        for epc in list(self._continuous_tags):
            if epc in seen:
                continue
            entry = self._continuous_tags[epc]
            entry["misses"] += 1
            if entry["misses"] >= self._continuous_miss_limit:
                del self._continuous_tags[epc]
        for tag in current:
            self._continuous_tags[tag["epc_hex"]] = {
                "tag": tag,
                "misses": 0,
            }
        return self._deduplicate_tags([
            entry["tag"] for entry in self._continuous_tags.values()
        ])

    def _on_inventory_done(self, tags: list):
        """显示单次结果，或将连续扫描的一轮结果合并到稳定列表。"""
        self._round_busy = False
        mode = self._inventory_request_mode
        self._inventory_request_mode = ""
        current = self._deduplicate_tags(tags)
        if mode == "periodic":
            merged = self._merge_continuous_tags(current)
            self._last_inventory_tags = merged
            visible = self._filter_tags(merged)
            message = (f"本轮识别 {len(current)} 张, "
                       f"当前保留 {len(merged)} 张标签")
        else:
            self._continuous_tags.clear()
            self._last_inventory_tags = current
            visible = self._filter_tags(current)
            message = f"本次识别 {len(current)} 张标签"
        if self.rssi_filter_check.isChecked():
            message += f", 过滤后显示 {len(visible)} 张"
        self._fill_tags(visible, message)

    def _on_read_done(self, ok: bool, epc_hex: str, data_hex: str, ascii_text: str):
        if ok:
            self.read_hex_label.setText(data_hex)
            self.read_ascii_label.setText(ascii_text or "(不可显示)")
            self._set_message(f"读取成功: {data_hex}")
        else:
            self.read_hex_label.setText("--")
            self.read_ascii_label.setText("--")
            self._set_message("读取失败: 标签不在范围内 / 存储区为空 / EPC 不正确")

    @staticmethod
    def _format_capacity(bank: str, result: dict) -> str:
        present = result.get("present")
        if present is False:
            return "不存在（0 Word / 0 字节）"
        if present is None or result.get("words") is None:
            return f"无法确认（{result.get('message') or '读取失败'}）"
        words = int(result["words"])
        byte_count = int(result.get("bytes", words * 2))
        if bank == "epc":
            data_words = max(0, words - 2)
            return (
                f"存在，总容量 {words} Word / {byte_count} 字节；"
                f"卡号数据 {data_words} Word / {data_words * 2} 字节"
            )
        return f"存在，容量 {words} Word / {byte_count} 字节"

    def _on_capacity_done(self, epc_hex: str, results: dict):
        self._capacity_busy = False
        self.detect_capacity_btn.setEnabled(self.connected)
        user_text = self._format_capacity("user", results.get("user", {}))
        epc_text = self._format_capacity("epc", results.get("epc", {}))
        if epc_hex != self.current_epc:
            self._reset_capacity_result()
            self._set_message("容量检测已完成，但当前选择的标签已经改变")
            return
        self.capacity_result_label.setText(
            f"User 区：{user_text}\nEPC 卡号区：{epc_text}"
        )
        message = (
            f"容量检测完成 [{epc_hex}]：User 区 {user_text}；"
            f"EPC 卡号区 {epc_text}"
        )
        self._set_message(message)
        self._append_log("info", message)

    def _on_write_done(self, ok: bool, message: str):
        self.write_result_label.setText(("✔ " if ok else "✘ ") + message)
        self._set_message(("写入成功: " if ok else "写入失败: ") + message)

    def _on_epc_write_done(self, ok: bool, message: str,
                           effective_epc: str):
        self._on_write_done(ok, message)
        if not ok:
            return
        old_epc = self.current_epc
        self.current_epc = effective_epc
        self.selected_epc_label.setText(effective_epc or "(请重新扫描)")
        self._reset_capacity_result()
        if not effective_epc:
            self.tag_table.setRowCount(0)
            return
        for row in range(self.tag_table.rowCount()):
            item = self.tag_table.item(row, 0)
            if item is not None and item.text() == old_epc:
                if effective_epc:
                    item.setText(effective_epc)
                    self.tag_table.selectRow(row)
                break

    def _on_operation_failed(self, operation: str, message: str):
        labels = {
            "connect": "连接",
            "disconnect": "断开",
            "inventory": "寻卡",
            "read": "读取",
            "capacity": "检测存储区容量",
            "write-user": "写入 User 区",
            "write-epc": "写入 EPC",
            "settings": "设置",
            "info": "读取模块信息",
            "debug": "调试指令",
        }
        text = f"{labels.get(operation, operation)}失败: {message}"
        if operation in ("connect", "disconnect"):
            self._on_connected(False, text)
            return
        if operation == "inventory":
            self._round_busy = False
            self._inventory_request_mode = ""
        elif operation == "read":
            self.read_hex_label.setText("--")
            self.read_ascii_label.setText("--")
        elif operation == "capacity":
            self._capacity_busy = False
            self.detect_capacity_btn.setEnabled(self.connected)
            self.capacity_result_label.setText(f"检测失败：{message}")
        elif operation.startswith("write-"):
            self.write_result_label.setText("✘ " + text)
        self._set_message(text)

    # ---- 操作 ---------------------------------------------------------------

    def is_connected(self) -> bool:
        return self.connected

    def disconnect_if_connected(self) -> None:
        if self.connected or self._connection_pending:
            self._set_periodic(False)
            self._set_controls_enabled(False)
            self.disconnect_btn.setEnabled(False)
            self.status_label.setText("正在断开… (当前命令结束后释放串口)")
            self.worker.disconnect_device()

    def shutdown_worker(self):
        self._set_periodic(False)
        self.worker.stop()
        self.worker.wait(15000)

    def _refresh_ports(self):
        self.port_combo.clear()
        try:
            ports = serial.tools.list_ports.comports()
        except Exception as exc:
            self._set_message(f"枚举串口失败: {exc}")
            return
        for port in sorted(ports, key=lambda item: item.device):
            self.port_combo.addItem(
                f"{port.device} - {port.description}", port.device)

    def _on_connect_clicked(self):
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "错误", "请选择串口")
            return
        self.connect_btn.setEnabled(False)
        self._connection_pending = True
        self.port_combo.setEnabled(False)
        self.baudrate_combo.setEnabled(False)
        self.refresh_ports_btn.setEnabled(False)
        self.status_label.setText("连接中…")
        self.worker.connect_device(port, int(self.baudrate_combo.currentText()))

    def _on_disconnect_clicked(self):
        """停止继续入队，并在当前串口命令完成后断开。"""
        self._set_periodic(False)
        self._set_controls_enabled(False)
        self.disconnect_btn.setEnabled(False)
        self.status_label.setText("正在断开… (当前命令结束后释放串口)")
        self.worker.disconnect_device()

    def _set_controls_enabled(self, enabled: bool):
        for widget in (
            self.tag_table, self.bank_combo, self.addr_spin, self.words_spin,
            self.write_mode_combo, self.write_data_edit, self.write_addr_spin,
            self.power_combo, self.region_combo, self.channel_spin,
            self.debug_edit, self.rssi_filter_check, self.rssi_threshold_spin,
            self.scan_interval, self.miss_limit_spin,
        ):
            widget.setEnabled(enabled)
        # 按钮通过 group 查找统一启停
        for group in self.findChildren(QGroupBox):
            if group.title() in ("标签识别", "标签数据读取", "写入标签",
                                 "模块信息与设置", "调试"):
                for button in group.findChildren(QPushButton):
                    button.setEnabled(enabled)
        self.write_addr_spin.setEnabled(
            enabled and self.write_mode_combo.currentData() != "epc")
        self.rssi_threshold_spin.setEnabled(
            enabled and self.rssi_filter_check.isChecked())
        self.detect_capacity_btn.setEnabled(enabled and not self._capacity_busy)

    def _on_tag_clicked(self, row: int, _column: int):
        item = self.tag_table.item(row, 0)
        if item is None:
            return
        self.current_epc = item.text()
        self.selected_epc_label.setText(self.current_epc)
        self._reset_capacity_result()

    def _clear_inventory_results(self):
        """清空显示与累计缓存，不改变连续扫描的运行状态。"""
        self._continuous_tags.clear()
        self._last_inventory_tags.clear()
        self.tag_table.setRowCount(0)
        self.current_epc = ""
        self.selected_epc_label.setText("(未选择)")
        self._reset_capacity_result()
        message = "标签识别结果已清空"
        if self._periodic:
            message += ", 连续扫描仍在运行"
        self._set_message(message)

    # ---- 连续扫描 (按间隔周期寻卡) ----------------------------------------

    def _on_single_clicked(self):
        """单次寻卡: 若连续扫描开启则自动停止, 切换为单次。"""
        if self._periodic:
            self._set_periodic(False)
        if self._round_busy:
            self._inventory_request_mode = "single"
            self._set_message("已停止连续扫描, 等待当前寻卡完成")
            return
        self._round_busy = True
        self._inventory_request_mode = "single"
        self.worker.single_inventory()

    def _on_scan_toggled(self, checked: bool):
        self._set_periodic(checked)

    def _set_periodic(self, active: bool):
        if active and not self.connected:
            active = False
        starting = active and not self._periodic
        self._periodic = active
        if active:
            if starting:
                self._continuous_tags.clear()
                self._last_inventory_tags.clear()
                self.tag_table.setRowCount(0)
            self._periodic_timer.start(int(self.scan_interval.value() * 1000))
            self._set_message(f"连续扫描中: 每 {self.scan_interval.value():g}s "
                              f"寻卡一次 (再点一次或点“单次寻卡”停止)")
        else:
            self._periodic_timer.stop()
        # 激活时按钮保持按下态并使用主色 (蓝) 背景
        self.scan_btn.setChecked(active)
        self.scan_btn.setProperty("class", "primary" if active else "")
        self.scan_btn.style().unpolish(self.scan_btn)
        self.scan_btn.style().polish(self.scan_btn)
        if active:
            self._periodic_tick()

    def _periodic_tick(self):
        if not self.connected:
            self._set_periodic(False)
            return
        if self._round_busy:
            return  # 上一轮还没返回, 跳过本拍, 避免命令在队列里堆积
        self._round_busy = True
        self._inventory_request_mode = "periodic"
        self.worker.single_inventory()

    def _on_interval_changed(self, _value: float):
        """连续扫描运行中修改间隔 -> 立即按新间隔重排定时器。"""
        if self._periodic:
            self._periodic_timer.start(int(self.scan_interval.value() * 1000))
            self._set_message(f"连续扫描间隔已改为 "
                              f"{self.scan_interval.value():g}s")

    def _on_read_clicked(self):
        if not self.current_epc:
            self._set_message("请先在标签列表中选择标签")
            return
        self.worker.read_tag(self.current_epc, self.bank_combo.currentData(),
                             self.addr_spin.value(), self.words_spin.value())

    def _reset_capacity_result(self):
        self.capacity_result_label.setText("User 区：--\nEPC 卡号区：--")

    def _on_detect_capacity_clicked(self):
        if not self.current_epc:
            self._set_message("请先在标签列表中选择标签")
            return
        if self._capacity_busy:
            return
        if self._periodic:
            self._set_periodic(False)
        self._capacity_busy = True
        self.detect_capacity_btn.setEnabled(False)
        self.capacity_result_label.setText("正在检测 User 区和 EPC 卡号区…")
        self._set_message("正在检测存储区是否存在及其容量，请保持标签位置不动")
        self.worker.detect_capacity(self.current_epc)

    def _on_write_mode_changed(self, _index: int):
        if self.write_mode_combo.currentData() == "epc":
            self.write_data_edit.setPlaceholderText(
                "新 EPC 卡号: 12 字节 (24 位十六进制)")
            self.write_addr_spin.setEnabled(False)
        else:
            self.write_data_edit.setPlaceholderText(
                "偶数字节的十六进制, 如 01 02 03 04")
            self.write_addr_spin.setEnabled(self.connected)

    def _on_write_clicked(self):
        if not self.current_epc:
            self._set_message("请先在标签列表中选择标签")
            return
        data_text = "".join(self.write_data_edit.text().split())
        if not data_text:
            self._set_message("请输入要写入的数据")
            return
        try:
            data = bytes.fromhex(data_text)
        except ValueError:
            self._set_message("请输入有效的十六进制数据 (仅 0-9、A-F)")
            return
        mode = self.write_mode_combo.currentData()
        if mode == "epc":
            if len(data) != 12:
                self._set_message("EPC 卡号必须是 24 位十六进制 (12 字节)")
                return
            normalized = data.hex().upper()
            warning = (f"将把标签 EPC 从\n{self.current_epc}\n改为\n"
                       f"{normalized}\n\n写入后旧卡号失效, 确定继续?")
        else:
            if len(data) % 2 != 0:
                self._set_message("数据长度必须是偶数字节 (十六进制字符数为 4 的倍数)")
                return
            normalized = data.hex().upper()
            warning = (f"将向标签 {self.current_epc} 的 User 区 "
                       f"Word {self.write_addr_spin.value()} 写入 "
                       f"{normalized}, 确定继续?")
        answer = QMessageBox.question(
            self, "确认写入", warning,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self._set_message("已取消写入")
            return
        if mode == "epc":
            self.worker.write_epc(self.current_epc, normalized)
        else:
            self.worker.write_user(self.current_epc, normalized,
                                   self.write_addr_spin.value())

    # ---- 消息 ---------------------------------------------------------------

    def _set_message(self, text: str):
        self.feedback_label.setText(text)

    def _append_log(self, direction: str, message: str):
        from datetime import datetime

        from GUI.pages.motion_control import LOG_PREFIX

        now = datetime.now()
        ts = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        channel = f"comm-{direction}" if direction in LOG_PREFIX else "comm-info"
        self.console_output.emit(
            channel, f"[{ts}] {LOG_PREFIX.get(direction, '--')} [RFID] {message}")
