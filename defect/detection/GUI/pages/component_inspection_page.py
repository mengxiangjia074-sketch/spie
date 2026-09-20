"""GUI page for the end-to-end PCB component completeness algorithm."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from component_inspection.config import (
    InspectionConfig,
    latest_capture_source,
    latest_ground_truth,
)
from component_inspection.ground_truth import load_ground_truth
from component_inspection.pipeline import default_output_directory, run_component_inspection
from GUI.core.paths import PROJECT_ROOT
from GUI.core.runner import TaskManager, open_in_explorer
from GUI.widgets.resultview import ImageView


class InspectionJob(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            result = self.operation(self.progress.emit)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class ComponentInspectionPage(QWidget):
    console_output = Signal(str, str)
    capture_requested = Signal()

    def __init__(self, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._job: InspectionJob | None = None
        self._result: dict | None = None
        self._output_dir: Path | None = None
        self._capture_pending = False
        self._settings = QSettings("LensDetect", "ComponentInspection")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_inputs())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._preview_panel("坐标 Mask 真值", "选择 component_masks.json", "truth"))
        splitter.addWidget(self._preview_panel("完整性检测结果", "运行后显示判定大图", "result"))
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 620])
        outer.addWidget(splitter, 1)

        summary = QFrame()
        summary.setObjectName("Card")
        summary_layout = QHBoxLayout(summary)
        summary_layout.setContentsMargins(12, 8, 12, 8)
        self.verdict_label = QLabel("尚未检测")
        self.verdict_label.setObjectName("SectionTitle")
        self.count_label = QLabel("存在 0  |  缺失 0  |  未覆盖 0")
        self.count_label.setObjectName("HintLabel")
        self.count_label.setMinimumWidth(280)
        self.open_button = QPushButton("打开结果目录")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        summary_layout.addWidget(self.verdict_label)
        summary_layout.addSpacing(16)
        summary_layout.addWidget(self.count_label)
        summary_layout.addStretch(1)
        summary_layout.addWidget(self.open_button)
        outer.addWidget(summary)

        self.status_label = QLabel("选择坐标 Mask 真值和实际 PCB 的高倍检查图。")
        self.status_label.setObjectName("HintLabel")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.manager.busy_changed.connect(self._manager_busy_changed)
        self.manager.task_started.connect(self._on_task_started)
        self.manager.task_finished.connect(self._on_task_finished)
        self._load_defaults()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(3)
        title = QLabel("元器件完整性检测")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "SAM3 从高倍图提取候选，分类头筛出元器件，RoMaV2 将位置映射到坐标 Mask 真值大图并判定缺件。"
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_inputs(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QGridLayout(panel)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(7)

        layout.addWidget(QLabel("坐标 Mask 真值"), 0, 0)
        self.truth_edit = QLineEdit()
        self.truth_edit.setPlaceholderText("mask_alignment/component_masks.json")
        self.truth_edit.editingFinished.connect(self._truth_changed)
        layout.addWidget(self.truth_edit, 0, 1, 1, 4)
        truth_browse = QPushButton("选择…")
        truth_browse.clicked.connect(self._browse_truth)
        layout.addWidget(truth_browse, 0, 5)
        latest_truth = QPushButton("最近真值")
        latest_truth.clicked.connect(self._use_latest_truth)
        layout.addWidget(latest_truth, 0, 6)

        layout.addWidget(QLabel("高倍检查输入"), 1, 0)
        self.inspection_edit = QLineEdit()
        self.inspection_edit.setPlaceholderText("单张高倍图、images 目录或整轮拍摄目录")
        layout.addWidget(self.inspection_edit, 1, 1, 1, 4)
        image_browse = QPushButton("单张图…")
        image_browse.clicked.connect(self._browse_image)
        layout.addWidget(image_browse, 1, 5)
        directory_browse = QPushButton("拍摄目录…")
        directory_browse.clicked.connect(self._browse_directory)
        layout.addWidget(directory_browse, 1, 6)

        layout.addWidget(QLabel("提示词"), 2, 0)
        self.prompt_edit = QLineEdit("component")
        self.prompt_edit.setMaximumWidth(180)
        layout.addWidget(self.prompt_edit, 2, 1)
        layout.addWidget(QLabel("SAM 阈值"), 2, 2)
        self.sam_threshold = self._threshold_spin(0.50)
        layout.addWidget(self.sam_threshold, 2, 3)
        layout.addWidget(QLabel("分类阈值"), 2, 4)
        self.classifier_threshold = self._threshold_spin(0.50)
        layout.addWidget(self.classifier_threshold, 2, 5)
        self.roma_setting = QComboBox()
        self.roma_setting.addItem("精确", "precise")
        self.roma_setting.addItem("基础", "base")
        self.roma_setting.addItem("快速", "fast")
        self.roma_setting.addItem("极速", "turbo")
        self.roma_setting.setToolTip("精确模式配准最稳，但显存占用和耗时最高")
        layout.addWidget(self.roma_setting, 2, 6)

        self.run_button = QPushButton("开始完整性检测")
        self.run_button.setProperty("class", "primary")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button, 3, 5, 1, 2)
        self.latest_capture_button = QPushButton("使用最近拍摄")
        self.latest_capture_button.clicked.connect(self._use_latest_capture)
        layout.addWidget(self.latest_capture_button, 3, 4)
        self.capture_button = QPushButton("拍摄一轮")
        self.capture_button.setToolTip("启动两倍率 PCB 拍摄，完成后自动载入高倍图")
        self.capture_button.clicked.connect(self._capture_once)
        layout.addWidget(self.capture_button, 3, 3)
        return panel

    @staticmethod
    def _threshold_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 0.99)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        spin.setFixedWidth(82)
        return spin

    def _preview_panel(self, title: str, placeholder: str, name: str) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        view = ImageView()
        view.setPlaceholder(placeholder)
        view.setMinimumSize(360, 300)
        setattr(self, f"{name}_view", view)
        layout.addWidget(label)
        layout.addWidget(view, 1)
        return panel

    def _load_defaults(self) -> None:
        truth = latest_ground_truth()
        capture = latest_capture_source()
        self.truth_edit.setText(str(truth) if truth is not None else "")
        self.inspection_edit.setText(str(capture) if capture is not None else "")
        self.truth_edit.setCursorPosition(0)
        self.inspection_edit.setCursorPosition(0)
        self.prompt_edit.setText(str(self._settings.value("prompt", "component")))
        self.sam_threshold.setValue(float(self._settings.value("sam_threshold", 0.50)))
        self.classifier_threshold.setValue(
            float(self._settings.value("classifier_threshold", 0.50))
        )
        setting = str(self._settings.value("roma_setting", "precise"))
        index = self.roma_setting.findData(setting)
        if index >= 0:
            self.roma_setting.setCurrentIndex(index)
        self._truth_changed()

    def _browse_truth(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择坐标 Mask 真值",
            str(PROJECT_ROOT / "working_data"),
            "坐标 Mask 元数据 (component_masks.json);;JSON 文件 (*.json)",
        )
        if selected:
            self.truth_edit.setText(selected)
            self.truth_edit.setCursorPosition(0)
            self._truth_changed()

    def _use_latest_truth(self) -> None:
        path = latest_ground_truth()
        if path is None:
            QMessageBox.warning(self, "没有真值", "尚未找到坐标 Mask 对齐导出结果。")
            return
        self.truth_edit.setText(str(path))
        self.truth_edit.setCursorPosition(0)
        self._truth_changed()

    def _browse_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择高倍检查图",
            str(PROJECT_ROOT / "working_data"),
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if selected:
            self.inspection_edit.setText(selected)
            self.inspection_edit.setCursorPosition(0)

    def _browse_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择高倍图或整轮拍摄目录", str(PROJECT_ROOT / "working_data")
        )
        if selected:
            self.inspection_edit.setText(selected)
            self.inspection_edit.setCursorPosition(0)

    def _use_latest_capture(self) -> None:
        path = latest_capture_source()
        if path is None:
            QMessageBox.warning(self, "没有拍摄结果", "尚未找到两倍率 PCB 拍摄结果。")
            return
        self._set_inspection_source(path)

    def _capture_once(self) -> None:
        if self.is_running():
            QMessageBox.warning(
                self,
                "完整性检测正在运行",
                "请等待当前完整性检测结束后再启动拍摄。",
            )
            return
        if self.manager.busy():
            QMessageBox.warning(self, "硬件忙", "相机或位移台正在执行其他任务。")
            return
        self._capture_pending = True
        self.capture_requested.emit()
        # The signal is handled synchronously. If the application rejected the
        # task (for example, a device control page still owns the hardware), do
        # not mistake a later mosaic task for this request.
        if not self.manager.busy():
            self._capture_pending = False

    def _on_task_started(self, spec) -> None:
        if self._capture_pending and spec.id == "mosaic":
            self._set_status("正在执行两倍率 PCB 拍摄；完成后将自动载入高倍图目录…")

    def _on_task_finished(self, task_id: str, result: str, exit_code: int) -> None:
        if task_id != "mosaic" or not self._capture_pending:
            return
        self._capture_pending = False
        if exit_code != 0:
            self._set_status(f"拍摄未完成：{result}", error=True)
            return
        path = latest_capture_source()
        if path is None:
            self._set_status("拍摄完成，但未找到本轮高倍图目录。", error=True)
            return
        self._set_inspection_source(path)
        self._set_status(f"拍摄完成，已载入 {path.name}；可以开始完整性检测。")

    def _set_inspection_source(self, path: str | Path) -> None:
        self.inspection_edit.setText(str(path))
        self.inspection_edit.setCursorPosition(0)

    def _truth_changed(self) -> None:
        value = self.truth_edit.text().strip()
        if not value:
            return
        try:
            truth = load_ground_truth(value)
        except Exception as exc:
            self.truth_view.setPlaceholder(f"无法加载真值: {exc}")
            return
        self.truth_view.load(truth.mask_overlay_image)
        self._set_status(f"已加载坐标 Mask 真值：{len(truth.components)} 个元器件。")

    def _run(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        truth = self.truth_edit.text().strip()
        inspection = self.inspection_edit.text().strip()
        if not truth or not inspection:
            QMessageBox.warning(self, "输入不完整", "请选择坐标 Mask 真值和高倍检查输入。")
            return
        output_dir = default_output_directory()
        config = InspectionConfig(
            prompt=self.prompt_edit.text().strip() or "component",
            sam_confidence_threshold=self.sam_threshold.value(),
            classifier_threshold=self.classifier_threshold.value(),
            roma_setting=str(self.roma_setting.currentData()),
        )
        self._settings.setValue("prompt", config.prompt)
        self._settings.setValue("sam_threshold", config.sam_confidence_threshold)
        self._settings.setValue("classifier_threshold", config.classifier_threshold)
        self._settings.setValue("roma_setting", config.roma_setting)
        self._result = None
        self._output_dir = output_dir
        self.result_view.setPlaceholder("算法运行中…")
        self.verdict_label.setText("检测中")
        self.count_label.setText("正在加载模型")
        self.open_button.setEnabled(False)
        self._set_running(True)

        def operation(progress):
            return run_component_inspection(
                ground_truth_metadata=truth,
                inspection_source=inspection,
                output_directory=output_dir,
                config=config,
                progress=progress,
            )

        job = InspectionJob(operation, self)
        self._job = job
        job.progress.connect(self._progress)
        job.succeeded.connect(self._finished)
        job.failed.connect(self._failed)
        job.finished.connect(job.deleteLater)
        job.start()

    def _progress(self, message: str) -> None:
        self._set_status(message)
        self.console_output.emit("stdout", message)

    def _finished(self, result: dict) -> None:
        self._job = None
        self._result = result
        self._set_running(False)
        counts = result["counts"]
        labels = {
            "complete": "器件完整",
            "incomplete": "发现缺件",
            "coverage_incomplete": "检查范围未覆盖整板",
        }
        self.verdict_label.setText(labels[result["status"]])
        self.count_label.setText(
            f"存在 {counts['present']}  |  缺失 {counts['missing']}  |  "
            f"未覆盖 {counts['uninspected']}"
        )
        overlay = Path(result["outputs"]["component_completeness_overlay"])
        self.result_view.load(overlay)
        self.open_button.setEnabled(True)
        self._set_status(f"检测完成：{result['outputs']['report_json']}")
        if result["status"] == "incomplete":
            missing = "、".join(result["missing_ids"][:20])
            suffix = "…" if len(result["missing_ids"]) > 20 else ""
            QMessageBox.warning(self, "发现缺件", f"缺失元器件：{missing}{suffix}")
        elif result["status"] == "coverage_incomplete":
            QMessageBox.information(
                self,
                "检查范围不完整",
                "已检查区域内没有发现缺件，但高倍图片未覆盖全部真值元器件。",
            )
        else:
            QMessageBox.information(self, "检测完成", "所有真值元器件均已检查且存在。")

    def _failed(self, message: str) -> None:
        self._job = None
        self._set_running(False)
        self.verdict_label.setText("检测失败")
        self.count_label.setText("未生成有效检测结果")
        self.result_view.setPlaceholder("检测失败，请查看运行控制台")
        self._set_status(f"检测失败：{message}", error=True)
        self.console_output.emit("comm-error", message)
        QMessageBox.critical(self, "检测失败", message)

    def _set_running(self, running: bool) -> None:
        enabled = not running and not self.manager.busy()
        self.run_button.setEnabled(enabled)
        self.latest_capture_button.setEnabled(enabled)
        self.capture_button.setEnabled(enabled)

    def _manager_busy_changed(self, busy: bool) -> None:
        self._update_action_buttons()

    def _update_action_buttons(self) -> None:
        enabled = not self.manager.busy() and not self.is_running()
        self.run_button.setEnabled(enabled)
        self.latest_capture_button.setEnabled(enabled)
        self.capture_button.setEnabled(enabled)

    def _open_output(self) -> None:
        if self._output_dir is not None:
            open_in_explorer(self._output_dir)

    def is_running(self) -> bool:
        return self._job is not None and self._job.isRunning()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setStyleSheet("color: #ef5f5f;" if error else "")
        self.status_label.setText(text)
