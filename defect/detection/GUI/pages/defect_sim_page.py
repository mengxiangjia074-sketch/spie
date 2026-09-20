"""Defect simulation page: box a component and remove it to fake a missing part.

The page shows a captured local (large-zoom) PCB image.  The operator draws a
bounding box around one component and clicks "生成缺陷".  By default the boxed
region is removed locally with OpenCV inpainting (no API); optional OpenAI
image-editing backends (``images/edits`` and ``responses``) are also available
for higher-fidelity fills.  See detection/defect_sim.py for the backend.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSettings, QThread, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from defect_sim import (
    DEFAULT_ACTOR,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DefectSimConfig,
    check_codex_image_capability,
    simulate_defect,
)
from GUI.core.paths import WORKING_DATA
from GUI.core.runner import open_in_explorer
from GUI.widgets.forms import NoWheelSpinBox
from GUI.widgets.resultview import ImageView

DEFECT_OUTPUT_DIR = WORKING_DATA / "defect_simulation"
PCB_CAPTURE_ROOT = WORKING_DATA / "pcb_two_zoom_capture"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

SIZE_OPTIONS = ("auto", "1024x1024", "1536x1024", "1024x1536")
WIRE_OPTIONS = (
    ("ChatGPT Pro 模型生图（codex，默认，真正出图）", "codex"),
    ("Images Edits API (/v1/images/edits) — 中转站生图", "edits"),
    ("Responses API (/v1/responses) — 中转站生图", "responses"),
    ("本地 OpenCV 修复（无需 API，纹理填充兜底）", "inpaint"),
)

# 各修复方式的预计耗时提示（显示在进度条旁边，让操作员心里有数）
WIRE_ETA_TEXT = {
    "codex": "目标约 1 分钟",
    "chatgpt": "目标约 1 分钟",
    "image_gen": "目标约 1 分钟",
    "edits": "通常 30–60 秒",
    "responses": "通常 30–60 秒",
    "inpaint": "通常几秒",
    "local": "通常几秒",
    "opencv": "通常几秒",
}


def latest_local_image(capture_root: Path = PCB_CAPTURE_ROOT) -> Path | None:
    """Return the newest usable large-Zoom image from the latest capture run."""
    capture_root = Path(capture_root)
    if not capture_root.is_dir():
        return None
    try:
        runs = sorted(
            (path for path in capture_root.iterdir() if path.is_dir()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
    except OSError:
        return None

    for run_dir in runs:
        stitched = run_dir / "pcb_stitched.png"
        if stitched.is_file():
            return stitched
        images_dir = run_dir / "images"
        if not images_dir.is_dir():
            continue
        try:
            images = [
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if images:
                return max(images, key=lambda path: (path.stat().st_mtime_ns, path.name))
        except OSError:
            continue
    return None


class BoxSelectView(QGraphicsView):
    """Zoomable image where dragging the left button draws a selection box."""

    box_changed = Signal(float, float, float, float)  # x0, y0, x1, y1 (image px)
    box_cleared = Signal()

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._box_item: QGraphicsRectItem | None = None
        self._draw_item: QGraphicsRectItem | None = None
        self._draw_origin = None
        self._box: QRectF | None = None
        self._placeholder = placeholder
        self.setMinimumSize(320, 300)
        self.set_placeholder(placeholder)

    def set_placeholder(self, text: str) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._box_item = None
        self._draw_item = None
        self._draw_origin = None
        self._box = None
        self._scene.addText(text).setDefaultTextColor(QColor("#98a2b8"))

    def load(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.set_placeholder(f"无法加载图像: {path.name}")
            return False
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._box_item = None
        self._draw_item = None
        self._draw_origin = None
        self._box = None
        self.fit_image()
        return True

    def image_size(self) -> tuple[int, int] | None:
        if self._pixmap_item is None:
            return None
        pixmap = self._pixmap_item.pixmap()
        return pixmap.width(), pixmap.height()

    def current_box(self) -> tuple[float, float, float, float] | None:
        if self._box is None:
            return None
        return (
            self._box.left(),
            self._box.top(),
            self._box.right(),
            self._box.bottom(),
        )

    def clear_box(self) -> None:
        self._box = None
        self._remove_box_item()
        self.box_cleared.emit()

    def _remove_box_item(self) -> None:
        if self._box_item is not None and self._box_item.scene() is self._scene:
            self._scene.removeItem(self._box_item)
        self._box_item = None

    def _render_box(self, rect: QRectF, temporary: bool = False) -> None:
        if not temporary:
            self._remove_box_item()
        color = QColor("#ffd166")
        brush = QColor("#ffd166")
        brush.setAlpha(46)
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        item = self._scene.addRect(rect, pen, QBrush(brush))
        item.setZValue(15)
        if temporary:
            self._draw_item = item
        else:
            self._box_item = item

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return
        factor = 1.22 if event.angleDelta().y() > 0 else 1.0 / 1.22
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit_image()
        event.accept()

    def _image_rect(self) -> QRectF:
        return QRectF(self._pixmap_item.pixmap().rect())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            point = self.mapToScene(event.position().toPoint())
            if self._image_rect().contains(point):
                self._draw_origin = point
                self._remove_box_item()
                self._box = None
                self._render_box(QRectF(point, point), temporary=True)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._draw_origin is not None and self._draw_item is not None:
            point = self.mapToScene(event.position().toPoint())
            rect = self._image_rect()
            point.setX(min(max(point.x(), rect.left()), rect.right()))
            point.setY(min(max(point.y(), rect.top()), rect.bottom()))
            self._draw_item.setRect(QRectF(self._draw_origin, point).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._draw_origin is not None:
            rect = self._draw_item.rect().normalized() if self._draw_item is not None else QRectF()
            if self._draw_item is not None and self._draw_item.scene() is self._scene:
                self._scene.removeItem(self._draw_item)
            self._draw_item = None
            self._draw_origin = None
            if rect.width() >= 8.0 and rect.height() >= 8.0:
                self._box = rect
                self._render_box(rect)
                self.box_changed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())
            else:
                self._box = None
                self.box_cleared.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _Job(QThread):
    """Run one blocking operation off the UI thread."""

    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, operation: Callable, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            result = self.operation(self.progress.emit)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class DefectSimPage(QWidget):
    console_output = Signal(str, str)  # channel, formatted message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("LensDetect", "GUI")
        self.image_path: Path | None = None
        self.last_outputs: dict[str, Path] | None = None
        self._job: _Job | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_file_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_source_panel())
        splitter.addWidget(self._build_result_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 620])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_control_panel())

        # 进度显示区：生成时可见（不确定进度条 + 实时计时），让操作员知道没卡住
        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 不确定模式 → 滚动动画
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.hide()
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("HintLabel")
        self.progress_label.hide()
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_label)
        outer.addLayout(progress_row)

        self.status_label = QLabel("先加载局部图，然后在原图上拖拽框选要移除的元器件。")
        self.status_label.setObjectName("HintLabel")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._tick_progress)
        self._start_time = 0.0
        self._eta_text = ""

        self._restore_settings()
        # Defer disk access until MainWindow has connected console_output.
        load_latest = self._load_latest_image
        QTimer.singleShot(0, lambda: load_latest(report_missing=False))

    # ---- construction -----------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(3)
        title = QLabel("缺陷模拟")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "在放大局部图上框选一个元器件，调用 GPT 图像编辑 API 把框选区域抹去，"
            "生成“元器件缺失”的模拟缺陷图。"
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_file_bar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)
        layout.addWidget(QLabel("局部图"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("加载最近拍摄的局部图，或选择一个图像文件")
        layout.addWidget(self.path_edit, 1)
        self.latest_button = QPushButton("加载最近局部图")
        self.latest_button.clicked.connect(lambda: self._load_latest_image())
        layout.addWidget(self.latest_button)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse_image)
        layout.addWidget(browse)
        return panel

    def _build_source_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(5)
        title = QLabel("1  原图：拖拽框选要移除的元器件")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.source_view = BoxSelectView("加载局部图后显示在这里")
        self.source_view.box_changed.connect(self._box_changed)
        self.source_view.box_cleared.connect(self._box_cleared)
        layout.addWidget(self.source_view, 1)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.setSpacing(5)
        title = QLabel("2  模拟结果（元器件缺失）")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.result_view = ImageView()
        layout.addWidget(self.result_view, 1)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # 主控件：用户只需关心「修复方式 + 生成」
        main_row = QHBoxLayout()
        main_row.addWidget(QLabel("修复方式"))
        self.wire_combo = QComboBox()
        for label, value in WIRE_OPTIONS:
            self.wire_combo.addItem(label, value)
        self.wire_combo.setToolTip(
            "ChatGPT Pro 模型生图（codex）：调用已登录的 ChatGPT 图像生成模型真正修复，"
            "不依赖中转站 key，最符合「模型生图」需求；\n"
            "Images Edits API：用透明 Mask 指定框选区域（images/edits 接口，需中转站生图权限）；\n"
            "Responses API：把框选区域画红框后发送给模型抹除（需中转站生图权限）；\n"
            "本地 OpenCV 修复：直接抹掉框选区域，无需 API，纹理填充兜底。"
        )
        main_row.addWidget(self.wire_combo, 1)
        self.advanced_button = QPushButton("高级设置")
        self.advanced_button.clicked.connect(self._show_advanced_settings)
        main_row.addWidget(self.advanced_button)
        layout.addLayout(main_row)

        self.advanced_dialog = self._build_advanced_settings_dialog()

        action_row = QHBoxLayout()
        self.box_label = QLabel("未框选")
        self.box_label.setObjectName("HintLabel")
        action_row.addWidget(self.box_label, 1)
        clear_box = QPushButton("清除框选")
        clear_box.clicked.connect(self._clear_box)
        action_row.addWidget(clear_box)
        open_dir = QPushButton("打开输出目录")
        open_dir.clicked.connect(lambda: open_in_explorer(DEFECT_OUTPUT_DIR))
        action_row.addWidget(open_dir)
        self.generate_button = QPushButton("生成缺陷")
        self.generate_button.setProperty("class", "primary")
        self.generate_button.clicked.connect(self._generate)
        action_row.addWidget(self.generate_button)
        layout.addLayout(action_row)

        return panel

    def _build_advanced_settings_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("缺陷模拟 - 高级设置")
        dialog.setModal(True)
        dialog.resize(720, 600)
        dialog.setMinimumSize(560, 440)

        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 2)
        content_layout.setSpacing(10)

        service_group = QGroupBox("图像生成服务")
        service_form = QFormLayout(service_group)
        service_form.setLabelAlignment(Qt.AlignRight)
        service_form.setFormAlignment(Qt.AlignTop)
        service_form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-…")
        service_form.addRow("API Key", self.api_key_edit)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText(DEFAULT_BASE_URL)
        service_form.addRow("Base URL", self.base_url_edit)

        self.actor_edit = QLineEdit()
        self.actor_edit.setPlaceholderText(DEFAULT_ACTOR)
        service_form.addRow("Actor 头", self.actor_edit)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(DEFAULT_MODEL)
        service_form.addRow("模型", self.model_edit)

        capability_row = QWidget()
        capability_layout = QHBoxLayout(capability_row)
        capability_layout.setContentsMargins(0, 0, 0, 0)
        capability_layout.setSpacing(8)
        self.capability_button = QPushButton("检测生图权限")
        self.capability_button.clicked.connect(self._check_image_capability)
        capability_layout.addWidget(self.capability_button)
        self.capability_label = QLabel("尚未检测")
        self.capability_label.setObjectName("HintLabel")
        self.capability_label.setWordWrap(True)
        capability_layout.addWidget(self.capability_label, 1)
        service_form.addRow("Codex", capability_row)

        self.size_combo = QComboBox()
        self.size_combo.addItems(SIZE_OPTIONS)
        service_form.addRow("输出尺寸", self.size_combo)

        self.max_side_spin = NoWheelSpinBox()
        self.max_side_spin.setRange(256, 4096)
        self.max_side_spin.setSingleStep(64)
        self.max_side_spin.setSuffix(" px")
        service_form.addRow("缩放上限", self.max_side_spin)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setObjectName("DefectPromptEdit")
        self.prompt_edit.setPlaceholderText(DEFAULT_PROMPT)
        self.prompt_edit.setMinimumHeight(120)
        service_form.addRow("提示词", self.prompt_edit)

        local_group = QGroupBox("本地 OpenCV 修复")
        local_form = QFormLayout(local_group)
        local_form.setLabelAlignment(Qt.AlignRight)
        local_form.setFormAlignment(Qt.AlignTop)

        self.radius_spin = NoWheelSpinBox()
        self.radius_spin.setRange(1, 20)
        self.radius_spin.setValue(3)
        self.radius_spin.setSuffix(" px")
        local_form.addRow("修复半径", self.radius_spin)

        self.feather_spin = NoWheelSpinBox()
        self.feather_spin.setRange(0, 60)
        self.feather_spin.setValue(12)
        self.feather_spin.setSuffix(" px")
        local_form.addRow("边缘羽化", self.feather_spin)

        content_layout.addWidget(service_group)
        content_layout.addWidget(local_group)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        save_button = QPushButton("保存设置")
        save_button.setProperty("class", "primary")
        save_button.clicked.connect(self._save_and_close_advanced_settings)
        buttons.addWidget(save_button)
        outer.addLayout(buttons)
        return dialog

    def _show_advanced_settings(self) -> None:
        self.advanced_dialog.exec()

    def _save_and_close_advanced_settings(self) -> None:
        self._save_settings()
        self.advanced_dialog.accept()

    def _check_image_capability(self) -> None:
        self.capability_button.setEnabled(False)
        self.capability_label.setText("正在检测当前 Codex 会话…")
        self._run_job(
            lambda: check_codex_image_capability(),
            self._capability_ready,
            "生图权限检测失败",
        )

    def _capability_ready(self, result: tuple[bool, str]) -> None:
        available, message = result
        self.capability_label.setText(("可用：" if available else "不可用：") + message)
        self._set_status(message, error=not available)

    # ---- settings ---------------------------------------------------------

    def _restore_settings(self) -> None:
        self.api_key_edit.setText(str(self.settings.value("defect_sim/api_key", "")))
        self.base_url_edit.setText(
            str(self.settings.value("defect_sim/base_url", DEFAULT_BASE_URL))
        )
        self.actor_edit.setText(
            str(self.settings.value("defect_sim/actor", DEFAULT_ACTOR))
        )
        self.model_edit.setText(str(self.settings.value("defect_sim/model", DEFAULT_MODEL)))
        prompt = str(self.settings.value("defect_sim/prompt", DEFAULT_PROMPT))
        self.prompt_edit.setPlainText(prompt)
        wire = str(self.settings.value("defect_sim/wire_api", "codex"))
        index = self.wire_combo.findData(wire)
        self.wire_combo.setCurrentIndex(index if index >= 0 else 0)
        self.radius_spin.setValue(int(self.settings.value("defect_sim/radius", 3)))
        self.feather_spin.setValue(int(self.settings.value("defect_sim/feather", 12)))
        size = str(self.settings.value("defect_sim/size", "auto"))
        index = self.size_combo.findText(size)
        self.size_combo.setCurrentIndex(index if index >= 0 else 0)
        self.max_side_spin.setValue(int(self.settings.value("defect_sim/max_side", 2048)))

    def _save_settings(self) -> None:
        self.settings.setValue("defect_sim/api_key", self.api_key_edit.text().strip())
        self.settings.setValue("defect_sim/base_url", self.base_url_edit.text().strip())
        self.settings.setValue("defect_sim/actor", self.actor_edit.text().strip())
        self.settings.setValue("defect_sim/model", self.model_edit.text().strip())
        self.settings.setValue("defect_sim/prompt", self.prompt_edit.toPlainText())
        self.settings.setValue("defect_sim/wire_api", self.wire_combo.currentData())
        self.settings.setValue("defect_sim/radius", self.radius_spin.value())
        self.settings.setValue("defect_sim/feather", self.feather_spin.value())
        self.settings.setValue("defect_sim/size", self.size_combo.currentText())
        self.settings.setValue("defect_sim/max_side", self.max_side_spin.value())

    def _build_config(self) -> DefectSimConfig:
        return DefectSimConfig(
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip() or DEFAULT_BASE_URL,
            model=self.model_edit.text().strip() or DEFAULT_MODEL,
            prompt=self.prompt_edit.toPlainText().strip() or DEFAULT_PROMPT,
            wire_api=str(self.wire_combo.currentData()),
            size=self.size_combo.currentText(),
            actor_authorization=self.actor_edit.text().strip(),
            max_side=self.max_side_spin.value(),
            inpaint_radius=self.radius_spin.value(),
            inpaint_feather=self.feather_spin.value(),
        )

    # ---- image loading ----------------------------------------------------

    def _browse_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择局部图",
            str(WORKING_DATA),
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if selected:
            self._set_image(Path(selected))

    def _load_latest_image(self, report_missing: bool = True) -> None:
        image_path = latest_local_image()
        if image_path is None:
            if report_missing:
                self._set_status(
                    "未找到最近的拍摄结果，请点击“选择…”手动打开局部图。",
                    error=True,
                )
            return
        self._set_image(image_path)

    def _set_image(self, path: Path) -> None:
        if not path.is_file():
            self._set_status(f"图像不存在: {path}", error=True)
            return
        if not self.source_view.load(path):
            self._set_status(f"无法加载图像: {path}", error=True)
            return
        self.image_path = path
        self.path_edit.setText(str(path))
        self.result_view.setPlaceholder("点击“生成缺陷”后在这里显示结果")
        self.last_outputs = None
        self._box_cleared()
        size = self.source_view.image_size()
        size_text = f"{size[0]}×{size[1]} px" if size else "未知尺寸"
        self._set_status(
            f"已加载：{path.name}（{size_text}）。滚轮放大，拖拽框选要移除的元器件。"
        )

    # ---- box / generate ---------------------------------------------------

    def _box_changed(self, x0: float, y0: float, x1: float, y1: float) -> None:
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        self.box_label.setText(f"框选区域: ({x0:.0f}, {y0:.0f}) → ({x1:.0f}, {y1:.0f})  {width:.0f}×{height:.0f} px")

    def _box_cleared(self) -> None:
        self.box_label.setText("未框选")

    def _clear_box(self) -> None:
        self.source_view.clear_box()

    def _generate(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        if self.image_path is None:
            message = "请先加载局部图。"
            self._set_status(message, error=True)
            return
        box = self.source_view.current_box()
        if box is None:
            message = "请先在原图上拖拽框选要移除的元器件。"
            self._set_status(message, error=True)
            return
        self._save_settings()
        cfg = self._build_config()
        wire = str(self.wire_combo.currentData())
        is_local = wire in ("inpaint", "local", "opencv")
        is_codex = wire in ("codex", "chatgpt", "image_gen")
        needs_key = not is_local and not is_codex
        if needs_key and not cfg.api_key:
            message = "当前图像编辑方式需要 API Key，请在高级设置中填写。"
            self._set_status(message, error=True)
            return
        DEFECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        image_path = self.image_path
        self._set_controls_enabled(False)
        self._start_progress(wire)
        self._run_job(
            lambda progress=None: simulate_defect(
                image_path, box, DEFECT_OUTPUT_DIR, cfg, on_progress=progress
            ),
            self._generate_ready,
            "缺陷模拟失败",
        )

    def _generate_ready(self, outputs: dict) -> None:
        self.last_outputs = outputs
        self.result_view.load(outputs["edited"])
        mm, ss = divmod(self._elapsed_seconds(), 60)
        self._set_status(f"缺陷模拟完成（用时 {mm:02d}:{ss:02d}）：{outputs['edited']}")

    # ---- job plumbing -----------------------------------------------------

    def _run_job(self, operation: Callable, done: Callable, fail_prefix: str) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self._set_controls_enabled(False)
        job = _Job(operation, self)
        self._job = job

        def succeeded(result) -> None:
            self._finish_job()
            done(result)

        job.succeeded.connect(succeeded)
        job.failed.connect(lambda message: self._job_failed(fail_prefix, message))
        job.progress.connect(self._on_progress)
        job.finished.connect(job.deleteLater)
        job.start()

    def _on_progress(self, message: str) -> None:
        self._set_status(message)

    def _finish_job(self) -> None:
        self._job = None
        self._set_controls_enabled(True)
        self._stop_progress()

    def _job_failed(self, prefix: str, message: str) -> None:
        self._finish_job()
        self._set_status(f"{prefix} {message}", error=True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.latest_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        if hasattr(self, "capability_button"):
            self.capability_button.setEnabled(enabled)

    def _set_status(self, text: str, error: bool = False) -> None:
        if error:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.console_output.emit(
                "comm-error", f"[{timestamp}] !! [缺陷模拟] {text}"
            )
            return
        self.status_label.setStyleSheet("")
        self.status_label.setText(text)

    # ---- progress display -------------------------------------------------

    def _start_progress(self, wire: str) -> None:
        self._start_time = time.time()
        self._eta_text = WIRE_ETA_TEXT.get(wire, "请稍候")
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_label.setText(f"已用 00:00 · {self._eta_text}")
        self._progress_timer.start()

    def _tick_progress(self) -> None:
        mm, ss = divmod(self._elapsed_seconds(), 60)
        self.progress_label.setText(f"已用 {mm:02d}:{ss:02d} · {self._eta_text}")

    def _elapsed_seconds(self) -> int:
        return max(0, int(time.time() - self._start_time))

    def _stop_progress(self) -> None:
        self._progress_timer.stop()
        self.progress_bar.hide()
        self.progress_label.hide()
