"""PnP coordinate preview, PCB capture, manual alignment and mask export page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pnp_mask_workflow import (
    PnpMaskError,
    build_pnp_preview,
    export_masks,
    fit_alignment,
    latest_small_global_capture,
    render_target_masks,
    transform_control_points,
)
from GUI.core.paths import PROJECT_ROOT, WORKING_DATA
from GUI.core.runner import TaskManager


PNP_BROWSE_ROOT = PROJECT_ROOT
DEFAULT_PNP_CANDIDATES = (
    PROJECT_ROOT / "P3S2_XW_L3_V1.1B_YJZZB(385).txt",
    PROJECT_ROOT.parent / "P3S2_XW_L3_V1.1B_YJZZB(385).txt",
)

MASK_COLORS = {
    "resistor": "#eb603a",
    "capacitor": "#3e96eb",
    "inductor": "#e6b446",
    "diode": "#58be5f",
    "transistor": "#be78c3",
    "ic": "#377de6",
    "connector": "#d7aa4b",
    "testpoint": "#bebebe",
    "fuse": "#dc50a0",
    "switch": "#82c85a",
    "other": "#46c8c8",
}


class EditableMaskItem(QGraphicsPolygonItem):
    """One movable component polygon with exportable scene coordinates."""

    def __init__(self, record: dict, opacity: float, changed: Callable):
        polygon = QPolygonF(
            [QPointF(float(x), float(y)) for x, y in record["target_polygon_px"]]
        )
        super().__init__(polygon)
        self.record = dict(record)
        self._changed = changed
        self._base_color = QColor(MASK_COLORS.get(record.get("category"), "#46c8c8"))
        self._label = QGraphicsSimpleTextItem(str(record.get("designator", "")), self)
        self._label.setBrush(QBrush(QColor("#ffffff")))
        self._label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._label.setAcceptedMouseButtons(Qt.NoButton)
        self._label.setZValue(2)
        self._position_label()
        self.setZValue(5)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setToolTip(str(record.get("designator", "人工新增框")))
        self.set_mask_opacity(opacity)
        self.set_editable(False)

    def _position_label(self) -> None:
        bounds = self.polygon().boundingRect()
        self._label.setPos(bounds.center() + QPointF(5.0, 5.0))

    def set_mask_opacity(self, opacity: float) -> None:
        color = QColor(self._base_color)
        color.setAlphaF(max(0.1, min(float(opacity), 0.8)))
        self.setBrush(QBrush(color))
        self._update_pen()

    def set_editable(self, enabled: bool) -> None:
        self.setFlag(QGraphicsItem.ItemIsSelectable, enabled)
        self.setFlag(QGraphicsItem.ItemIsMovable, enabled)
        if not enabled:
            self.setSelected(False)
        self._update_pen()

    def set_label_visible(self, visible: bool) -> None:
        self._label.setVisible(visible)

    def _update_pen(self) -> None:
        color = QColor("#ffd166") if self.isSelected() else QColor("#ffffff")
        pen = QPen(color, 3.0 if self.isSelected() else 1.8)
        pen.setCosmetic(True)
        self.setPen(pen)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._update_pen()
        elif change == QGraphicsItem.ItemPositionHasChanged and self.scene() is not None:
            self._changed()
        return result

    def export_record(self) -> dict:
        polygon = self.mapToScene(self.polygon())
        points = [[point.x(), point.y()] for point in polygon]
        center = np.mean(np.asarray(points, dtype=np.float64), axis=0)
        record = dict(self.record)
        record["target_polygon_px"] = points
        record["target_center_px"] = center.tolist()
        record["manually_edited"] = bool(
            self.pos().x() != 0.0 or self.pos().y() != 0.0 or record.get("manually_added")
        )
        return record


class ImagePointView(QGraphicsView):
    """Zoomable image that reports clicks in original image pixels."""

    image_clicked = Signal(float, float)
    mask_selection_changed = Signal(int)
    masks_changed = Signal()

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._marker_items = []
        self._mask_items: list[EditableMaskItem] = []
        self._mask_editing = False
        self._add_mask_mode = False
        self._draw_origin: QPointF | None = None
        self._draw_item: QGraphicsRectItem | None = None
        self._manual_counter = 0
        self._mask_opacity = 0.38
        self._show_mask_labels = True
        self._placeholder = placeholder
        self.setMinimumSize(320, 300)
        self._scene.selectionChanged.connect(self._selection_changed)
        self.set_placeholder(placeholder)

    def set_placeholder(self, text: str | None = None) -> None:
        self._mask_items = []
        self._scene.clear()
        self._pixmap_item = None
        self._marker_items = []
        self._mask_items = []
        self._mask_editing = False
        self._add_mask_mode = False
        label = self._scene.addText(text or self._placeholder)
        label.setDefaultTextColor(QColor("#98a2b8"))

    def load(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.set_placeholder(f"无法加载图像: {path.name}")
            return False
        self._mask_items = []
        self._scene.clear()
        self._marker_items = []
        self._mask_items = []
        self._mask_editing = False
        self._add_mask_mode = False
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_image()
        return True

    def set_mask_records(
        self,
        records: list[dict],
        opacity: float = 0.38,
        show_labels: bool = True,
    ) -> None:
        self.clear_masks()
        self._mask_opacity = opacity
        self._show_mask_labels = show_labels
        self._manual_counter = sum(bool(item.get("manually_added")) for item in records)
        for record in records:
            self._add_mask_item(record)
        self.masks_changed.emit()

    def _add_mask_item(self, record: dict) -> EditableMaskItem:
        item = EditableMaskItem(record, self._mask_opacity, self.masks_changed.emit)
        item.set_label_visible(self._show_mask_labels)
        self._scene.addItem(item)
        self._mask_items.append(item)
        return item

    def clear_masks(self) -> None:
        for item in self._mask_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._mask_items.clear()
        if self._draw_item is not None and self._draw_item.scene() is self._scene:
            self._scene.removeItem(self._draw_item)
        self._draw_item = None
        self._draw_origin = None
        self._mask_editing = False
        self._add_mask_mode = False
        self.mask_selection_changed.emit(0)

    def has_masks(self) -> bool:
        return bool(self._mask_items)

    def mask_records(self) -> list[dict]:
        return [item.export_record() for item in self._mask_items]

    def set_mask_editing(self, enabled: bool) -> None:
        self._mask_editing = enabled and bool(self._mask_items)
        if not self._mask_editing:
            self._add_mask_mode = False
        for item in self._mask_items:
            item.set_editable(self._mask_editing)
        self.viewport().setCursor(
            Qt.CrossCursor if self._mask_editing and self._add_mask_mode else Qt.ArrowCursor
        )

    def set_add_mask_mode(self, enabled: bool) -> None:
        self._add_mask_mode = enabled and self._mask_editing
        self.viewport().setCursor(Qt.CrossCursor if self._add_mask_mode else Qt.ArrowCursor)

    def set_mask_opacity(self, value: int | float) -> None:
        opacity = float(value) / 100.0 if float(value) > 1.0 else float(value)
        self._mask_opacity = opacity
        for item in self._mask_items:
            item.set_mask_opacity(opacity)

    def set_mask_labels_visible(self, visible: bool) -> None:
        self._show_mask_labels = visible
        for item in self._mask_items:
            item.set_label_visible(visible)

    def delete_selected_masks(self) -> int:
        selected = [item for item in self._mask_items if item.isSelected()]
        for item in selected:
            self._scene.removeItem(item)
            self._mask_items.remove(item)
        if selected:
            self.masks_changed.emit()
        return len(selected)

    def add_manual_mask(self, rect: QRectF) -> EditableMaskItem | None:
        rect = rect.normalized()
        if rect.width() < 3.0 or rect.height() < 3.0:
            return None
        self._manual_counter += 1
        record = {
            "instance_id": max(
                [int(item.record.get("instance_id", 0)) for item in self._mask_items]
                + [0]
            )
            + 1,
            "designator": f"MANUAL_{self._manual_counter:03d}",
            "comment": "manual mask",
            "footprint": "manual",
            "layer": "Manual",
            "category": "other",
            "target_polygon_px": [
                [rect.left(), rect.top()],
                [rect.right(), rect.top()],
                [rect.right(), rect.bottom()],
                [rect.left(), rect.bottom()],
            ],
            "target_center_px": [rect.center().x(), rect.center().y()],
            "visible": True,
            "manually_added": True,
        }
        item = self._add_mask_item(record)
        self._scene.clearSelection()
        item.set_editable(True)
        item.setSelected(True)
        self.masks_changed.emit()
        return item

    def _selection_changed(self) -> None:
        count = sum(item.isSelected() for item in self._mask_items)
        self.mask_selection_changed.emit(count)

    def set_markers(
        self,
        points: list[list[float]],
        color: str,
        pending: list[float] | None = None,
    ) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()
        if self._pixmap_item is None:
            return
        image_size = max(
            self._pixmap_item.pixmap().width(), self._pixmap_item.pixmap().height()
        )
        radius = max(5.0, image_size / 180.0)
        pen = QPen(QColor("#ffffff"), max(1.5, radius / 4.0))
        brush = QBrush(QColor(color))
        for index, point in enumerate(points, start=1):
            x, y = point
            circle = self._scene.addEllipse(
                x - radius, y - radius, radius * 2, radius * 2, pen, brush
            )
            circle.setZValue(10)
            text = self._scene.addSimpleText(str(index))
            text.setBrush(QBrush(QColor("#ffffff")))
            text.setPos(x + radius, y - radius * 1.8)
            text.setZValue(11)
            self._marker_items.extend((circle, text))
        if pending is not None:
            x, y = pending
            pending_pen = QPen(QColor("#ffd166"), max(2.0, radius / 3.0))
            circle = self._scene.addEllipse(
                x - radius * 1.2,
                y - radius * 1.2,
                radius * 2.4,
                radius * 2.4,
                pending_pen,
            )
            circle.setZValue(12)
            self._marker_items.append(circle)

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

    def mousePressEvent(self, event) -> None:
        if self._mask_editing:
            if event.button() == Qt.LeftButton and self._add_mask_mode:
                point = self.mapToScene(event.position().toPoint())
                image_rect = QRectF(self._pixmap_item.pixmap().rect())
                if image_rect.contains(point):
                    self._draw_origin = point
                    pen = QPen(QColor("#ffd166"), 3.0)
                    pen.setCosmetic(True)
                    self._draw_item = self._scene.addRect(QRectF(point, point), pen)
                    self._draw_item.setZValue(20)
                    event.accept()
                    return
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton and self._pixmap_item is not None:
            point = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._pixmap_item.pixmap().rect())
            if rect.contains(point):
                self.image_clicked.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._draw_origin is not None and self._draw_item is not None:
            point = self.mapToScene(event.position().toPoint())
            image_rect = QRectF(self._pixmap_item.pixmap().rect())
            point.setX(min(max(point.x(), image_rect.left()), image_rect.right()))
            point.setY(min(max(point.y(), image_rect.top()), image_rect.bottom()))
            self._draw_item.setRect(QRectF(self._draw_origin, point).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._draw_origin is not None:
            rect = self._draw_item.rect().normalized() if self._draw_item is not None else QRectF()
            if self._draw_item is not None:
                self._scene.removeItem(self._draw_item)
            self._draw_item = None
            self._draw_origin = None
            self.add_manual_mask(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._mask_editing and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_masks()
            event.accept()
            return
        super().keyPressEvent(event)


class WorkflowJob(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Callable, parent=None):
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            result = self.operation()
        except Exception as exc:  # reported in the page instead of losing the thread error
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class PnpMaskPage(QWidget):
    capture_requested = Signal()

    def __init__(self, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.session_dir = (
            WORKING_DATA
            / "pnp_mask_alignment"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.preview_path: Path | None = None
        self.components_path: Path | None = None
        self.stitched_path: Path | None = None
        self.capture_run_dir: Path | None = None
        self.components: list[dict] = []
        self.point_pairs: list[dict] = []
        self.pending_source: list[float] | None = None
        self.pending_designator = ""
        self.alignment: dict | None = None
        self.mask_preview_ready = False
        self._capture_pending = False
        self._job: WorkflowJob | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_file_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_source_panel())
        splitter.addWidget(self._build_target_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 620])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_alignment_panel())
        self.status_label = QLabel("先选择坐标文件并生成坐标图，然后拍摄或加载一轮 PCB 图像。")
        self.status_label.setObjectName("HintLabel")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.manager.task_started.connect(self._on_task_started)
        self.manager.task_finished.connect(self._on_task_finished)
        self.manager.busy_changed.connect(self._on_manager_busy)
        default_pnp = next((path for path in DEFAULT_PNP_CANDIDATES if path.is_file()), None)
        if default_pnp is not None:
            self.pnp_edit.setText(str(default_pnp))

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(3)
        title = QLabel("坐标 Mask 对齐")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "将 PnP 元器件坐标转换为布局图，拍摄并拼接 PCB；通过对应点把全部元器件 Mask 映射到实拍大图。"
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
        layout.addWidget(QLabel("坐标文件"))
        self.pnp_edit = QLineEdit()
        self.pnp_edit.setReadOnly(True)
        self.pnp_edit.setPlaceholderText("选择 Altium PnP .txt/.csv 文件")
        layout.addWidget(self.pnp_edit, 1)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse_pnp)
        layout.addWidget(browse)
        layout.addWidget(QLabel("PCB 面"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItem("正面（TopLayer）", "top")
        self.layer_combo.addItem("反面（BottomLayer）", "bottom")
        self.layer_combo.setToolTip("反面会按从 PCB 背面观察的方向水平镜像")
        self.layer_combo.currentIndexChanged.connect(self._board_side_changed)
        layout.addWidget(self.layer_combo)
        self.preview_button = QPushButton("生成坐标图")
        self.preview_button.clicked.connect(self._generate_preview)
        layout.addWidget(self.preview_button)
        self.capture_button = QPushButton("拍摄一轮")
        self.capture_button.setProperty("class", "primary")
        self.capture_button.clicked.connect(self._capture_once)
        layout.addWidget(self.capture_button)
        self.latest_button = QPushButton("加载最近拍摄")
        self.latest_button.clicked.connect(self._load_latest_capture)
        layout.addWidget(self.latest_button)
        return panel

    def _build_source_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(5)
        title = QLabel("1  坐标布局图：点击一个易识别元器件")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.source_view = ImagePointView("生成坐标图后显示在这里")
        self.source_view.image_clicked.connect(self._source_clicked)
        layout.addWidget(self.source_view, 1)
        return panel

    def _build_target_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.setSpacing(5)
        title = QLabel("2  实拍全局裁剪图：点击同一个元器件的位置")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.target_view = ImagePointView("拍摄或加载最近结果后显示在这里")
        self.target_view.image_clicked.connect(self._target_clicked)
        self.target_view.mask_selection_changed.connect(self._mask_selection_changed)
        self.target_view.masks_changed.connect(self._masks_changed)
        layout.addWidget(self.target_view, 1)
        return panel

    def _build_alignment_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Card")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)
        self.point_table = QTableWidget(0, 4)
        self.point_table.setHorizontalHeaderLabels(("# / 元件", "坐标图 x,y", "实拍图 x,y", "误差"))
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.point_table.setMaximumHeight(112)
        header = self.point_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        layout.addWidget(self.point_table, 1)

        controls = QVBoxLayout()
        row1 = QHBoxLayout()
        undo = QPushButton("撤销点")
        undo.clicked.connect(self._undo_point)
        clear = QPushButton("清空点")
        clear.clicked.connect(self._clear_points)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("局部相似（推荐）", "local_similarity")
        self.mode_combo.addItem("全局相似", "global_similarity")
        self.mode_combo.addItem("全局仿射", "global_affine")
        self.mode_combo.setToolTip(
            "局部相似：分散区域分别校正且保持元器件形状\n"
            "全局相似：整板统一旋转和等比例缩放\n"
            "全局仿射：允许横纵比例变化和剪切"
        )
        self.mode_combo.currentIndexChanged.connect(self._alignment_mode_changed)
        self.fit_button = QPushButton("拟合预览")
        self.fit_button.clicked.connect(self._fit_preview)
        row1.addWidget(undo)
        row1.addWidget(clear)
        row1.addWidget(self.mode_combo)
        row1.addWidget(self.fit_button)
        controls.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mask 透明度"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 80)
        self.opacity_slider.setValue(38)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.valueChanged.connect(self.target_view.set_mask_opacity)
        row2.addWidget(self.opacity_slider)
        self.label_check = QCheckBox("显示位号")
        self.label_check.setChecked(False)
        self.label_check.setToolTip("元器件较多时建议关闭，需要核对位号时再开启")
        self.label_check.toggled.connect(self.target_view.set_mask_labels_visible)
        row2.addWidget(self.label_check)
        self.mask_count_label = QLabel("0 个框")
        self.mask_count_label.setObjectName("HintLabel")
        row2.addWidget(self.mask_count_label)
        controls.addLayout(row2)

        row3 = QHBoxLayout()
        self.edit_mask_button = QPushButton("编辑框")
        self.edit_mask_button.setCheckable(True)
        self.edit_mask_button.setToolTip("开启后可选择并拖动 Mask；再次点击返回对应点模式")
        self.edit_mask_button.toggled.connect(self._toggle_mask_editing)
        self.add_mask_button = QPushButton("新增框")
        self.add_mask_button.setCheckable(True)
        self.add_mask_button.setToolTip("开启后在实拍图上拖出一个新的矩形 Mask")
        self.add_mask_button.toggled.connect(self._toggle_add_mask)
        self.delete_mask_button = QPushButton("删除选中")
        self.delete_mask_button.setToolTip("删除当前选中的框，也可按 Delete 或 Backspace")
        self.delete_mask_button.clicked.connect(self._delete_selected_masks)
        row3.addWidget(self.edit_mask_button)
        row3.addWidget(self.add_mask_button)
        row3.addWidget(self.delete_mask_button)
        controls.addLayout(row3)

        self.export_button = QPushButton("导出大图和 Mask")
        self.export_button.setProperty("class", "primary")
        self.export_button.clicked.connect(self._export)
        controls.addWidget(self.export_button)
        layout.addLayout(controls)
        return panel

    def _browse_pnp(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择元器件坐标文件",
            str(PNP_BROWSE_ROOT),
            "PnP 坐标文件 (*.txt *.csv);;所有文件 (*)",
        )
        if selected:
            self.pnp_edit.setText(selected)
            self._generate_preview()

    def _board_side_changed(self, _index: int = -1) -> None:
        if not hasattr(self, "source_view"):
            return
        self.preview_path = None
        self.components_path = None
        self.components = []
        self.source_view.set_placeholder("点击“生成坐标图”显示所选 PCB 面")
        self._clear_points()
        side = self.layer_combo.currentText()
        self._set_status(f"已选择{side}，请重新生成坐标图。")

    def _start_job(self, operation: Callable, done: Callable, message: str) -> None:
        if self._job is not None and self._job.isRunning():
            return
        self._set_controls_enabled(False)
        self._set_status(message)
        job = WorkflowJob(operation, self)
        self._job = job

        def succeeded(result) -> None:
            self._finish_job()
            done(result)

        job.succeeded.connect(succeeded)
        job.failed.connect(self._job_failed)
        job.finished.connect(job.deleteLater)
        job.start()

    def _finish_job(self) -> None:
        self._job = None
        self._set_controls_enabled(not self.manager.busy())

    def _job_failed(self, message: str) -> None:
        self._finish_job()
        self._set_status(f"处理失败：{message}", error=True)
        QMessageBox.critical(self, "处理失败", message)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.preview_button.setEnabled(enabled)
        self.latest_button.setEnabled(enabled)
        self.capture_button.setEnabled(enabled and not self.manager.busy())
        self.fit_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.edit_mask_button.setEnabled(enabled and self.mask_preview_ready)
        self.add_mask_button.setEnabled(
            enabled and self.mask_preview_ready and self.edit_mask_button.isChecked()
        )
        self.delete_mask_button.setEnabled(
            enabled
            and self.mask_preview_ready
            and self.edit_mask_button.isChecked()
            and bool(self.target_view.scene().selectedItems())
        )

    def _discard_mask_preview(self) -> None:
        self.mask_preview_ready = False
        self.target_view.clear_masks()
        for button in (self.edit_mask_button, self.add_mask_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        self.edit_mask_button.setEnabled(False)
        self.add_mask_button.setEnabled(False)
        self.delete_mask_button.setEnabled(False)
        self.mask_count_label.setText("0 个框")

    def _toggle_mask_editing(self, enabled: bool) -> None:
        if enabled and not self.mask_preview_ready:
            self.edit_mask_button.setChecked(False)
            return
        self.target_view.set_mask_editing(enabled)
        self.add_mask_button.setEnabled(enabled)
        if not enabled:
            self.add_mask_button.setChecked(False)
        self._mask_selection_changed(len(self.target_view.scene().selectedItems()))
        if self.mask_preview_ready:
            self._set_status(
                "Mask 编辑模式：可拖动、删除或新增框。"
                if enabled
                else "对应点模式：可继续在左右图添加配准点，重新拟合会重置框编辑。"
            )

    def _toggle_add_mask(self, enabled: bool) -> None:
        if enabled and not self.edit_mask_button.isChecked():
            self.add_mask_button.setChecked(False)
            return
        self.target_view.set_add_mask_mode(enabled)
        self.add_mask_button.setText("拖拽画框中" if enabled else "新增框")

    def _delete_selected_masks(self) -> None:
        count = self.target_view.delete_selected_masks()
        if count:
            self._set_status(f"已删除 {count} 个框；最终导出不会包含这些实例。")

    def _mask_selection_changed(self, count: int) -> None:
        self.delete_mask_button.setEnabled(
            self.mask_preview_ready and self.edit_mask_button.isChecked() and count > 0
        )

    def _masks_changed(self) -> None:
        if hasattr(self, "mask_count_label"):
            self.mask_count_label.setText(f"{len(self.target_view.mask_records())} 个框")

    def _generate_preview(self) -> None:
        pnp_path = Path(self.pnp_edit.text().strip())
        if not pnp_path.is_file():
            QMessageBox.warning(self, "未选择文件", "请先选择有效的 PnP 坐标文件。")
            return
        layer = str(self.layer_combo.currentData())
        self._start_job(
            lambda: build_pnp_preview(pnp_path, layer, self.session_dir),
            self._preview_ready,
            "正在解析坐标文件并生成布局图…",
        )

    def _preview_ready(self, result) -> None:
        self.preview_path, self.components_path = [Path(value) for value in result]
        document = json.loads(self.components_path.read_text(encoding="utf-8"))
        self.components = document.get("components", [])
        self.source_view.load(self.preview_path)
        self._clear_points()
        self._set_status(
            f"坐标图已生成：{len(self.components)} 个元器件。请拍摄一轮，或加载最近拍摄结果。"
        )

    def _capture_once(self) -> None:
        if self.manager.busy():
            QMessageBox.warning(self, "硬件忙", "相机或位移台正在执行其他任务。")
            return
        self._capture_pending = True
        self.capture_requested.emit()

    def _on_task_started(self, spec) -> None:
        if self._capture_pending and spec.id == "mosaic":
            self._set_status("正在拍摄一轮 PCB 图像；完成后将自动拼接大图…")

    def _on_task_finished(self, task_id: str, result: str, exit_code: int) -> None:
        if task_id != "mosaic" or not self._capture_pending:
            return
        self._capture_pending = False
        if exit_code != 0:
            self._set_status(f"拍摄未完成：{result}", error=True)
            return
        self._load_latest_capture()

    def _on_manager_busy(self, busy: bool) -> None:
        if self._job is None:
            self._set_controls_enabled(not busy)

    def _load_latest_capture(self) -> None:
        try:
            result = latest_small_global_capture()
        except PnpMaskError as exc:
            QMessageBox.warning(self, "没有全局裁剪图", str(exc))
            return
        if result is None:
            QMessageBox.warning(
                self, "没有拍摄结果", "pcb_two_zoom_capture 下没有拍摄轮次。"
            )
            return
        self._capture_image_ready(result)

    def _capture_image_ready(self, result) -> None:
        self._discard_mask_preview()
        self.capture_run_dir, self.stitched_path = [Path(value) for value in result]
        self.target_view.load(self.stitched_path)
        self._refresh_markers()
        self._set_status(
            f"已加载 {self.capture_run_dir.name}/small_zoom_image/"
            "small_global_cropped.png。依次在左、右图点击至少 3 组对应位置。"
        )

    def _source_clicked(self, x: float, y: float) -> None:
        if not self.components:
            return
        if self.mask_preview_ready:
            self._discard_mask_preview()
        point = np.asarray([x, y], dtype=np.float64)
        centers = np.asarray(
            [item["source_center_px"] for item in self.components], dtype=np.float64
        )
        nearest_index = int(np.argmin(np.linalg.norm(centers - point, axis=1)))
        nearest = self.components[nearest_index]
        self.pending_source = [float(value) for value in nearest["source_center_px"]]
        self.pending_designator = str(nearest["designator"])
        self._refresh_markers()
        self._set_status(
            f"已选择 {self.pending_designator}；现在点击右侧实拍图中的同一个元器件中心。"
        )

    def _target_clicked(self, x: float, y: float) -> None:
        if self.pending_source is None:
            self._set_status("请先点击左侧坐标图中的元器件。", error=True)
            return
        self.point_pairs.append(
            {
                "designator": self.pending_designator,
                "source": self.pending_source,
                "target": [float(x), float(y)],
            }
        )
        self.pending_source = None
        self.pending_designator = ""
        self.alignment = None
        self._refresh_markers()
        self._refresh_table()
        count = len(self.point_pairs)
        if count < 3:
            self._set_status(f"已添加 {count} 组对应点，还需要至少 {3 - count} 组。")
        else:
            self._set_status(f"已添加 {count} 组对应点，可以点击“拟合预览”。")

    def _refresh_markers(self) -> None:
        sources = [pair["source"] for pair in self.point_pairs]
        targets = [pair["target"] for pair in self.point_pairs]
        self.source_view.set_markers(sources, "#4d8dff", self.pending_source)
        self.target_view.set_markers(targets, "#ef5f5f")

    def _refresh_table(self) -> None:
        self.point_table.setRowCount(len(self.point_pairs))
        residuals = [None] * len(self.point_pairs)
        if self.alignment is not None and self.point_pairs:
            source = np.asarray([pair["source"] for pair in self.point_pairs])
            predicted = transform_control_points(source, self.alignment)
            target = np.asarray([pair["target"] for pair in self.point_pairs])
            residuals = np.linalg.norm(predicted - target, axis=1).tolist()
        for row, pair in enumerate(self.point_pairs):
            values = (
                f"{row + 1}  {pair['designator']}",
                f"{pair['source'][0]:.1f}, {pair['source'][1]:.1f}",
                f"{pair['target'][0]:.1f}, {pair['target'][1]:.1f}",
                "—" if residuals[row] is None else f"{residuals[row]:.1f} px",
            )
            for column, value in enumerate(values):
                self.point_table.setItem(row, column, QTableWidgetItem(value))

    def _undo_point(self) -> None:
        self._discard_mask_preview()
        if self.pending_source is not None:
            self.pending_source = None
            self.pending_designator = ""
        elif self.point_pairs:
            self.point_pairs.pop()
        self.alignment = None
        self._refresh_markers()
        self._refresh_table()
        self._set_status(f"当前有 {len(self.point_pairs)} 组对应点。")

    def _clear_points(self) -> None:
        self._discard_mask_preview()
        self.point_pairs.clear()
        self.pending_source = None
        self.pending_designator = ""
        self.alignment = None
        self._refresh_markers()
        self._refresh_table()

    def _alignment_mode_changed(self, _index: int = -1) -> None:
        self._discard_mask_preview()
        self.alignment = None
        self._refresh_table()
        if self.point_pairs:
            self._set_status("对齐方式已切换，请重新点击“拟合预览”。")

    def _calculate_alignment(self) -> dict:
        if len(self.point_pairs) < 3:
            raise ValueError("至少需要 3 组不共线的对应点。")
        return fit_alignment(
            [pair["source"] for pair in self.point_pairs],
            [pair["target"] for pair in self.point_pairs],
            str(self.mode_combo.currentData()),
        )

    def _fit_preview(self) -> None:
        if self.stitched_path is None or self.components_path is None:
            QMessageBox.warning(self, "缺少图像", "请先生成坐标图并加载实拍拼接图。")
            return
        try:
            alignment = self._calculate_alignment()
        except Exception as exc:
            QMessageBox.warning(self, "无法拟合", str(exc))
            return
        preview_dir = self.session_dir / "preview"
        opacity = self.opacity_slider.value() / 100.0
        draw_labels = self.label_check.isChecked()
        self._start_job(
            lambda: (
                alignment,
                export_masks(
                    self.stitched_path,
                    self.components_path,
                    alignment,
                    preview_dir,
                    opacity,
                    draw_labels,
                ),
            ),
            self._alignment_preview_ready,
            "正在计算对齐并生成 Mask 预览…",
        )

    def _alignment_preview_ready(self, result) -> None:
        self.alignment, outputs = result
        document = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))
        self.target_view.load(self.stitched_path)
        self.target_view.set_mask_records(
            document.get("components", []),
            self.opacity_slider.value() / 100.0,
            self.label_check.isChecked(),
        )
        self.mask_preview_ready = True
        self.edit_mask_button.setEnabled(True)
        self.edit_mask_button.setChecked(True)
        self._refresh_markers()
        self._refresh_table()
        source = np.asarray([pair["source"] for pair in self.point_pairs])
        predicted = transform_control_points(source, self.alignment)
        target = np.asarray([pair["target"] for pair in self.point_pairs])
        rmse = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
        self._set_status(f"Mask 预览已生成；对应点 RMSE = {rmse:.2f} px。确认后可导出。")

    def _export(self) -> None:
        if self.stitched_path is None or self.components_path is None:
            QMessageBox.warning(self, "缺少图像", "请先生成坐标图并加载实拍拼接图。")
            return
        try:
            alignment = self.alignment or self._calculate_alignment()
        except Exception as exc:
            QMessageBox.warning(self, "无法导出", str(exc))
            return
        output_dir = (self.capture_run_dir or self.session_dir) / "mask_alignment"
        opacity = self.opacity_slider.value() / 100.0
        draw_labels = self.label_check.isChecked()
        point_pairs = list(self.point_pairs)
        edited_records = self.target_view.mask_records() if self.mask_preview_ready else None

        def operation():
            if edited_records is not None:
                outputs = render_target_masks(
                    self.stitched_path,
                    edited_records,
                    output_dir,
                    opacity,
                    draw_labels,
                    alignment=alignment,
                    pnp_components=self.components_path,
                    manually_edited=True,
                )
            else:
                outputs = export_masks(
                    self.stitched_path,
                    self.components_path,
                    alignment,
                    output_dir,
                    opacity,
                    draw_labels,
                )
            (output_dir / "alignment_points.json").write_text(
                json.dumps(
                    {
                        "source_preview": str(self.preview_path),
                        "stitched_image": str(self.stitched_path),
                        "point_pairs": point_pairs,
                        "alignment": alignment,
                        "mask_editor_used": edited_records is not None,
                        "edited_instance_count": (
                            len(edited_records) if edited_records is not None else None
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return alignment, outputs, output_dir, edited_records is not None

        self._start_job(operation, self._export_ready, "正在导出叠加大图、二值 Mask 和实例 Mask…")

    def _export_ready(self, result) -> None:
        self.alignment, outputs, output_dir, editor_used = result
        if not editor_used:
            self.target_view.load(Path(outputs["overlay"]))
            self._refresh_markers()
            self._refresh_table()
        self._set_status(f"导出完成：{output_dir}")
        QMessageBox.information(
            self,
            "导出完成",
            "已生成：\n"
            "• 带彩色 Mask 的 PCB 大图\n"
            "• 二值 Mask\n"
            "• 16 位元器件实例 Mask\n"
            "• 元器件多边形和对齐参数 JSON\n\n"
            f"目录：{output_dir}",
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setStyleSheet("color: #ef5f5f;" if error else "")
        self.status_label.setText(text)
