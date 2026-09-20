"""Result viewing helpers: calibration summary table, JSON tree, image view."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

LENS_ENTRY_TYPES = (
    "lens_checkerboard_camera_calibration",
    "full_electric_lens_checkerboard_calibration",
)
BOARD_ENTRY_TYPE = "calibration_board_to_pixel_affine_core"
STAGE_ENTRY_TYPE = "stage_command_to_pixel_calibration"
ZOOM_TRANSFORM_ENTRY_TYPE = "zoom_pixel_homography_calibration"
HIDDEN_ENTRY_TYPES = frozenset((BOARD_ENTRY_TYPE,))

TYPE_LABELS = {entry_type: "相机内参标定" for entry_type in LENS_ENTRY_TYPES}
TYPE_LABELS.update(
    {
        BOARD_ENTRY_TYPE: "板→像素仿射标定",
        STAGE_ENTRY_TYPE: "位移台→像素标定",
        ZOOM_TRANSFORM_ENTRY_TYPE: "双 Zoom 像素变换",
    }
)


def visible_summary_document(document: Any) -> Any:
    """Return a GUI-only view without the hidden board calibration type."""
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        return document
    visible = dict(document)
    visible["entries"] = [
        entry
        for entry in document["entries"]
        if not isinstance(entry, dict) or entry.get("type") not in HIDDEN_ENTRY_TYPES
    ]
    return visible


def _mean_mm_per_pixel(pixel_to_board_matrix: Any) -> tuple[float, float] | None:
    try:
        columns_norm = []
        for column in range(2):
            dx = float(pixel_to_board_matrix[0][column])
            dy = float(pixel_to_board_matrix[1][column])
            columns_norm.append(math.hypot(dx, dy))
        return columns_norm[0], columns_norm[1]
    except (TypeError, ValueError, IndexError):
        return None


def _entry_headline(entry: dict[str, Any]) -> str:
    entry_type = entry.get("type")
    if entry_type in LENS_ENTRY_TYPES:
        parts = []
        for position in entry.get("positions", []):
            calibration = position.get("calibration") or position
            rms = calibration.get("rms_reprojection_error")
            views = calibration.get("views")
            name = position.get("name", "")
            if rms is not None:
                parts.append(f"{name}: RMS {float(rms):.4f} px × {views}")
        return "; ".join(parts) if parts else "—"
    if entry_type == BOARD_ENTRY_TYPE:
        scale = _mean_mm_per_pixel(entry.get("pixel_to_board", {}).get("matrix_2x3"))
        if scale:
            return f"像素比例 u={scale[0]:.6f} v={scale[1]:.6f} mm/px"
        return "—"
    if entry_type == STAGE_ENTRY_TYPE:
        section = entry.get("stage_command_to_pixel", {})
        try:
            matrix = section["matrix_2x2"]
            per_mm = [
                math.hypot(float(matrix[0][i]), float(matrix[1][i])) for i in range(2)
            ]
            rmse = section.get("fit_quality", {}).get("rmse_px")
            text = f"{0.5 * sum(per_mm):.3f} px/mm"
            if rmse is not None:
                text += f", RMSE {float(rmse):.4f} px"
            return text
        except (KeyError, TypeError, ValueError, IndexError):
            return "—"
    if entry_type == ZOOM_TRANSFORM_ENTRY_TYPE:
        try:
            source_zoom = entry["source_zoom"]
            target_zoom = entry["target_zoom"]
            text = f"{source_zoom} → {target_zoom}"
            source_scale = entry.get("source_pixel_scale", {}).get("mm_per_pixel")
            target_scale = entry.get("target_pixel_scale", {}).get("mm_per_pixel")
            if source_scale is not None and target_scale is not None:
                text += ", 像素长度 {}: {:.6f}, {}: {:.6f} mm/px".format(
                    source_zoom,
                    float(source_scale),
                    target_zoom,
                    float(target_scale),
                )
            quality = entry.get("fit_quality")
            if not quality:
                return text
            ratio = float(quality["inlier_ratio"])
            rmse = float(quality["forward_inliers"]["rmse_px"])
            return f"{text}, 内点 {ratio:.1%}, RMSE {rmse:.4f} px"
        except (KeyError, TypeError, ValueError):
            return "—"
    return "—"


class SummaryTable(QTableWidget):
    """calibrate.json entries: time / type / key numbers."""

    HEADERS = ("时间 (UTC)", "类型", "关键结果")

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.setWordWrap(False)

    def load(self, summary_path: Path) -> None:
        self.setRowCount(0)
        entries: list[dict[str, Any]] = []
        if summary_path.is_file():
            try:
                document = json.loads(summary_path.read_text(encoding="utf-8"))
                entries = visible_summary_document(document).get("entries", [])
            except (OSError, json.JSONDecodeError) as exc:
                self.setRowCount(1)
                item = QTableWidgetItem(f"读取失败: {exc}")
                item.setForeground(QBrush(QColor("#ef5f5f")))
                self.setItem(0, 2, item)
                return
        for entry in reversed(entries):
            if not isinstance(entry, dict):
                continue
            when = (
                entry.get("created_at_utc")
                or entry.get("calibrated_at_utc")
                or entry.get("saved_at_utc")
                or "—"
            )
            rows = [
                str(when),
                TYPE_LABELS.get(entry.get("type"), str(entry.get("type", "—"))),
                _entry_headline(entry),
            ]
            row_index = self.rowCount()
            self.insertRow(row_index)
            for column, text in enumerate(rows):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                if column < 2:
                    item.setForeground(QBrush(QColor("#98a2b8")))
                self.setItem(row_index, column, item)


class JsonTree(QTreeWidget):
    """Render parsed JSON as a three-column tree."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(("键", "值", "类型"))
        self.setAlternatingRowColors(True)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def load(self, path: Path) -> bool:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return self.load_document(document)

    def load_document(self, document: Any) -> bool:
        self.clear()
        self._add_node(self.invisibleRootItem(), "root", document, 0)
        self.expandToDepth(1)
        return True

    def _add_node(self, parent, name: str, value: Any, depth: int) -> None:
        if depth > 40:
            item = QTreeWidgetItem(parent, [name, "…", "深度限制"])
            return
        if isinstance(value, dict):
            item = QTreeWidgetItem(parent, [str(name), "", f"object({len(value)})"])
            for key, child in value.items():
                self._add_node(item, str(key), child, depth + 1)
        elif isinstance(value, list):
            item = QTreeWidgetItem(parent, [str(name), "", f"array({len(value)})"])
            for index, child in enumerate(value):
                self._add_node(item, f"[{index}]", child, depth + 1)
        else:
            item = QTreeWidgetItem(
                parent, [str(name), repr(value), type(value).__name__]
            )


class ImageView(QGraphicsView):
    """Zoomable image display, wheel to zoom, double-click to fit."""

    MIN_SCALE = 0.02
    MAX_SCALE = 32.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._info_text = ""
        self._fit_mode = True
        self.setPlaceholder("选择一个图像文件查看预览")

    def setPlaceholder(self, text: str) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._info_text = text
        self._fit_mode = True
        self._scene.addText(text, QFont("Microsoft YaHei UI", 10))

    def load(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.setPlaceholder(f"无法加载图像: {path.name}")
            return False
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._info_text = f"{path.name}  ({pixmap.width()}×{pixmap.height()} px)"
        self._fit_mode = True
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        return True

    def wheelEvent(self, event) -> None:
        if self._pixmap_item is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current = abs(self.transform().m11())
        target = current * factor
        if target < self.MIN_SCALE or target > self.MAX_SCALE:
            event.accept()
            return
        self._fit_mode = False
        self.scale(factor, factor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._pixmap_item is not None:
            self._fit_mode = True
            self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        if self._pixmap_item is not None and self._fit_mode:
            self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

    def info_text(self) -> str:
        return self._info_text


class SimpleCard(QWidget):
    """Rounded surface container used by the dashboard."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        layout.addLayout(self.body)
