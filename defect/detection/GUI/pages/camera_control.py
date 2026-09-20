"""Interactive camera preview, capture, recording, and UVC property controls."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from GUI.core.camera_worker import (
    CameraWorker,
    exposure_backend_value,
    exposure_milliseconds,
)
from GUI.core.paths import PROJECT_ROOT
from GUI.core.runner import TaskManager, open_in_explorer
from GUI.widgets.forms import NoWheelDoubleSpinBox


OUTPUT_ROOT = PROJECT_ROOT / "working_data" / "camera_control"
RESOLUTIONS = (
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 480),
)

COLOR_PROPERTIES = ("hue", "saturation", "contrast", "gamma")
EXPOSURE_PROPERTIES = ("exposure", "exposure_target")
WHITE_BALANCE_PROPERTIES = ("white_balance", "white_balance_red")
CAMSPC_EXPOSURE_BACKEND_LEVELS = tuple(range(-4, 1))
CAMSPC_EXPOSURE_TIME_LEVELS_MS = tuple(
    exposure_milliseconds(level) for level in CAMSPC_EXPOSURE_BACKEND_LEVELS
)
PREVIEW_ZOOM_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


class EqualWidthButtonRow(QWidget):
    """A responsive row that gives every button the exact same pixel width."""

    def __init__(self, buttons: tuple[QPushButton, ...], parent=None):
        super().__init__(parent)
        self._buttons = buttons
        self._spacing = 6
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for button in self._buttons:
            button.setParent(self)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        if not self._buttons:
            return QSize()
        width = max(button.sizeHint().width() for button in self._buttons)
        height = max(button.sizeHint().height() for button in self._buttons)
        return QSize(
            width * len(self._buttons) + self._spacing * (len(self._buttons) - 1),
            height,
        )

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.sizeHint().height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        count = len(self._buttons)
        if count == 0:
            return
        available = max(0, self.width() - self._spacing * (count - 1))
        button_width = available // count
        used = button_width * count + self._spacing * (count - 1)
        x = (self.width() - used) // 2
        for button in self._buttons:
            button.setGeometry(x, 0, button_width, self.height())
            x += button_width + self._spacing


class PropertySlider(QWidget):
    value_committed = Signal(str, float)

    def __init__(
        self,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        *,
        step: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.key = key
        self._factor = 10 ** decimals

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 4)
        layout.setSpacing(3)
        header = QHBoxLayout()
        self.label = QLabel(label)
        self.value_spin = NoWheelDoubleSpinBox()
        self.value_spin.setRange(minimum, maximum)
        self.value_spin.setDecimals(decimals)
        self.value_spin.setSingleStep(step)
        self.value_spin.setSuffix(suffix)
        self.value_spin.setFixedWidth(96)
        header.addWidget(self.label)
        header.addStretch(1)
        header.addWidget(self.value_spin)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(
            int(round(minimum * self._factor)),
            int(round(maximum * self._factor)),
        )
        self.slider.setSingleStep(max(1, int(round(step * self._factor))))
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._slider_value_changed)
        self.slider.sliderReleased.connect(self._slider_released)
        self.value_spin.editingFinished.connect(self._spin_committed)

    def _slider_value_changed(self, value: int) -> None:
        if self.slider.isSliderDown():
            return
        with QSignalBlocker(self.value_spin):
            self.value_spin.setValue(value / self._factor)
        self._emit_value()

    def _slider_released(self) -> None:
        with QSignalBlocker(self.value_spin):
            self.value_spin.setValue(self.slider.value() / self._factor)
        self._emit_value()

    def _spin_committed(self) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(round(self.value_spin.value() * self._factor)))
        self._emit_value()

    def _emit_value(self) -> None:
        self.value_committed.emit(self.key, self.value_spin.value())

    def set_value(self, value: float) -> None:
        value = min(self.value_spin.maximum(), max(self.value_spin.minimum(), value))
        with QSignalBlocker(self.value_spin), QSignalBlocker(self.slider):
            self.value_spin.setValue(value)
            self.slider.setValue(int(round(value * self._factor)))

    def set_supported(self, supported: bool, reason: str = "") -> None:
        self.setEnabled(supported)
        tooltip = reason if reason else ""
        self.setToolTip(tooltip)
        self.label.setToolTip(tooltip)


class ExposureTimeSlider(PropertySlider):
    """Discrete CamSPC exposure steps displayed as physical milliseconds."""

    def __init__(self, parent=None):
        self._levels = CAMSPC_EXPOSURE_TIME_LEVELS_MS
        super().__init__(
            "exposure",
            "曝光时间",
            0,
            len(self._levels) - 1,
            parent=parent,
        )
        self.value_spin.setRange(self._levels[0], self._levels[-1])
        self.value_spin.setDecimals(2)
        self.value_spin.setSingleStep(self._levels[0])
        self.value_spin.setSuffix(" ms")
        self.slider.setRange(0, len(self._levels) - 1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self._level_tooltip = (
            "CamSPC 支持的曝光档位: "
            + ", ".join(f"{value:g} ms" for value in self._levels)
        )
        self.setToolTip(self._level_tooltip)
        self.label.setToolTip(self._level_tooltip)
        self.set_value(self._levels[0])

    def _nearest_index(self, milliseconds: float) -> int:
        backend_value = exposure_backend_value(milliseconds)
        if not math.isfinite(backend_value):
            return 0
        return min(
            range(len(CAMSPC_EXPOSURE_BACKEND_LEVELS)),
            key=lambda index: abs(
                CAMSPC_EXPOSURE_BACKEND_LEVELS[index] - backend_value
            ),
        )

    def _set_index(self, index: int) -> None:
        index = max(0, min(len(self._levels) - 1, int(index)))
        with QSignalBlocker(self.value_spin), QSignalBlocker(self.slider):
            self.slider.setValue(index)
            self.value_spin.setValue(self._levels[index])

    def _slider_value_changed(self, value: int) -> None:
        if self.slider.isSliderDown():
            return
        self._set_index(value)
        self._emit_value()

    def _slider_released(self) -> None:
        self._set_index(self.slider.value())
        self._emit_value()

    def _spin_committed(self) -> None:
        self._set_index(self._nearest_index(self.value_spin.value()))
        self._emit_value()

    def set_value(self, value: float) -> None:
        self._set_index(self._nearest_index(value))

    def set_supported(self, supported: bool, reason: str = "") -> None:
        self.setEnabled(supported)
        tooltip = reason if not supported and reason else self._level_tooltip
        self.setToolTip(tooltip)
        self.label.setToolTip(tooltip)


class PreviewWidget(QLabel):
    points_changed = Signal(object)
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraPreview")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._source = QPixmap()
        self._native_size = (0, 0)
        self._points: list[tuple[int, int]] = []
        self._zoom_factor = 1.0
        self._pan_offset = QPointF()
        self._press_position: QPointF | None = None
        self._press_pan = QPointF()
        self._panning = False
        self.set_placeholder("相机未连接")

    def set_placeholder(self, text: str) -> None:
        self._source = QPixmap()
        self._native_size = (0, 0)
        self.clear_points()
        self.unsetCursor()
        self.setPixmap(QPixmap())
        self.setText(text)
        self.reset_zoom()

    def set_frame(self, image: QImage) -> None:
        self._source = QPixmap.fromImage(image)
        if self._native_size == (0, 0):
            self._native_size = (image.width(), image.height())
        self.setText("")
        self.setCursor(Qt.CrossCursor)
        self._fit_frame()

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def zoom_in(self) -> None:
        self._step_zoom(1)

    def zoom_out(self) -> None:
        self._step_zoom(-1)

    def reset_zoom(self) -> None:
        changed = not math.isclose(self._zoom_factor, 1.0)
        self._zoom_factor = 1.0
        self._pan_offset = QPointF()
        self._fit_frame()
        if changed:
            self.zoom_changed.emit(self._zoom_factor)

    def set_zoom(self, factor: float, anchor: QPointF | None = None) -> None:
        factor = min(PREVIEW_ZOOM_LEVELS[-1], max(PREVIEW_ZOOM_LEVELS[0], factor))
        if math.isclose(factor, self._zoom_factor):
            return

        old_display = self._display_rect()
        relative_x = relative_y = None
        if anchor is not None and old_display.contains(anchor):
            relative_x = (anchor.x() - old_display.left()) / old_display.width()
            relative_y = (anchor.y() - old_display.top()) / old_display.height()

        self._zoom_factor = factor
        fitted_width, fitted_height = self._fitted_dimensions()
        display_width = fitted_width * factor
        display_height = fitted_height * factor
        if relative_x is not None and relative_y is not None:
            content_center = QRectF(self.contentsRect()).center()
            self._pan_offset = QPointF(
                anchor.x() - content_center.x() - (relative_x - 0.5) * display_width,
                anchor.y() - content_center.y() - (relative_y - 0.5) * display_height,
            )
        self._clamp_pan(display_width, display_height)
        self.update()
        self.zoom_changed.emit(self._zoom_factor)

    def _step_zoom(self, direction: int, anchor: QPointF | None = None) -> None:
        if direction > 0:
            factor = next(
                (
                    level
                    for level in PREVIEW_ZOOM_LEVELS
                    if level > self._zoom_factor + 1e-9
                ),
                PREVIEW_ZOOM_LEVELS[-1],
            )
        else:
            factor = next(
                (
                    level
                    for level in reversed(PREVIEW_ZOOM_LEVELS)
                    if level < self._zoom_factor - 1e-9
                ),
                PREVIEW_ZOOM_LEVELS[0],
            )
        if anchor is None:
            anchor = QRectF(self.contentsRect()).center()
        self.set_zoom(factor, anchor)

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._points)

    def set_native_size(self, width: int, height: int) -> None:
        size = (max(0, int(width)), max(0, int(height)))
        if self._native_size not in ((0, 0), size):
            self.clear_points()
        self._native_size = size
        self.update()

    def undo_point(self) -> None:
        if not self._points:
            return
        self._points.pop()
        self.points_changed.emit(list(self._points))
        self.update()

    def clear_points(self) -> None:
        if not self._points:
            return
        self._points.clear()
        self.points_changed.emit([])
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_frame()

    def _fit_frame(self) -> None:
        if self._source.isNull():
            return
        fitted_width, fitted_height = self._fitted_dimensions()
        self._clamp_pan(
            fitted_width * self._zoom_factor,
            fitted_height * self._zoom_factor,
        )
        self.update()

    def _fitted_dimensions(self) -> tuple[float, float]:
        if self._source.isNull():
            return 0.0, 0.0
        content = QRectF(self.contentsRect())
        source_width = self._source.width() / self._source.devicePixelRatio()
        source_height = self._source.height() / self._source.devicePixelRatio()
        if (
            content.isEmpty()
            or source_width <= 0
            or source_height <= 0
        ):
            return 0.0, 0.0
        scale = min(content.width() / source_width, content.height() / source_height)
        return source_width * scale, source_height * scale

    def _clamp_pan(self, display_width: float, display_height: float) -> None:
        content = QRectF(self.contentsRect())
        max_x = max(0.0, (display_width - content.width()) / 2.0)
        max_y = max(0.0, (display_height - content.height()) / 2.0)
        self._pan_offset = QPointF(
            min(max_x, max(-max_x, self._pan_offset.x())),
            min(max_y, max(-max_y, self._pan_offset.y())),
        )

    def _display_rect(self) -> QRectF:
        fitted_width, fitted_height = self._fitted_dimensions()
        if fitted_width <= 0 or fitted_height <= 0:
            return QRectF()
        content = QRectF(self.contentsRect())
        width = fitted_width * self._zoom_factor
        height = fitted_height * self._zoom_factor
        return QRectF(
            content.center().x() - width / 2.0 + self._pan_offset.x(),
            content.center().y() - height / 2.0 + self._pan_offset.y(),
            width,
            height,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._source.isNull():
            super().mousePressEvent(event)
            return
        self._press_position = event.position()
        self._press_pan = QPointF(self._pan_offset)
        self._panning = False
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._press_position is None or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        delta = event.position() - self._press_position
        if not self._panning and delta.manhattanLength() < 4:
            event.accept()
            return
        display = self._display_rect()
        content = QRectF(self.contentsRect())
        if display.width() <= content.width() and display.height() <= content.height():
            event.accept()
            return
        self._panning = True
        self.setCursor(Qt.ClosedHandCursor)
        self._pan_offset = self._press_pan + delta
        self._clamp_pan(display.width(), display.height())
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._press_position is None:
            super().mouseReleaseEvent(event)
            return
        position = event.position()
        was_panning = self._panning
        self._press_position = None
        self._panning = False
        self.setCursor(Qt.CrossCursor)
        if was_panning:
            event.accept()
            return
        self._add_point_at(position)
        event.accept()

    def _add_point_at(self, position: QPointF) -> None:
        display = self._display_rect()
        width, height = self._native_size
        if not display.contains(position) or width <= 0 or height <= 0:
            return
        x = min(width - 1, max(0, int((position.x() - display.left()) * width / display.width())))
        y = min(height - 1, max(0, int((position.y() - display.top()) * height / display.height())))
        self._points.append((x, y))
        self.points_changed.emit(list(self._points))
        self.update()

    def wheelEvent(self, event) -> None:
        if self._source.isNull() or event.angleDelta().y() == 0:
            event.ignore()
            return
        self._step_zoom(1 if event.angleDelta().y() > 0 else -1, event.position())
        event.accept()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._source.isNull():
            return
        display = self._display_rect()
        if display.isEmpty():
            return

        painter = QPainter(self)
        painter.setClipRect(self.contentsRect())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(display, self._source, QRectF(self._source.rect()))
        if not self._points:
            painter.end()
            return
        width, height = self._native_size
        if width <= 0 or height <= 0:
            painter.end()
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        marker_color = QColor("#f59e0b")
        painter.setPen(QPen(marker_color, 2))
        for number, (x, y) in enumerate(self._points, 1):
            px = display.left() + (x + 0.5) * display.width() / width
            py = display.top() + (y + 0.5) * display.height() / height
            painter.drawEllipse(QRectF(px - 5, py - 5, 10, 10))
            painter.drawLine(int(px - 8), int(py), int(px + 8), int(py))
            painter.drawLine(int(px), int(py - 8), int(px), int(py + 8))

            text = f"{number}: ({x}, {y})"
            text_rect = QRectF(painter.fontMetrics().boundingRect(text)).adjusted(-5, -3, 5, 3)
            label_x = px + 10
            label_y = py - text_rect.height() - 8
            if label_x + text_rect.width() > display.right():
                label_x = px - text_rect.width() - 10
            if label_y < display.top():
                label_y = py + 8
            label_rect = QRectF(label_x, label_y, text_rect.width(), text_rect.height())
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 23, 42, 210))
            painter.drawRoundedRect(label_rect, 3, 3)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(label_rect, Qt.AlignCenter, text)
            painter.setPen(QPen(marker_color, 2))
        painter.end()


class CameraControlPage(QWidget):
    console_output = Signal(str, str)  # channel, text

    def __init__(
        self,
        manager: TaskManager,
        parent=None,
        *,
        worker: CameraWorker | None = None,
        auto_scan: bool = True,
    ):
        super().__init__(parent)
        self.manager = manager
        self.worker = worker or CameraWorker(self)
        self._owns_worker = worker is None
        self._connected = False
        self._recording = False
        self._task_busy = manager.busy()
        self._properties: dict[str, dict[str, Any]] = {}
        self._latest_size = (0, 0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_connection_bar())
        outer.addWidget(self._build_content(), 1)
        outer.addWidget(self._build_status_row())

        self._wire_worker()
        self._set_connected_controls(False)
        self.connect_button.setEnabled(False)
        self.manager.busy_changed.connect(self._on_task_busy)
        if self._owns_worker:
            self.worker.start()
        if auto_scan:
            QTimer.singleShot(0, self.scan_devices)

    # ---- construction ---------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        title = QLabel("相机控制")
        title.setObjectName("PageTitle")
        subtitle = QLabel("实时预览、拍照、录像及 UVC 曝光/白平衡/颜色参数")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _build_connection_bar(self) -> QWidget:
        group = QGroupBox("相机")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(250)
        self.scan_button = QPushButton("扫描")
        self.scan_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.scan_button.clicked.connect(self.scan_devices)
        self.preview_resolution_combo = self._resolution_combo(default=(1920, 1080))
        self.preview_resolution_combo.currentIndexChanged.connect(
            self._on_preview_resolution_changed
        )
        self.format_combo = QComboBox()
        self.format_combo.addItems(("MJPG", "YUY2", "驱动默认"))
        self.connect_button = QPushButton("连接")
        self.connect_button.setProperty("class", "primary")
        self.connect_button.clicked.connect(self._toggle_connection)

        layout.addWidget(QLabel("设备"))
        layout.addWidget(self.device_combo, 1)
        layout.addWidget(self.scan_button)
        layout.addWidget(QLabel("预览"))
        layout.addWidget(self.preview_resolution_combo)
        layout.addWidget(QLabel("格式"))
        layout.addWidget(self.format_combo)
        layout.addWidget(self.connect_button)
        return group

    def _build_content(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.content_splitter = splitter

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(345)
        controls_scroll.setMaximumWidth(430)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self._build_capture_group())
        controls_layout.addWidget(self._build_exposure_group())
        controls_layout.addWidget(self._build_white_balance_group())
        controls_layout.addWidget(self._build_color_group())
        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls)
        splitter.addWidget(controls_scroll)

        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        self.preview = PreviewWidget()
        preview_layout.addWidget(self.preview, 1)
        info = QHBoxLayout()
        self.preview_info = QLabel("-- × --")
        self.preview_info.setObjectName("PathLabel")
        self.coordinate_label = QLabel("像素坐标: --")
        self.coordinate_label.setObjectName("StatCaption")
        self.zoom_out_button = QToolButton()
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("缩小预览")
        self.zoom_out_button.setFixedSize(30, 28)
        self.zoom_out_button.clicked.connect(self.preview.zoom_out)
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setObjectName("StatCaption")
        self.zoom_value_label.setAlignment(Qt.AlignCenter)
        self.zoom_value_label.setFixedWidth(46)
        self.zoom_in_button = QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("放大预览")
        self.zoom_in_button.setFixedSize(30, 28)
        self.zoom_in_button.clicked.connect(self.preview.zoom_in)
        self.fit_zoom_button = QToolButton()
        self.fit_zoom_button.setText("适配")
        self.fit_zoom_button.setToolTip("将预览恢复为适配窗口")
        self.fit_zoom_button.setFixedSize(48, 28)
        self.fit_zoom_button.clicked.connect(self.preview.reset_zoom)
        self.undo_point_button = QPushButton("撤回")
        self.undo_point_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.undo_point_button.setEnabled(False)
        self.undo_point_button.clicked.connect(self.preview.undo_point)
        self.clear_points_button = QPushButton("清除")
        self.clear_points_button.setIcon(
            self.style().standardIcon(QStyle.SP_DialogDiscardButton)
        )
        self.clear_points_button.setEnabled(False)
        self.clear_points_button.clicked.connect(self.preview.clear_points)
        self.fps_label = QLabel("-- FPS")
        self.fps_label.setObjectName("StatCaption")
        self.recording_label = QLabel("")
        self.recording_label.setObjectName("RecordingStatus")
        info.addWidget(self.preview_info)
        info.addWidget(self.zoom_out_button)
        info.addWidget(self.zoom_value_label)
        info.addWidget(self.zoom_in_button)
        info.addWidget(self.fit_zoom_button)
        info.addWidget(self.coordinate_label)
        info.addStretch(1)
        info.addWidget(self.undo_point_button)
        info.addWidget(self.clear_points_button)
        info.addWidget(self.recording_label)
        info.addWidget(self.fps_label)
        preview_layout.addLayout(info)
        self.preview.points_changed.connect(self._on_preview_points_changed)
        self.preview.zoom_changed.connect(self._on_preview_zoom_changed)
        splitter.addWidget(preview_area)
        splitter.setSizes((370, 850))
        return splitter

    def _build_capture_group(self) -> QGroupBox:
        group = QGroupBox("拍摄与录像")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        resolution = QHBoxLayout()
        resolution.addWidget(QLabel("照片分辨率"))
        self.photo_resolution_combo = self._resolution_combo(
            default=(1920, 1080), include_follow=True
        )
        resolution.addWidget(self.photo_resolution_combo, 1)
        layout.addLayout(resolution)

        self.photo_button = QPushButton("拍照")
        self.photo_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.photo_button.clicked.connect(self._take_photo)
        self.record_button = QPushButton("开始录像")
        self.record_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.record_button.clicked.connect(self._toggle_recording)
        self.open_output_button = QPushButton("打开目录")
        self.open_output_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_output_button.clicked.connect(self._open_output_directory)
        self.capture_buttons = (
            self.photo_button,
            self.record_button,
            self.open_output_button,
        )
        self.capture_button_row = EqualWidthButtonRow(self.capture_buttons)
        layout.addWidget(self.capture_button_row)
        return group

    def _build_exposure_group(self) -> QGroupBox:
        group = QGroupBox("曝光控制")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        self.auto_exposure_check = QCheckBox("自动曝光")
        self.auto_exposure_check.setChecked(True)
        self.auto_exposure_check.toggled.connect(self._set_auto_exposure)
        layout.addWidget(self.auto_exposure_check)

        self.property_controls: dict[str, PropertySlider] = {}
        exposure = ExposureTimeSlider()
        exposure.value_committed.connect(self._set_property)
        self.property_controls["exposure"] = exposure
        layout.addWidget(exposure)

        target = PropertySlider("exposure_target", "曝光目标值", 0, 100)
        target.setToolTip("当前 CamSPC 驱动通过目标亮度属性调节自动曝光目标")
        target.value_committed.connect(self._set_property)
        self.property_controls["exposure_target"] = target
        layout.addWidget(target)
        reset = QPushButton("恢复连接时值")
        reset.clicked.connect(
            lambda: self.worker.reset_properties(EXPOSURE_PROPERTIES)
        )
        layout.addWidget(reset, 0, Qt.AlignRight)
        return group

    def _build_white_balance_group(self) -> QGroupBox:
        group = QGroupBox("白平衡")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        self.auto_wb_check = QCheckBox("自动白平衡")
        self.auto_wb_check.setChecked(True)
        self.auto_wb_check.toggled.connect(self._set_auto_white_balance)
        layout.addWidget(self.auto_wb_check)

        temperature = PropertySlider(
            "white_balance", "色温", 2000, 10000, step=100, suffix=" K"
        )
        temperature.value_committed.connect(self._set_property)
        self.property_controls["white_balance"] = temperature
        layout.addWidget(temperature)
        red = PropertySlider("white_balance_red", "红色分量", 0, 255)
        red.value_committed.connect(self._set_property)
        self.property_controls["white_balance_red"] = red
        layout.addWidget(red)

        buttons = QHBoxLayout()
        self.one_shot_wb_button = QPushButton("一次白平衡")
        self.one_shot_wb_button.clicked.connect(self.worker.white_balance_once)
        reset = QPushButton("恢复连接时值")
        reset.clicked.connect(
            lambda: self.worker.reset_properties(WHITE_BALANCE_PROPERTIES)
        )
        buttons.addWidget(self.one_shot_wb_button)
        buttons.addWidget(reset)
        layout.addLayout(buttons)
        return group

    def _build_color_group(self) -> QGroupBox:
        group = QGroupBox("颜色调节")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 8)
        for key, label in (
            ("hue", "色调"),
            ("saturation", "饱和度"),
            ("contrast", "对比度"),
            ("gamma", "伽马值"),
        ):
            control = PropertySlider(key, label, 0, 100)
            control.value_committed.connect(self._set_property)
            self.property_controls[key] = control
            layout.addWidget(control)
        reset = QPushButton("恢复连接时值")
        reset.clicked.connect(lambda: self.worker.reset_properties(COLOR_PROPERTIES))
        layout.addWidget(reset, 0, Qt.AlignRight)
        return group

    def _build_status_row(self) -> QWidget:
        row = QFrame()
        row.setObjectName("Card")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 6, 10, 6)
        self.status_label = QLabel("未连接")
        self.status_label.setObjectName("HintLabel")
        layout.addWidget(self.status_label, 1)
        self.backend_label = QLabel("")
        self.backend_label.setObjectName("PathLabel")
        layout.addWidget(self.backend_label)
        return row

    @staticmethod
    def _resolution_combo(
        *, default: tuple[int, int], include_follow: bool = False
    ) -> QComboBox:
        combo = QComboBox()
        if include_follow:
            combo.addItem("跟随预览", None)
        for width, height in RESOLUTIONS:
            combo.addItem(f"{width} × {height}", (width, height))
        index = combo.findData(default)
        combo.setCurrentIndex(max(0, index))
        return combo

    # ---- worker wiring and state ---------------------------------------

    def _wire_worker(self) -> None:
        self.worker.devices_ready.connect(self._on_devices)
        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.failed.connect(self._on_failed)
        self.worker.frame_ready.connect(self.preview.set_frame)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.properties_updated.connect(self._on_properties)
        self.worker.property_updated.connect(self._on_property_updated)
        self.worker.resolution_updated.connect(self._on_resolution_updated)
        self.worker.photo_saved.connect(self._on_photo_saved)
        self.worker.recording_changed.connect(self._on_recording_changed)

    def scan_devices(self) -> None:
        if self._connected:
            return
        self.scan_button.setEnabled(False)
        self.status_label.setText("正在扫描相机…")
        self.worker.scan()

    def _on_devices(self, ok: bool, devices: list, message: str) -> None:
        self.device_combo.clear()
        if ok:
            for device in devices:
                source = device.get("device", device["index"])
                self.device_combo.addItem(
                    f"[{device['index']}] {device['name']}", source
                )
        self.scan_button.setEnabled(not self._task_busy)
        self.connect_button.setEnabled(ok and bool(devices) and not self._task_busy)
        self.status_label.setText(message)

    def _toggle_connection(self) -> None:
        if self._connected:
            if self._recording:
                self.worker.stop_recording()
            self.connect_button.setEnabled(False)
            self.worker.disconnect_device()
            return
        if self.manager.busy():
            QMessageBox.warning(
                self,
                "相机正被任务占用",
                "请等待当前检测任务结束或先停止任务。",
            )
            return
        index = self.device_combo.currentData()
        size = self.preview_resolution_combo.currentData()
        if index is None or size is None:
            self.status_label.setText("请先扫描并选择相机")
            return
        pixel_format = self.format_combo.currentText()
        self.connect_button.setEnabled(False)
        self.status_label.setText("正在连接相机…")
        self.worker.connect_device(index, size[0], size[1], pixel_format)

    def _on_connected(self, snapshot: dict[str, Any]) -> None:
        self._connected = True
        self.connect_button.setText("断开")
        self.connect_button.setEnabled(True)
        self.backend_label.setText(snapshot.get("backend") or "OpenCV")
        self.status_label.setText("相机已连接")
        self._latest_size = (snapshot["width"], snapshot["height"])
        self.preview.set_native_size(*self._latest_size)
        self.preview_info.setText(f"{snapshot['width']} × {snapshot['height']}")
        self._select_resolution(self.preview_resolution_combo, self._latest_size)
        self._apply_properties(snapshot.get("properties") or {})
        self._set_connected_controls(True)

    def _on_disconnected(self, message: str) -> None:
        self._connected = False
        self._recording = False
        self.connect_button.setText("连接")
        self.connect_button.setEnabled(
            self.device_combo.count() > 0 and not self._task_busy
        )
        self.preview.set_placeholder("相机未连接")
        self.preview_info.setText("-- × --")
        self.fps_label.setText("-- FPS")
        self.recording_label.setText("")
        self.backend_label.setText("")
        self.status_label.setText(message)
        self._set_connected_controls(False)

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(message)
        self._append_console("comm-error", message)
        self.photo_button.setEnabled(self._connected and not self._recording)
        self.record_button.setEnabled(self._connected)
        self.photo_resolution_combo.setEnabled(
            self._connected and not self._recording
        )
        if not self._connected:
            self.connect_button.setEnabled(
                self.device_combo.count() > 0 and not self._task_busy
            )

    def _set_connected_controls(self, connected: bool) -> None:
        self.device_combo.setEnabled(not connected and not self._task_busy)
        self.scan_button.setEnabled(not connected and not self._task_busy)
        self.format_combo.setEnabled(not connected)
        self.photo_button.setEnabled(connected and not self._recording)
        self.record_button.setEnabled(connected)
        self.photo_resolution_combo.setEnabled(connected and not self._recording)
        self._update_zoom_controls()
        self.auto_exposure_check.setEnabled(
            connected and self._property_supported("auto_exposure")
        )
        self.auto_wb_check.setEnabled(
            connected and self._property_supported("auto_white_balance")
        )
        self._update_manual_property_states()

    def _on_task_busy(self, busy: bool) -> None:
        self._task_busy = busy
        if not self._connected:
            self.device_combo.setEnabled(not busy)
            self.scan_button.setEnabled(not busy)
            self.connect_button.setEnabled(not busy and self.device_combo.count() > 0)

    def is_connected(self) -> bool:
        return self._connected

    def disconnect_if_connected(self) -> None:
        if self._connected:
            if self._recording:
                self.worker.stop_recording()
            self.worker.disconnect_device()

    def shutdown_worker(self) -> None:
        if self._owns_worker:
            self.worker.shutdown()
            self.worker.wait(4000)

    # ---- properties -----------------------------------------------------

    def _property_supported(self, name: str) -> bool:
        return bool((self._properties.get(name) or {}).get("supported"))

    def _apply_properties(self, properties: dict[str, dict[str, Any]]) -> None:
        self._properties = properties
        for key, control in self.property_controls.items():
            record = properties.get(key) or {}
            supported = bool(record.get("supported"))
            control.set_supported(
                supported,
                "" if supported else "当前相机驱动不支持此参数",
            )
            value = record.get("value")
            if supported and value is not None and math.isfinite(float(value)):
                display_value = (
                    exposure_milliseconds(float(value))
                    if key == "exposure"
                    else float(value)
                )
                if math.isfinite(display_value):
                    control.set_value(display_value)

        auto_exposure = properties.get("auto_exposure") or {}
        auto_value = auto_exposure.get("value")
        with QSignalBlocker(self.auto_exposure_check):
            self.auto_exposure_check.setChecked(
                True if auto_value in (None, -1) else float(auto_value) >= 0.5
            )
        auto_wb = properties.get("auto_white_balance") or {}
        with QSignalBlocker(self.auto_wb_check):
            self.auto_wb_check.setChecked(float(auto_wb.get("value") or 0) >= 0.5)
        self._update_manual_property_states()

    def _on_properties(self, properties: dict[str, dict[str, Any]]) -> None:
        if self._connected:
            self._apply_properties(properties)

    def _set_property(self, name: str, value: float) -> None:
        backend_value = exposure_backend_value(value) if name == "exposure" else value
        if math.isfinite(backend_value):
            self.worker.set_property(name, backend_value)

    def _set_auto_exposure(self, checked: bool) -> None:
        if self._connected:
            self.worker.set_property("auto_exposure", 1.0 if checked else 0.0)
        self._update_manual_property_states()

    def _set_auto_white_balance(self, checked: bool) -> None:
        if self._connected:
            self.worker.set_property(
                "auto_white_balance", 1.0 if checked else 0.0
            )
        self._update_manual_property_states()

    def _update_manual_property_states(self) -> None:
        exposure = self.property_controls.get("exposure")
        if exposure is not None:
            exposure.setEnabled(
                self._connected
                and self._property_supported("exposure")
                and not self.auto_exposure_check.isChecked()
            )
        exposure_target = self.property_controls.get("exposure_target")
        if exposure_target is not None:
            exposure_target.setEnabled(
                self._connected and self._property_supported("exposure_target")
            )
        manual_wb = not self.auto_wb_check.isChecked()
        for name in WHITE_BALANCE_PROPERTIES:
            control = self.property_controls.get(name)
            if control is not None:
                control.setEnabled(
                    self._connected and self._property_supported(name) and manual_wb
                )
        self.one_shot_wb_button.setEnabled(
            self._connected
            and self._property_supported("auto_white_balance")
            and manual_wb
        )
        for name in COLOR_PROPERTIES:
            control = self.property_controls.get(name)
            if control is not None:
                control.setEnabled(
                    self._connected and self._property_supported(name)
                )

    def _on_property_updated(
        self, name: str, value: float, ok: bool, message: str
    ) -> None:
        if name in self.property_controls:
            display_value = exposure_milliseconds(value) if name == "exposure" else value
            if math.isfinite(display_value):
                self.property_controls[name].set_value(display_value)
        display_name = {
            "exposure": "曝光时间",
            "exposure_target": "曝光目标值",
            "white_balance": "色温",
            "white_balance_red": "红色分量",
            "hue": "色调",
            "saturation": "饱和度",
            "contrast": "对比度",
            "gamma": "伽马值",
        }.get(name, name)
        self.status_label.setText(f"{display_name}: {message}")

    # ---- capture --------------------------------------------------------

    def _on_preview_resolution_changed(self) -> None:
        if not self._connected:
            return
        size = self.preview_resolution_combo.currentData()
        if size is not None:
            self.worker.set_resolution(*size)

    def _on_resolution_updated(self, width: int, height: int) -> None:
        self._latest_size = (width, height)
        self.preview.set_native_size(width, height)
        self.preview_info.setText(f"{width} × {height}")
        self._select_resolution(self.preview_resolution_combo, self._latest_size)

    def _on_preview_points_changed(self, points: list[tuple[int, int]]) -> None:
        has_points = bool(points)
        self.undo_point_button.setEnabled(has_points)
        self.clear_points_button.setEnabled(has_points)
        if has_points:
            x, y = points[-1]
            self.coordinate_label.setText(
                f"像素坐标: ({x}, {y})  共 {len(points)} 点"
            )
        else:
            self.coordinate_label.setText("像素坐标: --")

    def _on_preview_zoom_changed(self, factor: float) -> None:
        self.zoom_value_label.setText(f"{round(factor * 100):d}%")
        self._update_zoom_controls()

    def _update_zoom_controls(self) -> None:
        factor = self.preview.zoom_factor
        self.zoom_out_button.setEnabled(
            self._connected and factor > PREVIEW_ZOOM_LEVELS[0] + 1e-9
        )
        self.zoom_in_button.setEnabled(
            self._connected and factor < PREVIEW_ZOOM_LEVELS[-1] - 1e-9
        )
        self.fit_zoom_button.setEnabled(
            self._connected and not math.isclose(factor, 1.0)
        )

    @staticmethod
    def _select_resolution(combo: QComboBox, size: tuple[int, int]) -> None:
        index = combo.findData(size)
        with QSignalBlocker(combo):
            if index < 0:
                combo.addItem(f"{size[0]} × {size[1]} (实际)", size)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

    def _take_photo(self) -> None:
        selected = self.photo_resolution_combo.currentData()
        size = selected if selected is not None else self._latest_size
        if not size or size == (0, 0):
            self.status_label.setText("相机尚未返回有效分辨率")
            return
        path = self._next_output_path("photo", ".png")
        self.photo_button.setEnabled(False)
        self.status_label.setText("正在拍照…")
        self.worker.take_photo(path, size[0], size[1])

    def _toggle_recording(self) -> None:
        if self._recording:
            self.worker.stop_recording()
        else:
            self.worker.start_recording(self._next_output_path("record", ".avi"))

    def _on_photo_saved(self, path: str) -> None:
        self.photo_button.setEnabled(self._connected and not self._recording)
        self.status_label.setText("拍照完成")
        self._append_console("comm-info", f"照片已保存: {path}")

    def _on_recording_changed(self, recording: bool, path: str) -> None:
        self._recording = recording
        self.record_button.setText("停止录像" if recording else "开始录像")
        self.record_button.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaStop if recording else QStyle.SP_MediaPlay
            )
        )
        self.recording_label.setText("● REC" if recording else "")
        self.photo_button.setEnabled(self._connected and not recording)
        self.photo_resolution_combo.setEnabled(self._connected and not recording)
        if path:
            self.status_label.setText("正在录像…" if recording else "录像已保存")
            self._append_console(
                "comm-info",
                ("开始录像，保存位置: " if recording else "录像已保存: ") + path,
            )

    def _on_stats(self, fps: float, width: int, height: int) -> None:
        self._latest_size = (width, height)
        self.preview.set_native_size(width, height)
        self.preview_info.setText(f"{width} × {height}")
        self.fps_label.setText(f"{fps:.1f} FPS")

    def _open_output_directory(self) -> None:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        open_in_explorer(OUTPUT_ROOT)

    def _append_console(self, channel: str, message: str) -> None:
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        self.console_output.emit(channel, f"[{timestamp}] -- [相机] {message}")

    @staticmethod
    def _next_output_path(prefix: str, suffix: str) -> Path:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = OUTPUT_ROOT / f"{prefix}_{stamp}{suffix}"
        counter = 2
        while path.exists():
            path = OUTPUT_ROOT / f"{prefix}_{stamp}_{counter:02d}{suffix}"
            counter += 1
        return path
