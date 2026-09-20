"""主界面: 程序启动后的落地页, 提供功能入口。

- 标定: 进入标定工作区。
- 图像采集: 进入检测/采集工作区 (任务页: PCB拍照、总览、采集结果等)。
- RFID检测: 进入 RFID 标签检测页。
- 镜头控制: 单独的交互式镜头控制页。
- 位移台控制: 单独的交互式位移台控制页。
- 相机控制: 实时预览、拍照、录像与 UVC 参数调节。

卡片为固定尺寸正方形, 图标 (detection/GUI/images 下的 SVG) 与名称居中,
卡片以 3×2 网格在页面中居中。
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from GUI.core.paths import ENTRY_ICONS
from GUI.core.theme import current_palette

CARD_SIZE = 180          # 功能卡片边长
CARD_SPACING = 16        # 卡片间距
ICON_SIZE = 88           # 卡片内图标边长 (px)
GRID_SPACING = 44
ANIMATION_INTERVAL_MS = 50


def load_icon(path, fallback_glyph: str = "") -> QPixmap:
    """把 SVG 渲染成透明背景 pixmap; 失败时返回空 pixmap (调用方回退文字)。"""
    renderer = QSvgRenderer(str(path))
    if not renderer.isValid():
        return QPixmap()
    image = QImage(ICON_SIZE, ICON_SIZE, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    return QPixmap.fromImage(image)


class EntryCard(QFrame):
    """一张可点击的正方形功能卡片: 居中图标 + 居中名称。"""

    clicked = Signal()

    def __init__(self, icon_path, title: str, fallback_glyph: str = "",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("EntryCard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(CARD_SIZE, CARD_SIZE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(18)

        pixmap = load_icon(icon_path)
        icon_label = QLabel()
        if pixmap.isNull():
            icon_label.setText(fallback_glyph)
            icon_label.setObjectName("EntryGlyph")
        else:
            icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignHCenter)
        layout.addStretch(4)
        layout.addWidget(icon_label, 0, Qt.AlignHCenter)
        layout.addStretch(3)

        title_label = QLabel(title)
        title_label.setObjectName("EntryTitle")
        title_label.setAlignment(Qt.AlignHCenter)
        layout.addWidget(title_label, 0, Qt.AlignHCenter)
        layout.addStretch(5)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class HomePage(QWidget):
    """主界面: 居中标题 + 居中的功能卡片行 + 右下角设置入口。"""

    capture_clicked = Signal()      # 进入图像采集工作区
    camera_clicked = Signal()       # 进入相机控制页
    lens_clicked = Signal()         # 进入镜头控制页
    calibration_clicked = Signal()  # 进入标定工作区
    motion_clicked = Signal()       # 进入位移台控制页
    rfid_clicked = Signal()         # 进入 RFID 检测页
    settings_clicked = Signal()     # 进入设置页

    def __init__(self, parent=None):
        super().__init__(parent)
        self._background_frame = 0
        self._background_timer = QTimer(self)
        self._background_timer.setInterval(ANIMATION_INTERVAL_MS)
        self._background_timer.timeout.connect(self._advance_background)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 28, 24, 24)
        outer.setSpacing(0)

        header = QWidget()
        header.setObjectName("HomeHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        eyebrow = QLabel("DefectDetection")
        eyebrow.setObjectName("HomeEyebrow")
        eyebrow.setAlignment(Qt.AlignHCenter)
        header_layout.addWidget(eyebrow)
        header_layout.addSpacing(7)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(14)
        title_row.addStretch(1)
        left_rule = QFrame()
        left_rule.setObjectName("HomeTitleRule")
        left_rule.setFixedSize(58, 2)
        title_row.addWidget(left_rule, 0, Qt.AlignVCenter)

        title = QLabel("主界面")
        title.setObjectName("HomeTitle")
        title.setAlignment(Qt.AlignCenter)
        title_row.addWidget(title)

        right_rule = QFrame()
        right_rule.setObjectName("HomeTitleRule")
        right_rule.setFixedSize(58, 2)
        title_row.addWidget(right_rule, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        header_layout.addLayout(title_row)

        outer.addWidget(header)

        outer.addStretch(3)

        # 3×2 网格整体居中，固定尺寸卡片不随窗口拉伸。
        self.cards_grid = QGridLayout()
        self.cards_grid.setSpacing(CARD_SPACING)
        entries = (
            ("calibration", "标定", "⊙", self.calibration_clicked),
            ("capture", "图像采集", "⊞", self.capture_clicked),
            ("rfid", "RFID检测", "◉", self.rfid_clicked),
            ("lens", "镜头控制", "◎", self.lens_clicked),
            ("motion", "位移台控制", "⇄", self.motion_clicked),
            ("camera", "相机控制", "▣", self.camera_clicked),
        )
        for index, (icon_key, name, fallback, signal) in enumerate(entries):
            card = EntryCard(ENTRY_ICONS[icon_key], name, fallback_glyph=fallback)
            card.clicked.connect(signal.emit)
            self.cards_grid.addWidget(card, index // 3, index % 3)
        cards_row = QHBoxLayout()
        cards_row.addStretch(1)
        cards_row.addLayout(self.cards_grid)
        cards_row.addStretch(1)
        outer.addLayout(cards_row)

        outer.addStretch(4)

        # 右下角设置入口 (小按钮, 不占功能卡片行)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        settings_button = QPushButton("⚙  设置")
        settings_button.setObjectName("BackButton")
        settings_button.setCursor(Qt.PointingHandCursor)
        settings_button.setToolTip("主题与目录设置")
        settings_button.clicked.connect(self.settings_clicked.emit)
        bottom.addWidget(settings_button)
        outer.addLayout(bottom)

    def _advance_background(self) -> None:
        self._background_frame = (self._background_frame + 1) % 100_000
        self.update()

    @staticmethod
    def _color_with_alpha(value: str, alpha: int) -> QColor:
        color = QColor(value)
        color.setAlpha(alpha)
        return color

    def paintEvent(self, event) -> None:
        del event
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(palette["bg"]))

        width = self.width()
        height = self.height()
        accent = palette["accent"]

        # Fixed engineering grid. Every fourth line is slightly stronger.
        for index, x in enumerate(range(0, width + GRID_SPACING, GRID_SPACING)):
            alpha = 28 if index % 4 == 0 else 13
            painter.setPen(QPen(self._color_with_alpha(accent, alpha), 1))
            painter.drawLine(x, 0, x, height)
        for index, y in enumerate(range(0, height + GRID_SPACING, GRID_SPACING)):
            alpha = 28 if index % 4 == 0 else 13
            painter.setPen(QPen(self._color_with_alpha(accent, alpha), 1))
            painter.drawLine(0, y, width, y)

        # Two restrained circuit traces frame the central function cards.
        painter.setPen(QPen(self._color_with_alpha(accent, 54), 1))
        upper_y = int(height * 0.27)
        lower_y = int(height * 0.76)
        painter.drawLine(0, upper_y, int(width * 0.18), upper_y)
        painter.drawLine(int(width * 0.18), upper_y,
                         int(width * 0.23), upper_y + 28)
        painter.drawLine(int(width * 0.23), upper_y + 28,
                         int(width * 0.34), upper_y + 28)
        painter.drawLine(width, lower_y, int(width * 0.82), lower_y)
        painter.drawLine(int(width * 0.82), lower_y,
                         int(width * 0.77), lower_y - 28)
        painter.drawLine(int(width * 0.77), lower_y - 28,
                         int(width * 0.66), lower_y - 28)

        node_color = self._color_with_alpha(accent, 90)
        painter.fillRect(int(width * 0.34) - 3, upper_y + 25, 7, 7, node_color)
        painter.fillRect(int(width * 0.66) - 3, lower_y - 31, 7, 7, node_color)

        # A slow scan line crosses the canvas with a short fading trail.
        scan_span = max(1, width + 240)
        scan_x = (self._background_frame * 4) % scan_span - 120
        for offset, alpha in ((0, 70), (-7, 38), (-14, 22), (-21, 12)):
            painter.setPen(QPen(
                self._color_with_alpha(accent, alpha),
                2 if offset == 0 else 1,
            ))
            painter.drawLine(scan_x + offset, 0, scan_x + offset, height)

        pulse_x = (self._background_frame * 5) % max(1, width + 80) - 40
        pulse_color = self._color_with_alpha(accent, 105)
        painter.fillRect(pulse_x, upper_y - 1, 26, 3, pulse_color)
        painter.fillRect(width - pulse_x - 26, lower_y - 1, 26, 3, pulse_color)
        painter.end()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._background_timer.start()

    def hideEvent(self, event) -> None:
        self._background_timer.stop()
        super().hideEvent(event)
