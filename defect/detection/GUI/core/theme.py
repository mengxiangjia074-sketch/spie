"""Application theme: Fusion style + token-based QSS, dark and light.

QSS uses @token placeholders substituted from the palette dict so braces in
the stylesheet never fight with string formatting.
"""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

UI_FONT = "Microsoft YaHei UI"
MONO_FONT = "Consolas"

DARK = {
    "bg": "#14171d",
    "surface": "#1b1f27",
    "surface2": "#232936",
    "surface3": "#2a3140",
    "border": "#2e3646",
    "text": "#e7eaf2",
    "text_dim": "#98a2b8",
    "control_bg": "#ffffff",
    "control_hover": "#eef1f6",
    "control_pressed": "#d5dbe6",
    "control_text": "#1d2330",
    "control_disabled_bg": "#3a4356",
    "accent": "#4d8dff",
    "accent_hover": "#6ba0ff",
    "accent_pressed": "#3f74d9",
    "accent_soft": "rgba(77, 141, 255, 0.16)",
    "on_accent": "#ffffff",
    "ok": "#3ecf8e",
    "ok_soft": "rgba(62, 207, 142, 0.13)",
    "warn": "#e8b339",
    "error": "#ef5f5f",
    "error_soft": "rgba(239, 95, 95, 0.13)",
    "sidebar_sel": "#2c3c5e",
    "log_bg": "#10131a",
    "scroll": "#3a4356",
}

LIGHT = {
    "bg": "#f2f4f8",
    "surface": "#ffffff",
    "surface2": "#eef1f6",
    "surface3": "#e3e8f0",
    "border": "#d5dbe6",
    "text": "#1d2330",
    "text_dim": "#5d6a7f",
    "control_bg": "#ffffff",
    "control_hover": "#eef1f6",
    "control_pressed": "#d5dbe6",
    "control_text": "#1d2330",
    "control_disabled_bg": "#e3e8f0",
    "accent": "#2f6fed",
    "accent_hover": "#4483f2",
    "accent_pressed": "#2861cc",
    "accent_soft": "rgba(47, 111, 237, 0.12)",
    "on_accent": "#ffffff",
    "ok": "#1fa36a",
    "ok_soft": "rgba(31, 163, 106, 0.10)",
    "warn": "#b97e14",
    "error": "#d64545",
    "error_soft": "rgba(214, 69, 69, 0.10)",
    "sidebar_sel": "#d8e4fb",
    "log_bg": "#fbfcfe",
    "scroll": "#c3cbd9",
}

THEMES = {"dark": DARK, "light": LIGHT}

# Updated by apply_theme(); widgets that paint colored text (e.g. the console)
# read from here so both themes stay readable.
_ACTIVE: dict = DARK


def current_palette() -> dict:
    return _ACTIVE

_QSS = """
* { outline: none; }
QMainWindow, QDialog { background: @bg; }
QWidget { color: @text; font-family: "%(ui_font)s"; font-size: 9pt; }
QToolTip { background: @surface3; color: @text; border: 1px solid @border;
           padding: 4px 6px; border-radius: 4px; }

/* ---- 侧边栏 ---- */
#Sidebar { background: @surface; border-right: 1px solid @border; }
#Sidebar::item { color: @text_dim; padding: 9px 14px; margin: 2px 10px;
                 border-radius: 8px; font-size: 10pt; }
#Sidebar::item:hover { background: @surface2; color: @text; }
#Sidebar::item:selected { background: @sidebar_sel; color: @text;
                          font-weight: 600; }

/* ---- 返回主界面按钮 ---- */
#BackButton { background: @control_bg; color: @control_text;
              border: 1px solid @border;
              border-radius: 8px; padding: 8px 14px; text-align: left;
              font-size: 10pt; }
#BackButton:hover { background: @control_hover; color: @control_text;
                    border-color: @accent; }
#BackButton:pressed { background: @control_pressed; }
#BackButton:disabled { background: @control_disabled_bg; color: @text_dim; }

#BrandTitle { color: @text; font-size: 13pt; font-weight: 700; padding: 2px; }
#BrandSub { color: @text_dim; font-size: 8pt; }

/* ---- 页面标题 ---- */
#PageTitle { font-size: 15pt; font-weight: 700; color: @text; }
#PageSubtitle { color: @text_dim; font-size: 9.5pt; }
#PathLabel { color: @text_dim; font-family: "%(mono)s"; font-size: 8pt; }
#HintLabel { color: @text_dim; font-size: 8.5pt; }
#HomeEyebrow { color: @accent; font-size: 8pt; font-weight: 700; }
#HomeTitle { color: @text; font-size: 24pt; font-weight: 700; padding: 0 4px; }
#HomeTitleRule { background: @accent; border: none; }
#SectionTitle { color: @text; font-weight: 600; font-size: 10.5pt; }
#ValueLabel { color: @text; }
#StatValue { font-size: 15pt; font-weight: 700; color: @text; }
#StatCaption { color: @text_dim; font-size: 8.5pt; }
#RecordingStatus { color: @error; font-weight: 700; }
#CameraPreview { background: @log_bg; color: @text_dim;
                 border: 1px solid @border; border-radius: 6px; }
#MotionStatusBadge { background: @surface2; border: 1px solid @border;
                     border-radius: 5px; }
#MotionStatusBadge QLabel#MotionStatusDot,
#MotionStatusBadge QLabel#MotionStatusText { color: @text_dim; }
#MotionStatusBadge[active="true"][severity="normal"] {
    background: @ok_soft; border-color: @ok; }
#MotionStatusBadge[active="true"][severity="normal"] QLabel#MotionStatusDot,
#MotionStatusBadge[active="true"][severity="normal"] QLabel#MotionStatusText {
    color: @ok; }
#MotionStatusBadge[active="true"][severity="alert"] {
    background: @error_soft; border-color: @error; }
#MotionStatusBadge[active="true"][severity="alert"] QLabel#MotionStatusDot,
#MotionStatusBadge[active="true"][severity="alert"] QLabel#MotionStatusText {
    color: @error; }
#MotionStatusBadge[active="true"] QLabel#MotionStatusText { font-weight: 600; }

/* ---- 卡片 ---- */
#Card { background: @surface; border: 1px solid @border; border-radius: 10px; }
#HeaderCard { background: @surface; border: 1px solid @border;
              border-radius: 10px; }

/* ---- 主界面功能入口卡片 (固定正方形, 内容居中) ---- */
#EntryCard { background: @surface; border: 1px solid @border;
             border-radius: 14px; }
#EntryCard:hover { background: @surface2; border-color: @accent; }
#EntryCard:pressed { background: @surface3; }
#EntryGlyph { font-size: 34pt; color: @accent; }
#EntryTitle { font-size: 12pt; font-weight: 600; color: @text; }

/* ---- 控件 ---- */
QGroupBox { background: @surface; border: 1px solid @border; border-radius: 8px;
            margin-top: 12px; padding: 8px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: @accent; font-weight: 600; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: @control_bg; color: @control_text;
    border: 1px solid @border; border-radius: 6px;
    padding: 4px 8px; selection-background-color: @accent;
    selection-color: @on_accent; min-height: 20px;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    background: @control_disabled_bg; color: @text_dim;
}
QLineEdit:read-only, QPlainTextEdit:read-only, QTextEdit:read-only {
    background: @control_disabled_bg; color: @text_dim; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid @accent; background: @control_bg;
}
#DefectPromptEdit { min-height: 120px; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: @surface; border: 1px solid @border;
    selection-background-color: @sidebar_sel; selection-color: @text; }
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: transparent; border: none; width: 0px; height: 0px; }

QCheckBox { spacing: 7px; color: @text; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid @border; background: @control_bg; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent;
    image: url(none); }
QCheckBox:disabled { color: @text_dim; }
QCheckBox::indicator:disabled { background: @control_disabled_bg; }

QSlider::groove:horizontal { height: 5px; background: @surface3;
    border-radius: 2px; }
QSlider::sub-page:horizontal { background: @accent; border-radius: 2px; }
QSlider::handle:horizontal { background: @control_bg; border: 2px solid @accent;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: @control_hover; }
QSlider:disabled::groove:horizontal { background: @control_disabled_bg; }
QSlider:disabled::sub-page:horizontal { background: @scroll; }
QSlider:disabled::handle:horizontal { border-color: @scroll;
    background: @control_disabled_bg; }

QPushButton { background: @control_bg; color: @control_text;
    border: 1px solid @border;
    border-radius: 6px; padding: 6px 16px; font-size: 9pt; }
QPushButton:hover { background: @control_hover; border-color: @accent; }
QPushButton:pressed { background: @control_pressed; }
QPushButton:disabled { color: @text_dim; background: @control_disabled_bg; }
QPushButton[class="primary"] { background: @accent; color: @on_accent;
    border: 2px solid transparent; font-weight: 600; padding: 5px 20px; }
QPushButton[class="primary"]:enabled:hover { background: @accent_hover;
    border-color: @on_accent; }
QPushButton[class="primary"]:enabled:pressed { background: @accent_pressed;
    border-color: @on_accent; }
QPushButton[class="primary"]:disabled { background: @control_disabled_bg;
    color: @text_dim; border-color: transparent; }
QPushButton[class="danger"] { background: @control_bg; color: @error;
    border: 1px solid @error; }
QPushButton[class="danger"]:hover { background: @error; color: @on_accent; }
QPushButton[class="danger"]:disabled { background: @control_disabled_bg;
    color: @text_dim; border-color: @border; }

QToolButton { background: @control_bg; border: 1px solid @border;
    border-radius: 6px; padding: 4px 8px; color: @control_text; }
QToolButton:hover { background: @control_hover; color: @control_text; }
QToolButton:disabled { background: @control_disabled_bg; color: @text_dim; }

/* ---- 表格 / 树 ---- */
QTableWidget, QTreeView, QTreeWidget, QListView {
    background: @surface; border: 1px solid @border; border-radius: 8px;
    gridline-color: @border; alternate-background-color: @surface2;
}
QHeaderView::section { background: @surface2; color: @text_dim; padding: 6px;
    border: none; border-bottom: 1px solid @border; font-weight: 600; }
QTableWidget::item, QTreeWidget::item { padding: 4px; }
QTreeWidget::item:selected, QTableWidget::item:selected, QTreeView::item:selected,
QListView::item:selected { background: @sidebar_sel; color: @text; }

/* ---- 标签页 ---- */
QTabWidget::pane { border: 1px solid @border; border-radius: 8px; top: -1px;
    background: @surface; }
QTabBar::tab { background: transparent; color: @text_dim; padding: 7px 16px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    border: 1px solid transparent; }
QTabBar::tab:hover { color: @text; }
QTabBar::tab:selected { color: @accent; background: @surface;
    border-color: @border; border-bottom: 2px solid @accent; }

/* ---- 日志 ---- */
#LogView { background: @log_bg; border: 1px solid @border; border-radius: 8px;
    font-family: "%(mono)s"; font-size: 9pt; }
#LogHeader { color: @text_dim; font-size: 8.5pt; }

/* ---- 滚动条 ---- */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: @scroll; border-radius: 5px;
    min-height: 24px; }
QScrollBar::handle:vertical:hover { background: @accent; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: @scroll; border-radius: 5px;
    min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- 状态栏 / 菜单 ---- */
QStatusBar { background: @surface; border-top: 1px solid @border;
    color: @text_dim; }
QStatusBar::item { border: none; }
QMenuBar { background: @surface; border-bottom: 1px solid @border; }
QMenuBar::item { padding: 5px 10px; border-radius: 6px; color: @text_dim; }
QMenuBar::item:selected { background: @surface2; color: @text; }
QMenu { background: @surface; border: 1px solid @border; padding: 6px; }
QMenu::item { padding: 6px 24px; border-radius: 6px; }
QMenu::item:selected { background: @sidebar_sel; }

/* ---- Dock ---- */
QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
QDockWidget::title { background: @surface; padding: 6px 10px;
    border-bottom: 1px solid @border; color: @text_dim; }

QProgressBar { background: @surface2; border: none; border-radius: 5px;
    height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: @accent; border-radius: 5px; }

QSplitter::handle { background: @border; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
"""


def build_qss(theme_name: str) -> str:
    tokens = dict(THEMES[theme_name])
    tokens["ui_font"] = UI_FONT
    tokens["mono"] = MONO_FONT
    qss = _QSS % {"ui_font": UI_FONT, "mono": MONO_FONT}
    # Replace longer names first: @accent must not consume the prefix of
    # @accent_hover, and the same applies to @surface/@surface2/@surface3.
    for key in sorted(tokens, key=len, reverse=True):
        value = tokens[key]
        qss = qss.replace("@" + key, value)
    return qss


def apply_theme(app: QApplication, theme_name: str) -> None:
    global _ACTIVE
    from PySide6.QtWidgets import QStyleFactory

    _ACTIVE = THEMES.get(theme_name, DARK)
    app.setStyle(QStyleFactory.create("Fusion"))
    palette = THEMES.get(theme_name, DARK)
    qpalette = QPalette()
    for role, hexcolor in (
        (QPalette.Window, palette["bg"]),
        (QPalette.Base, palette["surface"]),
        (QPalette.AlternateBase, palette["surface2"]),
        (QPalette.Text, palette["text"]),
        (QPalette.WindowText, palette["text"]),
        (QPalette.ButtonText, palette["text"]),
        (QPalette.Highlight, QColor(palette["accent"])),
        (QPalette.HighlightedText, QColor(palette["on_accent"])),
        (QPalette.ToolTipBase, QColor(palette["surface3"])),
        (QPalette.ToolTipText, QColor(palette["text"])),
        (QPalette.PlaceholderText, QColor(palette["text_dim"])),
    ):
        qpalette.setColor(role, QColor(hexcolor))
    app.setPalette(qpalette)
    app.setStyleSheet(build_qss(theme_name))
