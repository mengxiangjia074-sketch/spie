"""Settings page: theme, paths, about."""

from __future__ import annotations

import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QScrollArea,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from GUI import __version__
from GUI.core.paths import GUI_BACKUP_DIR, GUI_LOG_DIR, PROJECT_ROOT
from GUI.core.runner import open_in_explorer
from GUI.core.theme import THEMES
from GUI.widgets.resultview import SimpleCard


class SettingsPage(QWidget):
    theme_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # 内容包进滚动区: "关于"长文本会把页面最小高度撑到 400+,
        # 进而压缩 Dock 分隔条的可拖范围。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(18, 14, 18, 14)
        outer_layout.setSpacing(12)

        title = QLabel("设置与关于")
        title.setObjectName("PageTitle")
        outer_layout.addWidget(title)

        appearance = SimpleCard("外观")
        form = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("深色", "dark")
        self.theme_combo.addItem("浅色", "light")
        form.addRow("界面主题", self.theme_combo)
        appearance.body.addLayout(form)
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        outer_layout.addWidget(appearance)

        paths = SimpleCard("目录")
        for label, target in (
            ("任务日志", GUI_LOG_DIR),
            ("配置备份", GUI_BACKUP_DIR),
            ("项目根目录", PROJECT_ROOT),
        ):
            row = QHBoxLayout()
            name = QLabel(label)
            row.addWidget(name)
            row.addStretch(1)
            button = QPushButton("打开")
            button.clicked.connect(lambda _=False, t=target: open_in_explorer(t))
            row.addWidget(button)
            paths.body.addLayout(row)
        outer_layout.addWidget(paths)

        about = SimpleCard("关于")
        info = QLabel(
            "LensDetect 控制台 v{}\n\n"
            "· 本界面不修改 detection 下任何脚本; 仅以外科手术方式编辑 jsonnet 配置\n"
            "  (注释与 import 表达式保留原样, 每次保存前自动备份), 并以子进程运行\n"
            "  脚本、实时转发输出。\n"
            "· 相机、位移台、电动镜头为独占硬件, 同一时间只允许运行一个任务。\n"
            "· Python: {}".format(__version__, sys.executable)
        )
        info.setObjectName("HintLabel")
        info.setWordWrap(True)
        about.body.addWidget(info)
        outer_layout.addWidget(about)
        outer_layout.addStretch(1)

        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

    def set_theme(self, theme_name: str) -> None:
        index = self.theme_combo.findData(theme_name)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)

    def _on_theme(self) -> None:
        self.theme_changed.emit(self.theme_combo.currentData())
