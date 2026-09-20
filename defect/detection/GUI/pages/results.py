"""Results browser: working_data files, calibration summary, JSON & images."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from GUI.core.paths import CALIBRATION_SUMMARY, capture_result_roots
from GUI.core.runner import open_in_explorer
from GUI.widgets.resultview import (
    ImageView,
    JsonTree,
    SummaryTable,
    visible_summary_document,
)

SHOW_SUFFIXES = {".json", ".log", ".png", ".jpg", ".jpeg", ".bmp", ".txt", ".jsonnet"}
MAX_FILES = 4000

FOLDER_GLYPH = "📁"
FILE_GLYPHS = {".json": "🧾", ".jsonnet": "🧾", ".log": "📄", ".txt": "📄",
               ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".bmp": "🖼"}


class ResultsPage(QWidget):
    """文件浏览 + JSON/图像查看; 根目录与标定汇总由构造参数决定。

    标定工作区用 calibration_result_roots + 标定汇总; 图像采集工作区用
    capture_result_roots 且不含标定汇总页。
    """

    def __init__(self, title: str = "标定结果", roots_provider=capture_result_roots,
                 show_summary: bool = True, parent=None):
        super().__init__(parent)
        self._roots_provider = roots_provider
        self._show_summary = show_summary
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        header.addWidget(title_label)
        header.addStretch(1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("文件")
        self.file_tree.itemClicked.connect(self._on_file_clicked)
        splitter.addWidget(self.file_tree)

        self.tabs = QTabWidget()
        self._tab_widgets = {}
        if show_summary:
            self.summary_table = SummaryTable()
            self._tab_widgets["summary"] = self.summary_table
            self.tabs.addTab(self.summary_table, "标定汇总")
        self.json_tree = JsonTree()
        self._tab_widgets["json"] = self.json_tree
        self.tabs.addTab(self.json_tree, "JSON")
        self.raw_view = QPlainTextEdit()
        self.raw_view.setReadOnly(True)
        self._tab_widgets["raw"] = self.raw_view
        self.tabs.addTab(self.raw_view, "原始文本")
        self.image_view = ImageView()
        image_holder = QWidget()
        image_layout = QVBoxLayout(image_holder)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.image_caption = QLabel("")
        self.image_caption.setObjectName("HintLabel")
        image_layout.addWidget(self.image_view, 1)
        image_layout.addWidget(self.image_caption)
        self._tab_widgets["image"] = image_holder
        self.tabs.addTab(image_holder, "图像")
        splitter.addWidget(self.tabs)

        splitter.setSizes([300, 780])
        outer.addWidget(splitter, 1)

        self.refresh()

    # ---- tree -----------------------------------------------------------

    def refresh(self) -> None:
        self.file_tree.clear()
        for root in self._roots_provider():
            if not root.exists():
                continue
            top = QTreeWidgetItem(self.file_tree, [FOLDER_GLYPH + " " + root.name])
            top.setData(0, Qt.UserRole, root)
            self._populate(top, root, depth=0)
        self.file_tree.expandToDepth(0)
        if self._show_summary:
            self.summary_table.load(CALIBRATION_SUMMARY)

    def _populate(self, parent_item: QTreeWidgetItem, directory: Path, depth: int) -> None:
        if depth > 5:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for path in entries:
            if path.is_dir():
                if path.name.startswith(".") or path.name == "__pycache__":
                    continue
                child = QTreeWidgetItem(parent_item, [FOLDER_GLYPH + " " + path.name])
                child.setData(0, Qt.UserRole, path)
                self._populate(child, path, depth + 1)
            else:
                if path.suffix.lower() not in SHOW_SUFFIXES:
                    continue
                glyph = FILE_GLYPHS.get(path.suffix.lower(), "📄")
                child = QTreeWidgetItem(parent_item, [f"{glyph} {path.name}"])
                child.setData(0, Qt.UserRole, path)
                child.setToolTip(0, str(path))

    # ---- selection --------------------------------------------------------

    def _show_tab(self, key: str) -> None:
        widget = self._tab_widgets.get(key)
        if widget is not None:
            self.tabs.setCurrentWidget(widget)

    def _on_file_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, Qt.UserRole)
        if not isinstance(path, Path) or not path.is_file():
            return
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            if self.image_view.load(path):
                self.image_caption.setText(self.image_view.info_text())
                self._show_tab("image")
        elif suffix in {".json", ".jsonnet"}:
            if self._show_summary and path.resolve() == CALIBRATION_SUMMARY.resolve():
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    visible = visible_summary_document(document)
                    loaded = self.json_tree.load_document(visible)
                    raw_text = json.dumps(visible, indent=2, ensure_ascii=False)
                except (OSError, json.JSONDecodeError):
                    loaded = False
                    raw_text = path.read_text(encoding="utf-8", errors="replace")
            else:
                loaded = self.json_tree.load(path)
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            self.raw_view.setPlainText(raw_text)
            if loaded:
                self._show_tab("json")
            else:
                self._show_tab("raw")
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                text = f"读取失败: {exc}"
            self.raw_view.setPlainText(text)
            self._show_tab("raw")
