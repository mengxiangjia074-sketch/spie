"""One generic page per TaskSpec: header, settings form, run controls."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from GUI.core.jsonnet_store import (
    ConfigError,
    load_effective,
    set_values,
)
from GUI.core.paths import PROJECT_ROOT
from GUI.core.runner import TaskManager, TaskSpec
from GUI.widgets.forms import SettingsForm


def _relative(path) -> str:
    """路径显示为项目相对路径 (项目外则原样), 避免过长的绝对路径。

    路径是不含空格的单个长单词, wordWrap 压不下 QLabel 的最小宽度; 在分隔符
    后插入零宽空格 (ZWSP) 提供换行点, 显示效果不变。
    """
    if path is None:
        return ""
    try:
        text = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        text = str(path)
    return (
        text.replace("\\", "\\​").replace("/", "/​")
    )


class TaskPage(QWidget):
    """Settings + run controls for one detection script."""

    run_requested = Signal(str)  # task id

    def __init__(self, spec: TaskSpec, manager: TaskManager, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.manager = manager

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)

        outer.addWidget(self._build_header())

        # 只有配置表单在滚动区内; 状态行与按钮栏固定在页面底部,
        # 不需要滚到最底就能看到“保存并运行”。
        # 注意保留水平滚动条: 禁用会把表单最小宽度顶到整页, 挤压 Dock 分隔条。
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 6, 0)
        page_layout.setSpacing(10)

        self.form = SettingsForm(spec.fields)
        self.form.changed.connect(self._on_form_changed)
        page_layout.addWidget(self.form)
        page_layout.addStretch(1)
        scroll.setWidget(page)

        self.status_label = QLabel()
        self.status_label.setObjectName("HintLabel")
        outer.addWidget(scroll, 1)
        outer.addWidget(self.status_label)
        outer.addLayout(self._build_buttons())

        manager.busy_changed.connect(self._on_busy_changed)
        manager.task_finished.connect(self._on_task_finished)

        self.reload()

    # ---- UI ----------------------------------------------------------------

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("HeaderCard")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        title = QLabel(self.spec.title)
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(self.spec.subtitle)
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # 显示项目相对路径 (完整路径放 tooltip); 相对路径 + 零宽空格断行点,
        # 避免长路径把页面最小宽度撑大, 进而冻结 Dock 分隔条。
        script_label = QLabel(f"脚本  {_relative(self.spec.script)}")
        script_label.setObjectName("PathLabel")
        script_label.setWordWrap(True)
        script_label.setToolTip(str(self.spec.script))
        config_label = QLabel(f"配置  {_relative(self.spec.config)}")
        config_label.setObjectName("PathLabel")
        config_label.setWordWrap(True)
        config_label.setToolTip(str(self.spec.config))
        if self.spec.config is not None:
            layout.addWidget(script_label)
            layout.addWidget(config_label)
        else:
            layout.addWidget(script_label)
        return header

    def _build_buttons(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self.reload_button = QPushButton("重新读取")
        self.reload_button.setToolTip("放弃未保存的修改, 重新从配置文件读取")
        self.reload_button.clicked.connect(self.reload)

        self.save_button = QPushButton("保存设置")
        self.save_button.setToolTip(
            "把修改的键以外科手术方式写回配置文件:\n"
            "注释、import 与未修改内容保持原样。"
        )
        self.save_button.clicked.connect(self.save)

        self.run_button = QPushButton("保存并运行")
        self.run_button.setProperty("class", "primary")
        self.run_button.setToolTip("保存修改后, 以子进程运行脚本并在下方控制台显示输出")
        self.run_button.clicked.connect(self._on_run_clicked)

        bar.addWidget(self.reload_button)
        bar.addWidget(self.save_button)
        bar.addStretch(1)
        bar.addWidget(self.run_button)
        return bar

    # ---- actions -------------------------------------------------------------

    def reload(self) -> None:
        """(Re)load effective values from the config file into the form."""
        if self.spec.config is None:
            return
        try:
            values = load_effective(self.spec.config)
        except ConfigError as exc:
            self.form.apply({})
            self._set_status(f"配置读取失败: {exc}", error=True)
            self.run_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.reload_button.setEnabled(True)
            return
        self.form.apply(values)
        self.save_button.setEnabled(True)
        self._on_busy_changed(self.manager.busy())

    def save(self) -> bool:
        """Write changed keys back to the config file. Returns success."""
        if self.spec.config is None:
            return True
        updates = self.form.updates()
        if not updates:
            self._set_status("没有修改, 无需保存")
            return True
        try:
            values = set_values(self.spec.config, updates)
        except ConfigError as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            self._set_status(f"保存失败: {exc}", error=True)
            return False
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法写入配置文件: {exc}")
            return False
        self.form.apply(values)
        self._set_status(f"已保存 {len(updates)} 项修改")
        return True

    def _on_run_clicked(self) -> None:
        if not self.save():
            return
        self.run_requested.emit(self.spec.id)

    # ---- state -----------------------------------------------------------------

    def _set_status(self, text: str, error: bool = False) -> None:
        prefix = "✖ " if error else "● "
        self.status_label.setText(prefix + text)

    def _on_form_changed(self, count: int) -> None:
        if count:
            self._set_status(f"有 {count} 项未保存的修改")
        else:
            self._set_status("配置与文件一致")

    def _on_busy_changed(self, busy: bool) -> None:
        self.run_button.setEnabled(not busy)
        self.run_button.setText("任务运行中…" if busy else "保存并运行")

    def _on_task_finished(self, task_id: str, result: str, exit_code: int) -> None:
        if task_id != self.spec.id:
            return
        if result == "成功":
            self._set_status("上次运行成功")
        elif result == "已停止":
            self._set_status("上次运行被手动停止")
        else:
            self._set_status(f"上次运行失败: {result}", error=True)
