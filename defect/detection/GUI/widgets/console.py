"""Bottom dock showing live output of the running task."""

from __future__ import annotations

import html

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from GUI.core.paths import GUI_LOG_DIR, WORKING_DATA
from GUI.core.runner import TaskManager, open_in_explorer

MONO = "Consolas"
CALIBRATION_ERROR_PREFIX = "标定误差:"

# 非任务通道 -> 主题色键 (位移台通讯日志: 发送/接收/信息/错误)
_CHANNEL_COLOR_KEYS = {
    "comm-send": "accent",
    "comm-recv": "ok",
    "comm-info": "text_dim",
    "comm-error": "error",
}


class ConsolePanel(QWidget):
    """Live log view for one functional area."""

    def __init__(
        self,
        manager: TaskManager,
        parent=None,
        *,
        task_ids=None,
        listen_to_tasks: bool = True,
    ):
        super().__init__(parent)
        self._manager = manager
        self._task_ids = None if task_ids is None else frozenset(task_ids)
        self._listen_to_tasks = listen_to_tasks
        self._stop_button: QPushButton | None = None

        self._build_ui()

        if listen_to_tasks:
            manager.task_started.connect(self._on_started)
            manager.task_output.connect(self._on_output)
            manager.task_finished.connect(self._on_finished)
            manager.busy_changed.connect(self._on_busy_changed)

    # ---- UI ---------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addStretch(1)

        if self._listen_to_tasks:
            stop = QPushButton("强制停止")
            stop.setProperty("class", "danger")
            stop.setToolTip(
                "立即终止子进程。\n注意: 若位移台正在运动, "
                "脚本的安全停机逻辑不会执行,\n请尽量等当前动作结束后再停止。"
            )
            stop.setEnabled(False)
            stop.clicked.connect(self._manager.stop_current)
            bar.addWidget(stop)
            self._stop_button = stop

        for text, handler in (
            ("清空", self._clear),
            ("复制日志", self._copy_to_clipboard),
            ("日志目录", lambda: open_in_explorer(GUI_LOG_DIR)),
            ("输出目录", lambda: open_in_explorer(WORKING_DATA)),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            bar.addWidget(button)

        self.view = QPlainTextEdit()
        self.view.setObjectName("LogView")
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(20000)
        font = QFont(MONO)
        font.setStyleHint(QFont.Monospace)
        self.view.setFont(font)

        root.addLayout(bar)
        root.addWidget(self.view, 1)

    # ---- slots --------------------------------------------------------------

    def _accepts_task(self, task_id: str) -> bool:
        return self._task_ids is None or task_id in self._task_ids

    def _on_started(self, spec) -> None:
        if not self._accepts_task(spec.id):
            return
        self.view.clear()
        if spec.run_note:
            self._append_html(
                f'<span style="color:{self._colors()["accent"]}">'
                f"▶ {html.escape(spec.run_note)}</span>"
            )

    def _on_output(self, task_id: str, channel: str, text: str) -> None:
        if not text or (task_id and not self._accepts_task(task_id)):
            return
        escaped = html.escape(text)
        lowered = text.lstrip().lower()
        colors = self._colors()
        if channel in _CHANNEL_COLOR_KEYS:
            color = colors[_CHANNEL_COLOR_KEYS[channel]]
        elif channel == "stderr":
            color = colors["error"]
        elif lowered.startswith(("error:", "error ", "traceback")):
            color = colors["error"]
        elif lowered.startswith(("warning:", "warn:")):
            color = colors["warn"]
        elif text.lstrip().startswith(CALIBRATION_ERROR_PREFIX):
            color = colors["accent"]
        elif channel == "sys":
            color = colors["text_dim"]
        else:
            color = colors["text"]
        self._append_html(f'<span style="color:{color}">{escaped}</span>')
        scrollbar = self.view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        if at_bottom:
            self.view.moveCursor(QTextCursor.End)

    def append_line(self, channel: str, text: str) -> None:
        """非任务来源 (如位移台通讯日志) 追加一行到控制台。"""
        self._on_output("", channel, text)

    @staticmethod
    def _colors() -> dict:
        from GUI.core.theme import current_palette

        return current_palette()

    def _on_finished(self, task_id: str, result: str, exit_code: int) -> None:
        pass  # 结果显示在窗口状态栏, 控制台不再有头部小字

    def _on_busy_changed(self, busy: bool) -> None:
        if self._stop_button is not None:
            runner = self._manager.current()
            task_id = runner.spec.id if runner is not None else ""
            self._stop_button.setEnabled(
                busy and bool(task_id) and self._accepts_task(task_id)
            )

    # ---- helpers ------------------------------------------------------------

    def _append_html(self, fragment: str) -> None:
        self.view.appendHtml(fragment)

    def _clear(self) -> None:
        self.view.clear()

    def _copy_to_clipboard(self) -> None:
        QGuiApplication.clipboard().setText(self.view.toPlainText())
