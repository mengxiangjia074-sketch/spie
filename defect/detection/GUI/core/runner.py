"""Run detection scripts as child processes and stream their output.

One task at a time: camera, stage and lens are exclusive hardware, so the
TaskManager refuses to start a second script while one is running.  Output is
decoded as UTF-8 (children run with PYTHONUTF8=1) and mirrored into a log
file below working_data/gui_logs/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .paths import GUI_LOG_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class TaskSpec:
    """Declarative description of one runnable script."""

    id: str
    nav_label: str
    glyph: str
    title: str
    subtitle: str
    script: Path
    config: Path | None
    fields: list = field(default_factory=list)
    run_note: str = ""


class TaskRunner(QObject):
    output = Signal(str, str)  # channel ("stdout"/"stderr"/"sys"), text line
    finished = Signal(int, str)  # exit code, status text

    TERMINATE_GRACE_MS = 3000

    def __init__(self, spec: TaskSpec, log_path: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.spec = spec
        self.log_path = log_path
        self.started_at = datetime.now()
        self._stop_requested = False
        self._buffers = {"stdout": "", "stderr": ""}
        self._log_file = None
        self._process: QProcess | None = None

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        GUI_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w", encoding="utf-8", newline="")
        self._emit("sys", f"== 任务开始 {self.spec.title} ==")
        self._emit("sys", f"== 脚本: {self.spec.script}")
        self._emit("sys", f"== Python: {sys.executable}")
        self._emit("sys", f"== 日志: {self.log_path}")

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-u", str(self.spec.script)])
        process.setWorkingDirectory(str(PROJECT_ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_finished)
        self._process = process
        process.start()

    def request_stop(self) -> None:
        if self._process is None or self._process.state() == QProcess.NotRunning:
            return
        self._stop_requested = True
        self._emit("sys", "== 请求停止任务 (强制终止进程) ==")
        self._process.terminate()
        QTimer.singleShot(self.TERMINATE_GRACE_MS, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._emit("sys", "== 进程未自行退出, 强制结束 ==")
            self._process.kill()

    # ---- output ----------------------------------------------------------

    def _read_stdout(self) -> None:
        self._consume("stdout", self._process.readAllStandardOutput().data())

    def _read_stderr(self) -> None:
        self._consume("stderr", self._process.readAllStandardError().data())

    def _consume(self, channel: str, data: bytes) -> None:
        self._buffers[channel] += data.decode("utf-8", errors="replace")
        buffer = self._buffers[channel]
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            self._emit(channel, line.rstrip("\r"))
        self._buffers[channel] = buffer

    def _emit(self, channel: str, text: str) -> None:
        if self._log_file is not None:
            self._log_file.write(text + "\n")
            self._log_file.flush()
        self.output.emit(channel, text)

    def _on_process_error(self, error) -> None:
        if error == QProcess.Crashed and self._stop_requested:
            return  # handled by finished()
        if error in (QProcess.FailedToStart, QProcess.Crashed, QProcess.Timedout):
            self._emit("sys", f"== 进程错误: {error} ==")

    def _on_finished(self, exit_code: int, status) -> None:
        for channel in ("stdout", "stderr"):
            remainder = self._buffers[channel].rstrip("\r\n")
            if remainder:
                self._emit(channel, remainder)
                self._buffers[channel] = ""
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if self._stop_requested:
            result = "已停止"
        elif exit_code == 0:
            result = "成功"
        else:
            result = f"失败 (退出码 {exit_code})"
        self._emit("sys", f"== 任务结束: {result}, 用时 {elapsed:.1f} 秒 ==")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self.finished.emit(exit_code, result)


class TaskManager(QObject):
    """Owns the single active runner and the session run history."""

    task_started = Signal(object)  # TaskSpec
    task_output = Signal(str, str, str)  # task id, channel, text
    task_finished = Signal(str, str, int)  # task id, result, exit code
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._runner: TaskRunner | None = None
        self.history: list[dict[str, Any]] = []

    # ---- queries ---------------------------------------------------------

    def busy(self) -> bool:
        return self._runner is not None

    def current(self) -> TaskRunner | None:
        return self._runner

    # ---- control ---------------------------------------------------------

    def start(self, spec: TaskSpec) -> bool:
        if self._runner is not None:
            return False
        GUI_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = GUI_LOG_DIR / f"{stamp}_{spec.id}.log"
        runner = TaskRunner(spec, log_path, self)
        runner.output.connect(
            lambda channel, text, task_id=spec.id: self.task_output.emit(
                task_id, channel, text
            )
        )
        runner.finished.connect(lambda code, result, task_id=spec.id: self._finish(task_id, code, result))
        self._runner = runner
        self.history.append(
            {
                "id": spec.id,
                "title": spec.title,
                "started": runner.started_at,
                "log": log_path,
                "result": None,
                "exit_code": None,
            }
        )
        self.busy_changed.emit(True)
        self.task_started.emit(spec)
        runner.start()
        return True

    def stop_current(self) -> None:
        if self._runner is not None:
            self._runner.request_stop()

    def _finish(self, task_id: str, exit_code: int, result: str) -> None:
        runner = self._runner
        self._runner = None
        if self.history:
            self.history[-1]["result"] = result
            self.history[-1]["exit_code"] = exit_code
            self.history[-1]["ended"] = datetime.now()
        runner.deleteLater()
        self.busy_changed.emit(False)
        self.task_finished.emit(task_id, result, exit_code)


def open_in_explorer(path: Path) -> None:
    """Open a file or directory with the platform desktop shell."""
    path = Path(path)
    target = path.parent if path.is_file() else path
    if not target.is_dir():
        return
    if sys.platform == "win32":
        os.startfile(str(target))  # noqa: S606 (intended)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
