"""Main window: mode-based navigation.

顶层是主界面 (三个功能入口), 点击后整屏切换到对应功能界面:
- 图像采集工作区: 侧边栏 (总览/PCB拍照/采集结果/设置) + 页栈 + 控制台 Dock。
- 相机控制界面: 实时预览、拍照、录像与 UVC 参数。
- 镜头控制界面: 单独的 LensControlPage。
- 标定工作区: 可见标定任务页 + 已过滤的标定结果页。
- 位移台控制界面: 单独的 MotionControlPage (detection/motion_gui.py 的移植,
  双轴 Modbus RTU 串口控制)。
每个功能界面左上有"返回主界面"按钮, 主界面不显示侧边栏。
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from GUI import __version__
from GUI.core.paths import (
    CALIBRATION_DIR,
    calibration_result_roots,
    capture_result_roots,
)
from GUI.core.runner import TaskManager
from GUI.core.theme import THEMES, apply_theme, current_palette
from GUI.pages.home import HomePage
from GUI.pages.camera_control import CameraControlPage
from GUI.pages.component_inspection_page import ComponentInspectionPage
from GUI.pages.defect_sim_page import DefectSimPage
from GUI.pages.lens_control import LensControlPage
from GUI.pages.motion_control import MotionControlPage
from GUI.pages.pnp_mask_page import PnpMaskPage
from GUI.pages.registry import GUI_HIDDEN_TASK_IDS, TASKS, task_by_id
from GUI.pages.rfid_control import RfidControlPage
from GUI.pages.results import ResultsPage
from GUI.pages.settings_page import SettingsPage
from GUI.pages.task_page import TaskPage
from GUI.widgets.console import ConsolePanel

RESULTS_KEY = "results"            # 标定工作区的标定结果页
CAPTURE_RESULTS_KEY = "capture_results"  # 图像采集工作区的采集结果页
DEFECT_SIM_KEY = "defect_sim"
PNP_MASK_KEY = "pnp_mask"
COMPONENT_INSPECTION_KEY = "component_inspection"

# 顶层界面 (模式)
MODE_HOME = "home"
MODE_CAPTURE = "capture"
MODE_CAMERA = "camera"
MODE_LENS = "lens"
MODE_CALIBRATION = "calibration"
MODE_MOTION = "motion"
MODE_RFID = "rfid"
MODE_SETTINGS = "settings"

CONSOLE_MODES = (
    MODE_CAPTURE,
    MODE_CAMERA,
    MODE_LENS,
    MODE_CALIBRATION,
    MODE_MOTION,
    MODE_RFID,
)

MODE_TITLES = {
    MODE_HOME: "",
    MODE_CAPTURE: " — 图像采集",
    MODE_CAMERA: " — 相机控制",
    MODE_LENS: " — 镜头控制",
    MODE_CALIBRATION: " — 标定",
    MODE_MOTION: " — 位移台控制",
    MODE_RFID: " — RFID检测",
    MODE_SETTINGS: " — 设置",
}

NAV_ICON_SIZE = 20
NAV_FONT = "Segoe UI Symbol"
SIDEBAR_WIDTH = 200
MIN_WORKSPACE_HEIGHT = 260


def _nav_icon(glyph: str, color: str) -> QIcon:
    """Draw a nav glyph centered in a fixed-size pixmap.

    Centering each glyph in the same box keeps the icon column a constant
    width, so every label starts at the same x position regardless of the
    glyph's natural width.
    """
    pixmap = QPixmap(NAV_ICON_SIZE, NAV_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHints(
        QPainter.Antialiasing | QPainter.TextAntialiasing, True
    )
    font = QFont(NAV_FONT)
    font.setPixelSize(int(NAV_ICON_SIZE * 0.8))
    painter.setFont(font)
    painter.setPen(QColor(color))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, glyph)
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"LensDetect 视觉检测控制台 v{__version__}")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 680)

        self.manager = TaskManager(self)

        # 运行控制台默认隐藏, 由各功能界面的"运行控制台"按钮按需弹出
        self._current_mode = MODE_HOME
        self._console_visible_by_mode = {
            mode: False for mode in CONSOLE_MODES
        }
        self._console_buttons: dict[str, list[QPushButton]] = {
            mode: [] for mode in CONSOLE_MODES
        }

        self._build_central()
        self._build_console_dock()
        self._build_status_bar()

        self.settings = QSettings("LensDetect", "GUI")
        theme = self.settings.value("theme", "dark")
        if theme not in THEMES:
            theme = "dark"
        self.apply_theme(theme)
        self.enter_mode(MODE_HOME)

    # ---- construction -----------------------------------------------------

    def _build_central(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(central)

        self.pages: dict[str, QWidget] = {}
        self._workspaces: dict[str, tuple[QListWidget, QStackedWidget]] = {}

        self.mode_stack = QStackedWidget()
        # 放宽底部 Dock 的向上拖动范围；各页面内部滚动区负责处理压缩后的内容。
        self.mode_stack.setMinimumHeight(MIN_WORKSPACE_HEIGHT)
        self.home_page = self._build_home_screen()
        self._capture_screen = self._build_capture_screen()
        self._camera_screen = self._build_camera_screen()
        self._lens_screen = self._build_lens_screen()
        self._calibration_screen = self._build_calibration_screen()
        self._motion_screen = self._build_motion_screen()
        self._rfid_screen = self._build_rfid_screen()
        self._settings_screen = self._build_settings_screen()
        self.mode_stack.addWidget(self.home_page)
        self.mode_stack.addWidget(self._capture_screen)
        self.mode_stack.addWidget(self._camera_screen)
        self.mode_stack.addWidget(self._lens_screen)
        self.mode_stack.addWidget(self._calibration_screen)
        self.mode_stack.addWidget(self._motion_screen)
        self.mode_stack.addWidget(self._rfid_screen)
        self.mode_stack.addWidget(self._settings_screen)
        layout.addWidget(self.mode_stack)

    def _build_home_screen(self) -> QWidget:
        home = HomePage()
        home.capture_clicked.connect(lambda: self.enter_mode(MODE_CAPTURE))
        home.camera_clicked.connect(lambda: self.enter_mode(MODE_CAMERA))
        home.lens_clicked.connect(lambda: self.enter_mode(MODE_LENS))
        home.calibration_clicked.connect(
            lambda: self.enter_mode(MODE_CALIBRATION)
        )
        home.motion_clicked.connect(lambda: self.enter_mode(MODE_MOTION))
        home.rfid_clicked.connect(lambda: self.enter_mode(MODE_RFID))
        home.settings_clicked.connect(lambda: self.enter_mode(MODE_SETTINGS))
        return home

    def _build_capture_screen(self) -> QWidget:
        """图像采集工作区: PCB拍照 / 缺陷模拟 / 采集结果。"""
        pages: dict[str, QWidget] = {}
        nav_entries = []
        for spec in TASKS:
            if spec.id in GUI_HIDDEN_TASK_IDS:
                continue
            if not spec.script.is_relative_to(CALIBRATION_DIR):
                page = TaskPage(spec, self.manager)
                page.run_requested.connect(self._start_task)
                self.pages[spec.id] = page
                pages[spec.id] = page
                nav_entries.append((spec.id, spec.glyph, spec.nav_label))

        self.defect_sim_page = DefectSimPage()
        self.pages[DEFECT_SIM_KEY] = self.defect_sim_page
        pages[DEFECT_SIM_KEY] = self.defect_sim_page
        nav_entries.append((DEFECT_SIM_KEY, "◩", "缺陷模拟"))

        self.pnp_mask_page = PnpMaskPage(self.manager)
        self.pnp_mask_page.capture_requested.connect(
            lambda: self._start_task("mosaic")
        )
        self.pages[PNP_MASK_KEY] = self.pnp_mask_page
        pages[PNP_MASK_KEY] = self.pnp_mask_page
        nav_entries.append((PNP_MASK_KEY, "▦", "坐标 Mask 对齐"))

        self.component_inspection_page = ComponentInspectionPage(self.manager)
        self.component_inspection_page.capture_requested.connect(
            lambda: self._start_task("mosaic")
        )
        self.pages[COMPONENT_INSPECTION_KEY] = self.component_inspection_page
        pages[COMPONENT_INSPECTION_KEY] = self.component_inspection_page
        nav_entries.append((COMPONENT_INSPECTION_KEY, "✓", "元器件完整性检测"))

        self.pages[CAPTURE_RESULTS_KEY] = ResultsPage(
            title="采集结果",
            roots_provider=capture_result_roots,
            show_summary=False,
        )
        pages[CAPTURE_RESULTS_KEY] = self.pages[CAPTURE_RESULTS_KEY]
        nav_entries.append((CAPTURE_RESULTS_KEY, "▤", "采集结果"))

        screen, _, _ = self._build_workspace_screen(MODE_CAPTURE, pages, nav_entries)
        return screen

    def _build_calibration_screen(self) -> QWidget:
        """标定工作区: detection/calibration/ 下的任务页 + 标定结果。"""
        pages: dict[str, QWidget] = {}
        nav_entries = []
        for spec in TASKS:
            if spec.id in GUI_HIDDEN_TASK_IDS:
                continue
            if not spec.script.is_relative_to(CALIBRATION_DIR):
                continue
            page = TaskPage(spec, self.manager)
            page.run_requested.connect(self._start_task)
            self.pages[spec.id] = page
            pages[spec.id] = page
            nav_entries.append((spec.id, spec.glyph, spec.nav_label))

        self.pages[RESULTS_KEY] = ResultsPage(
            title="标定结果",
            roots_provider=calibration_result_roots,
            show_summary=True,
        )
        pages[RESULTS_KEY] = self.pages[RESULTS_KEY]
        nav_entries.append((RESULTS_KEY, "▤", "标定结果"))

        screen, _, _ = self._build_workspace_screen(
            MODE_CALIBRATION, pages, nav_entries
        )
        return screen

    def _build_workspace_screen(
        self, mode: str, pages: dict[str, QWidget], nav_entries: list
    ) -> tuple[QWidget, QListWidget, QStackedWidget]:
        """通用工作区: 返回按钮 + 本功能区侧边栏 + 页栈。"""
        screen = QWidget()
        layout = QHBoxLayout(screen)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        side = QVBoxLayout()
        side.setContentsMargins(10, 10, 0, 10)
        side.setSpacing(6)
        back = self._back_button()
        back.setFixedWidth(SIDEBAR_WIDTH)
        side.addWidget(back)
        console_button = self._console_button(mode)
        console_button.setFixedWidth(SIDEBAR_WIDTH)
        side.addWidget(console_button)

        sidebar = QListWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setIconSize(QSize(NAV_ICON_SIZE, NAV_ICON_SIZE))
        for key, glyph, label in nav_entries:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            item.setData(Qt.UserRole + 1, glyph)
            sidebar.addItem(item)
        side.addWidget(sidebar)
        layout.addLayout(side)

        stack = QStackedWidget()
        for key, page in pages.items():
            stack.addWidget(page)
        layout.addWidget(stack, 1)

        sidebar.currentRowChanged.connect(
            lambda row, sb=sidebar, st=stack: self._switch_page(sb, st, row)
        )
        sidebar.setCurrentRow(0)

        self._workspaces[mode] = (sidebar, stack)
        return screen, sidebar, stack

    def _build_lens_screen(self) -> QWidget:
        """镜头控制界面: 返回按钮 + 镜头控制页。"""
        return self._build_single_page_screen(
            MODE_LENS, "lens_page", lambda: LensControlPage(self.manager))

    def _build_camera_screen(self) -> QWidget:
        """相机控制界面: 返回/控制台按钮 + 不滚动的实时预览控制页。"""
        return self._build_single_page_screen(
            MODE_CAMERA,
            "camera_page",
            lambda: CameraControlPage(self.manager),
            scroll=False,
        )

    def _build_motion_screen(self) -> QWidget:
        """位移台控制界面: 返回按钮 + 位移台控制页。"""
        return self._build_single_page_screen(
            MODE_MOTION,
            "motion_page",
            lambda: MotionControlPage(self.manager),
        )

    def _build_rfid_screen(self) -> QWidget:
        """RFID 检测界面: 返回按钮 + E720 RFID 控制页。"""
        return self._build_single_page_screen(
            MODE_RFID, "rfid_page", RfidControlPage)

    def _build_settings_screen(self) -> QWidget:
        """设置界面 (从主界面进入): 返回按钮 + 设置页, 无控制台按钮。"""
        screen = self._build_single_page_screen(
            MODE_SETTINGS, "settings_page", SettingsPage, with_console=False)
        self.settings_page.theme_changed.connect(self.apply_theme)
        return screen

    def _build_single_page_screen(self, mode: str, attr: str, page_factory,
                                  with_console: bool = True,
                                  scroll: bool = True) -> QWidget:
        """独占整屏的功能界面: 返回按钮 + 单个控制页 (页面存到 self.<attr>)。

        默认把控制页包进 QScrollArea；实时相机页可关闭外层滚动，以便预览
        始终填充剩余空间。
        设置等非工作界面不生成"运行控制台"按钮 (with_console=False)。
        """
        screen = QWidget()
        layout = QVBoxLayout(screen)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 10, 10, 0)
        bar_layout.addWidget(self._back_button())
        bar_layout.addStretch(1)
        if with_console:
            bar_layout.addWidget(self._console_button(mode))
        layout.addWidget(bar)

        page = page_factory()
        setattr(self, attr, page)
        if scroll:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(page)
            layout.addWidget(scroll_area, 1)
        else:
            layout.addWidget(page, 1)
        return screen

    def _back_button(self) -> QPushButton:
        button = QPushButton("←  返回主界面")
        button.setObjectName("BackButton")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda: self.enter_mode(MODE_HOME))
        return button

    def _console_button(self, mode: str) -> QPushButton:
        """可勾选的"运行控制台"按钮: 点击弹出/收起控制台。"""
        button = QPushButton("▤  运行控制台")
        button.setObjectName("BackButton")
        button.setCheckable(True)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip("点击显示/隐藏底部运行控制台")
        button.clicked.connect(
            lambda checked, current_mode=mode: self._set_console_visible(
                current_mode, checked
            )
        )
        self._console_buttons.setdefault(mode, []).append(button)
        return button

    def _build_console_dock(self) -> None:
        capture_task_ids = {
            spec.id
            for spec in TASKS
            if spec.id not in GUI_HIDDEN_TASK_IDS
            and not spec.script.is_relative_to(CALIBRATION_DIR)
        }
        calibration_task_ids = {
            spec.id
            for spec in TASKS
            if spec.id not in GUI_HIDDEN_TASK_IDS
            and spec.script.is_relative_to(CALIBRATION_DIR)
        }

        self.console_stack = QStackedWidget()
        self.console_panels = {
            MODE_CAPTURE: ConsolePanel(
                self.manager, task_ids=capture_task_ids
            ),
            MODE_CAMERA: ConsolePanel(
                self.manager, listen_to_tasks=False
            ),
            MODE_LENS: ConsolePanel(
                self.manager, listen_to_tasks=False
            ),
            MODE_CALIBRATION: ConsolePanel(
                self.manager, task_ids=calibration_task_ids
            ),
            MODE_MOTION: ConsolePanel(
                self.manager, listen_to_tasks=False
            ),
            MODE_RFID: ConsolePanel(
                self.manager, listen_to_tasks=False
            ),
        }
        for mode in CONSOLE_MODES:
            self.console_stack.addWidget(self.console_panels[mode])

        dock = QDockWidget("运行控制台", self)
        dock.setObjectName("ConsoleDock")
        dock.setWidget(self.console_stack)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.visibilityChanged.connect(self._on_console_visibility_changed)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.console_dock = dock
        self.defect_sim_page.console_output.connect(self._on_defect_sim_output)
        self.component_inspection_page.console_output.connect(
            self._on_defect_sim_output
        )
        self.camera_page.console_output.connect(
            self.console_panels[MODE_CAMERA].append_line
        )
        self.motion_page.console_output.connect(
            self.console_panels[MODE_MOTION].append_line
        )
        self.rfid_page.console_output.connect(
            self.console_panels[MODE_RFID].append_line
        )

    def _on_defect_sim_output(self, channel: str, text: str) -> None:
        self.console_panels[MODE_CAPTURE].append_line(channel, text)
        if channel == "comm-error" and self._current_mode == MODE_CAPTURE:
            self._set_console_visible(MODE_CAPTURE, True)
            self.console_dock.raise_()

    def _build_status_bar(self) -> None:
        self.status_label = QLabel("空闲")
        self.statusBar().addWidget(self.status_label)
        env_label = QLabel(f"Python  {sys.executable}")
        env_label.setObjectName("PathLabel")
        self.statusBar().addPermanentWidget(env_label)

        self.manager.busy_changed.connect(
            lambda busy: self.status_label.setText(
                "任务运行中…" if busy else "空闲"
            )
        )
        self.manager.task_started.connect(
            lambda spec: self.status_label.setText(f"任务运行中: {spec.title}")
        )
        self.manager.task_finished.connect(
            lambda _id, result, _code: self.status_label.setText(
                f"上次任务: {result}"
            )
        )

    # ---- behavior -----------------------------------------------------------

    def enter_mode(self, mode: str) -> None:
        """整屏切换到主界面或指定的控制/任务工作区。"""
        if mode == MODE_HOME and not self._confirm_return_home():
            return
        screens = {
            MODE_HOME: self.home_page,
            MODE_CAPTURE: self._capture_screen,
            MODE_CAMERA: self._camera_screen,
            MODE_LENS: self._lens_screen,
            MODE_CALIBRATION: self._calibration_screen,
            MODE_MOTION: self._motion_screen,
            MODE_RFID: self._rfid_screen,
            MODE_SETTINGS: self._settings_screen,
        }
        target = screens.get(mode)
        if target is not None:
            self._current_mode = mode
            self.mode_stack.setCurrentWidget(target)
            self.setWindowTitle(
                f"LensDetect 视觉检测控制台 v{__version__}{MODE_TITLES.get(mode, '')}"
            )
            # 主界面不显示运行控制台; 功能界面按用户偏好 (含位移台通讯日志)
            if mode == MODE_HOME:
                self._release_hardware()
            self._sync_console_visibility(mode)

    def _confirm_return_home(self) -> bool:
        """返回主界面前处理占用相机的运行中任务; 拒绝则留在当前界面。"""
        if not self.manager.busy():
            return True
        answer = QMessageBox.question(
            self,
            "任务仍在运行",
            "有检测任务正在运行 (占用相机)。\n返回主界面将强制停止该任务, "
            "确定返回吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        self.manager.stop_current()
        return True

    def _release_hardware(self) -> None:
        """返回主界面时自动断开硬件连接。

        电动镜头、相机、位移台与 RFID 模块由各自交互页持有并排队断开；
        任务子进程占用的相机已在 _confirm_return_home 中确认并强制停止。
        """
        self.lens_page.disconnect_if_connected()
        self.camera_page.disconnect_if_connected()
        self.motion_page.disconnect_if_connected()
        self.rfid_page.disconnect_if_connected()

    def _set_console_visible(self, mode: str, visible: bool) -> None:
        """单独设置某个功能区的控制台显隐状态。"""
        if mode not in self.console_panels:
            return
        self._console_visible_by_mode[mode] = visible
        for button in self._console_buttons.get(mode, []):
            button.setChecked(visible)
        if self._current_mode == mode:
            self._activate_console(mode)
            self.console_dock.setVisible(visible)

    def _sync_console_visibility(self, mode: str) -> None:
        """切换到当前功能自己的控制台，其他功能的内容保持在各自面板。"""
        if mode in self.console_panels:
            self._activate_console(mode)
        visible = self._console_visible_by_mode.get(mode, False)
        self.console_dock.setVisible(visible)
        for button_mode, buttons in self._console_buttons.items():
            checked = self._console_visible_by_mode.get(button_mode, False)
            for button in buttons:
                button.setChecked(checked)

    def _activate_console(self, mode: str) -> None:
        panel = self.console_panels.get(mode)
        if panel is None:
            return
        self.console_stack.setCurrentWidget(panel)
        mode_title = MODE_TITLES.get(mode, "").replace("—", "").strip()
        self.console_dock.setWindowTitle(
            f"{mode_title} - 运行控制台" if mode_title else "运行控制台"
        )

    def _on_console_visibility_changed(self, visible: bool) -> None:
        """Dock 被自身关闭按钮收起时, 同步按钮与偏好。

        主界面触发的隐藏不算用户偏好变化 (回主界面时 _sync_console_visibility
        已在切换页面之后调用, 此时忽略), 否则再次进入功能界面会丢失弹开状态。
        """
        if visible or self._current_mode == MODE_HOME:
            return
        if self._current_mode not in self._console_visible_by_mode:
            return
        self._console_visible_by_mode[self._current_mode] = False
        for button in self._console_buttons.get(self._current_mode, []):
            button.setChecked(False)

    def apply_theme(self, theme_name: str) -> None:
        apply_theme(QApplication.instance(), theme_name)
        self.settings.setValue("theme", theme_name)
        self._refresh_nav_icons()
        settings_page = getattr(self, "settings_page", None)
        if isinstance(settings_page, SettingsPage):
            settings_page.set_theme(theme_name)

    def _refresh_nav_icons(self) -> None:
        """Redraw nav icons with the active theme's accent color."""
        for sidebar, _stack in self._workspaces.values():
            for row in range(sidebar.count()):
                item = sidebar.item(row)
                if item is not None:
                    item.setIcon(_nav_icon(item.data(Qt.UserRole + 1),
                                           current_palette()["accent"]))

    def _switch_page(
        self, sidebar: QListWidget, stack: QStackedWidget, row: int
    ) -> None:
        item = sidebar.item(row)
        if item is None:
            return
        key = item.data(Qt.UserRole)
        widget = self.pages.get(key)
        # 只切换属于该工作区页栈的页面, 防止把别的工作区的页面拉进来
        if widget is not None and stack.indexOf(widget) >= 0:
            stack.setCurrentWidget(widget)
        if isinstance(widget, ResultsPage):
            widget.refresh()

    def _start_task(self, task_id: str) -> None:
        spec = task_by_id(task_id)
        if spec is None:
            return
        if self.component_inspection_page.is_running():
            QMessageBox.warning(
                self,
                "完整性检测正在运行",
                "SAM / RoMa 完整性检测仍在运行，请等待检测完成后再启动拍摄或标定任务。",
            )
            return
        if self.manager.busy():
            QMessageBox.warning(
                self,
                "有任务正在运行",
                "相机 / 位移台 / 电动镜头为独占硬件, 同一时间只能运行一个任务。\n"
                "请等待当前任务结束或先停止它。",
            )
            return
        if self.lens_page.is_connected():
            QMessageBox.warning(
                self,
                "镜头控制页已连接",
                "镜头控制界面当前持有 LensConnect USB 连接。\n"
                "请先在镜头控制界面断开连接, 再启动检测任务。",
            )
            return
        if self.camera_page.is_connected():
            QMessageBox.warning(
                self,
                "相机控制页已连接",
                "相机控制界面当前持有摄像头。\n"
                "请先断开相机，再启动检测任务。",
            )
            return
        if self.motion_page.is_connected():
            QMessageBox.warning(
                self,
                "位移台控制页已连接",
                "位移台控制界面当前持有串口连接。\n"
                "请先在位移台控制界面断开连接, 再启动检测任务。",
            )
            return
        if not spec.script.is_file():
            QMessageBox.critical(self, "脚本不存在", f"找不到脚本:\n{spec.script}")
            return
        # 任务启动时自动弹出控制台并同步按钮
        self._set_console_visible(self._current_mode, True)
        self.console_dock.raise_()
        self.manager.start(spec)

    # ---- close ----------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.component_inspection_page.is_running():
            QMessageBox.information(
                self,
                "完整性检测正在运行",
                "GPU 检测任务仍在运行，请等待完成后再关闭程序。",
            )
            event.ignore()
            return
        if self.manager.busy():
            answer = QMessageBox.question(
                self,
                "任务仍在运行",
                "有任务正在运行, 关闭窗口将强制终止子进程。\n确定要关闭吗?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.manager.stop_current()
        self.lens_page.shutdown_worker()
        self.camera_page.shutdown_worker()
        self.motion_page.shutdown_worker()
        self.rfid_page.shutdown_worker()
        event.accept()
