"""Central path constants for the GUI package."""

from pathlib import Path

# GUI 包位于 <project>/detection/GUI/core/paths.py
GUI_DIR = Path(__file__).resolve().parents[1]
DETECTION_DIR = GUI_DIR.parent
PROJECT_ROOT = DETECTION_DIR.parent

CALIBRATION_DIR = DETECTION_DIR / "calibration"
CAPTURE_DIR = DETECTION_DIR / "capture"

# ---- 脚本 ----
SCRIPTS = {
    "height_check": CALIBRATION_DIR / "run_height_check.py",
    "lens": CALIBRATION_DIR / "calibrate_lens.py",
    "board": CALIBRATION_DIR / "calibrate_board_to_pixel.py",
    "zoom_transform": CALIBRATION_DIR / "calibrate_zoom_pixel_transform.py",
    "stage": CALIBRATION_DIR / "calibrate_stage_to_pixel.py",
    "autofocus": CALIBRATION_DIR / "autofocus_pcb.py",
    "mosaic": CAPTURE_DIR / "capture_pcb_two_zooms.py",
}

# ---- 配置文件 (GUI 只做外科手术式键值修改, 不重排文件) ----
CONFIGS = {
    "height_check": PROJECT_ROOT / "config" / "height_check" / "capture.jsonnet",
    "lens": CALIBRATION_DIR / "configs" / "lens" / "calibrate_lens.jsonnet",
    "board": CALIBRATION_DIR / "configs" / "board_to_pixel" / "board2pixel.jsonnet",
    "zoom_transform": CALIBRATION_DIR / "configs" / "zoom_pixel_transform" / "zoom_pixel_transform.jsonnet",
    "stage": CALIBRATION_DIR / "configs" / "stage_to_pixel" / "stage2pixel.jsonnet",
    "autofocus": CALIBRATION_DIR / "configs" / "autofocus" / "autofocus.jsonnet",
    "capture": PROJECT_ROOT / "config" / "pcb_two_zoom_capture" / "capture.jsonnet",
}

# ---- 输出 ----
WORKING_DATA = PROJECT_ROOT / "working_data"
GUI_LOG_DIR = WORKING_DATA / "gui_logs"
GUI_BACKUP_DIR = WORKING_DATA / "gui_backups"
CALIBRATION_OUTPUT_DIR = CALIBRATION_DIR / "output"
CALIBRATION_SUMMARY = CALIBRATION_OUTPUT_DIR / "calibrate.json"

# 主界面功能卡片图标 (SVG, 文件名即功能名)
IMAGES_DIR = GUI_DIR / "images"
ENTRY_ICONS = {
    "capture": IMAGES_DIR / "图像采集.svg",
    "camera": IMAGES_DIR / "相机控制.svg",
    "calibration": IMAGES_DIR / "标定.svg",
    "lens": IMAGES_DIR / "镜头.svg",
    "motion": IMAGES_DIR / "位移台控制.svg",
    "rfid": IMAGES_DIR / "RFID.svg",
}

# ---- 结果浏览 ----
# 标定工作区只展示这些结果目录; working_data 下其余内容归图像采集工作区
CALIBRATION_RESULT_NAMES = (
    "calibrate",
    "height_check",
    "pcb_autofocus",
    "stage_command_to_pixel",
    "zoom_pixel_transform",
)

# Retained because existing capture workflows consume this calibration, but
# intentionally hidden from both GUI result workspaces.
HIDDEN_CALIBRATION_RESULT_NAMES = (
    "board_pixel_affine",
)


def calibration_result_roots():
    """标定结果页的根目录: 指定的 working_data 子目录 + calibration/output。"""
    roots = [WORKING_DATA / name for name in CALIBRATION_RESULT_NAMES]
    roots.append(CALIBRATION_OUTPUT_DIR)
    return [root for root in roots if root.exists()]


def capture_result_roots():
    """采集结果页的根目录: working_data 下除标定目录外的全部内容。"""
    if not WORKING_DATA.is_dir():
        return []
    excluded = set(CALIBRATION_RESULT_NAMES) | set(HIDDEN_CALIBRATION_RESULT_NAMES)
    return sorted(
        (
            path
            for path in WORKING_DATA.iterdir()
            if path.name not in excluded and not path.name.startswith(".")
        ),
        key=lambda path: (not path.is_dir(), path.name.lower()),
    )
