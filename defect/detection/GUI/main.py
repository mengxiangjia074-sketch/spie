"""GUI entry point.

Run inside the plotly conda environment:

    conda activate plotly
    python detection/GUI/main.py
"""

import sys
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parent
DETECTION_DIR = GUI_DIR.parent

# GUI 以包形式导入 (from GUI.xxx import ...), LensCamera 包在 detection/ 下。
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from GUI.app import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LensDetect")
    app.setOrganizationName("LensDetect")
    app.setStyle("Fusion")

    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
