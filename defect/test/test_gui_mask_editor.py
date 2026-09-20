import os
import sys
from pathlib import Path

import cv2
import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from GUI.pages.pnp_mask_page import ImagePointView  # noqa: E402


def test_mask_items_can_move_delete_and_add(tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "image.png"
    assert cv2.imwrite(str(image_path), np.zeros((100, 140, 3), dtype=np.uint8))
    view = ImagePointView("test")
    assert view.load(image_path)
    view.set_mask_records(
        [
            {
                "instance_id": 1,
                "designator": "R1",
                "category": "resistor",
                "target_polygon_px": [[10, 10], [30, 10], [30, 20], [10, 20]],
            },
            {
                "instance_id": 2,
                "designator": "C1",
                "category": "capacitor",
                "target_polygon_px": [[50, 40], [65, 40], [65, 55], [50, 55]],
            },
        ]
    )
    view.set_mask_editing(True)

    first = view._mask_items[0]
    first.setPos(7.0, 9.0)
    moved = view.mask_records()[0]
    assert moved["target_polygon_px"][0] == [17.0, 19.0]
    assert moved["manually_edited"] is True

    first.setSelected(True)
    assert view.delete_selected_masks() == 1
    assert [record["designator"] for record in view.mask_records()] == ["C1"]

    added = view.add_manual_mask(QRectF(80, 60, 25, 18))
    assert added is not None
    records = view.mask_records()
    assert len(records) == 2
    assert records[-1]["designator"] == "MANUAL_001"
    assert records[-1]["target_polygon_px"] == [
        [80.0, 60.0],
        [105.0, 60.0],
        [105.0, 78.0],
        [80.0, 78.0],
    ]
    view.close()
    app.processEvents()
