import os
import json
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
sys.path.insert(0, str(DETECTION_DIR))

try:
    from PySide6.QtWidgets import QApplication

    from GUI.pages.lens_control import (
        DEFAULT_STEP_LENGTH,
        MOTORS,
        MotorCard,
        load_pcb_autofocus_positions,
    )
except ImportError:
    QApplication = None
    MotorCard = None


@unittest.skipIf(QApplication is None or MotorCard is None, "PySide6 is not installed")
class LensControlStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_every_motor_defaults_to_editable_step_100(self):
        for motor in MOTORS:
            with self.subTest(motor=motor):
                card = MotorCard(motor)
                self.assertEqual(card.step_spin.value(), DEFAULT_STEP_LENGTH)
                self.assertEqual(card.step_spin.minimum(), 1)
                self.assertEqual(card.step_spin.maximum(), 65535)
                self.assertTrue(card.step_spin.isEnabled())

    def test_motor_address_range_does_not_replace_step_range_or_value(self):
        card = MotorCard("zoom")
        card.step_spin.setValue(250)

        card.apply_block(
            {
                "supported": True,
                "min": 901,
                "max": 6353,
                "current": 1501,
                "initialized": False,
            }
        )

        self.assertEqual(card.step_spin.value(), 250)
        self.assertEqual((card.step_spin.minimum(), card.step_spin.maximum()), (1, 65535))
        self.assertTrue(card.step_spin.isEnabled())
        self.assertFalse(card.goto_spin.isEnabled())

        card.apply_bits(operating=False, initialized=True)
        self.assertTrue(card.step_spin.isEnabled())
        self.assertTrue(card.goto_spin.isEnabled())

    def test_disconnect_keeps_user_step_editable(self):
        card = MotorCard("focus")
        card.step_spin.setValue(375)
        card.set_enabled_connected(False)

        self.assertEqual(card.step_spin.value(), 375)
        self.assertTrue(card.step_spin.isEnabled())
        self.assertFalse(card.plus_btn.isEnabled())

    def test_zoom_card_shows_two_calibrated_positions_named_by_zoom(self):
        card = MotorCard("zoom")
        card.set_calibrated_positions(
            [
                {"zoom": 2800, "focus": 4936, "iris": 400},
                {"zoom": 6353, "focus": 5881, "iris": 400},
            ]
        )

        self.assertEqual(
            [button.text() for button in card.calibrated_zoom_buttons],
            ["2800", "6353"],
        )
        self.assertTrue(
            all(not button.isEnabled() for button in card.calibrated_zoom_buttons)
        )

        card.apply_block(
            {
                "supported": True,
                "min": 901,
                "max": 6353,
                "current": 2800,
                "initialized": True,
            }
        )
        self.assertTrue(
            all(button.isEnabled() for button in card.calibrated_zoom_buttons)
        )

    def test_autofocus_position_loader_requires_complete_three_motor_targets(self):
        document = {
            "entries": [
                {
                    "type": "pcb_autofocus_lens_position",
                    "lens_zoom": 6353,
                    "lens_focus": 5881,
                    "lens_iris": 400,
                    "device": 0,
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "lens_position": {"zoom": 2800, "focus": 4936, "iris": 400},
                    "device": 0,
                },
                {
                    "type": "pcb_autofocus_lens_position",
                    "lens_zoom": 4500,
                    "lens_focus": 5200,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrate.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            positions = load_pcb_autofocus_positions(path)

        self.assertEqual(
            positions,
            [
                {"zoom": 2800, "focus": 4936, "iris": 400, "device": 0},
                {"zoom": 6353, "focus": 5881, "iris": 400, "device": 0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
