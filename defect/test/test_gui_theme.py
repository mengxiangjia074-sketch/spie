import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

try:
    from GUI.core.theme import THEMES, build_qss
except ModuleNotFoundError:
    THEMES = None
    build_qss = None


@unittest.skipIf(build_qss is None, "PySide6 is not installed")
class GuiThemeTests(unittest.TestCase):
    def test_theme_tokens_with_common_prefixes_are_fully_replaced(self):
        for theme_name, palette in THEMES.items():
            with self.subTest(theme=theme_name):
                qss = build_qss(theme_name)
                self.assertNotIn("@", qss)
                self.assertIn(
                    f"background: {palette['accent_hover']};", qss
                )
                self.assertIn(
                    f"background: {palette['surface3']};", qss
                )

    def test_primary_hover_only_targets_enabled_buttons(self):
        qss = build_qss("dark")

        self.assertIn(
            'QPushButton[class="primary"]:enabled:hover', qss
        )
        self.assertIn(
            f"border-color: {THEMES['dark']['on_accent']};", qss
        )
        self.assertIn(
            'QPushButton[class="primary"]:disabled', qss
        )

    def test_numeric_spin_buttons_are_hidden(self):
        qss = build_qss("dark")

        self.assertIn("QSpinBox::up-button", qss)
        self.assertIn("QDoubleSpinBox::down-button", qss)
        self.assertIn("width: 0px; height: 0px;", qss)
        self.assertNotIn("QSpinBox::up-button:hover", qss)

    def test_defect_prompt_keeps_a_usable_height(self):
        qss = build_qss("dark")

        self.assertIn("#DefectPromptEdit { min-height: 120px; }", qss)

    def test_enabled_controls_are_white_and_unavailable_controls_are_gray(self):
        qss = build_qss("dark")

        self.assertEqual(THEMES["dark"]["control_bg"], "#ffffff")
        self.assertNotEqual(
            THEMES["dark"]["control_bg"],
            THEMES["dark"]["control_disabled_bg"],
        )
        self.assertEqual(THEMES["dark"]["control_text"], "#1d2330")
        self.assertIn("QLineEdit:read-only", qss)
        self.assertIn("QTextEdit:disabled", qss)
        self.assertIn("QToolButton:disabled", qss)

    def test_motion_status_indicators_have_normal_and_alert_styles(self):
        qss = build_qss("dark")

        self.assertIn('#MotionStatusBadge[active="true"][severity="normal"]', qss)
        self.assertIn('#MotionStatusBadge[active="true"][severity="alert"]', qss)
        self.assertIn(THEMES["dark"]["ok_soft"], qss)
        self.assertIn(THEMES["dark"]["error_soft"], qss)


if __name__ == "__main__":
    unittest.main()
