import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PROJECT_ROOT / "detection" / "calibration"
sys.path.insert(0, str(CALIBRATION_DIR))

import autofocus_pcb


def autofocus_result(*, zoom=6353, focus=5864, iris=506):
    return {
        "created_at_utc": "2026-08-29T12:00:00Z",
        "device": 0,
        "metric": "tenengrad",
        "autofocus_skipped": False,
        "best": {"score": 123.5},
        "lens_after": {
            "zoom": {"supported": zoom is not None, "current": zoom},
            "focus": {"supported": focus is not None, "current": focus},
            "iris": {"supported": iris is not None, "current": iris},
        },
    }


class AutofocusCalibrationSummaryTests(unittest.TestCase):
    def test_focus_reference_is_absent_from_settings_and_gui(self):
        settings = autofocus_pcb.load_settings()
        args = autofocus_pcb.load_args()

        from GUI.pages.registry import task_by_id

        field_keys = {field.key for field in task_by_id("autofocus").fields}
        self.assertNotIn("focus", settings)
        self.assertFalse(hasattr(args, "focus"))
        self.assertNotIn("focus", field_keys)
        self.assertIn("focus_min", field_keys)
        self.assertIn("focus_max", field_keys)

    def test_other_summary_entries_keep_default_saved_timestamp(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            autofocus_pcb.calibration_summary,
            "SUMMARY_ROOT",
            Path(directory),
        ):
            path = autofocus_pcb.calibration_summary.append_entry(
                {"type": "other_calibration", "name": "result"}
            )
            entry = json.loads(path.read_text(encoding="utf-8"))["entries"][0]

        self.assertIn("saved_at_utc", entry)

    def test_summary_entry_uses_final_actual_zoom_focus_and_iris(self):
        with mock.patch.object(
            autofocus_pcb.calibration_summary,
            "append_entry",
            return_value=Path("calibrate.json"),
        ) as append_entry:
            result = autofocus_pcb.append_autofocus_summary(autofocus_result())

        self.assertEqual(result, Path("calibrate.json"))
        entry = append_entry.call_args.args[0]
        self.assertEqual(entry["type"], "pcb_autofocus_lens_position")
        self.assertEqual(entry["lens_zoom"], 6353)
        self.assertEqual(entry["lens_focus"], 5864)
        self.assertEqual(entry["lens_iris"], 506)
        self.assertEqual(
            entry["lens_position"],
            {"zoom": 6353, "focus": 5864, "iris": 506},
        )
        self.assertEqual(
            set(entry),
            {
                "type",
                "created_at_utc",
                "device",
                "lens_zoom",
                "lens_focus",
                "lens_iris",
                "lens_position",
            },
        )
        self.assertEqual(
            append_entry.call_args.kwargs,
            {"zoom": 6353, "include_saved_at_utc": False},
        )

    def test_same_zoom_updates_autofocus_entry_and_preserves_other_results(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            autofocus_pcb.calibration_summary,
            "SUMMARY_ROOT",
            Path(directory),
        ):
            summary_path = Path(directory) / "calibrate.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [
                            {
                                "type": "other_calibration",
                                "name": "zoom_06353",
                                "lens_zoom": 6353,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            autofocus_pcb.append_autofocus_summary(autofocus_result(focus=5800))
            autofocus_pcb.append_autofocus_summary(autofocus_result(focus=5900))

            document = json.loads(summary_path.read_text(encoding="utf-8"))

        autofocus_entries = [
            entry
            for entry in document["entries"]
            if entry.get("type") == "pcb_autofocus_lens_position"
        ]
        self.assertEqual(len(autofocus_entries), 1)
        self.assertEqual(autofocus_entries[0]["name"], "zoom_06353")
        self.assertEqual(autofocus_entries[0]["lens_focus"], 5900)
        self.assertNotIn("saved_at_utc", autofocus_entries[0])
        for removed in (
            "autofocus_skipped",
            "metric",
            "best_score",
            "source_core_json",
            "best_image",
        ):
            self.assertNotIn(removed, autofocus_entries[0])
        self.assertEqual(document["entries"][0]["type"], "other_calibration")

    def test_summary_rejects_incomplete_final_lens_position(self):
        with self.assertRaisesRegex(
            autofocus_pcb.AutofocusError,
            "missing current iris",
        ):
            autofocus_pcb.append_autofocus_summary(autofocus_result(iris=None))


if __name__ == "__main__":
    unittest.main()
