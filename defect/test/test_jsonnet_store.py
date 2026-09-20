import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from GUI.core import jsonnet_store


class JsonnetStoreTests(unittest.TestCase):
    def test_set_values_allows_self_dependent_aliases_to_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "capture.jsonnet"
            config.write_text(
                """{
  \"capture_autofocus\": false,
  \"lens_autofocus\": self.capture_autofocus,
  \"second_alias\": self[\"lens_autofocus\"],
  \"unchanged\": 42,
}
""",
                encoding="utf-8",
            )

            original_backup_dir = jsonnet_store.GUI_BACKUP_DIR
            jsonnet_store.GUI_BACKUP_DIR = root / "backups"
            try:
                values = jsonnet_store.set_values(
                    config, {"capture_autofocus": True}
                )
            finally:
                jsonnet_store.GUI_BACKUP_DIR = original_backup_dir

            self.assertIs(values["capture_autofocus"], True)
            self.assertIs(values["lens_autofocus"], True)
            self.assertIs(values["second_alias"], True)
            self.assertEqual(values["unchanged"], 42)

            saved_text = config.read_text(encoding="utf-8")
            self.assertIn('\"capture_autofocus\": true', saved_text)
            self.assertIn('\"lens_autofocus\": self.capture_autofocus', saved_text)
            self.assertIn('\"second_alias\": self[\"lens_autofocus\"]', saved_text)


if __name__ == "__main__":
    unittest.main()
