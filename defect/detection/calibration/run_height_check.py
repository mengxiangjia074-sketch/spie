#!/usr/bin/env python3
"""Calibration-workspace entry point for the project height check."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_DIR = PROJECT_ROOT / "detection"
for module_path in (str(DETECTION_DIR), str(PROJECT_ROOT)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from height_check import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
