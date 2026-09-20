#!/usr/bin/env python3
"""Run-only test entry point for the trained SAM3 classification head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam3_fpic_scheme_b import run_pipeline


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "ground_truth" / "fpic_yolo"
DEFAULT_SAM_CHECKPOINT = ROOT.parents[1] / "sam3" / "models" / "sam3" / "sam3.pt"
DEFAULT_CLASSIFIER = ROOT / "runs" / "sam3_fpic_scheme_b_1000" / "classifier_best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "sam3_fpic_test_1000"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test a trained SAM3 classification head on FPIC s1")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    for path, description in (
        (args.dataset, "dataset"),
        (args.sam_checkpoint, "SAM3 checkpoint"),
        (args.classifier, "trained classifier checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    result = run_pipeline(
        args.dataset,
        args.sam_checkpoint,
        args.classifier,
        args.output,
        args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
