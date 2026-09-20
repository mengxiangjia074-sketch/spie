#!/usr/bin/env python3
"""Run PCB component completeness inspection from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from component_inspection.config import InspectionConfig
from component_inspection.pipeline import default_output_directory, run_component_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment high-magnification images with SAM3, filter candidates with the "
            "trained classifier, map them to the coordinate-Mask image with RoMaV2, "
            "and report missing components."
        )
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="component_masks.json exported by coordinate Mask alignment",
    )
    parser.add_argument(
        "--inspection",
        type=Path,
        required=True,
        help="one magnified image, an images directory, or a two-zoom capture run",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--classifier-checkpoint", type=Path, default=None)
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument("--roma-checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prompt", default="component")
    parser.add_argument("--sam-threshold", type=float, default=0.50)
    parser.add_argument("--classifier-threshold", type=float, default=0.50)
    parser.add_argument("--min-mask-area", type=int, default=60)
    parser.add_argument("--tile-size", type=int, default=1600)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--roma-setting", choices=("turbo", "fast", "base", "precise"), default="precise")
    parser.add_argument("--save-negative-patches", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = InspectionConfig(
        prompt=args.prompt,
        sam_confidence_threshold=args.sam_threshold,
        classifier_threshold=args.classifier_threshold,
        min_mask_area=args.min_mask_area,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        roma_setting=args.roma_setting,
        save_candidate_patches=args.save_negative_patches,
    )
    output_dir = args.output_dir or default_output_directory()
    result = run_component_inspection(
        ground_truth_metadata=args.ground_truth,
        inspection_source=args.inspection,
        output_directory=output_dir,
        config=config,
        classifier_checkpoint=args.classifier_checkpoint,
        sam3_checkpoint=args.sam3_checkpoint,
        roma_checkpoint=args.roma_checkpoint,
        device=args.device,
    )
    print(f"Status: {result['status']}")
    print(f"Missing IDs: {', '.join(result['missing_ids']) or '(none)'}")
    print(f"Report: {result['outputs']['report_json']}")
    print(f"Overlay: {result['outputs']['component_completeness_overlay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
