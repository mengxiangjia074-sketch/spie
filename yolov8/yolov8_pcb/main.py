#!/usr/bin/env python3
"""Train and run YOLOv8-Seg for PCB component instance segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcb_seg.dataset import find_latest_sam_results, prepare_dataset
from pcb_seg.fpic import prepare_fpic_dataset
from pcb_seg.evaluation import evaluate_split
from pcb_seg.inference import predict_images
from pcb_seg.training import train_model


PROGRAM_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROGRAM_ROOT.parents[1]
DEFAULT_SOURCE = WORKSPACE_ROOT / "data"
DEFAULT_SAM_RESULTS_ROOT = WORKSPACE_ROOT / "sam3" / "results" / "component_inspection"
DEFAULT_MODEL = PROGRAM_ROOT.parent / "yolov8l-seg.pt"
DEFAULT_DATASET = PROGRAM_ROOT / "dataset"
DEFAULT_TRAIN_PROJECT = PROGRAM_ROOT / "runs" / "train"
DEFAULT_PREDICT_OUTPUT = PROGRAM_ROOT / "runs" / "predict"


def _common_tile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv8-Seg PCB component dataset, training and inference pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create YOLO-Seg labels from SAM3 results")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--sam-results", type=Path)
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--validation-image")
    prepare.add_argument("--min-visible-ratio", type=float, default=0.35)
    prepare.add_argument("--force", action="store_true", help="rebuild a non-empty output directory")
    _common_tile_arguments(prepare)

    prepare_fpic = subparsers.add_parser(
        "prepare-fpic", help="convert local FPIC CSV annotations to YOLO-Seg data"
    )
    prepare_fpic.add_argument("--source", type=Path, default=WORKSPACE_ROOT / "data" / "s1")
    prepare_fpic.add_argument(
        "--output", type=Path, default=PROGRAM_ROOT / "ground_truth" / "fpic_yolo"
    )
    prepare_fpic.add_argument("--validation-group")
    prepare_fpic.add_argument("--min-visible-ratio", type=float, default=0.35)
    prepare_fpic.add_argument("--no-dslr", action="store_true")
    prepare_fpic.add_argument("--force", action="store_true")
    _common_tile_arguments(prepare_fpic)

    train = subparsers.add_parser("train", help="fine-tune YOLOv8-Seg")
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--data", type=Path, default=DEFAULT_DATASET / "data.yaml")
    train.add_argument("--project", type=Path, default=DEFAULT_TRAIN_PROJECT)
    train.add_argument("--name", default="pcb_components")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=1024)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--patience", type=int, default=30)
    train.add_argument("--amp", action="store_true", help="enable mixed precision (may download an AMP check model on first run)")
    train.add_argument("--resume", action="store_true")

    predict = subparsers.add_parser("predict", help="segment PCB components in images")
    predict.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_TRAIN_PROJECT / "pcb_components" / "weights" / "best.pt",
    )
    predict.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    predict.add_argument("--output", type=Path, default=DEFAULT_PREDICT_OUTPUT)
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.70)
    predict.add_argument("--merge-iou", type=float, default=0.50)
    predict.add_argument("--device", default="0")
    predict.add_argument("--batch", type=int, default=4)
    predict.add_argument("--edge-margin", type=int, default=4)
    _common_tile_arguments(predict)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a segmentation model on a YOLO split")
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, default=PROGRAM_ROOT / "runs" / "evaluation")
    evaluate.add_argument("--imgsz", type=int, default=1024)
    evaluate.add_argument("--batch", type=int, default=4)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--conf", type=float, default=0.25)
    evaluate.add_argument("--boundary-tolerance", type=int, default=2)
    evaluate.add_argument("--small-area", type=int, default=4096)

    all_command = subparsers.add_parser("all", help="prepare, train and predict")
    all_command.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    all_command.add_argument("--sam-results", type=Path)
    all_command.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    all_command.add_argument("--epochs", type=int, default=100)
    all_command.add_argument("--imgsz", type=int, default=1024)
    all_command.add_argument("--batch", type=int, default=2)
    all_command.add_argument("--device", default="0")
    all_command.add_argument("--validation-image")
    all_command.add_argument("--amp", action="store_true")
    all_command.add_argument("--force", action="store_true")
    _common_tile_arguments(all_command)
    return parser


def _sam_results(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"SAM3 results directory not found: {explicit}")
        return explicit
    return find_latest_sam_results(DEFAULT_SAM_RESULTS_ROOT)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare-fpic":
        result = prepare_fpic_dataset(
            args.source,
            args.output,
            args.validation_group,
            args.tile_size,
            args.overlap,
            args.min_visible_ratio,
            not args.no_dslr,
            args.force,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        result = prepare_dataset(
            args.source,
            _sam_results(args.sam_results),
            args.output,
            args.tile_size,
            args.overlap,
            args.validation_image,
            args.min_visible_ratio,
            args.force,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        best = train_model(
            args.model,
            args.data,
            args.project,
            args.name,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.patience,
            args.amp,
            args.resume,
        )
        print(f"best weights: {best}")
        return 0
    if args.command == "predict":
        result = predict_images(
            args.weights,
            args.source,
            args.output,
            args.tile_size,
            args.overlap,
            args.conf,
            args.iou,
            args.merge_iou,
            args.device,
            args.batch,
            args.edge_margin,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        result = evaluate_split(
            args.weights, args.data, args.output, args.imgsz, args.batch, args.device,
            args.conf, boundary_tolerance=args.boundary_tolerance,
            small_area_threshold=args.small_area,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    sam_results = _sam_results(args.sam_results)
    manifest = prepare_dataset(
        args.source,
        sam_results,
        DEFAULT_DATASET,
        args.tile_size,
        args.overlap,
        args.validation_image,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    best = train_model(
        args.model,
        DEFAULT_DATASET / "data.yaml",
        DEFAULT_TRAIN_PROJECT,
        "pcb_components",
        args.epochs,
        args.imgsz,
        args.batch,
        args.device,
        4,
        30,
        args.amp,
    )
    result = predict_images(
        best,
        args.source,
        DEFAULT_PREDICT_OUTPUT,
        args.tile_size,
        args.overlap,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
