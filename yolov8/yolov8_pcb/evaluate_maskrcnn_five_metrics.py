#!/usr/bin/env python3
"""Evaluate Mask R-CNN with the five paper metrics used by the S1 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from train_maskrcnn_s1 import (
    CLASS_NAMES,
    DEFAULT_DATA,
    DEFAULT_WEIGHTS,
    YoloPolygonDataset,
    build_model,
    collate_batch,
)


def mask_iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    intersection = np.count_nonzero(prediction & truth)
    union = np.count_nonzero(prediction | truth)
    return float(intersection / max(1, union))


def boundary_f1(prediction: np.ndarray, truth: np.ndarray, tolerance: int) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.morphologyEx(
        prediction.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)
    truth_boundary = cv2.morphologyEx(
        truth.astype(np.uint8), cv2.MORPH_GRADIENT, kernel
    ).astype(bool)
    band = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), band).astype(bool)
    truth_dilated = cv2.dilate(truth_boundary.astype(np.uint8), band).astype(bool)
    precision = np.count_nonzero(pred_boundary & truth_dilated) / max(
        1, np.count_nonzero(pred_boundary)
    )
    recall = np.count_nonzero(truth_boundary & pred_dilated) / max(
        1, np.count_nonzero(truth_boundary)
    )
    return float(2 * precision * recall / max(1e-9, precision + recall))


def greedy_matches(predictions: list[dict], truths: list[dict], threshold: float):
    candidates = sorted(
        (
            (mask_iou(prediction["mask"], truth["mask"]), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(truths)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches = []
    for iou, pred_index, truth_index in candidates:
        if iou < threshold:
            break
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
        matches.append((pred_index, truth_index))
    return matches, used_predictions, used_truths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--small-area", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = YoloPolygonDataset(args.data, "val", augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.pretrained_weights, len(CLASS_NAMES) + 1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    model.to(device).eval()

    ap_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    class_stats = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(len(CLASS_NAMES))
    }
    boundary_values: list[float] = []
    small_total = small_matched = 0
    predicted_total = matched_total = truth_total = 0

    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for output, target in zip(outputs, targets):
                all_masks = (output["masks"][:, 0] >= 0.5).detach().cpu()
                all_scores = output["scores"].detach().cpu()
                all_labels = output["labels"].detach().cpu()
                truth_masks = target["masks"].bool().cpu()
                truth_labels = target["labels"].cpu()
                ap_metric.update(
                    [{"masks": all_masks, "scores": all_scores, "labels": all_labels}],
                    [{"masks": truth_masks, "labels": truth_labels}],
                )

                keep = all_scores >= args.confidence
                predictions = [
                    {"class_id": int(label) - 1, "mask": mask.numpy()}
                    for label, mask in zip(all_labels[keep], all_masks[keep])
                ]
                truths = [
                    {"class_id": int(label) - 1, "mask": mask.numpy()}
                    for label, mask in zip(truth_labels, truth_masks)
                ]
                matches, used_predictions, used_truths = greedy_matches(
                    predictions, truths, args.matching_iou
                )
                predicted_total += len(predictions)
                matched_total += len(matches)
                truth_total += len(truths)
                for pred_index, truth_index in matches:
                    prediction = predictions[pred_index]
                    truth = truths[truth_index]
                    if prediction["class_id"] == truth["class_id"]:
                        class_stats[truth["class_id"]]["tp"] += 1
                    else:
                        class_stats[prediction["class_id"]]["fp"] += 1
                        class_stats[truth["class_id"]]["fn"] += 1
                    boundary_values.append(
                        boundary_f1(
                            prediction["mask"], truth["mask"], args.boundary_tolerance
                        )
                    )
                for pred_index, prediction in enumerate(predictions):
                    if pred_index not in used_predictions:
                        class_stats[prediction["class_id"]]["fp"] += 1
                for truth_index, truth in enumerate(truths):
                    if np.count_nonzero(truth["mask"]) < args.small_area:
                        small_total += 1
                        small_matched += int(truth_index in used_truths)
                    if truth_index not in used_truths:
                        class_stats[truth["class_id"]]["fn"] += 1

    class_f1 = {}
    for class_id, stats in class_stats.items():
        precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        class_f1[CLASS_NAMES[class_id]] = 2 * precision * recall / max(
            1e-9, precision + recall
        )
    class_tp = sum(stats["tp"] for stats in class_stats.values())
    class_fp = sum(stats["fp"] for stats in class_stats.values())
    class_fn = sum(stats["fn"] for stats in class_stats.values())
    ap = ap_metric.compute()
    result = {
        "model": "Mask R-CNN ResNet-50 FPN v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_images": len(dataset),
        "ground_truth_instances": truth_total,
        "predicted_instances": predicted_total,
        "matched_instances": matched_total,
        "joint_f1": 2 * class_tp / max(1, 2 * class_tp + class_fp + class_fn),
        "macro_f1": float(np.mean(list(class_f1.values()))),
        "mask_ap50_95": float(ap["map"]),
        "boundary_f1": float(np.mean(boundary_values)) if boundary_values else 0.0,
        "small_component_recall": small_matched / max(1, small_total),
        "class_f1": class_f1,
        "definitions": {
            "confidence_threshold": args.confidence,
            "matching_iou": args.matching_iou,
            "boundary_tolerance_px": args.boundary_tolerance,
            "small_area_threshold_px": args.small_area,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
