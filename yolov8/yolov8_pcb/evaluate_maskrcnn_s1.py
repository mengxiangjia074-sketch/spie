"""Evaluate the trained S1 Mask R-CNN with YOLOv8-Seg-style metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from train_maskrcnn_s1 import (
    CLASS_NAMES,
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    YoloPolygonDataset,
    build_model,
    collate_batch,
)


DEFAULT_CHECKPOINT = DEFAULT_OUTPUT / "best.pth"
DEFAULT_METRICS = DEFAULT_OUTPUT / "evaluation" / "metrics.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "MASKRCNN_S1_EVALUATION.md"
IOU_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="val")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def binary_mask_iou(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if not len(predictions) or not len(targets):
        return torch.zeros(
            (len(predictions), len(targets)), dtype=torch.float32, device=predictions.device
        )
    pred = predictions.flatten(1).float()
    truth = targets.flatten(1).float()
    intersection = pred @ truth.T
    union = pred.sum(1)[:, None] + truth.sum(1)[None, :] - intersection
    return intersection / union.clamp_min(1.0)


def append_matches(
    records: dict[int, list[tuple[float, bool]]],
    ground_truth_counts: dict[int, int],
    ious: torch.Tensor,
    scores: torch.Tensor,
    pred_labels: torch.Tensor,
    truth_labels: torch.Tensor,
) -> None:
    scores = scores.detach().cpu()
    pred_labels = pred_labels.detach().cpu()
    truth_labels = truth_labels.detach().cpu()
    ious = ious.detach().cpu()
    for class_id in range(1, len(CLASS_NAMES) + 1):
        pred_indices = torch.where(pred_labels == class_id)[0]
        truth_indices = torch.where(truth_labels == class_id)[0]
        ground_truth_counts[class_id] += len(truth_indices)
        if len(pred_indices):
            pred_indices = pred_indices[torch.argsort(scores[pred_indices], descending=True)]
        used_truths: set[int] = set()
        for pred_index in pred_indices.tolist():
            is_true_positive = False
            if len(truth_indices):
                candidate_ious = ious[pred_index, truth_indices]
                order = torch.argsort(candidate_ious, descending=True)
                for candidate in order.tolist():
                    truth_index = int(truth_indices[candidate])
                    if float(candidate_ious[candidate]) < IOU_THRESHOLD:
                        break
                    if truth_index not in used_truths:
                        used_truths.add(truth_index)
                        is_true_positive = True
                        break
            records[class_id].append((float(scores[pred_index]), is_true_positive))


def precision_recall_summary(
    records: dict[int, list[tuple[float, bool]]], ground_truth_counts: dict[int, int]
) -> tuple[dict, dict[int, dict]]:
    thresholds = np.linspace(0.0, 1.0, 1001)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_id in range(1, len(CLASS_NAMES) + 1):
        ordered = sorted(records[class_id], reverse=True)
        scores = np.asarray([item[0] for item in ordered], dtype=np.float64)
        true_positives = np.asarray([item[1] for item in ordered], dtype=np.int64)
        if len(scores):
            cumulative_tp = np.cumsum(true_positives)
            counts = np.searchsorted(-scores, -thresholds, side="right")
            tp = np.where(counts > 0, cumulative_tp[np.maximum(counts - 1, 0)], 0)
        else:
            counts = np.zeros_like(thresholds, dtype=int)
            tp = np.zeros_like(thresholds, dtype=int)
        fp = counts - tp
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(ground_truth_counts[class_id], 1)
        curves[class_id] = (precision, recall)

    mean_precision = np.mean([curve[0] for curve in curves.values()], axis=0)
    mean_recall = np.mean([curve[1] for curve in curves.values()], axis=0)
    mean_f1 = 2 * mean_precision * mean_recall / np.maximum(mean_precision + mean_recall, 1e-12)
    best_index = int(np.argmax(mean_f1))
    per_class = {}
    for class_id, (precision, recall) in curves.items():
        p = float(precision[best_index])
        r = float(recall[best_index])
        per_class[class_id] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
            "ground_truth_instances": ground_truth_counts[class_id],
        }
    overall = {
        "precision": float(mean_precision[best_index]),
        "recall": float(mean_recall[best_index]),
        "f1": float(mean_f1[best_index]),
        "confidence_threshold": float(thresholds[best_index]),
        "matching_iou": IOU_THRESHOLD,
    }
    return overall, per_class


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def extract_map(result: dict, pr_overall: dict, pr_per_class: dict[int, dict]) -> tuple[dict, dict]:
    overall = {
        **pr_overall,
        "map50": scalar(result["map_50"]),
        "map75": scalar(result["map_75"]),
        "map50_95": scalar(result["map"]),
        "mar100": scalar(result["mar_100"]),
    }
    class_ids = result["classes"].detach().cpu().tolist()
    class_maps = result["map_per_class"].detach().cpu().tolist()
    class_recalls = result["mar_100_per_class"].detach().cpu().tolist()
    per_class = {}
    for class_id, class_map, class_recall in zip(class_ids, class_maps, class_recalls):
        class_id = int(class_id)
        per_class[class_id] = {
            **pr_per_class[class_id],
            "map50_95": float(class_map),
            "mar100": float(class_recall),
        }
    return overall, per_class


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(metric: dict) -> str:
    return (
        f"{metric['precision']:.4f} | {metric['recall']:.4f} | {metric['f1']:.4f} | "
        f"{metric['map50']:.4f} | {metric['map75']:.4f} | {metric['map50_95']:.4f} | "
        f"{metric['mar100']:.4f} | {metric['confidence_threshold']:.3f}"
    )


def build_report(payload: dict) -> str:
    box = payload["box"]
    mask = payload["mask"]
    lines = [
        "# Mask R-CNN S1 Evaluation",
        "",
        "## Evaluation Setup",
        "",
        f"- Evaluated: {payload['evaluated_at']}",
        f"- Model: torchvision Mask R-CNN ResNet-50 FPN v2",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Best checkpoint epoch: {payload['checkpoint_epoch']}",
        f"- Dataset split: `{payload['test_split']}` (used as the test set because no separate test split exists)",
        f"- Test images: {payload['images']}",
        f"- Ground-truth instances: {payload['instances']}",
        f"- Classes: {len(CLASS_NAMES)}",
        f"- Device: {payload['device']}",
        f"- Mean inference time: {payload['inference_ms_per_image']:.2f} ms/image",
        "",
        "Precision and recall use class-aware one-to-one matching at IoU 0.50. The confidence threshold is selected by the highest mean F1 across classes. AP follows COCO's 101-point interpolation; mAP50:95 averages IoU thresholds 0.50 through 0.95 in steps of 0.05.",
        "",
        "## Overall Metrics",
        "",
        "| Type | Precision | Recall | F1 | mAP50 | mAP75 | mAP50:95 | mAR100 | Best confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Box | {metric_row(box)} |",
        f"| Mask | {metric_row(mask)} |",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Instances | Mask P | Mask R | Mask F1 | Mask mAP50:95 | Mask mAR100 | Box mAP50:95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for class_id, name in enumerate(CLASS_NAMES, 1):
        mask_class = payload["mask_per_class"][str(class_id)]
        box_class = payload["box_per_class"][str(class_id)]
        lines.append(
            f"| {name} | {mask_class['ground_truth_instances']} | "
            f"{mask_class['precision']:.4f} | {mask_class['recall']:.4f} | "
            f"{mask_class['f1']:.4f} | {mask_class['map50_95']:.4f} | "
            f"{mask_class['mar100']:.4f} | {box_class['map50_95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These metrics are directly comparable in structure to the Box and Mask metrics reported by YOLOv8-Seg, while the evaluated model is Mask R-CNN.",
            "- The held-out split is small and imbalanced. In particular, inductors and diodes have very few instances, so their per-class metrics have high uncertainty.",
            "- The report evaluates the best early-stopping checkpoint, not the final checkpoint.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/vision/miniconda3/envs/sam3/bin/python yolov8/yolov8_pcb/evaluate_maskrcnn_s1.py",
            "```",
            "",
            f"Raw metrics: `{payload['metrics_file']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    dataset = YoloPolygonDataset(args.data, args.split, augment=False)
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

    box_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    mask_metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)
    box_records: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    mask_records: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    box_truth_counts: dict[int, int] = defaultdict(int)
    mask_truth_counts: dict[int, int] = defaultdict(int)
    inference_seconds = 0.0
    instances = 0

    with torch.inference_mode():
        for images, targets in loader:
            images_gpu = [image.to(device, non_blocking=True) for image in images]
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            for output, target in zip(outputs, targets):
                pred_masks_gpu = output["masks"][:, 0] >= 0.5
                truth_masks_gpu = target["masks"].to(device, non_blocking=True).bool()
                mask_ious = binary_mask_iou(pred_masks_gpu, truth_masks_gpu)
                box_ious = box_iou(output["boxes"], target["boxes"].to(device, non_blocking=True))
                append_matches(
                    box_records,
                    box_truth_counts,
                    box_ious,
                    output["scores"],
                    output["labels"],
                    target["labels"],
                )
                append_matches(
                    mask_records,
                    mask_truth_counts,
                    mask_ious,
                    output["scores"],
                    output["labels"],
                    target["labels"],
                )

                pred_cpu = {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu(),
                    "masks": pred_masks_gpu.detach().cpu(),
                }
                target_cpu = {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                    "masks": target["masks"].bool().cpu(),
                }
                box_metric.update(
                    [{key: pred_cpu[key] for key in ("boxes", "scores", "labels")}],
                    [{key: target_cpu[key] for key in ("boxes", "labels")}],
                )
                mask_metric.update(
                    [{key: pred_cpu[key] for key in ("masks", "scores", "labels")}],
                    [{key: target_cpu[key] for key in ("masks", "labels")}],
                )
                instances += len(target["labels"])

    box_pr, box_class_pr = precision_recall_summary(box_records, box_truth_counts)
    mask_pr, mask_class_pr = precision_recall_summary(mask_records, mask_truth_counts)
    box_overall, box_per_class = extract_map(box_metric.compute(), box_pr, box_class_pr)
    mask_overall, mask_per_class = extract_map(mask_metric.compute(), mask_pr, mask_class_pr)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": "torchvision maskrcnn_resnet50_fpn_v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_split": str((args.data / "images" / args.split).resolve()),
        "images": len(dataset),
        "instances": instances,
        "class_names": list(CLASS_NAMES),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "inference_ms_per_image": inference_seconds * 1000 / len(dataset),
        "box": box_overall,
        "mask": mask_overall,
        "box_per_class": {str(key): value for key, value in box_per_class.items()},
        "mask_per_class": {str(key): value for key, value in mask_per_class.items()},
        "metrics_file": str(args.metrics.resolve()),
    }
    args.metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(build_report(payload), encoding="utf-8")
    print(json.dumps({"metrics": str(args.metrics), "report": str(args.report)}))


if __name__ == "__main__":
    main()
