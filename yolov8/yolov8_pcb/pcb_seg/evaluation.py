"""Detailed instance and boundary metrics for a YOLOv8-Seg validation split."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .training import _yolo_class


def _mask_from_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(intersection / max(1, union))


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if moments["m00"] <= 0:
        ys, xs = np.where(mask)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else (0.0, 0.0)
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _boundary_scores(pred: np.ndarray, truth: np.ndarray, tolerance: int) -> tuple[float, float]:
    pred_boundary = _boundary(pred)
    truth_boundary = _boundary(truth)
    pred_count = max(1, int(np.count_nonzero(pred_boundary)))
    truth_count = max(1, int(np.count_nonzero(truth_boundary)))
    precision = np.count_nonzero(pred_boundary & _dilate(truth_boundary, tolerance)) / pred_count
    recall = np.count_nonzero(truth_boundary & _dilate(pred_boundary, tolerance)) / truth_count
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    pred_band = _dilate(pred_boundary, tolerance)
    truth_band = _dilate(truth_boundary, tolerance)
    boundary_iou = np.count_nonzero(pred_band & truth_band) / max(1, np.count_nonzero(pred_band | truth_band))
    return float(f1), float(boundary_iou)


def _greedy_matches(predictions: list[dict], truths: list[dict], threshold: float = 0.5):
    candidates = []
    for pred_index, pred in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            iou = _mask_iou(pred["mask"], truth["mask"])
            if iou >= threshold:
                candidates.append((iou, pred_index, truth_index))
    candidates.sort(reverse=True)
    used_predictions = set()
    used_truths = set()
    matches = []
    for iou, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(pred_index)
        used_truths.add(truth_index)
        matches.append((pred_index, truth_index, iou))
    return matches, used_predictions, used_truths


def _load_truth(label_path: Path, image_shape: tuple[int, int]) -> list[dict]:
    height, width = image_shape
    truths = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        class_id = int(values[0])
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        truths.append({"class_id": class_id, "mask": _mask_from_points(points, image_shape)})
    return truths


def evaluate_split(
    weights: Path,
    data_yaml: Path,
    output: Path,
    imgsz: int = 1024,
    batch: int = 4,
    device: str = "0",
    confidence: float = 0.25,
    iou_threshold: float = 0.5,
    boundary_tolerance: int = 2,
    small_area_threshold: int = 4096,
) -> dict:
    import yaml

    YOLO = _yolo_class()
    model = YOLO(str(weights.resolve()))
    metrics = model.val(
        data=str(data_yaml.resolve()),
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
    )
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(data_yaml).parent
    val_path = dataset_root / data["val"]
    image_paths = sorted(val_path.glob("*"))
    class_names = data["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[index] for index in sorted(class_names, key=int)]

    class_stats = {index: {"tp": 0, "fp": 0, "fn": 0} for index in range(len(class_names))}
    total_tp = total_fp = total_fn = 0
    small_tp = small_total = 0
    boundary_f1 = []
    boundary_iou = []
    center_errors = []

    for image_path in image_paths:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not read validation image: {image_path}")
        height, width = image.shape[:2]
        label_path = dataset_root / "labels" / "val" / f"{image_path.stem}.txt"
        truths = _load_truth(label_path, (height, width))
        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            iou=0.7,
            device=device,
            retina_masks=True,
            verbose=False,
        )[0]
        predictions = []
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for class_id, score, points in zip(classes, scores, result.masks.xy):
                predictions.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "mask": _mask_from_points(np.asarray(points), (height, width)),
                    }
                )
        matches, used_predictions, used_truths = _greedy_matches(predictions, truths, iou_threshold)
        total_tp += len(matches)
        total_fp += len(predictions) - len(used_predictions)
        total_fn += len(truths) - len(used_truths)
        for pred_index, truth_index, _ in matches:
            pred = predictions[pred_index]
            truth = truths[truth_index]
            if pred["class_id"] == truth["class_id"]:
                class_stats[truth["class_id"]]["tp"] += 1
                class_stats[pred["class_id"]]["tp"] += 0
            else:
                class_stats[truth["class_id"]]["fn"] += 1
                class_stats[pred["class_id"]]["fp"] += 1
            f1, biou = _boundary_scores(pred["mask"], truth["mask"], boundary_tolerance)
            boundary_f1.append(f1)
            boundary_iou.append(biou)
            px, py = _centroid(pred["mask"])
            tx, ty = _centroid(truth["mask"])
            center_errors.append(float(np.hypot(px - tx, py - ty)))
        for truth_index, truth in enumerate(truths):
            area = int(np.count_nonzero(truth["mask"]))
            if area < small_area_threshold:
                small_total += 1
                if truth_index in used_truths:
                    small_tp += 1

        for pred_index, pred in enumerate(predictions):
            if pred_index not in used_predictions:
                class_stats[pred["class_id"]]["fp"] += 1
        for truth_index, truth in enumerate(truths):
            if truth_index not in used_truths:
                class_stats[truth["class_id"]]["fn"] += 1

    class_f1 = {}
    for index, stats in class_stats.items():
        precision = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        recall = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        class_f1[class_names[index]] = 2 * precision * recall / max(1e-9, precision + recall)
    macro_f1 = float(np.mean(list(class_f1.values()))) if class_f1 else 0.0
    joint_f1 = 2 * total_tp / max(1e-9, 2 * total_tp + total_fp + total_fn)
    result = {
        "weights": str(weights.resolve()),
        "test_split": str(val_path.resolve()),
        "definitions": {
            "matching_iou": iou_threshold,
            "boundary_tolerance_px": boundary_tolerance,
            "small_area_threshold_px": small_area_threshold,
            "macro_f1": "mean of class-aware instance F1 values",
            "joint_f1": "micro F1 of all IoU-matched instances with correct class",
            "center_error": "mean centroid Euclidean distance in pixels",
        },
        "mask_ap50": float(metrics.seg.map50),
        "mask_ap75": float(np.asarray(metrics.seg.all_ap)[:, 5].mean()),
        "mask_ap50_95": float(metrics.seg.map),
        "boundary_f1": float(np.mean(boundary_f1)) if boundary_f1 else 0.0,
        "boundary_iou": float(np.mean(boundary_iou)) if boundary_iou else 0.0,
        "macro_f1": macro_f1,
        "joint_f1": float(joint_f1),
        "small_component_recall": small_tp / max(1, small_total),
        "center_error_px_mean": float(np.mean(center_errors)) if center_errors else 0.0,
        "center_error_px_median": float(np.median(center_errors)) if center_errors else 0.0,
        "matched_instances": total_tp,
        "predicted_instances": total_tp + total_fp,
        "ground_truth_instances": total_tp + total_fn,
        "class_f1": class_f1,
        "class_stats": class_stats,
        "ultralytics": metrics.results_dict,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
