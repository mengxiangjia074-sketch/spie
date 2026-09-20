#!/usr/bin/env python3
"""Train and evaluate the engineering SAM3 + binary classifier model on s1.

SAM3 is frozen. It proposes component masks from the ``component`` text prompt;
the engineering ResNet18 head learns to accept or reject those candidates.
The output metrics use the same validation tiles and mask definitions as the
existing YOLOv8-Seg evaluation, while clearly marking the binary-classifier
versus five-class comparison limitation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
WORKSPACE = DETECTION_ROOT.parent
SAM3_DEFAULT = WORKSPACE / "models" / "sam3" / "sam3.pt"
YOLO_METRICS_DEFAULT = WORKSPACE.parent / "yolov8" / "yolov8_pcb" / "runs" / "fpic_evaluation_1000_es" / "metrics.json"
IMAGE_SIZE = (96, 224)
CLASS_NAMES = ["negative", "positive"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _imports():
    sys.path.insert(0, str(DETECTION_ROOT))
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import (
        SamPromptSegmenter,
        extract_rectified_patch,
    )

    return InspectionConfig, SamPromptSegmenter, extract_rectified_patch


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_from_points(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def load_truth(label_path: Path, shape: tuple[int, int]) -> list[dict]:
    height, width = shape
    truths = []
    if not label_path.is_file():
        return truths
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        truths.append({"class_id": int(values[0]), "mask": mask_from_points(points, shape)})
    return truths


def candidate_full_mask(candidate, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = map(int, candidate.mask_bbox_xyxy)
    full = np.zeros(shape, dtype=bool)
    ix1, iy1, ix2, iy2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return full
    crop = np.asarray(candidate.mask_crop, dtype=bool)
    src_x = ix1 - x1
    src_y = iy1 - y1
    full[iy1:iy2, ix1:ix2] = crop[src_y : src_y + iy2 - iy1, src_x : src_x + ix2 - ix1]
    return full


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def build_candidate_cache(
    dataset_root: Path,
    split: str,
    cache_path: Path,
    segmenter,
    label_iou: float,
) -> list[dict]:
    if cache_path.is_file():
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("label_iou") == label_iou:
            return payload["records"]
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    records = []
    for image_path in sorted(image_root.glob("*.jpg")):
        image = read_rgb(image_path)
        truths = load_truth(label_root / f"{image_path.stem}.txt", image.shape[:2])
        from PIL import Image

        candidates = segmenter.segment(Image.fromarray(image), image_name=image_path.name)
        for candidate in candidates:
            full_mask = candidate_full_mask(candidate, image.shape[:2])
            best_iou = max(
                (mask_iou(full_mask, truth["mask"]) for truth in truths),
                default=0.0,
            )
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "candidate": candidate,
                    "best_iou": float(best_iou),
                    "label": int(best_iou >= label_iou),
                }
            )
        print(f"{split}: {image_path.name} candidates={len(candidates)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        pickle.dump({"label_iou": label_iou, "records": records}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return records


def balance_training_records(records: list[dict], seed: int, negative_ratio: int) -> list[dict]:
    positives = [record for record in records if record["label"] == 1]
    negatives = [record for record in records if record["label"] == 0]
    rng = random.Random(seed)
    rng.shuffle(negatives)
    negatives = negatives[: max(len(positives) * negative_ratio, negative_ratio)]
    if not positives:
        raise RuntimeError("SAM3 produced no positive candidates on the training split")
    balanced = positives + negatives
    rng.shuffle(balanced)
    return balanced


def prepare_patches(records: list[dict], extract_patch) -> list[np.ndarray]:
    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["image_path"]].append((index, record))
    patches: list[np.ndarray | None] = [None] * len(records)
    for image_path, items in grouped.items():
        image = read_rgb(Path(image_path))
        for index, record in items:
            patch = extract_patch(image, record["candidate"], 0.15, 160)
            patches[index] = np.asarray(patch, dtype=np.uint8)
    return [patch for patch in patches if patch is not None]


def build_classifier():
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def classifier_metrics(predictions: list[int], targets: list[int]) -> dict:
    tp = sum(p == 1 and t == 1 for p, t in zip(predictions, targets))
    tn = sum(p == 0 and t == 0 for p, t in zip(predictions, targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(predictions, targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(predictions, targets))
    f1_negative = 2 * tn / max(1, 2 * tn + fp + fn)
    f1_positive = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "samples": len(targets),
        "accuracy": (tp + tn) / max(1, len(targets)),
        "precision_positive": tp / max(1, tp + fp),
        "recall_positive": tp / max(1, tp + fn),
        "f1_positive": f1_positive,
        "macro_f1": (f1_negative + f1_positive) / 2.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def boundary_scores(pred: np.ndarray, truth: np.ndarray, tolerance: int = 2) -> tuple[float, float]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    truth_boundary = cv2.morphologyEx(truth.astype(np.uint8), cv2.MORPH_GRADIENT, kernel).astype(bool)
    band = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), band).astype(bool)
    truth_dilated = cv2.dilate(truth_boundary.astype(np.uint8), band).astype(bool)
    precision = np.count_nonzero(pred_boundary & truth_dilated) / max(1, np.count_nonzero(pred_boundary))
    recall = np.count_nonzero(truth_boundary & pred_dilated) / max(1, np.count_nonzero(truth_boundary))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    biou = np.count_nonzero(pred_dilated & truth_dilated) / max(1, np.count_nonzero(pred_dilated | truth_dilated))
    return float(f1), float(biou)


def centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return (
        float(moments["m10"] / max(1e-9, moments["m00"])),
        float(moments["m01"] / max(1e-9, moments["m00"])),
    )


def average_precision(predictions_by_image: list[list[dict]], truths_by_image: list[list[dict]], threshold: float) -> float:
    records = []
    total_truth = sum(len(truths) for truths in truths_by_image)
    used = {index: set() for index in range(len(truths_by_image))}
    for image_index, predictions in enumerate(predictions_by_image):
        for prediction in predictions:
            records.append((prediction["confidence"], image_index, prediction))
    records.sort(key=lambda item: item[0], reverse=True)
    tp, fp = [], []
    for _, image_index, prediction in records:
        choices = [
            (truth_index, mask_iou(prediction["mask"], truth["mask"]))
            for truth_index, truth in enumerate(truths_by_image[image_index])
            if truth_index not in used[image_index]
        ]
        best = max(choices, key=lambda item: item[1], default=(-1, 0.0))
        if best[1] >= threshold:
            used[image_index].add(best[0])
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not total_truth:
        return 0.0
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total_truth
    precision = tp / np.maximum(1, tp + fp)
    return float(np.mean([np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0 for r in np.linspace(0, 1, 101)]))


def evaluate_predictions(predictions_by_image: list[list[dict]], truths_by_image: list[list[dict]]) -> dict:
    matches = []
    boundary_f1 = []
    boundary_iou = []
    center_errors = []
    matched = predicted = ground_truth = 0
    small_total = small_matched = 0
    for predictions, truths in zip(predictions_by_image, truths_by_image):
        predicted += len(predictions)
        ground_truth += len(truths)
        pairs = sorted(
            (mask_iou(prediction["mask"], truth["mask"]), pi, ti)
            for pi, prediction in enumerate(predictions)
            for ti, truth in enumerate(truths)
        )[::-1]
        used_predictions, used_truths = set(), set()
        for score, pi, ti in pairs:
            if score < 0.5 or pi in used_predictions or ti in used_truths:
                continue
            used_predictions.add(pi)
            used_truths.add(ti)
            matches.append(score)
            boundary = boundary_scores(predictions[pi]["mask"], truths[ti]["mask"])
            boundary_f1.append(boundary[0])
            boundary_iou.append(boundary[1])
            px, py = centroid(predictions[pi]["mask"])
            tx, ty = centroid(truths[ti]["mask"])
            center_errors.append(float(np.hypot(px - tx, py - ty)))
        for ti, truth in enumerate(truths):
            if np.count_nonzero(truth["mask"]) < 4096:
                small_total += 1
                if ti in used_truths:
                    small_matched += 1
    matched = len(matches)
    total_fp = predicted - matched
    total_fn = ground_truth - matched
    return {
        "mask_ap50": float(average_precision(predictions_by_image, truths_by_image, 0.50)),
        "mask_ap75": float(average_precision(predictions_by_image, truths_by_image, 0.75)),
        "mask_ap50_95": float(np.mean([average_precision(predictions_by_image, truths_by_image, threshold) for threshold in np.arange(0.50, 1.00, 0.05)])),
        "boundary_f1": float(np.mean(boundary_f1)) if boundary_f1 else 0.0,
        "boundary_iou": float(np.mean(boundary_iou)) if boundary_iou else 0.0,
        "instance_f1": 2 * matched / max(1, 2 * matched + total_fp + total_fn),
        "small_component_recall": small_matched / max(1, small_total),
        "center_error_px_mean": float(np.mean(center_errors)) if center_errors else 0.0,
        "center_error_px_median": float(np.median(center_errors)) if center_errors else 0.0,
        "matched_instances": matched,
        "predicted_instances": predicted,
        "ground_truth_instances": ground_truth,
        "candidate_mask_iou_mean_at_match": float(np.mean(matches)) if matches else 0.0,
    }


def write_report(result: dict, yolo_metrics_path: Path, report_path: Path) -> None:
    yolo = json.loads(yolo_metrics_path.read_text(encoding="utf-8")) if yolo_metrics_path.is_file() else {}
    sam = result["evaluation"]
    rows = [
        ("Mask AP50", sam["mask_ap50"], yolo.get("mask_ap50")),
        ("Mask AP75", sam["mask_ap75"], yolo.get("mask_ap75")),
        ("Mask AP50:95", sam["mask_ap50_95"], yolo.get("mask_ap50_95")),
        ("Boundary F1", sam["boundary_f1"], yolo.get("boundary_f1")),
        ("Boundary IoU", sam["boundary_iou"], yolo.get("boundary_iou")),
        ("Instance F1 / Joint F1", sam["instance_f1"], yolo.get("joint_f1")),
        ("Small component recall", sam["small_component_recall"], yolo.get("small_component_recall")),
        ("Center error mean (px)", sam["center_error_px_mean"], yolo.get("center_error_px_mean")),
    ]
    lines = [
        "# SAM3-Classify 与 YOLOv8-Seg 在 s1 上的对比",
        "",
        "## 模型定义",
        "",
        "- `SAM3-Classify`：冻结工程 SAM3 生成候选掩码，训练工程 ResNet18 二分类头筛选候选。",
        "- `YOLOv8-Seg`：读取已有 s1 验证结果 `fpic_evaluation_1000_es/metrics.json`。",
        "- 两者使用同一 `fpic_yolo` 验证划分和 IoU=0.50 的实例匹配口径。",
        "- SAM3-Classify 的分类头只有 `negative/positive`，因此不提供五类类别 F1；表中实例指标按所有元件统一类别比较。",
        "",
        "## 指标对比",
        "",
        "| 指标 | SAM3-Classify | YOLOv8-Seg | SAM3-Classify - YOLO |",
        "|---|---:|---:|---:|",
    ]
    for name, sam_value, yolo_value in rows:
        if yolo_value is None:
            lines.append(f"| {name} | {sam_value:.4f} | N/A | N/A |")
        else:
            delta = sam_value - float(yolo_value)
            lines.append(f"| {name} | {sam_value:.4f} | {float(yolo_value):.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "## SAM3-Classify 训练与评价",
            "",
            f"- 设备：`{result['device']}`",
            f"- SAM3 候选训练样本：`{result['train_candidate_count']}`",
            f"- 训练正样本：`{result['train_positive_count']}`",
            f"- 训练负样本：`{result['train_negative_count']}`",
            f"- 验证候选数：`{result['validation_candidate_count']}`",
            f"- 最佳 epoch：`{result['best_epoch']}`",
            f"- 分类头最佳 Macro-F1：`{result['best_classifier_macro_f1']:.4f}`",
            f"- 分类器权重：`{result['classifier_checkpoint']}`",
            f"- SAM3 权重：`{result['sam3_checkpoint']}`",
            "",
            "## 详细结果",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def train_and_evaluate(args: argparse.Namespace) -> dict:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3-Classify but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    InspectionConfig, SamPromptSegmenter, extract_patch = _imports()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = InspectionConfig(
        prompt="component",
        sam_confidence_threshold=args.sam_threshold,
        min_mask_area=30,
        edge_margin=0,
        tile_size=1600,
        tile_overlap=256,
        save_candidate_patches=False,
    )
    segmenter = SamPromptSegmenter(args.sam_checkpoint, config, device="cuda")
    try:
        train_records_all = build_candidate_cache(
            args.dataset, "train", output / "cache" / "train.pkl", segmenter, args.label_iou
        )
        val_records = build_candidate_cache(
            args.dataset, "val", output / "cache" / "val.pkl", segmenter, args.label_iou
        )
    finally:
        segmenter.close()
    train_records = balance_training_records(train_records_all, args.seed, args.negative_ratio)
    train_patches = prepare_patches(train_records, extract_patch)
    val_patches = prepare_patches(val_records, extract_patch)
    train_targets = [record["label"] for record in train_records]
    val_targets = [record["label"] for record in val_records]
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])

    class PatchDataset(Dataset):
        def __init__(self, patches, targets, transform):
            self.patches = patches
            self.targets = targets
            self.transform = transform

        def __len__(self):
            return len(self.patches)

        def __getitem__(self, index):
            return self.transform(Image.fromarray(self.patches[index])), self.targets[index]

    train_loader = DataLoader(PatchDataset(train_patches, train_targets, train_transform), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(PatchDataset(val_patches, val_targets, eval_transform), batch_size=args.batch, shuffle=False, num_workers=0)
    model = build_classifier().to(device)
    counts = np.bincount(train_targets, minlength=2).astype(np.float32)
    weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_f1 = -1.0
    best_epoch = 0
    history = []
    classifier_checkpoint = output / "classifier_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, targets in train_loader:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
        model.eval()
        predictions, targets_all = [], []
        with torch.inference_mode():
            for batch, targets in val_loader:
                predictions.extend(model(batch.to(device)).argmax(1).cpu().tolist())
                targets_all.extend(targets.tolist())
        metrics = classifier_metrics(predictions, targets_all)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_records)), **metrics}
        history.append(row)
        if metrics["macro_f1"] >= best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": list(IMAGE_SIZE),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                },
                classifier_checkpoint,
            )
        print(f"classifier epoch {epoch}/{args.epochs}: val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
    (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(classifier_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch, _ in val_loader:
            probabilities.extend(torch.softmax(model(batch.to(device)), dim=1)[:, 1].cpu().tolist())
    predictions_by_image: dict[str, list[dict]] = defaultdict(list)
    truths_by_image: dict[str, list[dict]] = {}
    val_image_order = []
    for record, probability in zip(val_records, probabilities):
        image_path = record["image_path"]
        if image_path not in truths_by_image:
            image = read_rgb(Path(image_path))
            label_path = args.dataset / "labels" / "val" / f"{Path(image_path).stem}.txt"
            truths_by_image[image_path] = load_truth(label_path, image.shape[:2])
            val_image_order.append(image_path)
        if probability >= args.classifier_threshold:
            image = read_rgb(Path(image_path))
            predictions_by_image[image_path].append(
                {
                    "confidence": float(record["candidate"].sam_score * probability),
                    "mask": candidate_full_mask(record["candidate"], image.shape[:2]),
                }
            )
    ordered_predictions = [predictions_by_image[path] for path in val_image_order]
    ordered_truths = [truths_by_image[path] for path in val_image_order]
    evaluation = evaluate_predictions(ordered_predictions, ordered_truths)
    classifier_val = classifier_metrics(
        [int(value >= args.classifier_threshold) for value in probabilities], val_targets
    )
    yolo_metrics_path = args.yolo_metrics.resolve()
    result = {
        "method": "SAM3-Classify: frozen engineering SAM3 + trainable engineering ResNet18 candidate filter",
        "device": str(device),
        "dataset": str(args.dataset.resolve()),
        "sam3_checkpoint": str(args.sam_checkpoint.resolve()),
        "classifier_checkpoint": str(classifier_checkpoint.resolve()),
        "train_candidate_count": len(train_records),
        "train_positive_count": sum(train_targets),
        "train_negative_count": len(train_targets) - sum(train_targets),
        "validation_candidate_count": len(val_records),
        "label_iou_threshold": args.label_iou,
        "classifier_threshold": args.classifier_threshold,
        "best_epoch": best_epoch,
        "best_classifier_macro_f1": best_f1,
        "classifier_validation": classifier_val,
        "evaluation": evaluation,
        "yolo_reference_metrics": str(yolo_metrics_path),
        "definitions": {
            "matching_iou": 0.5,
            "boundary_tolerance_px": 2,
            "small_area_threshold_px": 4096,
            "confidence": "SAM3 score multiplied by positive probability",
        },
        "limitations": [
            "SAM3 is frozen; only the engineering binary candidate-filter head is trained.",
            "SAM3-Classify reports all-component instance metrics, while YOLOv8-Seg also reports five-class metrics.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, yolo_metrics_path, output / "metrics.md")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM3_DEFAULT)
    parser.add_argument("--yolo-metrics", type=Path, default=YOLO_METRICS_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-threshold", type=float, default=0.25)
    parser.add_argument("--label-iou", type=float, default=0.30)
    parser.add_argument("--classifier-threshold", type=float, default=0.50)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train_and_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
