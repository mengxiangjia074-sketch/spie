#!/usr/bin/env python3
"""Scheme B: frozen SAM3 segmentation plus a trainable five-class head."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SAM3_PROGRAMS = WORKSPACE / "sam3" / "programs" / "detection"
SAM3_CHECKPOINT = WORKSPACE / "sam3" / "models" / "sam3" / "sam3.pt"
DATASET = ROOT / "ground_truth" / "fpic_yolo"
OUTPUT = ROOT / "runs" / "sam3_fpic_scheme_b"
CLASS_NAMES = ["resistors", "capacitors", "ICs", "inductors", "diodes"]


def _read_image(path: Path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


def _entries(dataset_root: Path, split: str):
    entries = []
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    for image_path in sorted(image_root.glob("*.jpg")):
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image = _read_image(image_path)
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 7:
                continue
            class_id = int(values[0])
            points = np.asarray([float(v) for v in values[1:]], dtype=np.float32).reshape(-1, 2)
            x1 = max(0, int(np.floor(points[:, 0].min() * width)))
            y1 = max(0, int(np.floor(points[:, 1].min() * height)))
            x2 = min(width, int(np.ceil(points[:, 0].max() * width)))
            y2 = min(height, int(np.ceil(points[:, 1].max() * height)))
            pad_x = max(2, int((x2 - x1) * 0.15))
            pad_y = max(2, int((y2 - y1) * 0.15))
            entries.append(
                {
                    "image": image_path,
                    "class_id": class_id,
                    "box": (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)),
                }
            )
    return entries


def _build_classifier(num_classes: int):
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ComponentCropDataset:
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from PIL import Image
        image = _read_image(self.entries[index]["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = self.entries[index]["box"]
        crop = image[y1:y2, x1:x2]
        return self.transform(Image.fromarray(crop)), self.entries[index]["class_id"]


def _macro_f1(predictions, targets, num_classes):
    values = []
    for class_id in range(num_classes):
        tp = sum(p == class_id and t == class_id for p, t in zip(predictions, targets))
        fp = sum(p == class_id and t != class_id for p, t in zip(predictions, targets))
        fn = sum(p != class_id and t == class_id for p, t in zip(predictions, targets))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-9, precision + recall))
    return float(np.mean(values))


def train_classifier(dataset_root: Path, output: Path, epochs: int, batch_size: int, device_name: str):
    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader

    random.seed(7)
    torch.manual_seed(7)
    device = torch.device((f"cuda:{device_name}" if str(device_name).isdigit() else device_name) if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    entries = _entries(dataset_root, "train")
    random.shuffle(entries)
    split = max(1, int(len(entries) * 0.15))
    validation_entries, train_entries = entries[:split], entries[split:]
    train_transform = transforms.Compose(
        [transforms.Resize((96, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((96, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    train_loader = DataLoader(ComponentCropDataset(train_entries, train_transform), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(ComponentCropDataset(validation_entries, eval_transform), batch_size=batch_size, shuffle=False, num_workers=4)
    counts = np.bincount([entry["class_id"] for entry in train_entries], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    model = _build_classifier(len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    best_f1 = -1.0
    history = []
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
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
                logits = model(batch.to(device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                targets_all.extend(targets.tolist())
        f1 = _macro_f1(predictions, targets_all, len(CLASS_NAMES))
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(train_entries)), "val_macro_f1": f1})
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": [96, 224],
                    "epoch": epoch,
                    "val_macro_f1": f1,
                },
                output / "classifier_best.pt",
            )
        print(f"classifier epoch {epoch}/{epochs}: loss={history[-1]['train_loss']:.4f} val_macro_f1={f1:.4f}")
    (output / "classifier_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return output / "classifier_best.pt"


def _mask_from_points(points, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def _load_truth(image_path: Path, label_path: Path):
    image = _read_image(image_path)
    height, width = image.shape[:2]
    truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        points = np.asarray([float(v) for v in values[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        truth.append({"class_id": int(values[0]), "mask": _mask_from_points(points, (height, width))})
    return image, truth


def _iou(a, b):
    return float(np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b)))


def _boundary(mask):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)


def _boundary_metrics(pred, truth, tolerance=2):
    pb, tb = _boundary(pred), _boundary(truth)
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8), kernel).astype(bool)
    td = cv2.dilate(tb.astype(np.uint8), kernel).astype(bool)
    precision = np.count_nonzero(pb & td) / max(1, np.count_nonzero(pb))
    recall = np.count_nonzero(tb & pd) / max(1, np.count_nonzero(tb))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    biou = np.count_nonzero(pd & td) / max(1, np.count_nonzero(pd | td))
    return float(f1), float(biou)


def _centroid(mask):
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return (m["m10"] / max(1e-9, m["m00"]), m["m01"] / max(1e-9, m["m00"]))


def _ap(predictions_by_image, truths_by_image, class_id, threshold):
    records = []
    total_truth = 0
    used = {index: set() for index in range(len(truths_by_image))}
    for image_index, truths in enumerate(truths_by_image):
        total_truth += sum(t["class_id"] == class_id for t in truths)
        for prediction in predictions_by_image[image_index]:
            if prediction["class_id"] == class_id:
                records.append((prediction["confidence"], image_index, prediction))
    records.sort(key=lambda item: item[0], reverse=True)
    tp, fp = [], []
    for _, image_index, prediction in records:
        candidates = [
            (index, _iou(prediction["mask"], truth["mask"]))
            for index, truth in enumerate(truths_by_image[image_index])
            if truth["class_id"] == class_id and index not in used[image_index]
        ]
        best = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
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


def run_pipeline(dataset_root: Path, sam_checkpoint: Path, classifier_checkpoint: Path, output: Path, device_name: str):
    sys.path.insert(0, str(SAM3_PROGRAMS))
    import types
    package = types.ModuleType("component_inspection")
    package.__path__ = [str(SAM3_PROGRAMS / "component_inspection")]
    sys.modules["component_inspection"] = package
    from PIL import Image
    import torch
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import SamPromptSegmenter, extract_rectified_patch
    from torchvision import transforms

    device = torch.device((f"cuda:{device_name}" if str(device_name).isdigit() else device_name) if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier_data = torch.load(classifier_checkpoint, map_location=device, weights_only=True)
    classifier = _build_classifier(len(CLASS_NAMES)).to(device)
    classifier.load_state_dict(classifier_data["model_state_dict"])
    classifier.eval()
    transform = transforms.Compose([transforms.Resize((96, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    config = InspectionConfig(prompt="component", sam_confidence_threshold=0.25, min_mask_area=30, edge_margin=0, tile_size=1600, tile_overlap=256)
    sam_device = "cuda" if device.type == "cuda" else "cpu"
    segmenter = SamPromptSegmenter(sam_checkpoint, config, device=sam_device)
    predictions_by_image, truths_by_image = [], []
    val_images = sorted((dataset_root / "images" / "val").glob("*.jpg"))
    output.mkdir(parents=True, exist_ok=True)
    for image_path in val_images:
        image_bgr, truths = _load_truth(image_path, dataset_root / "labels" / "val" / f"{image_path.stem}.txt")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidates = segmenter.segment(Image.fromarray(image_rgb), image_path.name)
        patches = [extract_rectified_patch(image_rgb, candidate, 0.15, 160) for candidate in candidates]
        predictions = []
        with torch.inference_mode():
            for candidate, patch in zip(candidates, patches):
                tensor = transform(Image.fromarray(patch.astype(np.uint8))).unsqueeze(0).to(device)
                probabilities = torch.softmax(classifier(tensor), dim=1)[0].cpu().numpy()
                class_id = int(probabilities.argmax())
                x1, y1, x2, y2 = candidate.mask_bbox_xyxy
                full_mask = np.zeros(image_bgr.shape[:2], dtype=bool)
                ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
                ix1, iy1 = max(0, ix1), max(0, iy1)
                ix2, iy2 = min(image_bgr.shape[1], ix2), min(image_bgr.shape[0], iy2)
                crop = candidate.mask_crop[: max(0, iy2 - int(y1)), : max(0, ix2 - int(x1))]
                if crop.size and ix2 > ix1 and iy2 > iy1:
                    full_mask[iy1:iy2, ix1:ix2] = crop[: iy2 - iy1, : ix2 - ix1]
                predictions.append({"class_id": class_id, "confidence": float(candidate.sam_score * probabilities[class_id]), "mask": full_mask})
        predictions_by_image.append(predictions)
        truths_by_image.append(truths)
        print(f"{image_path.name}: SAM3 candidates={len(candidates)}")
    segmenter.close()

    ap_values = {f"AP@{threshold:.2f}": float(np.mean([_ap(predictions_by_image, truths_by_image, class_id, threshold) for class_id in range(len(CLASS_NAMES))])) for threshold in [0.5, 0.75]}
    ap_values["AP50:95"] = float(np.mean([_ap(predictions_by_image, truths_by_image, class_id, threshold) for class_id in range(len(CLASS_NAMES)) for threshold in np.arange(0.5, 1.0, 0.05)]))
    class_stats = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(len(CLASS_NAMES))}
    boundary_f1, boundary_iou, center_errors = [], [], []
    matched = total_pred = total_truth = small_total = small_tp = 0
    for predictions, truths in zip(predictions_by_image, truths_by_image):
        total_pred += len(predictions); total_truth += len(truths)
        pairs = sorted(((_iou(p["mask"], t["mask"]), pi, ti) for pi, p in enumerate(predictions) for ti, t in enumerate(truths)), reverse=True)
        used_p, used_t = set(), set()
        for iou, pi, ti in pairs:
            if iou < 0.5 or pi in used_p or ti in used_t:
                continue
            used_p.add(pi); used_t.add(ti); matched += 1
            prediction, truth = predictions[pi], truths[ti]
            if prediction["class_id"] == truth["class_id"]:
                class_stats[truth["class_id"]]["tp"] += 1
            else:
                class_stats[truth["class_id"]]["fn"] += 1; class_stats[prediction["class_id"]]["fp"] += 1
            f1, biou = _boundary_metrics(prediction["mask"], truth["mask"]); boundary_f1.append(f1); boundary_iou.append(biou)
            px, py = _centroid(prediction["mask"]); tx, ty = _centroid(truth["mask"]); center_errors.append(float(np.hypot(px - tx, py - ty)))
        for pi, prediction in enumerate(predictions):
            if pi not in used_p: class_stats[prediction["class_id"]]["fp"] += 1
        for ti, truth in enumerate(truths):
            area = np.count_nonzero(truth["mask"])
            if area < 4096:
                small_total += 1
                if ti in used_t: small_tp += 1
            if ti not in used_t: class_stats[truth["class_id"]]["fn"] += 1
    class_f1 = {}
    for class_id, stats in class_stats.items():
        p = stats["tp"] / max(1, stats["tp"] + stats["fp"]); r = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        class_f1[CLASS_NAMES[class_id]] = 2 * p * r / max(1e-9, p + r)
    class_tp = sum(s["tp"] for s in class_stats.values()); class_fp = sum(s["fp"] for s in class_stats.values()); class_fn = sum(s["fn"] for s in class_stats.values())
    result = {"method": "frozen SAM3 + trained five-class ResNet18 head", "test_images": len(val_images), "ground_truth_instances": total_truth, "predicted_instances": total_pred, "matched_instances": matched, **ap_values, "Boundary F1": float(np.mean(boundary_f1)) if boundary_f1 else 0.0, "Boundary IoU": float(np.mean(boundary_iou)) if boundary_iou else 0.0, "Macro-F1": float(np.mean(list(class_f1.values()))), "Joint-F1": 2 * class_tp / max(1, 2 * class_tp + class_fp + class_fn), "Small Component Recall": small_tp / max(1, small_total), "Center Error Mean px": float(np.mean(center_errors)) if center_errors else 0.0, "Center Error Median px": float(np.median(center_errors)) if center_errors else 0.0, "class_f1": class_f1, "classifier_checkpoint": str(classifier_checkpoint.resolve()), "definitions": {"match_iou": 0.5, "small_area_px": 4096, "boundary_tolerance_px": 2}}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train-head", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM3_CHECKPOINT)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.command == "train-head":
        checkpoint = train_classifier(args.dataset, args.output, args.epochs, args.batch, args.device)
        print(checkpoint)
    else:
        classifier = args.classifier or args.output / "classifier_best.pt"
        print(json.dumps(run_pipeline(args.dataset, args.sam_checkpoint, classifier, args.output, args.device), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
