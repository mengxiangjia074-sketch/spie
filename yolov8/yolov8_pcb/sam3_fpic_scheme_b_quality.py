#!/usr/bin/env python3
"""Train a SAM3 candidate quality and acceptance head on FPIC S1."""

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


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SAM3_PROGRAMS = WORKSPACE / "sam3" / "programs" / "detection"
SAM3_CHECKPOINT = WORKSPACE / "sam3" / "models" / "sam3" / "sam3.pt"
DATASET = ROOT / "ground_truth" / "fpic_yolo"
OUTPUT = ROOT / "runs" / "sam3_fpic_scheme_b_quality"
CLASS_NAMES = ["resistors", "capacitors", "ICs", "inductors", "diodes"]
DEFAULT_DEV_GROUP = "Microscope_s1_p1_2x_40_ring"


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


class QualityResNet18(__import__("torch").nn.Module):
    def __init__(self, num_classes: int):
        import torch.nn as nn
        from torchvision.models import resnet18

        super().__init__()
        self.backbone = resnet18(weights=None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(features, num_classes)
        self.quality = nn.Linear(features, 1)

    def forward(self, inputs):
        features = self.backbone(inputs)
        return self.classifier(features), self.quality(features).squeeze(1)


def _build_classifier(num_classes: int):
    return QualityResNet18(num_classes)


def _install_component_package():
    import types

    sys.path.insert(0, str(SAM3_PROGRAMS))
    package = types.ModuleType("component_inspection")
    package.__path__ = [str(SAM3_PROGRAMS / "component_inspection")]
    sys.modules["component_inspection"] = package


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


def _candidate_records(dataset_root: Path, split: str, segmenter, label_iou: float):
    from PIL import Image

    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    records = []
    for image_path in sorted(image_root.glob("*.jpg")):
        image = _read_image(image_path)
        _, truths = _load_truth(image_path, label_root / f"{image_path.stem}.txt")
        candidates = segmenter.segment(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), image_path.name)
        for candidate in candidates:
            full_mask = _candidate_full_mask(candidate, image.shape[:2])
            overlaps = [_iou(full_mask, truth["mask"]) for truth in truths]
            best_index = int(np.argmax(overlaps)) if overlaps else -1
            best_iou = float(overlaps[best_index]) if overlaps else 0.0
            class_target = int(truths[best_index]["class_id"] + 1) if best_index >= 0 and best_iou >= label_iou else 0
            records.append({
                "image_path": str(image_path.resolve()),
                "candidate": candidate,
                "class_target": class_target,
                "quality_target": best_iou,
            })
        print(f"{split}: {image_path.name} candidates={len(candidates)}", flush=True)
    return records


def _prediction_mask(candidate, shape, mode: str):
    if mode == "raw":
        return _candidate_full_mask(candidate, shape)
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    if mode == "bbox":
        x1, y1, x2, y2 = map(int, candidate.mask_bbox_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    elif mode == "rotated-rect":
        points = cv2.boxPoints(candidate.rotated_rect)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    else:
        raise ValueError(f"unsupported mask mode: {mode}")
    return mask.astype(bool)


def _candidate_full_mask(candidate, shape):
    height, width = shape
    x1, y1, x2, y2 = map(int, candidate.mask_bbox_xyxy)
    full = np.zeros(shape, dtype=bool)
    ix1, iy1, ix2, iy2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return full
    crop = np.asarray(candidate.mask_crop, dtype=bool)
    full[iy1:iy2, ix1:ix2] = crop[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1]
    return full


def _load_or_build_records(dataset_root: Path, split: str, cache_path: Path, segmenter, label_iou: float, sam_threshold: float):
    if cache_path.is_file():
        with cache_path.open("rb") as source:
            payload = pickle.load(source)
        if payload.get("label_iou") == label_iou and payload.get("sam_threshold") == sam_threshold:
            return payload["records"]
    records = _candidate_records(dataset_root, split, segmenter, label_iou)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as target:
        pickle.dump({"label_iou": label_iou, "sam_threshold": sam_threshold, "records": records}, target, protocol=pickle.HIGHEST_PROTOCOL)
    return records


def _balance_records(records, seed: int, negative_ratio: int, negative_iou: float):
    positives = [record for record in records if record["class_target"] > 0]
    negatives = [record for record in records if record["quality_target"] <= negative_iou]
    if not positives:
        raise RuntimeError("SAM3 produced no positive candidates")
    rng = random.Random(seed)
    rng.shuffle(negatives)
    negatives = negatives[: max(len(positives) * negative_ratio, negative_ratio)]
    balanced = positives + negatives
    rng.shuffle(balanced)
    return balanced


def _prepare_patches(records, extract_patch):
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["image_path"]].append((index, record))
    patches = [None] * len(records)
    for image_path, items in grouped.items():
        image = cv2.cvtColor(_read_image(Path(image_path)), cv2.COLOR_BGR2RGB)
        for index, record in items:
            patches[index] = np.asarray(extract_patch(image, record["candidate"], 0.15, 160), dtype=np.uint8)
    return patches


def _foreground_macro_f1(predictions, targets):
    values = []
    for class_id in range(1, len(CLASS_NAMES) + 1):
        tp = sum(p == class_id and t == class_id for p, t in zip(predictions, targets))
        fp = sum(p == class_id and t != class_id for p, t in zip(predictions, targets))
        fn = sum(p != class_id and t == class_id for p, t in zip(predictions, targets))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-9, precision + recall))
    return float(np.mean(values))


def train_classifier(dataset_root: Path, output: Path, sam_checkpoint: Path, epochs: int, batch_size: int, device_name: str,
                     sam_threshold: float, label_iou: float, negative_iou: float, negative_ratio: int,
                     dev_group: str, seed: int):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms

    _install_component_package()
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import SamPromptSegmenter, extract_rectified_patch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device((f"cuda:{device_name}" if str(device_name).isdigit() else device_name) if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = InspectionConfig(prompt="component", sam_confidence_threshold=sam_threshold, min_mask_area=30,
                              edge_margin=0, tile_size=1600, tile_overlap=256)
    segmenter = SamPromptSegmenter(sam_checkpoint, config, device="cuda" if device.type == "cuda" else "cpu")
    try:
        train_records_all = _load_or_build_records(dataset_root, "train", output / "cache" / "train.pkl", segmenter, label_iou, sam_threshold)
    finally:
        segmenter.close()
    dev_marker = f"fpic_{dev_group}_"
    dev_records = [record for record in train_records_all if dev_marker in Path(record["image_path"]).name]
    train_pool = [record for record in train_records_all if dev_marker not in Path(record["image_path"]).name]
    if not dev_records:
        raise RuntimeError(f"no candidate records found for dev group: {dev_group}")
    train_records = _balance_records(train_pool, seed, negative_ratio, negative_iou)
    train_patches = _prepare_patches(train_records, extract_rectified_patch)
    dev_patches = _prepare_patches(dev_records, extract_rectified_patch)
    num_classes = len(CLASS_NAMES) + 1
    train_targets = [record["class_target"] for record in train_records]
    dev_targets = [record["class_target"] for record in dev_records]
    train_quality = [record["quality_target"] for record in train_records]
    dev_quality = [record["quality_target"] for record in dev_records]
    train_transform = transforms.Compose([transforms.Resize((96, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    eval_transform = transforms.Compose([transforms.Resize((96, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class CandidateDataset(Dataset):
        def __init__(self, patches, targets, qualities, transform):
            self.patches, self.targets, self.qualities, self.transform = patches, targets, qualities, transform
        def __len__(self):
            return len(self.patches)
        def __getitem__(self, index):
            return self.transform(Image.fromarray(self.patches[index])), self.targets[index], self.qualities[index]

    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    sample_weights = [1.0 / np.sqrt(max(counts[target], 1.0)) for target in train_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(CandidateDataset(train_patches, train_targets, train_quality, train_transform), batch_size=batch_size, sampler=sampler, num_workers=0)
    dev_loader = DataLoader(CandidateDataset(dev_patches, dev_targets, dev_quality, eval_transform), batch_size=batch_size, shuffle=False, num_workers=0)
    class_weights = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    model = _build_classifier(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_selection = (-1.0, float("-inf"))
    history = []
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "classifier_quality_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, targets, qualities in train_loader:
            batch = batch.to(device)
            targets = targets.to(device, dtype=torch.long)
            qualities = qualities.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits, quality_logits = model(batch)
            classification_loss = criterion(logits, targets)
            quality_loss = torch.nn.functional.smooth_l1_loss(torch.sigmoid(quality_logits), qualities)
            loss = classification_loss + quality_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
        model.eval()
        predictions, targets_all, quality_predictions = [], [], []
        with torch.inference_mode():
            for batch, targets, qualities in dev_loader:
                logits, quality_logits = model(batch.to(device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                targets_all.extend(targets.tolist())
                quality_predictions.extend(torch.sigmoid(quality_logits).cpu().tolist())
        macro_f1 = _foreground_macro_f1(predictions, targets_all)
        quality_mae = float(np.mean(np.abs(np.asarray(quality_predictions) - np.asarray(dev_quality))))
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_records)), "val_foreground_macro_f1": macro_f1, "val_quality_mae": quality_mae}
        history.append(row)
        selection = (macro_f1, -quality_mae)
        if selection > best_selection:
            best_selection = selection
            torch.save({"model_state_dict": model.state_dict(), "model_type": "quality_resnet18", "class_names": ["background", *CLASS_NAMES], "image_size": [96, 224], "epoch": epoch, "dev_group": dev_group, "dev_foreground_macro_f1": macro_f1, "dev_quality_mae": quality_mae}, checkpoint_path)
        print(f"quality epoch {epoch}/{epochs}: loss={row['train_loss']:.4f} val_macro_f1={macro_f1:.4f} quality_mae={quality_mae:.4f}", flush=True)
    (output / "classifier_quality_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return checkpoint_path


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


def run_pipeline(dataset_root: Path, sam_checkpoint: Path, classifier_checkpoint: Path, output: Path, device_name: str,
                 classifier_threshold: float, quality_threshold: float, mask_mode: str):
    _install_component_package()
    from PIL import Image
    import torch
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import SamPromptSegmenter, extract_rectified_patch
    from torchvision import transforms

    device = torch.device((f"cuda:{device_name}" if str(device_name).isdigit() else device_name) if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(classifier_checkpoint, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names", ["background", *CLASS_NAMES])
    model = _build_classifier(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = transforms.Compose([transforms.Resize((96, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    config = InspectionConfig(prompt="component", sam_confidence_threshold=0.25, min_mask_area=30, edge_margin=0, tile_size=1600, tile_overlap=256)
    segmenter = SamPromptSegmenter(sam_checkpoint, config, device="cuda" if device.type == "cuda" else "cpu")
    predictions_by_image, truths_by_image = [], []
    val_images = sorted((dataset_root / "images" / "val").glob("*.jpg"))
    output.mkdir(parents=True, exist_ok=True)
    for image_path in val_images:
        image_bgr, truths = _load_truth(image_path, dataset_root / "labels" / "val" / f"{image_path.stem}.txt")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidates = segmenter.segment(Image.fromarray(image_rgb), image_path.name)
        predictions = []
        with torch.inference_mode():
            for candidate in candidates:
                patch = extract_rectified_patch(image_rgb, candidate, 0.15, 160)
                tensor = transform(Image.fromarray(patch.astype(np.uint8))).unsqueeze(0).to(device)
                logits, quality_logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                quality = float(torch.sigmoid(quality_logits)[0].cpu())
                foreground_probability = float(1.0 - probabilities[0])
                class_id = int(np.argmax(probabilities[1:])) if foreground_probability >= classifier_threshold else -1
                if class_id >= 0 and quality >= quality_threshold:
                    class_id += 0
                    x1, y1, x2, y2 = candidate.mask_bbox_xyxy
                    full_mask = _prediction_mask(candidate, image_bgr.shape[:2], mask_mode)
                    class_probability = float(probabilities[class_id + 1])
                    predictions.append({"class_id": class_id, "confidence": float(candidate.sam_score * foreground_probability * class_probability * quality), "mask": full_mask})
        predictions_by_image.append(predictions)
        truths_by_image.append(truths)
        print(f"{image_path.name}: accepted={len(predictions)}", flush=True)
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
            if np.count_nonzero(truth["mask"]) < 4096:
                small_total += 1
                if ti in used_t: small_tp += 1
            if ti not in used_t: class_stats[truth["class_id"]]["fn"] += 1
    class_f1 = {}
    for class_id, stats in class_stats.items():
        p = stats["tp"] / max(1, stats["tp"] + stats["fp"]); r = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        class_f1[CLASS_NAMES[class_id]] = 2 * p * r / max(1e-9, p + r)
    class_tp = sum(s["tp"] for s in class_stats.values()); class_fp = sum(s["fp"] for s in class_stats.values()); class_fn = sum(s["fn"] for s in class_stats.values())
    result = {"method": "quality-aware SAM3 + background/five-class head", "mask_mode": mask_mode, "test_images": len(val_images), "ground_truth_instances": total_truth, "predicted_instances": total_pred, "matched_instances": matched, **ap_values, "Boundary F1": float(np.mean(boundary_f1)) if boundary_f1 else 0.0, "Boundary IoU": float(np.mean(boundary_iou)) if boundary_iou else 0.0, "Macro-F1": float(np.mean(list(class_f1.values()))), "Joint-F1": 2 * class_tp / max(1, 2 * class_tp + class_fp + class_fn), "Small Component Recall": small_tp / max(1, small_total), "Center Error Mean px": float(np.mean(center_errors)) if center_errors else 0.0, "Center Error Median px": float(np.median(center_errors)) if center_errors else 0.0, "class_f1": class_f1, "classifier_checkpoint": str(classifier_checkpoint.resolve()), "classifier_threshold": classifier_threshold, "quality_threshold": quality_threshold, "definitions": {"match_iou": 0.5, "small_area_px": 4096, "boundary_tolerance_px": 2, "confidence": "SAM score * foreground probability * class probability * predicted mask IoU"}}
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
    parser.add_argument("--sam-threshold", type=float, default=0.25)
    parser.add_argument("--label-iou", type=float, default=0.30)
    parser.add_argument("--negative-iou", type=float, default=0.10)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--dev-group", default=DEFAULT_DEV_GROUP)
    parser.add_argument("--classifier-threshold", type=float, default=0.50)
    parser.add_argument("--quality-threshold", type=float, default=0.30)
    parser.add_argument("--mask-mode", choices=["raw", "bbox", "rotated-rect"], default="raw")
    args = parser.parse_args()
    if args.command == "train-head":
        checkpoint = train_classifier(args.dataset, args.output, args.sam_checkpoint, args.epochs, args.batch, args.device, args.sam_threshold, args.label_iou, args.negative_iou, args.negative_ratio, args.dev_group, 7)
        print(checkpoint)
    else:
        classifier = args.classifier or args.output / "classifier_quality_best.pt"
        print(json.dumps(run_pipeline(args.dataset, args.sam_checkpoint, classifier, args.output, args.device, args.classifier_threshold, args.quality_threshold, args.mask_mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
