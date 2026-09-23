#!/usr/bin/env python3
"""Train and evaluate a local mask refiner for quality-aware SAM3 candidates."""

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

import sam3_fpic_scheme_b_quality as quality


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "ground_truth" / "fpic_yolo"
SAM3_CHECKPOINT = ROOT.parents[1] / "sam3" / "models" / "sam3" / "sam3.pt"
QUALITY_RUN = ROOT / "runs" / "sam3_fpic_scheme_b_quality"
OUTPUT = ROOT / "runs" / "sam3_fpic_mask_refiner"
INPUT_SIZE = (128, 256)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def expanded_box(box, shape, padding: float = 0.50):
    height, width = shape
    x1, y1, x2, y2 = map(int, box)
    pad_x = max(4, int(round((x2 - x1) * padding)))
    pad_y = max(4, int(round((y2 - y1) * padding)))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def candidate_bbox_mask(candidate, shape):
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, candidate.mask_bbox_xyxy)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask.astype(bool)


def resize_sample(image_rgb, raw_mask, box_mask, target_mask, box):
    x1, y1, x2, y2 = box
    size = (INPUT_SIZE[1], INPUT_SIZE[0])
    image = cv2.resize(image_rgb[y1:y2, x1:x2], size, interpolation=cv2.INTER_AREA)
    raw = cv2.resize(raw_mask[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    bbox = cv2.resize(box_mask[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    target = cv2.resize(target_mask[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    return image, raw, bbox, target


def build_samples(records, label_iou: float):
    grouped = defaultdict(list)
    for record in records:
        if record["quality_target"] >= label_iou:
            grouped[record["image_path"]].append(record)
    samples = []
    for image_name, image_records in grouped.items():
        image_path = Path(image_name)
        image_bgr, truths = quality._load_truth(
            image_path,
            image_path.parents[2] / "labels" / image_path.parent.name / f"{image_path.stem}.txt",
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        for record in image_records:
            candidate = record["candidate"]
            raw_mask = quality._candidate_full_mask(candidate, image_bgr.shape[:2])
            overlaps = [quality._iou(raw_mask, truth["mask"]) for truth in truths]
            if not overlaps:
                continue
            target = truths[int(np.argmax(overlaps))]["mask"]
            box = expanded_box(candidate.mask_bbox_xyxy, image_bgr.shape[:2])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            sample = resize_sample(
                image_rgb,
                raw_mask,
                candidate_bbox_mask(candidate, image_bgr.shape[:2]),
                target,
                box,
            )
            samples.append({"image_path": image_name, "data": sample})
    return samples


def make_input(image, raw_mask, box_mask):
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    return np.concatenate(
        [image.transpose(2, 0, 1), raw_mask[None].astype(np.float32), box_mask[None].astype(np.float32)],
        axis=0,
    )


def conv_block(in_channels: int, out_channels: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MaskRefiner(__import__("torch").nn.Module):
    def __init__(self):
        import torch.nn as nn

        super().__init__()
        self.encoder1 = conv_block(5, 32)
        self.encoder2 = conv_block(32, 64)
        self.bottleneck = conv_block(64, 128)
        self.decoder2 = conv_block(128 + 64, 64)
        self.decoder1 = conv_block(64 + 32, 32)
        self.output = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        import torch.nn.functional as F

        level1 = self.encoder1(inputs)
        level2 = self.encoder2(self.pool(level1))
        features = self.bottleneck(self.pool(level2))
        features = F.interpolate(features, size=level2.shape[-2:], mode="bilinear", align_corners=False)
        features = self.decoder2(__import__("torch").cat([features, level2], dim=1))
        features = F.interpolate(features, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder1(__import__("torch").cat([features, level1], dim=1))).squeeze(1)


def mask_loss(logits, targets):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).flatten(1).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1) + 1.0)).mean()
    pred_edge = F.max_pool2d(probabilities[:, None], 3, 1, 1) + F.max_pool2d(-probabilities[:, None], 3, 1, 1)
    target_edge = F.max_pool2d(targets[:, None], 3, 1, 1) + F.max_pool2d(-targets[:, None], 3, 1, 1)
    edge_intersection = (pred_edge * target_edge).flatten(1).sum(1)
    edge_dice = 1.0 - ((2.0 * edge_intersection + 1.0) / (pred_edge.flatten(1).sum(1) + target_edge.flatten(1).sum(1) + 1.0)).mean()
    return bce + dice + 0.5 * edge_dice


def train(args):
    import torch
    from torch.utils.data import DataLoader, Dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    quality._install_component_package()
    with args.candidate_cache.open("rb") as source:
        payload = pickle.load(source)
    records = payload["records"]
    marker = f"fpic_{args.dev_group}_"
    train_samples = build_samples([r for r in records if marker not in Path(r["image_path"]).name], args.label_iou)
    dev_samples = build_samples([r for r in records if marker in Path(r["image_path"]).name], args.label_iou)
    if not train_samples or not dev_samples:
        raise RuntimeError("mask refiner train/dev samples are empty")

    class RefinementDataset(Dataset):
        def __init__(self, samples, augment):
            self.samples = samples
            self.augment = augment
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, index):
            image, raw, bbox, target = self.samples[index]["data"]
            image, raw, bbox, target = image.copy(), raw.copy(), bbox.copy(), target.copy()
            if self.augment and random.random() < 0.5:
                image, raw, bbox, target = image[:, ::-1].copy(), raw[:, ::-1].copy(), bbox[:, ::-1].copy(), target[:, ::-1].copy()
            if self.augment and random.random() < 0.5:
                image, raw, bbox, target = image[::-1].copy(), raw[::-1].copy(), bbox[::-1].copy(), target[::-1].copy()
            if self.augment:
                gain = random.uniform(0.8, 1.2)
                image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            return torch.from_numpy(make_input(image, raw, bbox)), torch.from_numpy(target.astype(np.float32))

    train_loader = DataLoader(RefinementDataset(train_samples, True), batch_size=args.batch, shuffle=True, num_workers=0)
    dev_loader = DataLoader(RefinementDataset(dev_samples, False), batch_size=args.batch, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = MaskRefiner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "mask_refiner_best.pt"
    best_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mask_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        intersections = sums = 0.0
        with torch.inference_mode():
            for inputs, targets in dev_loader:
                predictions = torch.sigmoid(model(inputs.to(device))).cpu()
                intersections += float((predictions * targets).sum()) * 2.0
                sums += float(predictions.sum() + targets.sum())
        dice = intersections / max(sums, 1.0)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_soft_dice": dice}
        history.append(row)
        if dice > best_dice:
            best_dice = dice
            torch.save({"model_state_dict": model.state_dict(), "input_size": list(INPUT_SIZE), "dev_group": args.dev_group, "epoch": epoch, "dev_soft_dice": dice}, checkpoint_path)
        print(f"refiner epoch {epoch}/{args.epochs}: loss={row['train_loss']:.4f} dev_dice={dice:.4f}", flush=True)
    (output / "mask_refiner_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(checkpoint_path)


def refine_candidate(model, image_rgb, candidate, device, threshold):
    import torch

    shape = image_rgb.shape[:2]
    raw = quality._candidate_full_mask(candidate, shape)
    bbox_mask = candidate_bbox_mask(candidate, shape)
    box = expanded_box(candidate.mask_bbox_xyxy, shape)
    x1, y1, x2, y2 = box
    size = (INPUT_SIZE[1], INPUT_SIZE[0])
    image = cv2.resize(image_rgb[y1:y2, x1:x2], size, interpolation=cv2.INTER_AREA)
    raw_small = cv2.resize(raw[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    bbox_small = cv2.resize(bbox_mask[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    inputs = torch.from_numpy(make_input(image, raw_small, bbox_small))[None].to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(inputs))[0].cpu().numpy()
    probability = cv2.resize(probability, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    full = np.zeros(shape, dtype=bool)
    full[y1:y2, x1:x2] = probability >= threshold
    return full


def summarize(predictions_by_image, truths_by_image):
    ap_values = {f"AP@{threshold:.2f}": float(np.mean([quality._ap(predictions_by_image, truths_by_image, class_id, threshold) for class_id in range(len(quality.CLASS_NAMES))])) for threshold in [0.5, 0.75]}
    ap_values["AP50:95"] = float(np.mean([quality._ap(predictions_by_image, truths_by_image, class_id, threshold) for class_id in range(len(quality.CLASS_NAMES)) for threshold in np.arange(0.5, 1.0, 0.05)]))
    stats = {class_id: {"tp": 0, "fp": 0, "fn": 0} for class_id in range(len(quality.CLASS_NAMES))}
    boundary_f1, boundary_iou, center_errors = [], [], []
    matched = predicted = ground_truth = small_total = small_matched = 0
    for predictions, truths in zip(predictions_by_image, truths_by_image):
        predicted += len(predictions); ground_truth += len(truths)
        pairs = sorted(((quality._iou(p["mask"], t["mask"]), pi, ti) for pi, p in enumerate(predictions) for ti, t in enumerate(truths)), reverse=True)
        used_p, used_t = set(), set()
        for iou, pi, ti in pairs:
            if iou < 0.5 or pi in used_p or ti in used_t:
                continue
            used_p.add(pi); used_t.add(ti); matched += 1
            p, t = predictions[pi], truths[ti]
            if p["class_id"] == t["class_id"]: stats[t["class_id"]]["tp"] += 1
            else: stats[t["class_id"]]["fn"] += 1; stats[p["class_id"]]["fp"] += 1
            bf1, biou = quality._boundary_metrics(p["mask"], t["mask"]); boundary_f1.append(bf1); boundary_iou.append(biou)
            px, py = quality._centroid(p["mask"]); tx, ty = quality._centroid(t["mask"]); center_errors.append(float(np.hypot(px - tx, py - ty)))
        for pi, p in enumerate(predictions):
            if pi not in used_p: stats[p["class_id"]]["fp"] += 1
        for ti, t in enumerate(truths):
            if np.count_nonzero(t["mask"]) < 4096:
                small_total += 1; small_matched += int(ti in used_t)
            if ti not in used_t: stats[t["class_id"]]["fn"] += 1
    class_f1 = {}
    for class_id, value in stats.items():
        precision = value["tp"] / max(1, value["tp"] + value["fp"])
        recall = value["tp"] / max(1, value["tp"] + value["fn"])
        class_f1[quality.CLASS_NAMES[class_id]] = 2 * precision * recall / max(precision + recall, 1e-9)
    tp = sum(v["tp"] for v in stats.values()); fp = sum(v["fp"] for v in stats.values()); fn = sum(v["fn"] for v in stats.values())
    return {**ap_values, "Boundary F1": float(np.mean(boundary_f1)), "Boundary IoU": float(np.mean(boundary_iou)), "Macro-F1": float(np.mean(list(class_f1.values()))), "Joint-F1": 2 * tp / max(1, 2 * tp + fp + fn), "Small Component Recall": small_matched / max(1, small_total), "Center Error Mean px": float(np.mean(center_errors)), "Center Error Median px": float(np.median(center_errors)), "ground_truth_instances": ground_truth, "predicted_instances": predicted, "matched_instances": matched, "class_f1": class_f1}


def evaluate(args):
    import torch
    from PIL import Image
    from torchvision import transforms

    quality._install_component_package()
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import SamPromptSegmenter, extract_rectified_patch

    device = torch.device(args.device)
    refiner_data = torch.load(args.refiner, map_location=device, weights_only=True)
    refiner = MaskRefiner().to(device)
    refiner.load_state_dict(refiner_data["model_state_dict"])
    refiner.eval()
    classifier_data = torch.load(args.classifier, map_location=device, weights_only=True)
    classifier = quality._build_classifier(len(classifier_data["class_names"])).to(device)
    classifier.load_state_dict(classifier_data["model_state_dict"])
    classifier.eval()
    transform = transforms.Compose([transforms.Resize((96, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    config = InspectionConfig(prompt="component", sam_confidence_threshold=args.sam_threshold, min_mask_area=30, edge_margin=0, tile_size=1600, tile_overlap=256)
    segmenter = SamPromptSegmenter(args.sam_checkpoint, config, device="cuda")
    predictions_by_image, truths_by_image = [], []
    image_paths = sorted((args.dataset / "images" / args.eval_split).glob("*.jpg"))
    if args.eval_group:
        marker = f"fpic_{args.eval_group}_"
        image_paths = [path for path in image_paths if marker in path.name]
    for image_path in image_paths:
        image_bgr, truths = quality._load_truth(image_path, args.dataset / "labels" / args.eval_split / f"{image_path.stem}.txt")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        candidates = segmenter.segment(Image.fromarray(image_rgb), image_path.name)
        predictions = []
        with torch.inference_mode():
            for candidate in candidates:
                patch = extract_rectified_patch(image_rgb, candidate, 0.15, 160)
                logits, quality_logits = classifier(transform(Image.fromarray(patch.astype(np.uint8)))[None].to(device))
                probabilities = torch.softmax(logits, 1)[0]
                foreground = float(1.0 - probabilities[0])
                predicted_quality = float(torch.sigmoid(quality_logits)[0])
                if foreground < args.classifier_threshold or predicted_quality < args.quality_threshold:
                    continue
                class_id = int(probabilities[1:].argmax())
                class_probability = float(probabilities[class_id + 1])
                mask = refine_candidate(refiner, image_rgb, candidate, device, args.mask_threshold)
                confidence = float(candidate.sam_score * foreground * class_probability * predicted_quality)
                predictions.append({"class_id": class_id, "confidence": confidence, "mask": mask})
        predictions_by_image.append(predictions); truths_by_image.append(truths)
        print(f"{image_path.name}: accepted={len(predictions)}", flush=True)
    segmenter.close()
    result = {"method": "quality-aware SAM3 + learned mask refiner", "refiner_checkpoint": str(args.refiner.resolve()), "mask_threshold": args.mask_threshold, "eval_split": args.eval_split, "eval_group": args.eval_group, **summarize(predictions_by_image, truths_by_image)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--candidate-cache", type=Path, default=QUALITY_RUN / "cache" / "train.pkl")
    parser.add_argument("--classifier", type=Path, default=QUALITY_RUN / "classifier_quality_best.pt")
    parser.add_argument("--refiner", type=Path, default=OUTPUT / "mask_refiner_best.pt")
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM3_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dev-group", default=quality.DEFAULT_DEV_GROUP)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--label-iou", type=float, default=0.30)
    parser.add_argument("--sam-threshold", type=float, default=0.25)
    parser.add_argument("--classifier-threshold", type=float, default=0.50)
    parser.add_argument("--quality-threshold", type=float, default=0.30)
    parser.add_argument("--mask-threshold", type=float, default=0.50)
    parser.add_argument("--eval-split", choices=["train", "val"], default="val")
    parser.add_argument("--eval-group")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
