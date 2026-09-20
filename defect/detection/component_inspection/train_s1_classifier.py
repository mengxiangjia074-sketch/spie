#!/usr/bin/env python3
"""Train the engineering binary component classifier on the prepared s1 split.

The production pipeline freezes SAM3 and trains a ResNet18 candidate filter. This
script reproduces the classifier checkpoint contract without replacing the
existing production weights. SAM3 inference is intentionally not run here; the
machine running this job currently has no usable CUDA driver.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["negative", "positive"]
IMAGE_SIZE = (96, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)


def parse_labels(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
        x1 = max(0, int(np.floor(points[:, 0].min() * width)))
        y1 = max(0, int(np.floor(points[:, 1].min() * height)))
        x2 = min(width, int(np.ceil(points[:, 0].max() * width)))
        y2 = min(height, int(np.ceil(points[:, 1].max() * height)))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def padded_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = max(2, int((x2 - x1) * 0.15))
    pad_y = max(2, int((y2 - y1) * 0.15))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def negative_box(
    rng: random.Random,
    width: int,
    height: int,
    positive_boxes: list[tuple[int, int, int, int]],
    reference: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    ref_width = max(8, reference[2] - reference[0])
    ref_height = max(8, reference[3] - reference[1])
    for _ in range(80):
        crop_width = min(width, max(8, int(ref_width * rng.uniform(0.8, 1.2))))
        crop_height = min(height, max(8, int(ref_height * rng.uniform(0.8, 1.2))))
        x1 = rng.randint(0, max(0, width - crop_width))
        y1 = rng.randint(0, max(0, height - crop_height))
        candidate = (x1, y1, x1 + crop_width, y1 + crop_height)
        if all(iou(candidate, box) < 0.02 for box in positive_boxes):
            return candidate
    return (0, 0, min(width, ref_width), min(height, ref_height))


def collect_samples(dataset_root: Path, split: str, seed: int) -> list[tuple[Path, tuple[int, int, int, int], int]]:
    rng = random.Random(seed)
    samples = []
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    for image_path in sorted(image_root.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        image = read_image(image_path)
        height, width = image.shape[:2]
        boxes = parse_labels(label_root / f"{image_path.stem}.txt", width, height)
        for box in boxes:
            samples.append((image_path, padded_box(box, width, height), 1))
        for box in boxes:
            samples.append((image_path, negative_box(rng, width, height, boxes, padded_box(box, width, height)), 0))
    rng.shuffle(samples)
    return samples


def macro_f1(predictions: list[int], targets: list[int]) -> float:
    values = []
    for class_id in (0, 1):
        tp = sum(pred == class_id and target == class_id for pred, target in zip(predictions, targets))
        fp = sum(pred == class_id and target != class_id for pred, target in zip(predictions, targets))
        fn = sum(pred != class_id and target == class_id for pred, target in zip(predictions, targets))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        values.append(2 * precision * recall / max(1e-9, precision + recall))
    return float(sum(values) / len(values))


def classification_metrics(predictions: list[int], targets: list[int], probabilities: list[float]) -> dict:
    tp = sum(pred == 1 and target == 1 for pred, target in zip(predictions, targets))
    tn = sum(pred == 0 and target == 0 for pred, target in zip(predictions, targets))
    fp = sum(pred == 1 and target == 0 for pred, target in zip(predictions, targets))
    fn = sum(pred == 0 and target == 1 for pred, target in zip(predictions, targets))
    return {
        "samples": len(targets),
        "accuracy": (tp + tn) / max(1, len(targets)),
        "precision_positive": tp / max(1, tp + fp),
        "recall_positive": tp / max(1, tp + fn),
        "f1_positive": 2 * tp / max(1, 2 * tp + fp + fn),
        "macro_f1": macro_f1(predictions, targets),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "positive_probability_mean": float(np.mean(probabilities)) if probabilities else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.models import resnet18

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 1)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Restore the NVIDIA driver/CUDA runtime before starting GPU training."
        )
    device = torch.device(args.device)

    train_samples = collect_samples(args.dataset, "train", args.seed)
    val_samples = collect_samples(args.dataset, "val", args.seed + 1)
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    class CropDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, (x1, y1, x2, y2), target = self.samples[index]
            image = read_image(path)
            crop = image[y1:y2, x1:x2]
            return self.transform(Image.fromarray(crop)), target

    train_loader = DataLoader(CropDataset(train_samples, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_samples, eval_transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    history = []

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
        predictions, targets_all, probabilities = [], [], []
        with torch.inference_mode():
            for batch, targets in val_loader:
                probs = torch.softmax(model(batch.to(device)), dim=1)[:, 1]
                predictions.extend((probs >= 0.5).long().cpu().tolist())
                targets_all.extend(targets.tolist())
                probabilities.extend(probs.cpu().tolist())
        metrics = classification_metrics(predictions, targets_all, probabilities)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, len(train_samples)), **metrics}
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
                    "val_macro_f1": metrics["macro_f1"],
                },
                args.output / "classifier_best.pt",
            )
        print(f"epoch {epoch}/{args.epochs} loss={row['train_loss']:.5f} val_macro_f1={metrics['macro_f1']:.5f}", flush=True)

    final = history[-1]
    result = {
        "method": "engineering SAM3 candidate filter classifier trained on s1 crops",
        "sam3_checkpoint": str((Path(__file__).resolve().parents[2] / "models" / "sam3" / "sam3.pt").resolve()),
        "sam3_status": "frozen; SAM3 inference is not part of classifier-only training",
        "classifier_checkpoint": str((args.output / "classifier_best.pt").resolve()),
        "dataset": str(args.dataset.resolve()),
        "device": str(device),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "best_validation_metrics": max(history, key=lambda row: row["macro_f1"]),
        "final_validation_metrics": final,
        "history_file": str((args.output / "history.json").resolve()),
        "limitations": [
            "The production SAM3 segmenter requires CUDA and was not run on this host.",
            "The reported metrics are classifier-only; no SAM3 mask AP or end-to-end detection metrics are claimed.",
        ],
    }
    (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
