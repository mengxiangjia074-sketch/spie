"""Fine-tune torchvision Mask R-CNN on the FPIC S1 polygon dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(__file__).resolve().parent / "ground_truth" / "fpic_yolo"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "models" / "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "maskrcnn_s1_100"
CLASS_NAMES = ("resistors", "capacitors", "ICs", "inductors", "diodes")


class YoloPolygonDataset(Dataset):
    def __init__(self, root: Path, split: str, augment: bool) -> None:
        self.image_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        self.augment = augment
        self.images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if not self.images:
            raise FileNotFoundError(f"no images found in {self.image_dir}")
        missing = [path.name for path in self.images if not (self.label_dir / f"{path.stem}.txt").is_file()]
        if missing:
            raise FileNotFoundError(f"missing labels for {len(missing)} images; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        masks: list[np.ndarray] = []
        labels: list[int] = []

        label_path = self.label_dir / f"{image_path.stem}.txt"
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 7 or (len(fields) - 1) % 2:
                raise ValueError(f"invalid polygon at {label_path}:{line_number}")
            class_id = int(fields[0])
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"invalid class {class_id} at {label_path}:{line_number}")
            coordinates = [float(value) for value in fields[1:]]
            points = [
                (
                    min(width - 1, max(0, round(coordinates[i] * width))),
                    min(height - 1, max(0, round(coordinates[i + 1] * height))),
                )
                for i in range(0, len(coordinates), 2)
            ]
            mask_image = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask_image).polygon(points, outline=1, fill=1)
            mask = np.asarray(mask_image, dtype=np.uint8)
            if mask.any():
                masks.append(mask)
                labels.append(class_id + 1)

        image_tensor = pil_to_tensor(image).float().div_(255.0)
        if masks:
            mask_tensor = torch.from_numpy(np.stack(masks))
            label_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            mask_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            label_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.augment and random.random() < 0.5:
            image_tensor = image_tensor.flip(-1)
            mask_tensor = mask_tensor.flip(-1)
        if self.augment and random.random() < 0.2:
            image_tensor = image_tensor.flip(-2)
            mask_tensor = mask_tensor.flip(-2)

        if len(mask_tensor):
            boxes = masks_to_boxes(mask_tensor)
            keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
            boxes = boxes[keep]
            mask_tensor = mask_tensor[keep]
            label_tensor = label_tensor[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": label_tensor,
            "masks": mask_tensor,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image_tensor, target


def collate_batch(batch):
    return tuple(zip(*batch))


def move_targets(targets, device: torch.device):
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def build_model(weights_path: Path, num_classes: int):
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def run_train_epoch(model, loader, optimizer, scaler, device, amp: bool) -> float:
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {float(loss.detach())}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
    return total / len(loader)


@torch.inference_mode()
def run_validation_loss(model, loader, device, amp: bool) -> float:
    # Detection models return their loss dictionaries only while in training mode.
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        with torch.amp.autocast("cuda", enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite validation loss: {float(loss)}")
        total += float(loss)
    return total / len(loader)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"pretrained weights not found: {args.weights}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    train_set = YoloPolygonDataset(args.data, "train", augment=True)
    val_set = YoloPolygonDataset(args.data, "val", augment=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_options)

    model = build_model(args.weights, num_classes=len(CLASS_NAMES) + 1).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=4, min_lr=1e-6
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_val_loss = math.inf
    bad_epochs = 0
    history = []
    started_at = time.time()

    config = {
        **vars(args),
        "data": str(args.data.resolve()),
        "weights": str(args.weights.resolve()),
        "output": str(args.output.resolve()),
        "classes": CLASS_NAMES,
        "train_images": len(train_set),
        "val_images": len(val_set),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    write_json(args.output / "config.json", config)

    stop_reason = "max_epochs"
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = run_train_epoch(model, train_loader, optimizer, scaler, device, amp)
        val_loss = run_validation_loss(model, val_loader, device, amp)
        scheduler.step(val_loss)
        epoch_seconds = time.time() - epoch_started
        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "bad_epochs": bad_epochs,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "class_names": CLASS_NAMES,
            "config": config,
        }
        atomic_torch_save(checkpoint, args.output / "last.pth")
        if improved:
            atomic_torch_save(checkpoint, args.output / "best.pth")
        write_json(args.output / "history.json", history)
        write_json(
            args.output / "status.json",
            {
                **record,
                "status": "running",
                "elapsed_seconds": time.time() - started_at,
                "estimated_remaining_seconds": epoch_seconds * (args.epochs - epoch),
            },
        )
        print(json.dumps(record), flush=True)

        if bad_epochs >= args.patience:
            stop_reason = "early_stopping"
            break

    write_json(
        args.output / "status.json",
        {
            "status": "complete",
            "stop_reason": stop_reason,
            "epochs_completed": history[-1]["epoch"],
            "best_val_loss": best_val_loss,
            "elapsed_seconds": time.time() - started_at,
        },
    )


if __name__ == "__main__":
    main()
