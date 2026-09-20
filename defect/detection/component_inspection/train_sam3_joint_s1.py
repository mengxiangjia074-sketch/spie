#!/usr/bin/env python3
"""Full SAM3 + engineering classifier fine-tuning on the prepared s1 split.

The trimmed local SAM3 checkout does not include its original training matcher.
This script therefore calls SAM3's differentiable grounding forward directly and
uses a deterministic mask/box/objectness loss against the s1 polygons. The
engineering ResNet18 head is trained alongside it on positive component crops
and sampled background crops.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
WORKSPACE = DETECTION_ROOT.parent
SAM3_DEFAULT = WORKSPACE / "models" / "sam3" / "sam3.pt"
YOLO_METRICS_DEFAULT = WORKSPACE.parent / "yolov8" / "yolov8_pcb" / "runs" / "fpic_evaluation_1000_es" / "metrics.json"
IMAGE_SIZE = (96, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def local_imports():
    sys.path.insert(0, str(DETECTION_ROOT))
    from component_inspection.config import InspectionConfig
    from component_inspection.segmentation import (
        extract_rectified_patch,
    )
    from sam3.model.data_misc import FindStage

    return InspectionConfig, FindStage, extract_rectified_patch


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
        mask = mask_from_points(points, shape)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        truths.append(
            {
                "class_id": int(values[0]),
                "mask": mask,
                "box": (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            }
        )
    return truths


def build_classifier():
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def make_dummy_target(device):
    empty_boxes = __import__("torch").zeros((0, 4), device=device)
    return SimpleNamespace(
        boxes=empty_boxes,
        boxes_padded=__import__("torch").zeros((1, 0, 4), device=device),
        num_boxes=__import__("torch").zeros((1,), dtype=__import__("torch").long, device=device),
        segments=None,
        semantic_segments=None,
        is_valid_segment=None,
        is_exhaustive=__import__("torch").zeros((1,), dtype=__import__("torch").bool, device=device),
        object_ids=__import__("torch").zeros((0,), dtype=__import__("torch").long, device=device),
        object_ids_padded=__import__("torch").zeros((1, 0), dtype=__import__("torch").long, device=device),
    )


def prepare_sam_input(model, image_rgb: np.ndarray, find_stage, device):
    import torch
    from torchvision.transforms import v2

    image = v2.functional.to_image(image_rgb).to(device)
    transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(1008, 1008)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    backbone_out = model.backbone.forward_image(transform(image).unsqueeze(0))
    backbone_out.update(model.backbone.forward_text(["component"], device=device))
    prompt = model._get_dummy_prompt()
    return backbone_out, find_stage, prompt


def normalized_boxes(truths: list[dict], shape: tuple[int, int], device):
    import torch

    height, width = shape
    values = []
    for truth in truths:
        x1, y1, x2, y2 = truth["box"]
        values.append(
            [
                ((x1 + x2) / 2) / width,
                ((y1 + y2) / 2) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
        )
    return torch.tensor(values, dtype=torch.float32, device=device) if values else torch.zeros((0, 4), device=device)


def match_queries(pred_logits, truths_small):
    """Greedily assign distinct SAM3 queries to truth masks without gradients."""
    import torch
    import torch.nn.functional as F

    if not len(truths_small):
        return []
    probabilities = pred_logits.sigmoid().detach()
    truth_tensor = torch.stack([torch.from_numpy(mask.astype(np.float32)) for mask in truths_small]).to(pred_logits.device)
    truth_tensor = truth_tensor.unsqueeze(1)
    truth_tensor = F.interpolate(truth_tensor, size=probabilities.shape[-2:], mode="nearest").squeeze(1) > 0.5
    pred_binary = probabilities > 0.5
    intersections = (pred_binary[:, None] & truth_tensor[None]).flatten(2).sum(-1).float()
    unions = (pred_binary[:, None] | truth_tensor[None]).flatten(2).sum(-1).float().clamp_min(1.0)
    ious = intersections / unions
    assignments = []
    used = set()
    for truth_index in range(len(truths_small)):
        order = torch.argsort(ious[:, truth_index], descending=True).tolist()
        query_index = next((value for value in order if value not in used), order[0])
        used.add(query_index)
        assignments.append((query_index, truth_index))
    return assignments


def sam_loss(out: dict, truths: list[dict], shape: tuple[int, int], device):
    import torch
    import torch.nn.functional as F

    pred_masks = out["pred_masks"][0]
    pred_logits = out["pred_logits"][0, :, 0]
    pred_boxes = out["pred_boxes"][0]
    truth_masks = [truth["mask"] for truth in truths]
    assignments = match_queries(pred_masks, truth_masks)
    object_targets = torch.zeros_like(pred_logits)
    mask_loss = pred_masks.sum() * 0.0
    box_loss = pred_boxes.sum() * 0.0
    for query_index, truth_index in assignments:
        truth_small = torch.from_numpy(truth_masks[truth_index].astype(np.float32)).to(device)
        truth_small = F.interpolate(truth_small[None, None], size=pred_masks.shape[-2:], mode="nearest").squeeze()
        logits = pred_masks[query_index]
        mask_loss = mask_loss + F.binary_cross_entropy_with_logits(logits, truth_small)
        probability = logits.sigmoid()
        dice = 1.0 - (2.0 * (probability * truth_small).sum() + 1.0) / (probability.sum() + truth_small.sum() + 1.0)
        mask_loss = mask_loss + dice
        object_targets[query_index] = 1.0
    if assignments:
        box_targets = normalized_boxes(truths, shape, device)
        box_loss = F.l1_loss(pred_boxes[[q for q, _ in assignments]], box_targets[[t for _, t in assignments]])
        mask_loss = mask_loss / len(assignments)
    positive_count = int(object_targets.sum().detach().cpu())
    if positive_count:
        negative_count = max(1, int(object_targets.numel()) - positive_count)
        positive_weight = torch.tensor(negative_count / positive_count, device=device)
        object_loss = F.binary_cross_entropy_with_logits(
            pred_logits, object_targets, pos_weight=positive_weight
        )
    else:
        object_loss = F.binary_cross_entropy_with_logits(pred_logits, object_targets)
    presence_target = torch.tensor([1.0 if truths else 0.0], device=device)
    presence_loss = F.binary_cross_entropy_with_logits(out["presence_logit_dec"].reshape(-1), presence_target)
    total = mask_loss + 0.5 * object_loss + 0.25 * box_loss + 0.5 * presence_loss
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()),
        "presence_loss": float(presence_loss.detach().cpu()),
        "matched_queries": len(assignments),
    }


def crop_patch(image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = max(2, int((x2 - x1) * 0.15))
    pad_y = max(2, int((y2 - y1) * 0.15))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)
    return image_rgb[y1:y2, x1:x2]


def classifier_batch(image_rgb, truths, rng, device):
    import torch
    from PIL import Image
    from torchvision import transforms

    samples = []
    for truth in truths:
        samples.append((crop_patch(image_rgb, truth["box"]), 1))
    boxes = [truth["box"] for truth in truths]
    height, width = image_rgb.shape[:2]
    for truth in truths:
        x1, y1, x2, y2 = truth["box"]
        crop_width, crop_height = max(8, x2 - x1), max(8, y2 - y1)
        for _ in range(30):
            nx1 = rng.randint(0, max(0, width - crop_width))
            ny1 = rng.randint(0, max(0, height - crop_height))
            candidate = (nx1, ny1, nx1 + crop_width, ny1 + crop_height)
            if all(mask_iou_box(candidate, other) < 0.02 for other in boxes):
                samples.append((crop_patch(image_rgb, candidate), 0))
                break
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tensors = torch.stack([transform(Image.fromarray(patch.astype(np.uint8))) for patch, _ in samples]).to(device)
    targets = torch.tensor([target for _, target in samples], dtype=torch.long, device=device)
    return tensors, targets


def mask_iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)


def train(args):
    import torch
    import torch.nn.functional as F

    if str(DETECTION_ROOT) not in sys.path:
        sys.path.insert(0, str(DETECTION_ROOT))
    from sam3 import build_sam3_image_model
    from torch.amp import autocast

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full SAM3 fine-tuning")
    device = torch.device(args.device)
    InspectionConfig, FindStage, _ = local_imports()
    from train_sam3_classify_s1 import evaluate_predictions

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model(
        bpe_path=str(DETECTION_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        checkpoint_path=str(args.sam_checkpoint.resolve()),
        load_from_HF=False,
        device=str(device),
        eval_mode=True,
    )
    # Keep all SAM3 parameters trainable. Eval mode avoids the unavailable
    # training-only matcher; autograd remains enabled for the direct forward.
    model.requires_grad_(True)
    model.transformer.decoder.dac = False
    model._compute_matching = lambda out, targets: None
    classifier = build_classifier().to(device)
    if args.initial_classifier and args.initial_classifier.is_file():
        checkpoint = torch.load(args.initial_classifier, map_location=device, weights_only=True)
        classifier.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.sam_lr},
            {"params": classifier.parameters(), "lr": args.classifier_lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    _, _, _, = local_imports()
    train_images = sorted((args.dataset / "images" / "train").glob("*.jpg"))
    if args.max_train_images:
        train_images = train_images[: args.max_train_images]
    rng = random.Random(args.seed)
    train_history = []
    dummy_target = make_dummy_target(device)
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_images)
        model.eval()
        classifier.train()
        epoch_loss = []
        for index, image_path in enumerate(train_images, start=1):
            image_rgb = read_rgb(image_path)
            truths = load_truth(args.dataset / "labels" / "train" / f"{image_path.stem}.txt", image_rgb.shape[:2])
            img_ids = torch.tensor([0], device=device, dtype=torch.long)
            text_ids = torch.tensor([0], device=device, dtype=torch.long)
            find_stage = FindStage(img_ids=img_ids, text_ids=text_ids, input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                backbone_out, find_stage, prompt = prepare_sam_input(model, image_rgb, find_stage, device)
                out = model.forward_grounding(backbone_out, find_stage, dummy_target, prompt)
                loss, details = sam_loss(out, truths, image_rgb.shape[:2], device)
                if truths:
                    classifier_inputs, classifier_targets = classifier_batch(image_rgb, truths, rng, device)
                    classifier_loss = F.cross_entropy(classifier(classifier_inputs), classifier_targets)
                    loss = loss + args.classifier_weight * classifier_loss
                    details["classifier_loss"] = float(classifier_loss.detach().cpu())
                else:
                    details["classifier_loss"] = 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss.append(float(loss.detach().cpu()))
            if index % args.status_interval == 0:
                print(f"epoch {epoch}/{args.epochs} image {index}/{len(train_images)} loss={epoch_loss[-1]:.4f}", flush=True)
        row = {"epoch": epoch, "loss_mean": float(np.mean(epoch_loss)), "loss_last": epoch_loss[-1] if epoch_loss else 0.0}
        train_history.append(row)
        (output / "history.json").write_text(json.dumps(train_history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)

    checkpoint_path = output / "sam3_classify_joint_best.pt"
    torch.save(
        {
            "sam3_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "class_names": ["negative", "positive"],
            "image_size": list(IMAGE_SIZE),
            "epochs": args.epochs,
            "sam_lr": args.sam_lr,
            "classifier_lr": args.classifier_lr,
        },
        checkpoint_path,
    )
    model.eval()
    classifier.eval()
    val_images = sorted((args.dataset / "images" / "val").glob("*.jpg"))
    predictions_by_image, truths_by_image = [], []
    from torchvision.transforms import v2
    preprocess = v2.Compose([v2.ToDtype(torch.uint8, scale=True), v2.Resize((1008, 1008)), v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)])
    transform = __import__("torchvision").transforms.Compose([__import__("torchvision").transforms.Resize(IMAGE_SIZE), __import__("torchvision").transforms.ToTensor(), __import__("torchvision").transforms.Normalize(MEAN, STD)])
    from PIL import Image
    with torch.no_grad():
        for image_path in val_images:
            image_rgb = read_rgb(image_path)
            truths = load_truth(args.dataset / "labels" / "val" / f"{image_path.stem}.txt", image_rgb.shape[:2])
            img_ids = torch.tensor([0], device=device, dtype=torch.long)
            find_stage = FindStage(img_ids=img_ids, text_ids=torch.tensor([0], device=device, dtype=torch.long), input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
            image_tensor = preprocess(v2.functional.to_image(image_rgb).to(device)).unsqueeze(0)
            backbone_out = model.backbone.forward_image(image_tensor)
            backbone_out.update(model.backbone.forward_text(["component"], device=device))
            out = model.forward_grounding(backbone_out, find_stage, None, model._get_dummy_prompt())
            masks = F.interpolate(out["pred_masks"].float().sigmoid(), size=image_rgb.shape[:2], mode="bilinear", align_corners=False)[0]
            scores = (out["pred_logits"].float().sigmoid()[0, :, 0] * out["presence_logit_dec"].float().sigmoid().reshape(-1)[0]).cpu().numpy()
            boxes = out["pred_boxes"][0].float().cpu().numpy()
            predictions = []
            height, width = image_rgb.shape[:2]
            for query_index, score in enumerate(scores):
                if score < args.sam_threshold:
                    continue
                cx, cy, bw, bh = boxes[query_index]
                x1, y1 = int(max(0, (cx - bw / 2) * width)), int(max(0, (cy - bh / 2) * height))
                x2, y2 = int(min(width, (cx + bw / 2) * width)), int(min(height, (cy + bh / 2) * height))
                if x2 <= x1 or y2 <= y1:
                    continue
                patch = transform(Image.fromarray(crop_patch(image_rgb, (x1, y1, x2, y2)))).unsqueeze(0).to(device)
                positive_probability = float(torch.softmax(classifier(patch), dim=1)[0, 1].cpu())
                if positive_probability < args.classifier_threshold:
                    continue
                predictions.append({"confidence": float(score * positive_probability), "mask": masks[query_index].cpu().numpy() > 0.5})
            predictions_by_image.append(predictions)
            truths_by_image.append(truths)
    evaluation = evaluate_predictions(predictions_by_image, truths_by_image)
    result = {
        "method": "joint unfrozen SAM3 + engineering ResNet18 classifier",
        "device": str(device),
        "dataset": str(args.dataset.resolve()),
        "sam3_checkpoint": str(args.sam_checkpoint.resolve()),
        "joint_checkpoint": str(checkpoint_path.resolve()),
        "train_images": len(train_images),
        "validation_images": len(val_images),
        "epochs": args.epochs,
        "sam_lr": args.sam_lr,
        "classifier_lr": args.classifier_lr,
        "evaluation": evaluation,
        "yolo_reference_metrics": str(args.yolo_metrics.resolve()),
        "limitations": [
            "SAM3 is fully trainable; the local official matcher is unavailable, so a direct mask/box/objectness loss is used.",
            "The engineering classifier remains binary; class-aware five-class F1 is not reported for this model.",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    yolo = json.loads(args.yolo_metrics.read_text(encoding="utf-8"))
    rows = [
        ("Mask AP50", evaluation["mask_ap50"], yolo.get("mask_ap50")),
        ("Mask AP75", evaluation["mask_ap75"], yolo.get("mask_ap75")),
        ("Mask AP50:95", evaluation["mask_ap50_95"], yolo.get("mask_ap50_95")),
        ("Boundary F1", evaluation["boundary_f1"], yolo.get("boundary_f1")),
        ("Boundary IoU", evaluation["boundary_iou"], yolo.get("boundary_iou")),
        ("Instance F1 / Joint F1", evaluation["instance_f1"], yolo.get("joint_f1")),
        ("Small component recall", evaluation["small_component_recall"], yolo.get("small_component_recall")),
        ("Center error mean (px)", evaluation["center_error_px_mean"], yolo.get("center_error_px_mean")),
    ]
    lines = [
        "# Joint Unfrozen SAM3-Classify 与 YOLOv8-Seg 对比", "",
        "- Joint Unfrozen SAM3-Classify：全量解冻 SAM3，联合训练工程 ResNet18 二分类头。",
        "- YOLOv8-Seg：使用已有 s1 `fpic_evaluation_1000_es/metrics.json`。",
        "- 两者使用相同验证切分与 IoU=0.50 匹配定义；SAM3-Classify 按所有元件统一类别统计。", "",
        "| 指标 | Joint SAM3-Classify | YOLOv8-Seg | 差值 |", "|---|---:|---:|---:|",
    ]
    for name, value, reference in rows:
        lines.append(f"| {name} | {value:.4f} | {float(reference):.4f} | {value - float(reference):+.4f} |")
    lines.extend(["", "## 详细结果", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""])
    (output / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM3_DEFAULT)
    parser.add_argument("--yolo-metrics", type=Path, default=YOLO_METRICS_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam-lr", type=float, default=1e-7)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-weight", type=float, default=0.5)
    parser.add_argument("--sam-threshold", type=float, default=0.25)
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--initial-classifier", type=Path)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--status-interval", type=int, default=10)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
