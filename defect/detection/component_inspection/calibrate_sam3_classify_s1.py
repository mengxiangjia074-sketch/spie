#!/usr/bin/env python3
"""Calibrate SAM3 and engineering-classifier thresholds on the s1 validation set."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DETECTION_ROOT = HERE.parent
WORKSPACE = DETECTION_ROOT.parent
YOLO_METRICS_DEFAULT = WORKSPACE.parent / "yolov8" / "yolov8_pcb" / "runs" / "fpic_evaluation_1000_es" / "metrics.json"
SAM_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
CLASSIFIER_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def low_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def load_model(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM3 threshold calibration")
    if str(DETECTION_ROOT) not in sys.path:
        sys.path.insert(0, str(DETECTION_ROOT))
    from sam3 import build_sam3_image_model
    from train_sam3_joint_s1 import build_classifier

    model = build_sam3_image_model(
        bpe_path=str(DETECTION_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        checkpoint_path=str(args.sam_checkpoint.resolve()),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )
    checkpoint = torch.load(args.joint_checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(checkpoint["sam3_state_dict"], strict=True)
    model.transformer.decoder.dac = False
    model.eval()
    classifier = build_classifier().cuda().eval()
    classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    return model, classifier


def collect_predictions(args, model, classifier):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import v2
    from sam3.model.data_misc import FindStage
    from train_sam3_joint_s1 import load_truth, read_rgb

    transform = transforms.Compose([
        transforms.Resize((96, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    preprocess = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5] * 3, [0.5] * 3),
    ])
    cache_path = args.output / "raw_validation_predictions.pkl"
    if cache_path.is_file():
        with cache_path.open("rb") as stream:
            return pickle.load(stream)
    records = []
    image_paths = sorted((args.dataset / "images" / "val").glob("*.jpg"))
    with torch.no_grad():
        for image_path in image_paths:
            image_rgb = read_rgb(image_path)
            height, width = image_rgb.shape[:2]
            truths = load_truth(args.dataset / "labels" / "val" / f"{image_path.stem}.txt", image_rgb.shape[:2])
            find = FindStage(
                img_ids=torch.tensor([0], device="cuda", dtype=torch.long),
                text_ids=torch.tensor([0], device="cuda", dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            image_tensor = preprocess(v2.functional.to_image(image_rgb).cuda()).unsqueeze(0)
            backbone_out = model.backbone.forward_image(image_tensor)
            backbone_out.update(model.backbone.forward_text(["component"], device="cuda"))
            out = model.forward_grounding(backbone_out, find, None, model._get_dummy_prompt())
            mask_low = (out["pred_masks"][0].float().sigmoid() > 0.5).cpu().numpy()
            sam_scores = (out["pred_logits"].float().sigmoid()[0, :, 0] * out["presence_logit_dec"].float().sigmoid().reshape(-1)[0]).cpu().numpy()
            boxes = out["pred_boxes"][0].float().cpu().numpy()
            order = np.argsort(sam_scores)[::-1][: args.max_queries]
            patches = []
            kept = []
            for query_index in order:
                score = float(sam_scores[query_index])
                cx, cy, bw, bh = boxes[query_index]
                x1 = int(max(0, (cx - bw / 2) * width))
                y1 = int(max(0, (cy - bh / 2) * height))
                x2 = int(min(width, (cx + bw / 2) * width))
                y2 = int(min(height, (cy + bh / 2) * height))
                if x2 <= x1 or y2 <= y1:
                    continue
                pad_x = max(2, int((x2 - x1) * 0.15))
                pad_y = max(2, int((y2 - y1) * 0.15))
                crop = image_rgb[max(0, y1 - pad_y) : min(height, y2 + pad_y), max(0, x1 - pad_x) : min(width, x2 + pad_x)]
                patches.append(transform(Image.fromarray(crop)))
                kept.append((query_index, score))
            if patches:
                positive_probabilities = torch.softmax(classifier(torch.stack(patches).cuda()), dim=1)[:, 1].cpu().numpy()
            else:
                positive_probabilities = np.zeros((0,), dtype=np.float32)
            predictions = [
                {
                    "sam_score": score,
                    "positive_probability": float(probability),
                    "mask_low": mask_low[query_index].astype(np.uint8),
                }
                for (query_index, score), probability in zip(kept, positive_probabilities)
            ]
            truth_low = [
                {"class_id": truth["class_id"], "mask": cv2.resize(truth["mask"].astype(np.uint8), (mask_low.shape[-1], mask_low.shape[-2]), interpolation=cv2.INTER_NEAREST).astype(bool)}
                for truth in truths
            ]
            records.append({"image": str(image_path), "shape": [height, width], "predictions": predictions, "truths_low": truth_low})
    args.output.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as stream:
        pickle.dump(records, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return records


def evaluate_threshold(records, sam_threshold: float, classifier_threshold: float):
    if str(DETECTION_ROOT) not in sys.path:
        sys.path.insert(0, str(DETECTION_ROOT))
    from train_sam3_classify_s1 import evaluate_predictions

    predictions_by_image = []
    truths_by_image = []
    for record in records:
        shape = tuple(record["truths_low"][0]["mask"].shape) if record["truths_low"] else (288, 288)
        predictions = []
        for prediction in record["predictions"]:
            if prediction["sam_score"] < sam_threshold or prediction["positive_probability"] < classifier_threshold:
                continue
            predictions.append({
                "confidence": prediction["sam_score"] * prediction["positive_probability"],
                "mask": prediction["mask_low"].astype(bool),
            })
        predictions_by_image.append(predictions)
        truths_by_image.append(record["truths_low"])
    return evaluate_predictions(predictions_by_image, truths_by_image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-metrics", type=Path, default=YOLO_METRICS_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, classifier = load_model(args)
    records = collect_predictions(args, model, classifier)
    candidates = []
    for sam_threshold in SAM_THRESHOLDS:
        for classifier_threshold in CLASSIFIER_THRESHOLDS:
            metrics = evaluate_threshold(records, sam_threshold, classifier_threshold)
            candidates.append({"sam_threshold": sam_threshold, "classifier_threshold": classifier_threshold, **metrics})
    best_instance = max(candidates, key=lambda item: (item["instance_f1"], item["mask_ap50"], item["boundary_f1"]))
    best_ap = max(candidates, key=lambda item: (item["mask_ap50"], item["instance_f1"]))
    best_boundary = max(candidates, key=lambda item: (item["boundary_f1"], item["instance_f1"]))
    baseline = next(item for item in candidates if item["sam_threshold"] == 0.25 and item["classifier_threshold"] == 0.50)
    result = {
        "model": str(args.joint_checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "grid": {"sam_thresholds": SAM_THRESHOLDS, "classifier_thresholds": CLASSIFIER_THRESHOLDS},
        "baseline": baseline,
        "recommended_by_instance_f1": best_instance,
        "best_by_mask_ap50": best_ap,
        "best_by_boundary_f1": best_boundary,
        "all_candidates": candidates,
    }
    (args.output / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "sam_confidence_threshold": best_instance["sam_threshold"],
        "classifier_threshold": best_instance["classifier_threshold"],
        "objective": "instance_f1, then mask_ap50, then boundary_f1",
        "checkpoint": str(args.joint_checkpoint.resolve()),
    }
    (args.output / "recommended_thresholds.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    yolo = json.loads(args.yolo_metrics.read_text(encoding="utf-8"))
    lines = [
        "# SAM3-Classify 验证集阈值校准报告", "",
        "校准目标：优先最大化实例 F1，其次 Mask AP50，再其次 Boundary F1。", "",
        "| 指标 | 校准前 | 推荐阈值后 | YOLOv8-Seg |", "|---|---:|---:|---:|",
    ]
    for label, key, yolo_key in [
        ("SAM3 confidence threshold", "sam_threshold", None),
        ("Classifier positive threshold", "classifier_threshold", None),
        ("Mask AP50", "mask_ap50", "mask_ap50"),
        ("Mask AP75", "mask_ap75", "mask_ap75"),
        ("Mask AP50:95", "mask_ap50_95", "mask_ap50_95"),
        ("Boundary F1", "boundary_f1", "boundary_f1"),
        ("Boundary IoU", "boundary_iou", "boundary_iou"),
        ("Instance F1", "instance_f1", "joint_f1"),
        ("Small component recall", "small_component_recall", "small_component_recall"),
        ("Center error mean (px)", "center_error_px_mean", "center_error_px_mean"),
    ]:
        before = baseline[key]
        after = best_instance[key]
        reference = "N/A" if yolo_key is None else f"{float(yolo[yolo_key]):.4f}"
        lines.append(f"| {label} | {before:.4f} | {after:.4f} | {reference} |")
    lines.extend([
        "", "## 推荐配置", "", "```json", json.dumps(config, ensure_ascii=False, indent=2), "```", "",
        "## 其他最优点", "",
        f"- Mask AP50 最优：SAM3={best_ap['sam_threshold']:.2f}, classifier={best_ap['classifier_threshold']:.2f}, AP50={best_ap['mask_ap50']:.4f}, Instance F1={best_ap['instance_f1']:.4f}",
        f"- Boundary F1 最优：SAM3={best_boundary['sam_threshold']:.2f}, classifier={best_boundary['classifier_threshold']:.2f}, Boundary F1={best_boundary['boundary_f1']:.4f}, Instance F1={best_boundary['instance_f1']:.4f}",
        "", "详细扫描结果保存在 `calibration.json`。", "",
    ])
    (args.output / "calibration.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
