#!/usr/bin/env python3
"""Run tiled Mask R-CNN segmentation and export inspection-style overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pcb_seg.geometry import box_iou, generate_tiles
from train_maskrcnn_s1 import CLASS_NAMES, DEFAULT_WEIGHTS, build_model


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PALETTE = (
    (40, 190, 70),
    (40, 150, 230),
    (210, 120, 45),
    (180, 70, 190),
    (60, 60, 220),
)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def touches_internal_edge(box, tile, image_size, margin: int) -> bool:
    width, height = image_size
    x1, y1, x2, y2 = box
    return bool(
        (tile.x1 > 0 and x1 <= margin)
        or (tile.y1 > 0 and y1 <= margin)
        or (tile.x2 < width and x2 >= tile.width - margin)
        or (tile.y2 < height and y2 >= tile.height - margin)
    )


def deduplicate(detections: list[dict], threshold: float) -> list[dict]:
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for detection in ordered:
        if any(
            box_iou(detection["bbox_xyxy"], previous["bbox_xyxy"]) >= threshold
            for previous in kept
        ):
            continue
        kept.append(detection)
    kept.sort(key=lambda item: (item["center_xy"][1], item["center_xy"][0]))
    for index, detection in enumerate(kept, 1):
        detection["id"] = f"component_{index:04d}"
    return kept


def render(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image.copy()
    mask_layer = image.copy()
    for detection in detections:
        points = np.rint(detection["segmentation"]).astype(np.int32)
        color = PALETTE[detection["class_id"] % len(PALETTE)]
        cv2.fillPoly(mask_layer, [points.reshape(-1, 1, 2)], color)
    canvas = cv2.addWeighted(canvas, 0.68, mask_layer, 0.32, 0)
    for detection in detections:
        points = np.rint(detection["segmentation"]).astype(np.int32)
        color = PALETTE[detection["class_id"] % len(PALETTE)]
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        x1, y1, _, _ = map(lambda value: int(round(value)), detection["bbox_xyxy"])
        label = (
            f"{detection['id']} {detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def predict(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(args.pretrained_weights, len(CLASS_NAMES) + 1)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.detections_per_img = 100
    model.to(device).eval()

    image_paths = sorted(
        path
        for path in args.source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found in {args.source}")
    summary = {
        "model": "Mask R-CNN ResNet-50 FPN v2",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "images": [],
    }
    with torch.inference_mode():
        for image_path in image_paths:
            image = read_image(image_path)
            height, width = image.shape[:2]
            tiles = generate_tiles(width, height, args.tile_size, args.overlap)
            detections = []
            for start in range(0, len(tiles), args.batch_size):
                tile_batch = tiles[start : start + args.batch_size]
                tensors = []
                for tile in tile_batch:
                    crop = image[tile.y1 : tile.y2, tile.x1 : tile.x2]
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    tensors.append(
                        torch.from_numpy(rgb.copy())
                        .permute(2, 0, 1)
                        .float()
                        .div_(255.0)
                        .to(device)
                    )
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    outputs = model(tensors)
                for tile, output in zip(tile_batch, outputs):
                    scores = output["scores"].detach().cpu().numpy()
                    labels = output["labels"].detach().cpu().numpy().astype(int)
                    boxes = output["boxes"].detach().cpu().numpy()
                    masks = (output["masks"][:, 0] >= 0.5).detach().cpu().numpy()
                    for score, label, box, mask in zip(scores, labels, boxes, masks):
                        if score < args.confidence:
                            continue
                        mask_area = int(np.count_nonzero(mask))
                        maximum_area = (
                            args.max_ic_mask_area if int(label) == 3
                            else args.max_small_mask_area
                        )
                        if mask_area > maximum_area:
                            continue
                        if touches_internal_edge(
                            box, tile, (width, height), args.edge_margin
                        ):
                            continue
                        contours, _ = cv2.findContours(
                            mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if not contours:
                            continue
                        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        if len(contour) < 3:
                            continue
                        global_points = contour.astype(np.float32) + np.asarray(
                            [tile.x1, tile.y1], dtype=np.float32
                        )
                        global_box = box.astype(np.float32) + np.asarray(
                            [tile.x1, tile.y1, tile.x1, tile.y1], dtype=np.float32
                        )
                        class_id = int(label) - 1
                        detections.append(
                            {
                                "id": "",
                                "class_id": class_id,
                                "class_name": CLASS_NAMES[class_id],
                                "confidence": float(score),
                                "bbox_xyxy": global_box.tolist(),
                                "center_xy": [
                                    float((global_box[0] + global_box[2]) / 2),
                                    float((global_box[1] + global_box[3]) / 2),
                                ],
                                "mask_area_px": mask_area,
                                "segmentation": global_points.tolist(),
                            }
                        )
            raw_count = len(detections)
            detections = deduplicate(detections, args.merge_iou)
            image_output = args.output / image_path.stem
            overlay_path = image_output / "maskrcnn_mask_overlay.png"
            json_path = image_output / "detections.json"
            write_image(overlay_path, render(image, detections))
            payload = {
                "image_name": image_path.name,
                "candidate_count": raw_count,
                "accepted_count": len(detections),
                "inference": {
                    "tile_size": args.tile_size,
                    "overlap": args.overlap,
                    "confidence_threshold": args.confidence,
                    "merge_iou_threshold": args.merge_iou,
                    "max_small_mask_area_px": args.max_small_mask_area,
                    "max_ic_mask_area_px": args.max_ic_mask_area,
                },
                "outputs": {
                    "mask_overlay": str(overlay_path.resolve()),
                    "detections_json": str(json_path.resolve()),
                },
                "detections": detections,
            }
            image_output.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary["images"].append(
                {
                    "image": image_path.name,
                    "detections": len(detections),
                    "output": str(image_output.resolve()),
                }
            )
            print(f"{image_path.name}: detections={len(detections)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--edge-margin", type=int, default=4)
    parser.add_argument("--max-small-mask-area", type=int, default=20000)
    parser.add_argument("--max-ic-mask-area", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(predict(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
