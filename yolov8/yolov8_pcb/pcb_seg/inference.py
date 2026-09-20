"""Tiled YOLOv8-Seg inference and result export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Tile, box_iou, generate_tiles, polygon_area
from .training import _yolo_class


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_image(path: Path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    return image


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def _touches_internal_edge(points: np.ndarray, tile: Tile, image_size, margin: int) -> bool:
    width, height = image_size
    return bool(
        (tile.x1 > 0 and points[:, 0].min() <= margin)
        or (tile.y1 > 0 and points[:, 1].min() <= margin)
        or (tile.x2 < width and points[:, 0].max() >= tile.width - margin)
        or (tile.y2 < height and points[:, 1].max() >= tile.height - margin)
    )


def _deduplicate(detections: list[dict], iou_threshold: float) -> list[dict]:
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for detection in ordered:
        if any(box_iou(detection["bbox_xyxy"], other["bbox_xyxy"]) >= iou_threshold for other in kept):
            continue
        kept.append(detection)
    kept.sort(key=lambda item: (item["center_xy"][1], item["center_xy"][0]))
    for index, detection in enumerate(kept, start=1):
        detection["id"] = f"component_{index:04d}"
    return kept


def _render(image, detections: list[dict]):
    canvas = image.copy()
    mask_layer = image.copy()
    palette = ((40, 190, 70), (40, 150, 230), (210, 120, 45), (180, 70, 190))
    for index, detection in enumerate(detections):
        points = np.rint(detection["segmentation"]).astype(np.int32)
        color = palette[index % len(palette)]
        cv2.fillPoly(mask_layer, [points.reshape(-1, 1, 2)], color)
    canvas = cv2.addWeighted(canvas, 0.68, mask_layer, 0.32, 0)
    for index, detection in enumerate(detections):
        points = np.rint(detection["segmentation"]).astype(np.int32)
        color = palette[index % len(palette)]
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        x1, y1, _, _ = [int(round(value)) for value in detection["bbox_xyxy"]]
        label = f"{detection['id']} {detection['confidence']:.2f}"
        cv2.putText(
            canvas,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def predict_images(
    weights: Path,
    source: Path,
    output: Path,
    tile_size: int = 1024,
    overlap: int = 256,
    confidence: float = 0.25,
    model_iou: float = 0.7,
    merge_iou: float = 0.5,
    device: str = "0",
    batch: int = 4,
    edge_margin: int = 4,
) -> dict:
    if not weights.is_file():
        raise FileNotFoundError(f"trained YOLOv8-Seg weights not found: {weights}")
    paths = [source] if source.is_file() else sorted(
        path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no input images found at {source}")
    model = _yolo_class()(str(weights.resolve()))
    summary = {"weights": str(weights.resolve()), "images": []}

    for image_path in paths:
        image = _read_image(image_path)
        height, width = image.shape[:2]
        tiles = generate_tiles(width, height, tile_size, overlap)
        crops = [image[tile.y1 : tile.y2, tile.x1 : tile.x2] for tile in tiles]
        results = model.predict(
            source=crops,
            task="segment",
            imgsz=tile_size,
            conf=confidence,
            iou=model_iou,
            device=device,
            batch=batch,
            retina_masks=True,
            verbose=False,
        )
        detections = []
        for tile, result in zip(tiles, results):
            if result.boxes is None or result.masks is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id, points in zip(boxes, scores, classes, result.masks.xy):
                local_points = np.asarray(points, dtype=np.float32)
                if len(local_points) < 3 or _touches_internal_edge(
                    local_points, tile, (width, height), edge_margin
                ):
                    continue
                global_points = local_points + np.asarray([tile.x1, tile.y1], dtype=np.float32)
                global_box = np.asarray(box, dtype=np.float32) + np.asarray(
                    [tile.x1, tile.y1, tile.x1, tile.y1], dtype=np.float32
                )
                detections.append(
                    {
                        "id": "",
                        "class_id": int(class_id),
                        "class_name": str(result.names[int(class_id)]),
                        "confidence": float(score),
                        "bbox_xyxy": global_box.tolist(),
                        "center_xy": [
                            float((global_box[0] + global_box[2]) / 2.0),
                            float((global_box[1] + global_box[3]) / 2.0),
                        ],
                        "mask_area_px": float(polygon_area(global_points.tolist())),
                        "segmentation": global_points.tolist(),
                    }
                )
        detections = _deduplicate(detections, merge_iou)
        image_output = output / image_path.stem
        overlay_path = image_output / "yolo_mask_overlay.png"
        json_path = image_output / "detections.json"
        _write_image(overlay_path, _render(image, detections))
        payload = {
            "image_name": image_path.name,
            "detection_count": len(detections),
            "inference": {
                "tile_size": tile_size,
                "overlap": overlap,
                "confidence_threshold": confidence,
                "model_iou_threshold": model_iou,
                "merge_iou_threshold": merge_iou,
            },
            "outputs": {
                "overlay": str(overlay_path.resolve()),
                "detections_json": str(json_path.resolve()),
            },
            "detections": detections,
        }
        image_output.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary["images"].append(
            {"image": image_path.name, "detections": len(detections), "output": str(image_output)}
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
