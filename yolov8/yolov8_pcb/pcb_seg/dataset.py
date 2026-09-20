"""Build a tiled YOLO segmentation dataset from accepted SAM3 detections."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import Tile, clip_polygon_to_tile, generate_tiles, polygon_area


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def find_latest_sam_results(results_root: Path) -> Path:
    runs = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and list(path.glob("detections/*/detections.json"))
    ]
    if not runs:
        raise FileNotFoundError(f"no complete SAM3 result run found under {results_root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _load_annotations(result_root: Path, image_path: Path) -> list[list[tuple[float, float]]]:
    annotation_path = result_root / "detections" / image_path.stem / "detections.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"SAM3 annotation is missing for {image_path.name}: {annotation_path}"
        )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    polygons = []
    for detection in payload.get("accepted_detections", []):
        points = detection.get("segmentation") or detection.get("rotated_box_points")
        if points and len(points) >= 3:
            polygons.append([(float(x), float(y)) for x, y in points])
    return polygons


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def _format_label(points: list[tuple[float, float]], tile: Tile) -> str:
    values = ["0"]
    for x, y in points:
        values.extend(
            (
                f"{min(1.0, max(0.0, x / tile.width)):.6f}",
                f"{min(1.0, max(0.0, y / tile.height)):.6f}",
            )
        )
    return " ".join(values)


def prepare_dataset(
    source: Path,
    sam_results: Path,
    output: Path,
    tile_size: int = 1024,
    overlap: int = 256,
    validation_image: str | None = None,
    min_visible_ratio: float = 0.35,
    force: bool = False,
) -> dict:
    source = source.resolve()
    sam_results = sam_results.resolve()
    output = output.resolve()
    images = sorted(
        path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"no source images found under {source}")
    validation_name = validation_image or images[-1].name
    if validation_name not in {path.name for path in images}:
        raise ValueError(f"validation image not found in source: {validation_name}")

    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"dataset output is not empty: {output}; pass force=True to rebuild it")
        for child in output.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    counts = {"train_tiles": 0, "val_tiles": 0, "instances": 0}
    split_sources = {"train": [], "val": []}

    for image_path in images:
        image = cv2.imdecode(
            np.fromfile(str(image_path), dtype="uint8"), cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"could not read image: {image_path}")
        height, width = image.shape[:2]
        polygons = _load_annotations(sam_results, image_path)
        split = "val" if image_path.name == validation_name else "train"
        split_sources[split].append(image_path.name)
        for index, tile in enumerate(generate_tiles(width, height, tile_size, overlap)):
            labels = []
            for polygon in polygons:
                original_area = polygon_area(polygon)
                clipped = clip_polygon_to_tile(polygon, tile)
                visible_area = polygon_area(clipped)
                if len(clipped) < 3 or visible_area < 4.0:
                    continue
                if visible_area / max(original_area, 1e-9) < min_visible_ratio:
                    continue
                labels.append(_format_label(clipped, tile))

            stem = f"{image_path.stem}_tile_{index:03d}_{tile.x1}_{tile.y1}"
            image_out = output / "images" / split / f"{stem}.jpg"
            label_out = output / "labels" / split / f"{stem}.txt"
            crop = image[tile.y1 : tile.y2, tile.x1 : tile.x2]
            _write_image(image_out, crop)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            counts[f"{split}_tiles"] += 1
            counts["instances"] += len(labels)

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: component\n",
        encoding="utf-8",
    )
    manifest = {
        "source": str(source),
        "sam_results": str(sam_results),
        "tile_size": tile_size,
        "overlap": overlap,
        "min_visible_ratio": min_visible_ratio,
        "split_sources": split_sources,
        **counts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
