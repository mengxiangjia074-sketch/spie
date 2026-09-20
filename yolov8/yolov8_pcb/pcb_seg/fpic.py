"""Convert the local FPIC CSV annotations into a tiled YOLOv8-Seg dataset."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .geometry import Tile, clip_polygon_to_tile, generate_tiles, polygon_area


FPIC_CLASSES = ["resistors", "capacitors", "ICs", "inductors", "diodes"]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _parse_shape(value: str) -> tuple[str, list[tuple[float, float]]]:
    try:
        shape = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FPIC component_location: {value!r}") from exc
    shape_type = shape.get("name")
    if shape_type == "rect":
        try:
            x = float(shape["x"])
            y = float(shape["y"])
            width = float(shape["width"])
            height = float(shape["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid FPIC rectangle: {value!r}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"FPIC rectangle must have positive size: {value!r}")
        return shape_type, [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ]
    if shape_type == "polygon":
        xs = shape.get("all_points_x") or []
        ys = shape.get("all_points_y") or []
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError(f"invalid FPIC polygon: {value!r}")
        return shape_type, [(float(x), float(y)) for x, y in zip(xs, ys)]
    raise ValueError(f"unsupported FPIC shape type {shape_type!r}: {value!r}")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"could not encode image: {path}")
    encoded.tofile(str(path))


def _find_image(root: Path, camera: str, image_name: str) -> Path | None:
    if camera == "DSLR":
        candidate = root / "DSLR" / "img" / image_name
        return candidate if candidate.is_file() else None
    matches = list((root / "Microscope" / "img" / "front").rglob(image_name))
    return matches[0] if matches else None


def _collect_groups(source: Path) -> list[dict]:
    groups = []
    for annotation in sorted((source / "DSLR" / "annotation").glob("*.csv")):
        groups.append({"id": f"DSLR_{annotation.stem}", "camera": "DSLR", "annotation": annotation})
    for annotation in sorted((source / "Microscope" / "annotation" / "front").glob("*.csv")):
        groups.append({"id": f"Microscope_{annotation.stem}", "camera": "Microscope", "annotation": annotation})
    if not groups:
        raise FileNotFoundError(f"no FPIC annotation CSV files found under {source}")
    return groups


def _read_group(source: Path, group: dict) -> dict:
    records: dict[str, list[dict]] = defaultdict(list)
    with group["annotation"].open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_name = (row.get("image_name") or "").strip()
            class_name = (row.get("component_type") or "").strip()
            if not image_name or class_name not in FPIC_CLASSES:
                raise ValueError(f"unsupported FPIC row in {group['annotation']}: {row}")
            image_path = _find_image(source, group["camera"], image_name)
            if image_path is None:
                raise FileNotFoundError(
                    f"FPIC image {image_name!r} referenced by {group['annotation']} was not found"
                )
            shape_type, polygon = _parse_shape(row["component_location"])
            records[str(image_path.resolve())].append(
                {
                    "class_id": FPIC_CLASSES.index(class_name),
                    "shape_type": shape_type,
                    "polygon": polygon,
                }
            )
    return {**group, "records": records}


def _label_line(class_id: int, points: list[tuple[float, float]], tile: Tile) -> str:
    values = [str(class_id)]
    for x, y in points:
        values.extend(
            (
                f"{min(1.0, max(0.0, x / tile.width)):.6f}",
                f"{min(1.0, max(0.0, y / tile.height)):.6f}",
            )
        )
    return " ".join(values)


def _clear_output(output: Path, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"output is not empty: {output}; pass --force to rebuild it")
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def prepare_fpic_dataset(
    source: Path,
    output: Path,
    validation_group: str | None = None,
    tile_size: int = 1024,
    overlap: int = 256,
    min_visible_ratio: float = 0.35,
    include_dslr: bool = True,
    force: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    groups = _collect_groups(source)
    if not include_dslr:
        groups = [group for group in groups if group["camera"] != "DSLR"]
    if not groups:
        raise ValueError("no FPIC groups remain after applying camera filters")
    if validation_group is None:
        microscope_groups = [group for group in groups if group["camera"] == "Microscope"]
        validation_group = microscope_groups[-1]["id"] if microscope_groups else groups[-1]["id"]
    valid_group_ids = {group["id"] for group in groups}
    if validation_group not in valid_group_ids:
        raise ValueError(f"unknown validation group {validation_group!r}; choose one of {sorted(valid_group_ids)}")

    loaded_groups = [_read_group(source, group) for group in groups]
    _clear_output(output, force)
    counts = {"train_tiles": 0, "val_tiles": 0, "instances": 0}
    split_groups = {"train": [], "val": []}
    unlabeled_tiles = 0
    annotation_shapes = {"rect": 0, "polygon": 0}

    for group in loaded_groups:
        split = "val" if group["id"] == validation_group else "train"
        split_groups[split].append(group["id"])
        for image_name, records in group["records"].items():
            for record in records:
                annotation_shapes[record["shape_type"]] += 1
            image_path = Path(image_name)
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not read FPIC image: {image_path}")
            height, width = image.shape[:2]
            for tile_index, tile in enumerate(generate_tiles(width, height, tile_size, overlap)):
                labels = []
                for record in records:
                    original_area = polygon_area(record["polygon"])
                    clipped = clip_polygon_to_tile(record["polygon"], tile)
                    visible_area = polygon_area(clipped)
                    if len(clipped) < 3 or visible_area < 4.0:
                        continue
                    if visible_area / max(original_area, 1e-9) < min_visible_ratio:
                        continue
                    labels.append(_label_line(record["class_id"], clipped, tile))
                if not labels:
                    unlabeled_tiles += 1
                stem = (
                    f"fpic_{_safe_name(group['id'])}_{_safe_name(image_path.stem)}"
                    f"_tile_{tile_index:03d}_{tile.x1}_{tile.y1}"
                )
                image_out = output / "images" / split / f"{stem}.jpg"
                label_out = output / "labels" / split / f"{stem}.txt"
                _write_image(image_out, image[tile.y1 : tile.y2, tile.x1 : tile.x2])
                label_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                counts[f"{split}_tiles"] += 1
                counts["instances"] += len(labels)

    output.mkdir(parents=True, exist_ok=True)
    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(FPIC_CLASSES))
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": str(source),
        "label_source": "FPIC CSV shapes converted to YOLO polygons",
        "warning": "Most FPIC shapes are rectangles and are weak segmentation labels.",
        "tile_size": tile_size,
        "overlap": overlap,
        "min_visible_ratio": min_visible_ratio,
        "validation_group": validation_group,
        "split_groups": split_groups,
        "classes": FPIC_CLASSES,
        "source_annotation_shapes": annotation_shapes,
        "unlabeled_tiles": unlabeled_tiles,
        **counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
