"""Load component truth directly from coordinate Mask alignment output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class GroundTruthError(RuntimeError):
    pass


@dataclass(frozen=True)
class GroundTruthComponent:
    component_id: str
    designator: str
    category: str
    polygon_xy: tuple[tuple[float, float], ...]
    center_xy: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    source: dict


@dataclass(frozen=True)
class GroundTruth:
    metadata_path: Path
    registration_image: Path
    mask_overlay_image: Path
    image_size: tuple[int, int]
    components: tuple[GroundTruthComponent, ...]
    source: dict


def _resolve_artifact(path_value: str, metadata_path: Path) -> Path:
    supplied = Path(path_value).expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.insert(0, metadata_path.parent / supplied)
    candidates.extend(
        [
            metadata_path.parent / supplied.name,
            metadata_path.parent.parent / supplied.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise GroundTruthError(f"ground-truth image not found: {path_value}")


def _component_from_record(record: dict, index: int) -> GroundTruthComponent:
    try:
        polygon = np.asarray(record["target_polygon_px"], dtype=np.float64).reshape(-1, 2)
    except (KeyError, TypeError, ValueError) as exc:
        raise GroundTruthError(f"component #{index} has an invalid target polygon") from exc
    if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
        raise GroundTruthError(f"component #{index} has an invalid target polygon")
    center_value = record.get("target_center_px")
    center = (
        np.asarray(center_value, dtype=np.float64).reshape(2)
        if center_value is not None
        else np.mean(polygon, axis=0)
    )
    if not np.all(np.isfinite(center)):
        raise GroundTruthError(f"component #{index} has an invalid center")
    minimum = np.min(polygon, axis=0)
    maximum = np.max(polygon, axis=0)
    designator = str(record.get("designator") or f"component_{index:04d}")
    instance_id = record.get("instance_id", index)
    component_id = str(designator or instance_id)
    return GroundTruthComponent(
        component_id=component_id,
        designator=designator,
        category=str(record.get("category") or "other"),
        polygon_xy=tuple((float(x), float(y)) for x, y in polygon),
        center_xy=(float(center[0]), float(center[1])),
        bbox_xyxy=(
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ),
        source=dict(record),
    )


def load_ground_truth(metadata_path: str | Path) -> GroundTruth:
    metadata = Path(metadata_path).expanduser().resolve()
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GroundTruthError(f"component_masks.json not found: {metadata}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundTruthError(f"cannot read ground truth {metadata}: {exc}") from exc
    if payload.get("type") != "pcb_component_instance_masks":
        raise GroundTruthError(
            "ground truth must be component_masks.json exported by coordinate Mask alignment"
        )
    stitched_value = payload.get("stitched_image")
    if not stitched_value:
        raise GroundTruthError("ground truth does not contain stitched_image")
    registration_image = _resolve_artifact(str(stitched_value), metadata)
    overlay_candidate = metadata.parent / "pcb_with_component_masks.png"
    mask_overlay = overlay_candidate.resolve() if overlay_candidate.is_file() else registration_image

    image = cv2.imdecode(np.fromfile(registration_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise GroundTruthError(f"cannot read ground-truth image: {registration_image}")
    actual_size = (int(image.shape[1]), int(image.shape[0]))
    declared_size = payload.get("image_size_px")
    if declared_size is not None and tuple(map(int, declared_size)) != actual_size:
        raise GroundTruthError(
            f"ground-truth image size is {actual_size}, metadata declares {declared_size}"
        )

    components = tuple(
        _component_from_record(record, index)
        for index, record in enumerate(payload.get("components") or [], start=1)
        if isinstance(record, dict) and record.get("visible", True)
    )
    if not components:
        raise GroundTruthError("ground truth contains no visible component polygons")
    identifiers = [item.component_id for item in components]
    if len(set(identifiers)) != len(identifiers):
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        raise GroundTruthError(f"ground truth has duplicate component IDs: {duplicates}")
    return GroundTruth(
        metadata_path=metadata,
        registration_image=registration_image,
        mask_overlay_image=mask_overlay,
        image_size=actual_size,
        components=components,
        source=payload,
    )
