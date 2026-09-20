"""SAM3 text-prompt segmentation and component candidate post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .config import DETECTION_ROOT, InspectionConfig


ProgressCallback = Callable[[str], None]
TARGET_PATCH_ASPECT_RATIO = 2.4


@dataclass
class SegmentCandidate:
    sam_score: float
    sam_bbox_xyxy: tuple[int, int, int, int]
    mask_bbox_xyxy: tuple[int, int, int, int]
    mask_crop: np.ndarray
    center_xy: tuple[float, float]
    inner_point_xy: tuple[float, float]
    rotated_rect: tuple[tuple[float, float], tuple[float, float], float]
    source_mask_index: int
    component_index: int
    component_count: int
    original_mask_area: int

    @property
    def mask_area(self) -> int:
        return int(np.count_nonzero(self.mask_crop))

    @property
    def mask_fill_ratio(self) -> float:
        x1, y1, x2, y2 = self.mask_bbox_xyxy
        return self.mask_area / max(1, (x2 - x1) * (y2 - y1))


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def normalize_rotated_rect(rect):
    (center_x, center_y), (width, height), angle = rect
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    angle = float(angle)
    if height > width:
        width, height = height, width
        angle += 90.0
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return ((float(center_x), float(center_y)), (width, height), angle)


def rect_corners(rect) -> np.ndarray:
    (center_x, center_y), (width, height), angle = rect
    theta = np.deg2rad(float(angle))
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    corners = np.asarray(
        [
            [-width / 2.0, -height / 2.0],
            [width / 2.0, -height / 2.0],
            [width / 2.0, height / 2.0],
            [-width / 2.0, height / 2.0],
        ],
        dtype=np.float32,
    )
    corners = corners @ rotation.T
    corners[:, 0] += float(center_x)
    corners[:, 1] += float(center_y)
    return corners


def rect_from_mask(mask_crop: np.ndarray, mask_box) -> tuple | None:
    ys, xs = np.where(mask_crop)
    if xs.size == 0:
        return None
    x1, y1, _, _ = mask_box
    points = np.column_stack([xs + x1, ys + y1]).astype(np.float32)
    return normalize_rotated_rect(cv2.minAreaRect(points))


def component_centroid(mask_crop: np.ndarray, mask_box) -> tuple[float, float]:
    moments = cv2.moments(mask_crop.astype(np.uint8), binaryImage=True)
    x1, y1, x2, y2 = mask_box
    if moments["m00"] <= 0:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (
        float(x1) + moments["m10"] / moments["m00"],
        float(y1) + moments["m01"] / moments["m00"],
    )


def component_inner_point(mask_crop: np.ndarray, mask_box) -> tuple[float, float]:
    padded = np.pad(mask_crop.astype(np.uint8), 1, mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    padded_y, padded_x = np.unravel_index(np.argmax(distance), distance.shape)
    x1, y1, _, _ = mask_box
    return float(x1 + padded_x - 1), float(y1 + padded_y - 1)


def open_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return mask.astype(bool, copy=True)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    return opened if np.any(opened) else mask.astype(bool, copy=True)


def split_mask_components(
    mask: np.ndarray,
    min_component_area: int,
    min_component_area_ratio: float,
    mask_open_kernel: int,
) -> list[dict]:
    cleaned = open_binary_mask(mask, mask_open_kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    threshold = max(
        1,
        int(min_component_area),
        int(np.ceil(largest * max(0.0, float(min_component_area_ratio)))),
    )
    kept_labels = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= threshold
    ]
    kept_labels.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    components = []
    for rank, label in enumerate(kept_labels, start=1):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        components.append(
            {
                "component_index": rank,
                "component_count": len(kept_labels),
                "mask_box": (x, y, x + width, y + height),
                "mask_crop": labels[y : y + height, x : x + width] == label,
            }
        )
    return components


def _tensor_numpy(value) -> np.ndarray:
    return value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def collect_candidates_from_state(
    state: dict,
    config: InspectionConfig,
    offset_xy: tuple[int, int] = (0, 0),
    tile_size: tuple[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> list[SegmentCandidate]:
    if state.get("boxes") is None or state.get("masks") is None:
        return []
    boxes = _tensor_numpy(state["boxes"])
    scores = _tensor_numpy(state["scores"]).reshape(-1)
    masks = _tensor_numpy(state["masks"])
    offset_x, offset_y = map(int, offset_xy)
    candidates: list[SegmentCandidate] = []
    for source_index, (box, score, raw_mask) in enumerate(
        zip(boxes, scores, masks), start=1
    ):
        mask = np.asarray(raw_mask).squeeze() > 0.5
        original_area = int(np.count_nonzero(mask))
        sam_x1, sam_y1, sam_x2, sam_y2 = [int(round(value)) for value in box.tolist()]
        pieces = split_mask_components(
            mask,
            min_component_area=config.min_mask_area,
            min_component_area_ratio=config.min_component_area_ratio,
            mask_open_kernel=config.mask_open_kernel,
        )
        for piece in pieces:
            x1, y1, x2, y2 = piece["mask_box"]
            if tile_size is not None and image_size is not None:
                tile_width, tile_height = tile_size
                image_width, image_height = image_size
                margin = config.edge_margin
                touches_internal_boundary = (
                    (x1 < margin and offset_x > 0)
                    or (y1 < margin and offset_y > 0)
                    or (x2 > tile_width - margin and offset_x + tile_width < image_width)
                    or (y2 > tile_height - margin and offset_y + tile_height < image_height)
                )
                if touches_internal_boundary:
                    continue
            mask_box = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            crop = np.asarray(piece["mask_crop"], dtype=bool)
            rect = rect_from_mask(crop, mask_box)
            if rect is None:
                continue
            candidates.append(
                SegmentCandidate(
                    sam_score=float(score),
                    sam_bbox_xyxy=(
                        sam_x1 + offset_x,
                        sam_y1 + offset_y,
                        sam_x2 + offset_x,
                        sam_y2 + offset_y,
                    ),
                    mask_bbox_xyxy=mask_box,
                    mask_crop=crop,
                    center_xy=component_centroid(crop, mask_box),
                    inner_point_xy=component_inner_point(crop, mask_box),
                    rotated_rect=rect,
                    source_mask_index=source_index,
                    component_index=int(piece["component_index"]),
                    component_count=int(piece["component_count"]),
                    original_mask_area=original_area,
                )
            )
    return candidates


def generate_tiles(width: int, height: int, tile_size: int, overlap: int):
    step = max(1, tile_size - overlap)
    last_x = max(0, width - tile_size)
    last_y = max(0, height - tile_size)
    x_starts = list(range(0, last_x + 1, step)) or [0]
    y_starts = list(range(0, last_y + 1, step)) or [0]
    if x_starts[-1] != last_x:
        x_starts.append(last_x)
    if y_starts[-1] != last_y:
        y_starts.append(last_y)
    for y1 in y_starts:
        for x1 in x_starts:
            yield x1, y1, min(width, x1 + tile_size), min(height, y1 + tile_size)


def _overlapping_mask_views(a: SegmentCandidate, b: SegmentCandidate):
    ax1, ay1, ax2, ay2 = a.mask_bbox_xyxy
    bx1, by1, bx2, by2 = b.mask_bbox_xyxy
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if x1 >= x2 or y1 >= y2:
        return None
    return (
        a.mask_crop[y1 - ay1 : y2 - ay1, x1 - ax1 : x2 - ax1],
        b.mask_crop[y1 - by1 : y2 - by1, x1 - bx1 : x2 - bx1],
    )


def masks_mutually_overlap(a: SegmentCandidate, b: SegmentCandidate, threshold: float) -> bool:
    views = _overlapping_mask_views(a, b)
    if views is None:
        return False
    intersection = int(np.count_nonzero(views[0] & views[1]))
    if intersection <= 0:
        return False
    return (
        intersection / max(1, a.mask_area) >= threshold
        and intersection / max(1, b.mask_area) >= threshold
    )


def filter_and_suppress_candidates(
    candidates: list[SegmentCandidate],
    image_size: tuple[int, int],
    config: InspectionConfig,
) -> list[SegmentCandidate]:
    width, height = image_size
    maximum_area = width * height * config.max_mask_area_ratio
    filtered = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.mask_bbox_xyxy
        if candidate.mask_area < config.min_mask_area or candidate.mask_area > maximum_area:
            continue
        if candidate.mask_fill_ratio < config.min_mask_fill_ratio:
            continue
        if config.edge_margin > 0 and (
            x1 < config.edge_margin
            or y1 < config.edge_margin
            or x2 > width - config.edge_margin
            or y2 > height - config.edge_margin
        ):
            continue
        filtered.append(candidate)
    ordered = sorted(filtered, key=lambda item: (item.mask_area, item.sam_score), reverse=True)
    kept: list[SegmentCandidate] = []
    for candidate in ordered:
        if any(
            masks_mutually_overlap(candidate, previous, config.mutual_overlap_threshold)
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item.center_xy[1], item.center_xy[0]))


def extract_rectified_patch(
    image_rgb: np.ndarray,
    candidate: SegmentCandidate,
    padding_ratio: float,
    min_patch_size: int,
) -> np.ndarray:
    center, (width, height), angle = candidate.rotated_rect
    padding = max(1.0, min(width, height) * max(0.0, float(padding_ratio)))
    width += padding * 2.0
    height += padding * 2.0
    if width / height < TARGET_PATCH_ASPECT_RATIO:
        width = height * TARGET_PATCH_ASPECT_RATIO
    else:
        height = width / TARGET_PATCH_ASPECT_RATIO
    expanded = (center, (width, height), angle)
    source = rect_corners(expanded).astype(np.float32)
    output_width = max(1, int(round(width)))
    output_height = max(1, int(round(height)))
    target = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    patch = cv2.warpPerspective(
        image_rgb,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    longest = max(patch.shape[:2])
    if longest < min_patch_size:
        scale = float(min_patch_size) / max(1, longest)
        patch = cv2.resize(
            patch,
            (max(1, int(round(patch.shape[1] * scale))), max(1, int(round(patch.shape[0] * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return patch


class SamPromptSegmenter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config: InspectionConfig,
        device: str = "auto",
        progress: ProgressCallback | None = None,
    ):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.config = config
        self.progress = progress
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAM3 component segmentation requires an available CUDA GPU")
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bpe_path = DETECTION_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.is_file():
            raise FileNotFoundError(f"SAM3 tokenizer not found: {bpe_path}")
        _emit(progress, "正在加载 SAM3 文本提示分割模型…")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.model = build_sam3_image_model(
                bpe_path=str(bpe_path),
                checkpoint_path=str(Path(checkpoint_path).resolve()),
                load_from_HF=False,
                device=device,
            )
        self.processor = Sam3Processor(
            model=self.model,
            confidence_threshold=config.sam_confidence_threshold,
            device=device,
        )

    def _ground(self, image: Image.Image):
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ):
            state = self.processor.set_image(image)
            return self.processor.set_text_prompt(state=state, prompt=self.config.prompt)

    def segment(self, image: Image.Image, image_name: str = "image") -> list[SegmentCandidate]:
        width, height = image.size
        _emit(self.progress, f"{image_name}: SAM3 整图文本提示分割")
        raw = collect_candidates_from_state(self._ground(image), self.config)
        if max(width, height) > self.config.tile_size or not raw:
            tiles = list(
                generate_tiles(
                    width,
                    height,
                    self.config.tile_size,
                    self.config.tile_overlap,
                )
            )
            for index, (x1, y1, x2, y2) in enumerate(tiles, start=1):
                _emit(self.progress, f"{image_name}: SAM3 分块 {index}/{len(tiles)}")
                tile = image.crop((x1, y1, x2, y2))
                raw.extend(
                    collect_candidates_from_state(
                        self._ground(tile),
                        self.config,
                        offset_xy=(x1, y1),
                        tile_size=(x2 - x1, y2 - y1),
                        image_size=(width, height),
                    )
                )
        candidates = filter_and_suppress_candidates(raw, (width, height), self.config)
        _emit(self.progress, f"{image_name}: SAM 候选 {len(candidates)} 个")
        return candidates

    def close(self) -> None:
        self.processor = None
        self.model = None
