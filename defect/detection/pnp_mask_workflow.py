#!/usr/bin/env python3
"""PnP coordinate preview, captured-grid stitching, and component-mask export."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pnp_layout


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAPTURE_ROOT = PROJECT_ROOT / "working_data" / "pcb_two_zoom_capture"


class PnpMaskError(RuntimeError):
    pass


def _load_pnp_module():
    return pnp_layout


def latest_capture_run() -> Path | None:
    if not CAPTURE_ROOT.is_dir():
        return None
    candidates = [
        path.parent
        for path in CAPTURE_ROOT.glob("*/pcb_mosaic.json")
        if path.is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def latest_small_global_capture(
    capture_root: Path = CAPTURE_ROOT,
) -> tuple[Path, Path] | None:
    """Return the newest dated run and its cropped small-Zoom global image."""
    capture_root = Path(capture_root)
    if not capture_root.is_dir():
        return None
    try:
        runs = [path for path in capture_root.iterdir() if path.is_dir()]
    except OSError as exc:
        raise PnpMaskError(f"cannot list capture root {capture_root}: {exc}") from exc
    if not runs:
        return None
    run_dir = max(
        runs,
        key=lambda path: (path.name, path.stat().st_mtime_ns),
    )
    image_path = run_dir / "small_zoom_image" / "small_global_cropped.png"
    if not image_path.is_file():
        raise PnpMaskError(
            f"latest capture {run_dir.name} has no "
            "small_zoom_image/small_global_cropped.png"
        )
    return run_dir, image_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PnpMaskError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PnpMaskError(f"JSON root must be an object: {path}")
    return value


def _zoom_from_report(run_dir: Path, report: dict[str, Any]) -> int:
    two_zoom_path = run_dir / "pcb_two_zoom_capture.json"
    if two_zoom_path.is_file():
        two_zoom = _read_json(two_zoom_path)
        zoom = (two_zoom.get("large") or {}).get("lens_zoom")
        if zoom is not None:
            return int(zoom)
    name = (report.get("camera_intrinsics") or {}).get("position_name", "")
    match = re.search(r"zoom_(\d+)", str(name))
    if match:
        return int(match.group(1))
    raise PnpMaskError("capture report does not identify the large-image lens zoom")


def _stage_to_pixel(run_dir: Path, report: dict[str, Any]) -> np.ndarray:
    calibration_path = Path(report["stage_calibration"])
    if not calibration_path.is_absolute():
        calibration_path = PROJECT_ROOT / calibration_path
    document = _read_json(calibration_path)
    zoom = _zoom_from_report(run_dir, report)
    matches = [
        entry
        for entry in document.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("type") == "stage_command_to_pixel_calibration"
        and (
            entry.get("lens_zoom") == zoom
            or entry.get("name") == f"zoom_{zoom:05d}"
        )
    ]
    if not matches:
        raise PnpMaskError(
            f"no stage-to-pixel calibration for lens zoom {zoom} in {calibration_path}"
        )
    try:
        matrix = np.asarray(
            matches[-1]["stage_command_to_pixel"]["matrix_2x2"],
            dtype=np.float64,
        ).reshape(2, 2)
    except (KeyError, TypeError, ValueError) as exc:
        raise PnpMaskError("invalid stage-to-pixel matrix in calibration summary") from exc
    return matrix


def _registration_features(image: np.ndarray) -> dict[str, Any] | None:
    """Extract scale-independent features for adjacent-frame registration."""
    height, width = image.shape[:2]
    scale = min(1.0, 1600.0 / max(width, height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(32, int(round(width * scale))), max(32, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
        norm = cv2.NORM_L2
        ratio = 0.72
        method = "SIFT"
    else:
        detector = cv2.ORB_create(nfeatures=10000)
        norm = cv2.NORM_HAMMING
        ratio = 0.78
        method = "ORB"
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 12:
        return None
    points = np.asarray([point.pt for point in keypoints], dtype=np.float64) / scale
    return {
        "points": points,
        "descriptors": descriptors,
        "norm": norm,
        "ratio": ratio,
        "method": method,
    }


def _estimate_relative_placement(
    reference: dict[str, Any] | None,
    current: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Estimate where the current frame belongs relative to the reference."""
    if reference is None or current is None or reference["norm"] != current["norm"]:
        return None
    matcher = cv2.BFMatcher(current["norm"])
    pairs = matcher.knnMatch(current["descriptors"], reference["descriptors"], k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < current["ratio"] * pair[1].distance
    ]
    if len(good) < 12:
        return None
    source = np.asarray(
        [current["points"][match.queryIdx] for match in good], dtype=np.float64
    )
    target = np.asarray(
        [reference["points"][match.trainIdx] for match in good], dtype=np.float64
    )
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if matrix is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 12:
        return None
    linear = np.asarray(matrix[:, :2], dtype=np.float64)
    scale = math.sqrt(abs(float(np.linalg.det(linear))))
    rotation_degrees = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    if not 0.96 <= scale <= 1.04 or abs(rotation_degrees) > 2.0:
        return None
    displacements = target[inliers] - source[inliers]
    placement = np.median(displacements, axis=0)
    residuals = np.linalg.norm(displacements - placement, axis=1)
    return placement, {
        "status": "registered",
        "method": current["method"],
        "match_count": len(good),
        "inlier_count": inlier_count,
        "median_residual_px": float(np.median(residuals)),
        "estimated_scale": scale,
        "estimated_rotation_degrees": rotation_degrees,
    }


def _registered_frame_shifts(
    frames: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Refine calibrated placements and reject V4L2 frames from an old position."""
    predicted = [np.asarray(shift, dtype=np.float64) for _, shift, _ in frames]
    features = [_registration_features(image) for image, _, _ in frames]
    registered = [predicted[0].copy()]
    diagnostics = [{"status": "reference"}]
    for index in range(1, len(frames)):
        expected_delta = predicted[index] - predicted[index - 1]
        estimate = _estimate_relative_placement(features[index - 1], features[index])
        if estimate is None:
            registered.append(
                predicted[index] + (registered[index - 1] - predicted[index - 1])
            )
            diagnostics.append({"status": "calibration_fallback"})
            continue
        measured_delta, record = estimate
        expected_distance = float(np.linalg.norm(expected_delta))
        correction = float(np.linalg.norm(measured_delta - expected_delta))
        correction_limit = max(160.0, expected_distance * 0.25)
        record.update(
            {
                "expected_delta_px": expected_delta.tolist(),
                "measured_delta_px": measured_delta.tolist(),
                "correction_px": correction,
                "correction_limit_px": correction_limit,
            }
        )
        if expected_distance > 200.0 and correction > correction_limit:
            previous = frames[index - 1][2]
            current = frames[index][2]
            raise PnpMaskError(
                "captured frame r{row:02d}_c{col:02d} does not match its stage "
                "position: expected a ({ex:.1f}, {ey:.1f}) px step, but image "
                "content moved ({mx:.1f}, {my:.1f}) px. The camera likely returned "
                "a stale V4L2 buffer frame after r{prow:02d}_c{pcol:02d}; capture "
                "the PCB again with the updated buffer handling.".format(
                    row=int(current.get("row", -1)),
                    col=int(current.get("col", -1)),
                    ex=expected_delta[0],
                    ey=expected_delta[1],
                    mx=measured_delta[0],
                    my=measured_delta[1],
                    prow=int(previous.get("row", -1)),
                    pcol=int(previous.get("col", -1)),
                )
            )
        registered.append(registered[index - 1] + measured_delta)
        diagnostics.append(record)
    return registered, diagnostics


def stitch_capture(run_dir: Path, output_path: Path | None = None) -> Path:
    """Stitch a capture grid using stage calibration plus image registration."""
    run_dir = Path(run_dir).resolve()
    report_path = run_dir / "pcb_mosaic.json"
    report = _read_json(report_path)
    observations = report.get("observations")
    if not isinstance(observations, list) or not observations:
        raise PnpMaskError(f"no captured observations in {report_path}")
    stage_to_pixel = _stage_to_pixel(run_dir, report)
    initial = np.asarray(report["motion"]["initial_position_mm"], dtype=np.float64)

    frames: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for observation in observations:
        image_text = observation.get("undistorted_image") or observation.get("raw_image")
        if not image_text:
            raise PnpMaskError("capture observation has no saved image")
        image_path = Path(image_text)
        if not image_path.is_absolute():
            image_path = run_dir / image_path
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise PnpMaskError(f"cannot read captured image: {image_path}")
        actual = np.asarray(observation["actual_position_mm"], dtype=np.float64)
        shift = -(stage_to_pixel @ (actual - initial))
        frames.append((image, shift, observation))

    registered_shifts, registrations = _registered_frame_shifts(frames)

    min_xy = np.floor(np.min(registered_shifts, axis=0)).astype(int)
    max_xy = np.ceil(
        np.max(
            [
                shift + np.asarray([image.shape[1], image.shape[0]])
                for (image, _, _), shift in zip(frames, registered_shifts)
            ],
            axis=0,
        )
    ).astype(int)
    canvas_size = max_xy - min_xy
    if np.any(canvas_size <= 0) or int(canvas_size[0]) * int(canvas_size[1]) > 150_000_000:
        raise PnpMaskError(f"invalid stitched canvas size: {canvas_size.tolist()}")

    width, height = int(canvas_size[0]), int(canvas_size[1])
    accum = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    placement_records = []
    for (image, calibrated_shift, observation), shift, registration in zip(
        frames, registered_shifts, registrations
    ):
        x, y = np.rint(shift - min_xy).astype(int)
        image_h, image_w = image.shape[:2]
        yy = np.minimum(np.arange(image_h) + 1, np.arange(image_h, 0, -1))
        xx = np.minimum(np.arange(image_w) + 1, np.arange(image_w, 0, -1))
        feather = np.minimum(yy[:, None], xx[None, :]).astype(np.float32)
        feather = np.minimum(feather, 128.0) / 128.0
        roi = np.s_[y : y + image_h, x : x + image_w]
        accum[roi] += image.astype(np.float32) * feather[..., None]
        weights[roi] += feather
        placement_records.append(
            {
                "row": observation.get("row"),
                "col": observation.get("col"),
                "canvas_xy": [int(x), int(y)],
                "calibrated_shift_px": calibrated_shift.tolist(),
                "registered_shift_px": shift.tolist(),
                "registration": registration,
                "actual_position_mm": observation.get("actual_position_mm"),
            }
        )

    result = np.zeros_like(accum, dtype=np.uint8)
    valid = weights > 0
    result[valid] = np.clip(accum[valid] / weights[valid, None], 0, 255).astype(np.uint8)
    output_path = output_path or (run_dir / "pcb_stitched.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise PnpMaskError("OpenCV failed to encode stitched PCB image")
    encoded.tofile(str(output_path))
    metadata = {
        "type": "pcb_stitched_image",
        "source_report": str(report_path),
        "image": str(output_path),
        "width_px": width,
        "height_px": height,
        "stage_to_pixel": stage_to_pixel.tolist(),
        "image_registration": "adjacent SIFT/ORB translation with calibration fallback",
        "canvas_origin_shift_px": (-min_xy).tolist(),
        "placements": placement_records,
    }
    (output_path.with_suffix(".json")).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def _font(size: int):
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def build_pnp_preview(
    pnp_path: Path,
    layer: str,
    output_dir: Path,
    encoding: str = "gb18030",
    max_side: int = 1800,
) -> tuple[Path, Path]:
    module = _load_pnp_module()
    parts = module.parse_pnp(Path(pnp_path), encoding)
    layer_name = {"top": "TopLayer", "bottom": "BottomLayer"}.get(layer, layer)
    selected = [part for part in parts if part.layer == layer_name]
    if not selected:
        raise PnpMaskError(f"PnP file has no parts on {layer_name}")
    extent = module.board_extent(selected, margin_mm=2.0)
    x0, y0, x1, y1 = [float(value) for value in extent]
    scale = min(max_side / max(x1 - x0, 1e-6), max_side / max(y1 - y0, 1e-6))
    width = max(320, int(math.ceil((x1 - x0) * scale)))
    height = max(240, int(math.ceil((y1 - y0) * scale)))
    mirrored_horizontally = layer_name == "BottomLayer"

    def to_px(x_mm: float, y_mm: float) -> list[float]:
        x_px = (x1 - x_mm) * scale if mirrored_horizontally else (x_mm - x0) * scale
        return [x_px, (y1 - y_mm) * scale]

    image = Image.new("RGB", (width, height), (25, 29, 36))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(9, int(scale * 0.45)))
    records = []
    for index, part in enumerate(sorted(selected, key=lambda item: item.size[0] * item.size[1], reverse=True), start=1):
        polygon = [to_px(x, y) for x, y in module.part_corners(part)]
        rgb = module.CATEGORY_META[part.category][1]
        draw.polygon([tuple(point) for point in polygon], fill=(*rgb, 145), outline=(*rgb, 255), width=max(1, int(scale * 0.08)))
        center = to_px(part.cx, part.cy)
        draw.text(tuple(center), part.designator, fill=(255, 255, 255, 235), font=font, anchor="mm")
        records.append(
            {
                "instance_id": index,
                "designator": part.designator,
                "comment": part.comment,
                "footprint": part.footprint,
                "layer": part.layer,
                "category": part.category,
                "center_mm": [part.cx, part.cy],
                "rotation_deg": part.rotation,
                "size_mm": list(part.size),
                "source_polygon_px": polygon,
                "source_center_px": center,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "pnp_component_preview.png"
    metadata_path = output_dir / "pnp_components.json"
    image.save(preview_path)
    metadata_path.write_text(
        json.dumps(
            {
                "type": "pnp_component_coordinates",
                "source_file": str(Path(pnp_path).resolve()),
                "layer": layer_name,
                "board_side": "front" if layer_name == "TopLayer" else "back",
                "mirrored_horizontally": mirrored_horizontally,
                "encoding": encoding,
                "extent_mm": [x0, y0, x1, y1],
                "scale_px_per_mm": scale,
                "preview_size_px": [width, height],
                "component_count": len(records),
                "components": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return preview_path, metadata_path


def fit_affine(source_points: list[list[float]], target_points: list[list[float]]) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise PnpMaskError("source and target points must be matching Nx2 arrays")
    if len(source) < 3:
        raise PnpMaskError("at least three point pairs are required")
    centered = source - np.mean(source, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        raise PnpMaskError("alignment points must not be collinear")
    matrix, inliers = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC if len(source) > 3 else cv2.LMEDS,
        ransacReprojThreshold=8.0,
    )
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise PnpMaskError("cannot fit an affine transform from the selected points")
    return matrix


def _spatial_balance_weights(source: np.ndarray) -> np.ndarray:
    """Give a dense cluster roughly the same total influence as a sparse area."""
    count = len(source)
    if count <= 2:
        return np.ones(count, dtype=np.float64)
    distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
    extent = float(np.linalg.norm(np.ptp(source, axis=0)))
    bandwidth = float(np.median(positive) * 1.8) if len(positive) else extent * 0.1
    bandwidth = max(bandwidth, extent * 0.025, 1.0)
    density = np.sum(
        np.exp(-np.square(distances) / (2.0 * bandwidth * bandwidth)), axis=1
    ) + 1.0
    weights = 1.0 / density
    return weights / np.mean(weights)


def _weighted_affine(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    centered = source - np.mean(source, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < 2:
        raise PnpMaskError("alignment points must not be collinear")
    design = np.column_stack((source, np.ones(len(source))))
    root_weights = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * root_weights, target * root_weights, rcond=None
    )
    if rank < 3:
        raise PnpMaskError("cannot fit an affine transform from the selected points")
    matrix = coefficients.T
    if not np.all(np.isfinite(matrix)):
        raise PnpMaskError("cannot fit an affine transform from the selected points")
    return matrix


def _weighted_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    reflection: bool | None = None,
) -> np.ndarray:
    if len(source) < 2 or np.max(np.linalg.norm(source - source[0], axis=1)) < 1e-6:
        raise PnpMaskError("similarity alignment needs two distinct source points")

    def solve(reflected: bool) -> tuple[np.ndarray, float]:
        rows = []
        values = []
        row_weights = []
        for (x, y), (u, v), weight in zip(source, target, weights):
            if reflected:
                rows.extend(([x, y, 1.0, 0.0], [-y, x, 0.0, 1.0]))
            else:
                rows.extend(([x, -y, 1.0, 0.0], [y, x, 0.0, 1.0]))
            values.extend((u, v))
            row_weights.extend((weight, weight))
        design = np.asarray(rows, dtype=np.float64)
        values_array = np.asarray(values, dtype=np.float64)
        root_weights = np.sqrt(np.asarray(row_weights, dtype=np.float64))
        parameters, _, rank, _ = np.linalg.lstsq(
            design * root_weights[:, None],
            values_array * root_weights,
            rcond=None,
        )
        if rank < 4:
            raise PnpMaskError("cannot fit a similarity transform from the selected points")
        a, b, tx, ty = parameters
        if reflected:
            matrix = np.asarray([[a, b, tx], [b, -a, ty]], dtype=np.float64)
        else:
            matrix = np.asarray([[a, -b, tx], [b, a, ty]], dtype=np.float64)
        predicted = source @ matrix[:, :2].T + matrix[:, 2]
        error = float(np.sum(weights * np.sum(np.square(predicted - target), axis=1)))
        return matrix, error

    candidates = [solve(False), solve(True)] if reflection is None else [solve(reflection)]
    matrix, _ = min(candidates, key=lambda candidate: candidate[1])
    if not np.all(np.isfinite(matrix)):
        raise PnpMaskError("cannot fit a similarity transform from the selected points")
    return matrix


def fit_alignment(
    source_points: list[list[float]],
    target_points: list[list[float]],
    mode: str = "local_similarity",
) -> dict[str, Any]:
    """Build a serializable alignment model with spatially balanced controls."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise PnpMaskError("source and target points must be matching Nx2 arrays")
    if len(source) < 3:
        raise PnpMaskError("at least three point pairs are required")
    weights = _spatial_balance_weights(source)
    if mode == "global_affine":
        matrix = _weighted_affine(source, target, weights)
        return {"mode": mode, "matrix": matrix.tolist()}
    if mode not in ("global_similarity", "local_similarity"):
        raise PnpMaskError(f"unknown alignment mode: {mode}")
    global_matrix = _weighted_similarity(source, target, weights)
    if mode == "global_similarity":
        return {"mode": mode, "matrix": global_matrix.tolist()}

    distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1e-6)]
    extent = float(np.linalg.norm(np.ptp(source, axis=0)))
    local_radius = float(np.median(positive) * 2.5) if len(positive) else extent * 0.2
    local_radius = max(local_radius, extent * 0.08, 30.0)
    return {
        "mode": mode,
        "matrix": global_matrix.tolist(),
        "source_points": source.tolist(),
        "target_points": target.tolist(),
        "control_weights": weights.tolist(),
        "neighbors": min(6, len(source)),
        "local_radius_px": local_radius,
        "reflection": bool(np.linalg.det(global_matrix[:, :2]) < 0),
    }


def _matrix_for_anchor(alignment: np.ndarray | dict[str, Any], anchor: np.ndarray) -> np.ndarray:
    if not isinstance(alignment, dict):
        return np.asarray(alignment, dtype=np.float64).reshape(2, 3)
    global_matrix = np.asarray(alignment["matrix"], dtype=np.float64).reshape(2, 3)
    if alignment.get("mode") != "local_similarity":
        return global_matrix
    source = np.asarray(alignment["source_points"], dtype=np.float64)
    target = np.asarray(alignment["target_points"], dtype=np.float64)
    distances = np.linalg.norm(source - anchor, axis=1)
    order = np.argsort(distances)[: int(alignment.get("neighbors", 6))]
    radius = float(alignment.get("local_radius_px", 100.0))
    epsilon = max(radius * 0.04, 1.0)
    local_weights = 1.0 / (np.square(distances[order]) + epsilon * epsilon)
    local_weights *= np.asarray(alignment.get("control_weights", np.ones(len(source))))[order]
    try:
        local_matrix = _weighted_similarity(
            source[order],
            target[order],
            local_weights,
            reflection=bool(alignment.get("reflection", False)),
        )
    except PnpMaskError:
        return global_matrix
    influence = math.exp(-((float(distances[order[0]]) / radius) ** 2))
    return global_matrix * (1.0 - influence) + local_matrix * influence


def transform_points(
    points: list[list[float]] | np.ndarray,
    alignment: np.ndarray | dict[str, Any],
    anchor: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    anchor_array = np.asarray(anchor, dtype=np.float64) if anchor is not None else np.mean(values, axis=0)
    matrix = _matrix_for_anchor(alignment, anchor_array)
    return values @ matrix[:, :2].T + matrix[:, 2]


def transform_control_points(
    points: list[list[float]] | np.ndarray, alignment: np.ndarray | dict[str, Any]
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.asarray(
        [transform_points([point], alignment, anchor=point)[0] for point in values]
    )


CATEGORY_COLORS = {
    "resistor": (58, 96, 235),
    "capacitor": (235, 150, 62),
    "inductor": (70, 180, 230),
    "diode": (95, 190, 88),
    "transistor": (195, 120, 190),
    "ic": (230, 125, 55),
    "connector": (75, 170, 215),
    "testpoint": (190, 190, 190),
    "fuse": (160, 80, 220),
    "switch": (90, 200, 130),
    "other": (70, 200, 200),
}


def render_target_masks(
    stitched_path: Path,
    transformed_records: list[dict[str, Any]],
    output_dir: Path,
    opacity: float = 0.38,
    draw_labels: bool = True,
    alignment: dict[str, Any] | None = None,
    pnp_components: Path | None = None,
    manually_edited: bool = False,
) -> dict[str, Path]:
    """Render already-positioned component polygons into final mask artifacts."""
    image = cv2.imdecode(np.fromfile(stitched_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise PnpMaskError(f"cannot read stitched image: {stitched_path}")
    height, width = image.shape[:2]
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    union_mask = np.zeros((height, width), dtype=np.uint8)
    color_mask = np.zeros_like(image)
    output_records = []
    for output_id, item in enumerate(transformed_records, start=1):
        target = np.asarray(item["target_polygon_px"], dtype=np.float64).reshape(-1, 2)
        if len(target) < 3 or not np.all(np.isfinite(target)):
            raise PnpMaskError(f"invalid target polygon for {item.get('designator', output_id)}")
        polygon = np.rint(target).astype(np.int32)
        color = CATEGORY_COLORS.get(item.get("category"), CATEGORY_COLORS["other"])
        cv2.fillPoly(instance_mask, [polygon], output_id)
        cv2.fillPoly(union_mask, [polygon], 255)
        cv2.fillPoly(color_mask, [polygon], color)
        cv2.polylines(color_mask, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
        center = np.mean(target, axis=0)
        record = dict(item)
        record["source_instance_id"] = item.get("instance_id")
        record["instance_id"] = output_id
        record["target_polygon_px"] = target.tolist()
        record["target_center_px"] = center.tolist()
        record["visible"] = bool(
            np.any((target[:, 0] >= 0) & (target[:, 0] < width) & (target[:, 1] >= 0) & (target[:, 1] < height))
        )
        output_records.append(record)

    alpha = np.clip(float(opacity), 0.0, 1.0)
    overlay = image.copy()
    active = union_mask > 0
    overlay[active] = np.clip(
        image[active].astype(np.float32) * (1.0 - alpha)
        + color_mask[active].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    for record in output_records:
        polygon = np.rint(record["target_polygon_px"]).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, (255, 255, 255), 2, cv2.LINE_AA)
        center = np.asarray(record["target_center_px"])
        if draw_labels and 0 <= center[0] < width and 0 <= center[1] < height:
            cv2.putText(
                overlay,
                str(record.get("designator", record["instance_id"])),
                tuple(np.rint(center).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overlay": output_dir / "pcb_with_component_masks.png",
        "binary_mask": output_dir / "component_mask.png",
        "instance_mask": output_dir / "component_instances.png",
        "metadata": output_dir / "component_masks.json",
    }
    for key, value in (("overlay", overlay), ("binary_mask", union_mask), ("instance_mask", instance_mask)):
        ok, encoded = cv2.imencode(".png", value)
        if not ok:
            raise PnpMaskError(f"cannot encode {key}")
        encoded.tofile(str(outputs[key]))
    outputs["metadata"].write_text(
        json.dumps(
            {
                "type": "pcb_component_instance_masks",
                "stitched_image": str(Path(stitched_path).resolve()),
                "pnp_components": str(Path(pnp_components).resolve()) if pnp_components else None,
                "alignment": alignment,
                "manually_edited": manually_edited,
                "image_size_px": [width, height],
                "instance_count": len(output_records),
                "components": output_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return outputs


def export_masks(
    stitched_path: Path,
    components_path: Path,
    alignment: np.ndarray | dict[str, Any],
    output_dir: Path,
    opacity: float = 0.38,
    draw_labels: bool = True,
) -> dict[str, Path]:
    document = _read_json(components_path)
    transformed_records = []
    for item in document["components"]:
        source = np.asarray(item["source_polygon_px"], dtype=np.float64).reshape(-1, 2)
        source_center = np.asarray(item.get("source_center_px", np.mean(source, axis=0)))
        target = transform_points(source, alignment, anchor=source_center)
        record = dict(item)
        record["target_polygon_px"] = target.tolist()
        record["target_center_px"] = np.mean(target, axis=0).tolist()
        transformed_records.append(record)
    serialized_alignment = (
        alignment
        if isinstance(alignment, dict)
        else {"mode": "global_affine", "matrix": np.asarray(alignment).tolist()}
    )
    return render_target_masks(
        stitched_path,
        transformed_records,
        output_dir,
        opacity,
        draw_labels,
        alignment=serialized_alignment,
        pnp_components=components_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    stitch = sub.add_parser("stitch")
    stitch.add_argument("run_dir", type=Path)
    preview = sub.add_parser("preview")
    preview.add_argument("pnp", type=Path)
    preview.add_argument("--layer", choices=("top", "bottom"), default="top")
    preview.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stitch":
        print(stitch_capture(args.run_dir))
    else:
        print(build_pnp_preview(args.pnp, args.layer, args.output_dir)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
