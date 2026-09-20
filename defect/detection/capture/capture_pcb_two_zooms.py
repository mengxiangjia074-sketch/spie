"""Capture a white-marker-aligned PCB at two lens zoom positions.

The small-zoom phase detects the four white corner blocks, averages their four
inward-facing corners, moves that point to the resolution-dependent image
centre, captures the full view again, and crops the PCB area bounded by those
four inward corners.
The top-left block's bottom-right corner is then transformed into large-zoom
pixels with the calibrated homography. A second relative stage move places
that point at ``large_target_pixel`` before the serpentine local-image sweep.
"""

from __future__ import annotations

import math
import sys
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
from typing import Any


CAPTURE_DIR = Path(__file__).resolve().parent
DETECTION_DIR = CAPTURE_DIR.parent
for module_path in (str(CAPTURE_DIR), str(DETECTION_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import LensCamera as lens_camera  # noqa: E402
import capture_pcb  # noqa: E402
from LensCamera import paths  # noqa: E402
from LensCamera.settings import Settings as JsonnetSettings  # noqa: E402
from LensCamera.settings import SettingsError  # noqa: E402


DEFAULT_SETTINGS_PATH = (
    paths.PROJECT_ROOT / "config" / "pcb_two_zoom_capture" / "capture.jsonnet"
)
AUTOFOCUS_ENTRY_TYPE = "pcb_autofocus_lens_position"
ZOOM_TRANSFORM_ENTRY_TYPE = "zoom_pixel_homography_calibration"
BLOCK_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
MAX_WHITE_BLOCK_HOLE_FRACTION = 0.10
MIN_WHITE_BLOCK_LAYOUT_AREA_FRACTION = 0.12
MIN_WHITE_BLOCK_OPPOSITE_EDGE_RATIO = 0.45


class TwoZoomCaptureError(capture_pcb.PcbMosaicError):
    pass


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise TwoZoomCaptureError(
            f"{DEFAULT_SETTINGS_PATH} missing required setting {name!r}"
        )
    item = settings[name]
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def load_args(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    if settings is None:
        try:
            settings = JsonnetSettings(
                DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)
            ).values
        except SettingsError as exc:
            raise TwoZoomCaptureError(str(exc)) from exc
    return SimpleNamespace(**{name: setting_value(settings, name) for name in settings})


def calibration_path(args: SimpleNamespace) -> Path:
    if args.calibration:
        return Path(args.calibration).resolve()
    return capture_pcb.DEFAULT_CALIBRATION_SUMMARY_PATH


def autofocus_position_for_zoom(
    document: dict[str, Any], summary_path: Path, zoom: Any
) -> dict[str, int | None]:
    requested_zoom = capture_pcb.zoom_address(zoom)
    entry = capture_pcb.newest_summary_entry_by_lens_zoom(
        document, AUTOFOCUS_ENTRY_TYPE, requested_zoom, summary_path
    )
    position = entry.get("lens_position")
    if not isinstance(position, dict):
        position = {}
    focus = entry.get("lens_focus", position.get("focus"))
    iris = entry.get("lens_iris", position.get("iris"))
    if focus is None:
        raise TwoZoomCaptureError(
            "autofocus result {!r} for lens_zoom={} in {} has no lens_focus; "
            "rerun PCB autofocus at this zoom".format(
                entry.get("name"), requested_zoom, summary_path
            )
        )
    return {
        "zoom": requested_zoom,
        "focus": capture_pcb.zoom_address(focus, "lens_focus"),
        "iris": (
            capture_pcb.zoom_address(iris, "lens_iris") if iris is not None else None
        ),
    }


def finite_pair(values: Any, name: str) -> list[float]:
    if values is None:
        raise TwoZoomCaptureError(f"{name} is required")
    try:
        result = [float(values[0]), float(values[1])]
    except (TypeError, ValueError, IndexError) as exc:
        raise TwoZoomCaptureError(f"{name} must contain two numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise TwoZoomCaptureError(f"{name} must contain finite values")
    return result


def validate_detection_settings(args: SimpleNamespace) -> None:
    if not 1 <= int(args.white_block_threshold) <= 255:
        raise TwoZoomCaptureError("white_block_threshold must be in [1, 255]")
    if float(args.white_block_min_area_px) <= 0:
        raise TwoZoomCaptureError("white_block_min_area_px must be above 0")
    if not 0 < float(args.white_block_max_area_fraction) < 1:
        raise TwoZoomCaptureError(
            "white_block_max_area_fraction must be between 0 and 1"
        )
    if not 0 < float(args.white_block_min_square_ratio) <= 1:
        raise TwoZoomCaptureError("white_block_min_square_ratio must be in (0, 1]")
    if not 0 < float(args.white_block_min_fill_ratio) <= 1:
        raise TwoZoomCaptureError("white_block_min_fill_ratio must be in (0, 1]")
    kernel = int(args.white_block_morph_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise TwoZoomCaptureError(
            "white_block_morph_kernel must be a positive odd integer"
        )
    try:
        int(args.global_crop_padding_px)
    except (TypeError, ValueError) as exc:
        raise TwoZoomCaptureError("global_crop_padding_px must be an integer") from exc


def build_capture_args(
    args: SimpleNamespace,
    document: dict[str, Any],
    summary_path: Path,
) -> SimpleNamespace:
    small_zoom = capture_pcb.zoom_address(args.small_zoom, "small_zoom")
    large_zoom = capture_pcb.zoom_address(args.large_zoom, "large_zoom")
    if small_zoom >= large_zoom:
        raise TwoZoomCaptureError("small_zoom must be below large_zoom")

    small_position = autofocus_position_for_zoom(document, summary_path, small_zoom)
    large_position = autofocus_position_for_zoom(document, summary_path, large_zoom)
    runtime = SimpleNamespace(**vars(args))
    runtime.small_zoom = small_zoom
    runtime.small_focus = small_position["focus"]
    runtime.small_iris = small_position["iris"]
    runtime.large_zoom = large_zoom
    runtime.large_focus = large_position["focus"]
    runtime.large_iris = large_position["iris"]
    runtime.large_target_pixel = finite_pair(
        args.large_target_pixel, "large_target_pixel"
    )
    runtime.lens_zoom = large_zoom
    runtime.undistort = True
    validate_detection_settings(runtime)
    return runtime


def phase_values(args: SimpleNamespace, prefix: str) -> dict[str, Any]:
    return {
        "name": prefix,
        "zoom": getattr(args, f"{prefix}_zoom"),
        "focus": getattr(args, f"{prefix}_focus"),
        "iris": getattr(args, f"{prefix}_iris"),
    }


def lens_snapshot_position(snapshot: dict[str, Any]) -> dict[str, Any]:
    position = {}
    for name in lens_camera.MOTOR_ORDER:
        item = snapshot.get(name) or {}
        position[name] = item.get("current") if item.get("supported") else None
    return position


def move_lens_motor(
    name: str, target: int, capabilities: Any, args: SimpleNamespace
) -> dict[str, Any]:
    try:
        lens_camera.ensure_ready(name, init_if_needed=not bool(args.lens_no_init))
        move = lens_camera.move_connected_motor(
            name,
            target,
            capabilities,
            settle_seconds=float(args.lens_settle_seconds),
            init_if_needed=None,
        )
    except lens_camera.LensControlError as exc:
        raise TwoZoomCaptureError(str(exc)) from exc
    print(
        "lens {}: target {} -> actual {} (error {})".format(
            name, move["target"], move["actual"], move["error"]
        )
    )
    return {
        "target": move["target"],
        "before": move["before"],
        "actual": move["actual"],
        "error": move["error"],
        "moved": move["moved"],
    }


def move_lens_to_phase(args: SimpleNamespace, phase: dict[str, Any]) -> dict[str, Any]:
    try:
        capabilities = lens_camera.connect(args.lens_device)
    except lens_camera.LensControlError as exc:
        raise TwoZoomCaptureError(str(exc)) from exc
    device_number = lens_camera.get_last_connected_device_number(args.lens_device)
    try:
        targets = {"zoom": phase["zoom"], "focus": phase["focus"]}
        if phase.get("iris") is not None:
            targets["iris"] = phase["iris"]
        ranges = lens_camera.read_ranges(capabilities)
        lens_camera.print_ranges(ranges, targets)
        lens_camera.validate_targets(targets, ranges)
        moves = {
            name: move_lens_motor(name, targets[name], capabilities, args)
            for name in lens_camera.MOTOR_ORDER
            if name in targets
        }
        snapshot = lens_camera.read_lens_snapshot(capabilities)
        return {
            "enabled": True,
            "device": device_number,
            "targets": targets,
            "moves": moves,
            "snapshot": snapshot,
            "position": lens_snapshot_position(snapshot),
        }
    except TwoZoomCaptureError:
        raise
    except Exception as exc:
        raise TwoZoomCaptureError(
            f"lens move failed for {phase['name']}: {exc}"
        ) from exc
    finally:
        lens_camera.close()


def load_visual_phase_calibrations(
    np: Any,
    summary_path: Path,
    phase_name: str,
    lens_zoom: Any,
    document: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase_name,
        "lens_zoom": capture_pcb.zoom_address(lens_zoom),
        "stage": capture_pcb.load_stage_calibration_for_zoom(
            np, document, summary_path, lens_zoom
        ),
        "intrinsics": capture_pcb.load_intrinsics_from_summary(
            np, summary_path, lens_zoom=lens_zoom, document=document
        ),
    }


def load_zoom_transform(
    np: Any,
    document: dict[str, Any],
    summary_path: Path,
    source_zoom: Any,
    target_zoom: Any,
) -> dict[str, Any]:
    source = capture_pcb.zoom_address(source_zoom, "source_zoom")
    target = capture_pcb.zoom_address(target_zoom, "target_zoom")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise TwoZoomCaptureError(f"{summary_path} has no entries array")

    matches: list[tuple[dict[str, Any], str]] = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or entry.get("type") != ZOOM_TRANSFORM_ENTRY_TYPE
        ):
            continue
        try:
            entry_source = capture_pcb.zoom_address(entry.get("source_zoom"))
            entry_target = capture_pcb.zoom_address(entry.get("target_zoom"))
        except capture_pcb.PcbMosaicError:
            continue
        if (entry_source, entry_target) == (source, target):
            matches.append((entry, "source_to_target"))
        elif (entry_source, entry_target) == (target, source):
            matches.append((entry, "target_to_source"))
    if not matches:
        raise TwoZoomCaptureError(
            f"{summary_path} has no Zoom pixel transform between {source} and {target}"
        )

    entry, direction = matches[-1]
    section = entry.get(direction)
    if not isinstance(section, dict) or "matrix_3x3" not in section:
        raise TwoZoomCaptureError(
            f"Zoom transform {entry.get('name')!r} has no {direction}.matrix_3x3"
        )
    try:
        matrix = np.asarray(section["matrix_3x3"], dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError) as exc:
        raise TwoZoomCaptureError("Zoom transform matrix must be 3 x 3") from exc
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise TwoZoomCaptureError("Zoom transform matrix is non-finite or singular")
    stored_source_scale = parse_zoom_pixel_scale(
        entry, "source_pixel_scale", summary_path
    )
    stored_target_scale = parse_zoom_pixel_scale(
        entry, "target_pixel_scale", summary_path
    )
    if direction == "source_to_target":
        source_pixel_scale = stored_source_scale
        target_pixel_scale = stored_target_scale
    else:
        source_pixel_scale = stored_target_scale
        target_pixel_scale = stored_source_scale
    return {
        "source_zoom": source,
        "target_zoom": target,
        "calibration_name": entry.get("name"),
        "stored_direction": direction,
        "matrix": matrix,
        "matrix_3x3": matrix.tolist(),
        "source_pixel_scale": source_pixel_scale,
        "target_pixel_scale": target_pixel_scale,
    }


def parse_zoom_pixel_scale(
    entry: dict[str, Any], key: str, summary_path: Path
) -> dict[str, Any]:
    section = entry.get(key)
    if not isinstance(section, dict):
        raise TwoZoomCaptureError(
            f"Zoom transform {entry.get('name')!r} in {summary_path} has no "
            f"{key}; rerun Zoom pixel transform calibration"
        )
    result = dict(section)
    for name in ("horizontal", "vertical"):
        axis = section.get(name)
        try:
            mm_per_pixel = float(axis["mm_per_pixel"])
        except (TypeError, ValueError, KeyError) as exc:
            raise TwoZoomCaptureError(
                f"{key}.{name}.mm_per_pixel is missing or invalid"
            ) from exc
        if not math.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
            raise TwoZoomCaptureError(
                f"{key}.{name}.mm_per_pixel must be finite and above 0"
            )
        result[name] = dict(axis)
        result[name]["mm_per_pixel"] = mm_per_pixel
    try:
        representative = float(section["mm_per_pixel"])
    except (TypeError, ValueError, KeyError) as exc:
        raise TwoZoomCaptureError(f"{key}.mm_per_pixel is missing or invalid") from exc
    if not math.isfinite(representative) or representative <= 0:
        raise TwoZoomCaptureError(f"{key}.mm_per_pixel must be finite and above 0")
    result["mm_per_pixel"] = representative
    return result


def transform_zoom_pixel(np: Any, transform: dict[str, Any], point: Any) -> list[float]:
    source = np.asarray(point, dtype=np.float64).reshape(2)
    homogeneous = np.asarray([source[0], source[1], 1.0], dtype=np.float64)
    mapped = transform["matrix"] @ homogeneous
    denominator = float(mapped[2])
    if not np.all(np.isfinite(mapped)) or abs(denominator) < 1e-12:
        raise TwoZoomCaptureError(
            "Zoom transform produced an invalid homogeneous point"
        )
    result = mapped[:2] / denominator
    if not np.all(np.isfinite(result)):
        raise TwoZoomCaptureError("Zoom transform produced NaN or infinity")
    return [float(result[0]), float(result[1])]


def _mean_edge_length(np: Any, first: Any, second: Any) -> float:
    return 0.5 * (
        float(np.linalg.norm(np.asarray(first[0]) - np.asarray(first[1])))
        + float(np.linalg.norm(np.asarray(second[0]) - np.asarray(second[1])))
    )


def _mean_unit_direction(np: Any, first: Any, second: Any, name: str) -> Any:
    vectors = []
    for start, end in (first, second):
        vector = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(vector))
        if not math.isfinite(length) or length <= 1e-9:
            raise TwoZoomCaptureError(f"{name} edge has zero or invalid length")
        vectors.append(vector / length)
    direction = vectors[0] + vectors[1]
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-9:
        raise TwoZoomCaptureError(f"{name} edges point in opposite directions")
    return direction / length


def calculate_pcb_geometry_from_white_blocks(
    np: Any,
    inner_corners: dict[str, Any],
    zoom_transform: dict[str, Any],
) -> dict[str, Any]:
    try:
        small = {
            name: np.asarray(inner_corners[name], dtype=np.float64).reshape(2)
            for name in BLOCK_NAMES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise TwoZoomCaptureError(
            "white-block detection has incomplete inward-corner coordinates"
        ) from exc
    if not all(np.all(np.isfinite(point)) for point in small.values()):
        raise TwoZoomCaptureError("white-block inward corners contain NaN or infinity")

    small_width_px = _mean_edge_length(
        np,
        (small["top_left"], small["top_right"]),
        (small["bottom_left"], small["bottom_right"]),
    )
    small_height_px = _mean_edge_length(
        np,
        (small["top_left"], small["bottom_left"]),
        (small["top_right"], small["bottom_right"]),
    )
    source_scale = zoom_transform["source_pixel_scale"]
    target_scale = zoom_transform["target_pixel_scale"]
    pcb_width_mm = small_width_px * source_scale["horizontal"]["mm_per_pixel"]
    pcb_height_mm = small_height_px * source_scale["vertical"]["mm_per_pixel"]
    large_width_px = pcb_width_mm / target_scale["horizontal"]["mm_per_pixel"]
    large_height_px = pcb_height_mm / target_scale["vertical"]["mm_per_pixel"]

    mapped = {
        name: np.asarray(
            transform_zoom_pixel(np, zoom_transform, small[name]),
            dtype=np.float64,
        )
        for name in BLOCK_NAMES
    }
    width_direction = _mean_unit_direction(
        np,
        (mapped["top_left"], mapped["top_right"]),
        (mapped["bottom_left"], mapped["bottom_right"]),
        "PCB width",
    )
    height_direction = _mean_unit_direction(
        np,
        (mapped["top_left"], mapped["bottom_left"]),
        (mapped["top_right"], mapped["bottom_right"]),
        "PCB height",
    )
    direction_cross = abs(
        float(
            width_direction[0] * height_direction[1]
            - width_direction[1] * height_direction[0]
        )
    )
    if direction_cross < 0.2:
        raise TwoZoomCaptureError(
            "mapped PCB width and height directions are nearly parallel"
        )
    mapped_width_px = _mean_edge_length(
        np,
        (mapped["top_left"], mapped["top_right"]),
        (mapped["bottom_left"], mapped["bottom_right"]),
    )
    mapped_height_px = _mean_edge_length(
        np,
        (mapped["top_left"], mapped["bottom_left"]),
        (mapped["top_right"], mapped["bottom_right"]),
    )
    return {
        "source": "four_white_block_inward_corners",
        "small_inner_corners_px": {name: small[name].tolist() for name in BLOCK_NAMES},
        "small_size_px": {
            "width": small_width_px,
            "height": small_height_px,
        },
        "physical_size_mm": {
            "width": pcb_width_mm,
            "height": pcb_height_mm,
        },
        "large_size_px": {
            "width": large_width_px,
            "height": large_height_px,
        },
        "large_homography_edge_size_px": {
            "width": mapped_width_px,
            "height": mapped_height_px,
        },
        "large_mapped_inner_corners_px": {
            name: mapped[name].tolist() for name in BLOCK_NAMES
        },
        "large_width_direction": width_direction.tolist(),
        "large_height_direction": height_direction.tolist(),
        "source_pixel_scale": source_scale,
        "target_pixel_scale": target_scale,
    }


def _positive_ray_extent_to_frame(
    point: Any, direction: Any, frame_width: int, frame_height: int
) -> float:
    extents = []
    for coordinate, component, upper in zip(
        point, direction, (float(frame_width), float(frame_height))
    ):
        if component > 1e-12:
            extents.append((upper - coordinate) / component)
        elif component < -1e-12:
            extents.append((0.0 - coordinate) / component)
    positive = [float(value) for value in extents if value >= 0]
    if not positive:
        raise TwoZoomCaptureError("PCB direction does not intersect the camera frame")
    return min(positive)


def _maximum_overlap_step(
    direction: Any,
    frame_width: int,
    frame_height: int,
    overlap_fraction: float,
) -> float:
    retained = 1.0 - float(overlap_fraction)
    limits = []
    for component, size in zip(direction, (frame_width, frame_height)):
        if abs(float(component)) > 1e-12:
            limits.append(float(size) * retained / abs(float(component)))
    if not limits or min(limits) <= 0:
        raise TwoZoomCaptureError("cannot calculate a positive grid step")
    return min(limits)


def _automatic_axis_grid(
    span_px: float,
    margin_px: float,
    first_frame_coverage_px: float,
    maximum_step_px: float,
) -> tuple[int, float, float]:
    required_travel = max(
        0.0, float(span_px) + float(margin_px) - float(first_frame_coverage_px)
    )
    if required_travel <= 1e-9:
        return 1, 0.0, 0.0
    count = int(math.ceil(required_travel / maximum_step_px - 1e-12)) + 1
    spacing = required_travel / (count - 1)
    return count, spacing, required_travel


def build_automatic_capture_grid(
    np: Any,
    args: SimpleNamespace,
    pcb_geometry: dict[str, Any],
    stage_calibration: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = np.asarray(args.large_target_pixel, dtype=np.float64).reshape(2)
    width_direction = np.asarray(
        pcb_geometry["large_width_direction"], dtype=np.float64
    ).reshape(2)
    height_direction = np.asarray(
        pcb_geometry["large_height_direction"], dtype=np.float64
    ).reshape(2)
    large_width_px = float(pcb_geometry["large_size_px"]["width"])
    large_height_px = float(pcb_geometry["large_size_px"]["height"])
    target_scale = pcb_geometry["target_pixel_scale"]
    margin_mm = float(args.capture_margin_mm)
    width_margin_px = margin_mm / target_scale["horizontal"]["mm_per_pixel"]
    height_margin_px = margin_mm / target_scale["vertical"]["mm_per_pixel"]
    width_first_coverage = _positive_ray_extent_to_frame(
        target, width_direction, frame_width, frame_height
    )
    height_first_coverage = _positive_ray_extent_to_frame(
        target, height_direction, frame_width, frame_height
    )
    width_maximum_step = _maximum_overlap_step(
        width_direction,
        frame_width,
        frame_height,
        float(args.overlap_fraction),
    )
    height_maximum_step = _maximum_overlap_step(
        height_direction,
        frame_width,
        frame_height,
        float(args.overlap_fraction),
    )
    columns, spacing_width_px, width_travel_px = _automatic_axis_grid(
        large_width_px,
        width_margin_px,
        width_first_coverage,
        width_maximum_step,
    )
    rows, spacing_height_px, height_travel_px = _automatic_axis_grid(
        large_height_px,
        height_margin_px,
        height_first_coverage,
        height_maximum_step,
    )
    pixel_to_stage = np.asarray(
        stage_calibration["pixel_to_stage"], dtype=np.float64
    ).reshape(2, 2)
    cells = []
    for row in range(rows):
        column_order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        for column in column_order:
            width_offset_px = column * spacing_width_px
            height_offset_px = row * spacing_height_px
            pixel_offset = (
                width_direction * width_offset_px + height_direction * height_offset_px
            )
            stage_offset = pixel_to_stage @ (-pixel_offset)
            cells.append(
                {
                    "row": row,
                    "col": column,
                    "board_offset_mm": [
                        width_offset_px * target_scale["horizontal"]["mm_per_pixel"],
                        height_offset_px * target_scale["vertical"]["mm_per_pixel"],
                    ],
                    "pixel_offset_px": pixel_offset.tolist(),
                    "stage_offset_mm": stage_offset.tolist(),
                }
            )

    horizontal_mm_per_pixel = target_scale["horizontal"]["mm_per_pixel"]
    vertical_mm_per_pixel = target_scale["vertical"]["mm_per_pixel"]
    grid = {
        "size_source": "four_white_block_inward_corners",
        "pcb_width_mm": float(pcb_geometry["physical_size_mm"]["width"]),
        "pcb_height_mm": float(pcb_geometry["physical_size_mm"]["height"]),
        "pcb_width_px_at_large_zoom": large_width_px,
        "pcb_height_px_at_large_zoom": large_height_px,
        "capture_margin_mm": margin_mm,
        "overlap_fraction": float(args.overlap_fraction),
        "fov_width_mm": width_maximum_step
        / (1.0 - float(args.overlap_fraction))
        * horizontal_mm_per_pixel,
        "fov_height_mm": height_maximum_step
        / (1.0 - float(args.overlap_fraction))
        * vertical_mm_per_pixel,
        "columns": columns,
        "rows": rows,
        "spacing_x_mm": spacing_width_px * horizontal_mm_per_pixel,
        "spacing_y_mm": spacing_height_px * vertical_mm_per_pixel,
        "spacing_width_px": spacing_width_px,
        "spacing_height_px": spacing_height_px,
        "first_frame_positive_coverage_width_px": width_first_coverage,
        "first_frame_positive_coverage_height_px": height_first_coverage,
        "maximum_spacing_width_px": width_maximum_step,
        "maximum_spacing_height_px": height_maximum_step,
        "capture_target_pixel": target.tolist(),
        "width_direction": width_direction.tolist(),
        "height_direction": height_direction.tolist(),
        "coverage_bounds_board_mm": {
            "x_min": 0.0,
            "x_max": float(pcb_geometry["physical_size_mm"]["width"]) + margin_mm,
            "y_min": 0.0,
            "y_max": float(pcb_geometry["physical_size_mm"]["height"]) + margin_mm,
        },
        "minimum_overlap_x": (
            1.0 - spacing_width_px / width_maximum_step * (1.0 - args.overlap_fraction)
            if columns > 1
            else 0.0
        ),
        "minimum_overlap_y": (
            1.0
            - spacing_height_px / height_maximum_step * (1.0 - args.overlap_fraction)
            if rows > 1
            else 0.0
        ),
        "capture_order": "serpentine: left-to-right, one row down, right-to-left",
        "frames_inside_pcb": bool(margin_mm <= 1e-9),
        "required_width_travel_px": width_travel_px,
        "required_height_travel_px": height_travel_px,
    }
    return cells, grid


def order_block_corners(np: Any, points: Any) -> Any:
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = points[:, 1] - points[:, 0]
    indices = (
        int(np.argmin(coordinate_sum)),
        int(np.argmin(coordinate_difference)),
        int(np.argmax(coordinate_sum)),
        int(np.argmax(coordinate_difference)),
    )
    if len(set(indices)) != 4:
        by_y = points[np.argsort(points[:, 1])]
        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bottom = by_y[2:][np.argsort(by_y[2:, 0])]
        return np.asarray([top[0], top[1], bottom[1], bottom[0]])
    return points[list(indices)]


def _detect_four_white_blocks_at_threshold(
    cv2: Any,
    np: Any,
    image: Any,
    args: SimpleNamespace,
    threshold: int,
) -> tuple[dict[str, Any], Any]:
    if image is None or getattr(image, "size", 0) == 0:
        raise TwoZoomCaptureError("white-block detection received an empty image")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel_size = int(args.white_block_morph_kernel)
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # CCOMP keeps foreground components enclosed by the white fixture border.
    # Hole boundaries are not candidates themselves, and top-level contours
    # with a substantial enclosed hole are rejected as PCB mounting features.
    contour_result = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[-2]
    hierarchy = contour_result[-1]
    hierarchy_items = hierarchy[0] if hierarchy is not None else None
    image_area = float(width * height)
    maximum_area = image_area * float(args.white_block_max_area_fraction)
    candidates: list[dict[str, Any]] = []
    rejected_hollow_candidate_count = 0
    for contour_index, contour in enumerate(contours):
        if hierarchy_items is not None and int(hierarchy_items[contour_index][3]) != -1:
            continue
        area = float(cv2.contourArea(contour))
        if not float(args.white_block_min_area_px) <= area <= maximum_area:
            continue
        hole_area = 0.0
        child_index = (
            int(hierarchy_items[contour_index][2])
            if hierarchy_items is not None
            else -1
        )
        while child_index != -1:
            hole_area += float(cv2.contourArea(contours[child_index]))
            child_index = int(hierarchy_items[child_index][0])
        hole_fraction = hole_area / area
        if hole_fraction > MAX_WHITE_BLOCK_HOLE_FRACTION:
            rejected_hollow_candidate_count += 1
            continue
        rectangle = cv2.minAreaRect(contour)
        rectangle_width, rectangle_height = map(float, rectangle[1])
        if rectangle_width <= 1e-6 or rectangle_height <= 1e-6:
            continue
        square_ratio = min(rectangle_width, rectangle_height) / max(
            rectangle_width, rectangle_height
        )
        rectangle_area = rectangle_width * rectangle_height
        fill_ratio = area / rectangle_area
        if square_ratio < float(args.white_block_min_square_ratio):
            continue
        if fill_ratio < float(args.white_block_min_fill_ratio):
            continue
        corners = order_block_corners(np, cv2.boxPoints(rectangle))
        center = corners.mean(axis=0)
        candidates.append(
            {
                "area_px": area,
                "square_ratio": float(square_ratio),
                "fill_ratio": float(fill_ratio),
                "hole_fraction": float(hole_fraction),
                "center": center,
                "corners": corners,
            }
        )

    image_center = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    corner_targets = {
        "top_left": np.asarray([0.0, 0.0]),
        "top_right": np.asarray([width - 1.0, 0.0]),
        "bottom_right": np.asarray([width - 1.0, height - 1.0]),
        "bottom_left": np.asarray([0.0, height - 1.0]),
    }
    if len(candidates) < 4:
        raise TwoZoomCaptureError(
            "white-block detection found only {} accepted square candidate(s); "
            "four are required. Adjust threshold/area/square settings and make "
            "sure all four blocks are visible".format(len(candidates))
        )

    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float64)
    shortlist = sorted(
        candidates,
        key=lambda item: min(
            float(np.linalg.norm((item["center"] - target) / scale))
            for target in corner_targets.values()
        ),
    )[:16]
    best_assignment = None
    best_score = math.inf
    for assigned in permutations(shortlist, 4):
        top_left, top_right, bottom_right, bottom_left = assigned
        if not (
            top_left["center"][0] < top_right["center"][0]
            and bottom_left["center"][0] < bottom_right["center"][0]
            and top_left["center"][1] < bottom_left["center"][1]
            and top_right["center"][1] < bottom_right["center"][1]
        ):
            continue
        corner_score = sum(
            float(np.linalg.norm((item["center"] - corner_targets[name]) / scale))
            for name, item in zip(BLOCK_NAMES, assigned)
        )
        areas = np.asarray([item["area_px"] for item in assigned])
        area_variation = float(np.std(areas) / np.mean(areas))
        row_column_skew = (
            abs(top_left["center"][1] - top_right["center"][1]) / scale[1]
            + abs(bottom_left["center"][1] - bottom_right["center"][1]) / scale[1]
            + abs(top_left["center"][0] - bottom_left["center"][0]) / scale[0]
            + abs(top_right["center"][0] - bottom_right["center"][0]) / scale[0]
        )
        score = corner_score + area_variation + 0.25 * float(row_column_skew)
        if score < best_score:
            best_score = score
            best_assignment = assigned
    if best_assignment is None:
        raise TwoZoomCaptureError(
            "white-block candidates do not form a valid four-corner layout; "
            "make sure all four marker blocks are visible"
        )
    selected = dict(zip(BLOCK_NAMES, best_assignment))

    block_centers = np.asarray([selected[name]["center"] for name in BLOCK_NAMES])
    blocks_center = block_centers.mean(axis=0)
    blocks: dict[str, Any] = {}
    corner_labels = ("top_left", "top_right", "bottom_right", "bottom_left")
    for name in BLOCK_NAMES:
        item = selected[name]
        blocks[name] = {
            "center_px": [float(value) for value in item["center"]],
            "corners_px": {
                label: [float(value) for value in point]
                for label, point in zip(corner_labels, item["corners"])
            },
            "area_px": item["area_px"],
            "square_ratio": item["square_ratio"],
            "fill_ratio": item["fill_ratio"],
            "hole_fraction": item["hole_fraction"],
        }
    inner_corners = {
        "top_left": blocks["top_left"]["corners_px"]["bottom_right"],
        "top_right": blocks["top_right"]["corners_px"]["bottom_left"],
        "bottom_right": blocks["bottom_right"]["corners_px"]["top_left"],
        "bottom_left": blocks["bottom_left"]["corners_px"]["top_right"],
    }
    inner_corners_center = np.asarray(
        [inner_corners[name] for name in BLOCK_NAMES], dtype=np.float64
    ).mean(axis=0)
    centre_delta = image_center - inner_corners_center
    return {
        "image_size_px": [int(width), int(height)],
        "threshold": int(threshold),
        "standalone_candidate_count": len(candidates),
        "rejected_hollow_candidate_count": rejected_hollow_candidate_count,
        "maximum_marker_hole_fraction": MAX_WHITE_BLOCK_HOLE_FRACTION,
        "accepted_candidate_count": len(blocks),
        "blocks": blocks,
        "blocks_center_px": [float(value) for value in blocks_center],
        "inner_corners_center_px": [float(value) for value in inner_corners_center],
        "alignment_center_px": [float(value) for value in inner_corners_center],
        "alignment_center_definition": "mean_of_four_inward_block_corners",
        "image_center_px": [float(value) for value in image_center],
        "center_to_image_delta_px": [float(value) for value in centre_delta],
        "center_to_image_distance_px": float(np.linalg.norm(centre_delta)),
        "inner_corners_px": inner_corners,
        "top_left_block_bottom_right_px": inner_corners["top_left"],
    }, mask


def white_block_layout_quality(np: Any, result: dict[str, Any]) -> dict[str, Any]:
    centers = np.asarray(
        [result["blocks"][name]["center_px"] for name in BLOCK_NAMES],
        dtype=np.float64,
    )
    image_width, image_height = result["image_size_px"]
    top_length = float(np.linalg.norm(centers[1] - centers[0]))
    right_length = float(np.linalg.norm(centers[2] - centers[1]))
    bottom_length = float(np.linalg.norm(centers[2] - centers[3]))
    left_length = float(np.linalg.norm(centers[3] - centers[0]))

    def opposite_ratio(first: float, second: float) -> float:
        maximum = max(first, second)
        return min(first, second) / maximum if maximum > 1e-9 else 0.0

    width_edge_ratio = opposite_ratio(top_length, bottom_length)
    height_edge_ratio = opposite_ratio(left_length, right_length)
    x = centers[:, 0]
    y = centers[:, 1]
    polygon_area = 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )
    layout_area_fraction = polygon_area / float(image_width * image_height)
    cross_products = []
    for index in range(4):
        first = centers[(index + 1) % 4] - centers[index]
        second = centers[(index + 2) % 4] - centers[(index + 1) % 4]
        cross_products.append(float(first[0] * second[1] - first[1] * second[0]))
    convex = all(value > 1e-6 for value in cross_products) or all(
        value < -1e-6 for value in cross_products
    )
    plausible = bool(
        convex
        and layout_area_fraction >= MIN_WHITE_BLOCK_LAYOUT_AREA_FRACTION
        and width_edge_ratio >= MIN_WHITE_BLOCK_OPPOSITE_EDGE_RATIO
        and height_edge_ratio >= MIN_WHITE_BLOCK_OPPOSITE_EDGE_RATIO
    )
    return {
        "plausible": plausible,
        "convex": convex,
        "layout_area_px": polygon_area,
        "layout_area_fraction": layout_area_fraction,
        "minimum_layout_area_fraction": MIN_WHITE_BLOCK_LAYOUT_AREA_FRACTION,
        "top_edge_px": top_length,
        "bottom_edge_px": bottom_length,
        "left_edge_px": left_length,
        "right_edge_px": right_length,
        "width_opposite_edge_ratio": width_edge_ratio,
        "height_opposite_edge_ratio": height_edge_ratio,
        "minimum_opposite_edge_ratio": MIN_WHITE_BLOCK_OPPOSITE_EDGE_RATIO,
    }


def detect_four_white_blocks(
    cv2: Any, np: Any, image: Any, args: SimpleNamespace
) -> tuple[dict[str, Any], Any]:
    requested_threshold = int(args.white_block_threshold)
    minimum_threshold = max(1, requested_threshold - 60)
    thresholds = list(range(requested_threshold, minimum_threshold - 1, -10))
    if thresholds[-1] != minimum_threshold:
        thresholds.append(minimum_threshold)

    attempts: list[dict[str, Any]] = []
    for threshold in thresholds:
        try:
            result, mask = _detect_four_white_blocks_at_threshold(
                cv2, np, image, args, threshold
            )
        except TwoZoomCaptureError as exc:
            attempts.append(
                {"threshold": threshold, "status": "failed", "error": str(exc)}
            )
            continue
        quality = white_block_layout_quality(np, result)
        if quality["plausible"]:
            attempts.append(
                {"threshold": threshold, "status": "accepted", "quality": quality}
            )
            result["requested_threshold"] = requested_threshold
            result["threshold_fallback_used"] = threshold != requested_threshold
            result["threshold_attempts"] = attempts
            result["layout_quality"] = quality
            return result, mask
        attempts.append(
            {"threshold": threshold, "status": "rejected", "quality": quality}
        )

    attempted_values = ", ".join(str(item["threshold"]) for item in attempts)
    raise TwoZoomCaptureError(
        "white-block candidates did not form four solid outer marker squares "
        f"at thresholds [{attempted_values}]; adjust lighting or the white-block "
        "threshold and make sure all four blocks are fully visible"
    )


def draw_white_block_detection(
    cv2: Any, np: Any, image: Any, result: dict[str, Any]
) -> Any:
    annotated = image.copy()
    colors = {
        "top_left": (255, 120, 30),
        "top_right": (30, 210, 255),
        "bottom_right": (40, 220, 80),
        "bottom_left": (220, 80, 220),
    }
    for name in BLOCK_NAMES:
        block = result["blocks"][name]
        points = np.rint(list(block["corners_px"].values())).astype(np.int32)
        cv2.polylines(annotated, [points], True, colors[name], 3, cv2.LINE_AA)
        center = tuple(map(int, np.rint(block["center_px"])))
        cv2.circle(annotated, center, 7, colors[name], -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            name,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colors[name],
            2,
            cv2.LINE_AA,
        )
        inward_corner = tuple(map(int, np.rint(result["inner_corners_px"][name])))
        cv2.drawMarker(
            annotated, inward_corner, colors[name], cv2.MARKER_TILTED_CROSS, 18, 2
        )
    alignment_center = tuple(map(int, np.rint(result["alignment_center_px"])))
    image_center = tuple(map(int, np.rint(result["image_center_px"])))
    cv2.drawMarker(annotated, alignment_center, (255, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.drawMarker(annotated, image_center, (255, 255, 0), cv2.MARKER_CROSS, 30, 3)
    cv2.line(annotated, alignment_center, image_center, (255, 255, 0), 2, cv2.LINE_AA)
    return annotated


def crop_global_image(
    np: Any, image: Any, detection: dict[str, Any], padding_px: int
) -> tuple[Any, list[int]]:
    del np
    height, width = image.shape[:2]
    corners = detection["inner_corners_px"]
    left = max(corners["top_left"][0], corners["bottom_left"][0])
    right = min(corners["top_right"][0], corners["bottom_right"][0])
    top = max(corners["top_left"][1], corners["top_right"][1])
    bottom = min(corners["bottom_left"][1], corners["bottom_right"][1])
    # Positive values retain pixels outside the marker corners; negative values
    # deliberately contract the crop further into the PCB on all four sides.
    padding = int(padding_px)
    x0 = max(0, int(math.floor(left)) - padding)
    y0 = max(0, int(math.floor(top)) - padding)
    x1 = min(width, int(math.ceil(right)) + padding + 1)
    y1 = min(height, int(math.ceil(bottom)) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        raise TwoZoomCaptureError(
            "the four inward white-block corners do not define a valid crop: "
            f"[{x0}, {y0}, {x1}, {y1}]"
        )
    cropped = image[y0:y1, x0:x1].copy()
    if cropped.size == 0:
        raise TwoZoomCaptureError("the calculated global-image crop is empty")
    return cropped, [x0, y0, x1, y1]


def calculate_pixel_alignment_plan(
    np: Any,
    source_pixel: Any,
    target_pixel: Any,
    stage_calibration: dict[str, Any],
) -> dict[str, Any]:
    source = np.asarray(source_pixel, dtype=np.float64).reshape(2)
    target = np.asarray(target_pixel, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise TwoZoomCaptureError("alignment pixels contain NaN or infinity")
    pixel_delta = target - source
    pixel_to_stage = np.asarray(
        stage_calibration["pixel_to_stage"], dtype=np.float64
    ).reshape(2, 2)
    stage_to_pixel = np.asarray(
        stage_calibration["stage_to_pixel"], dtype=np.float64
    ).reshape(2, 2)
    stage_delta = pixel_to_stage @ pixel_delta
    reproduced = stage_to_pixel @ stage_delta
    conversion_error = float(np.linalg.norm(reproduced - pixel_delta))
    if conversion_error > 1e-6:
        raise TwoZoomCaptureError(
            f"stage pixel conversion is inconsistent; error {conversion_error:.6g} px"
        )
    return {
        "source_pixel": source.tolist(),
        "target_pixel": target.tolist(),
        "requested_pixel_delta_px": pixel_delta.tolist(),
        "pixel_distance_px": float(np.linalg.norm(pixel_delta)),
        "requested_relative_stage_delta_mm": stage_delta.tolist(),
        "stage_distance_mm": float(np.linalg.norm(stage_delta)),
        "reproduced_pixel_delta_px": reproduced.tolist(),
        "conversion_error_px": conversion_error,
    }


def validate_target_pixel(
    point: Any, width: int, height: int, allow_outside: bool, name: str
) -> None:
    x, y = finite_pair(point, name)
    if not allow_outside and not (0 <= x < width and 0 <= y < height):
        raise TwoZoomCaptureError(
            f"{name}=({x:.3f}, {y:.3f}) is outside {width} x {height} image"
        )


def execute_relative_stage_move(
    np: Any, args: SimpleNamespace, requested_delta: Any, label: str
) -> dict[str, Any]:
    try:
        from motion_controller import MODE_ABSOLUTE, XYMotionController
    except Exception as exc:
        raise TwoZoomCaptureError(f"cannot import motion controller: {exc}") from exc

    delta = np.asarray(requested_delta, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(delta)):
        raise TwoZoomCaptureError("requested stage delta contains NaN or infinity")
    stage = None
    try:
        stage = XYMotionController(
            args.port,
            baudrate=args.baudrate,
            x_lead_mm_per_rev=args.x_lead_mm_per_rev,
            y_lead_mm_per_rev=args.y_lead_mm_per_rev,
            x_slave_address=args.x_slave_address,
            y_slave_address=args.y_slave_address,
        )
        initial = capture_pcb.read_stage_position(np, stage, f"before {label}")
        target = initial + delta
        if float(np.linalg.norm(delta)) <= 1e-12:
            return {
                "before_position_mm": initial.tolist(),
                "requested_position_mm": target.tolist(),
                "actual_position_mm": initial.tolist(),
                "error_vector_mm": [0.0, 0.0],
                "error_distance_mm": 0.0,
                "travel_distance_mm": 0.0,
            }
        return capture_pcb.move_stage_absolute(
            np,
            stage,
            target,
            args,
            MODE_ABSOLUTE,
            label,
            float(args.max_move_mm),
        )
    except TwoZoomCaptureError:
        raise
    except Exception as exc:
        raise TwoZoomCaptureError(f"{label} failed: {exc}") from exc
    finally:
        if stage is not None:
            try:
                stage.close()
            except Exception:
                pass


def execute_absolute_stage_move(
    np: Any,
    args: SimpleNamespace,
    requested_position: Any,
    label: str,
    maximum_distance_mm: float,
) -> dict[str, Any]:
    try:
        from motion_controller import MODE_ABSOLUTE, XYMotionController
    except Exception as exc:
        raise TwoZoomCaptureError(f"cannot import motion controller: {exc}") from exc

    target = np.asarray(requested_position, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(target)):
        raise TwoZoomCaptureError("requested stage position contains NaN or infinity")
    stage = None
    try:
        stage = XYMotionController(
            args.port,
            baudrate=args.baudrate,
            x_lead_mm_per_rev=args.x_lead_mm_per_rev,
            y_lead_mm_per_rev=args.y_lead_mm_per_rev,
            x_slave_address=args.x_slave_address,
            y_slave_address=args.y_slave_address,
        )
        initial = capture_pcb.read_stage_position(np, stage, f"before {label}")
        if float(np.linalg.norm(target - initial)) <= 1e-12:
            return {
                "before_position_mm": initial.tolist(),
                "requested_position_mm": target.tolist(),
                "actual_position_mm": initial.tolist(),
                "error_vector_mm": [0.0, 0.0],
                "error_distance_mm": 0.0,
                "travel_distance_mm": 0.0,
            }
        return capture_pcb.move_stage_absolute(
            np,
            stage,
            target,
            args,
            MODE_ABSOLUTE,
            label,
            float(maximum_distance_mm),
        )
    except TwoZoomCaptureError:
        raise
    except Exception as exc:
        raise TwoZoomCaptureError(f"{label} failed: {exc}") from exc
    finally:
        if stage is not None:
            try:
                stage.close()
            except Exception:
                pass


def capture_small_global_image(
    cv2: Any,
    np: Any,
    args: SimpleNamespace,
    intrinsics: dict[str, Any],
    stage_calibration: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    images_dir = run_dir / "small_zoom_image"
    images_dir.mkdir()
    camera = None
    try:
        camera, camera_record = capture_pcb.open_camera(cv2, args)
        before_raw = capture_pcb.capture_frame(camera, args, "small zoom alignment")
        height, width = before_raw.shape[:2]
        map1, map2 = capture_pcb.build_undistort_maps(cv2, intrinsics, width, height)
        before = cv2.remap(before_raw, map1, map2, cv2.INTER_LINEAR)
        before_path = images_dir / "small_alignment_before_undistorted.png"
        capture_pcb.write_png(cv2, before_path, before)
        before_raw_path = None
        if args.save_raw:
            before_raw_path = images_dir / "small_alignment_before_raw.png"
            capture_pcb.write_png(cv2, before_raw_path, before_raw)
        before_detection, before_mask = detect_four_white_blocks(cv2, np, before, args)
        print(
            "Initial white blocks: threshold {}; rejected {} hollow candidates; "
            "layout area {:.1%}".format(
                before_detection["threshold"],
                before_detection["rejected_hollow_candidate_count"],
                before_detection["layout_quality"]["layout_area_fraction"],
            )
        )
        before_annotated = draw_white_block_detection(cv2, np, before, before_detection)
        before_mask_path = images_dir / "small_alignment_before_mask.png"
        before_annotated_path = images_dir / "small_alignment_before_annotated.png"
        capture_pcb.write_png(cv2, before_mask_path, before_mask)
        capture_pcb.write_png(cv2, before_annotated_path, before_annotated)

        alignment_plan = calculate_pixel_alignment_plan(
            np,
            before_detection["alignment_center_px"],
            before_detection["image_center_px"],
            stage_calibration,
        )
        print(
            "Small zoom inward-corner centre: ({:.2f}, {:.2f}) px; image centre: "
            "({:.2f}, {:.2f}) px; distance {:.2f} px".format(
                *before_detection["alignment_center_px"],
                *before_detection["image_center_px"],
                before_detection["center_to_image_distance_px"],
            )
        )
        print(
            "Small zoom relative stage move: ({:.6f}, {:.6f}) mm".format(
                *alignment_plan["requested_relative_stage_delta_mm"]
            )
        )
        alignment_motion = execute_relative_stage_move(
            np,
            args,
            alignment_plan["requested_relative_stage_delta_mm"],
            "small zoom white-block centring",
        )

        final_raw = capture_pcb.capture_frame(camera, args, "small zoom global image")
        final_height, final_width = final_raw.shape[:2]
        if (final_width, final_height) != (width, height):
            map1, map2 = capture_pcb.build_undistort_maps(
                cv2, intrinsics, final_width, final_height
            )
        final = cv2.remap(final_raw, map1, map2, cv2.INTER_LINEAR)
        final_path = images_dir / "small_undistorted.png"
        capture_pcb.write_png(cv2, final_path, final)
        final_raw_path = None
        if args.save_raw:
            final_raw_path = images_dir / "small_final_raw.png"
            capture_pcb.write_png(cv2, final_raw_path, final_raw)
        final_detection, final_mask = detect_four_white_blocks(cv2, np, final, args)
        print(
            "Final white blocks: threshold {}; rejected {} hollow candidates; "
            "layout area {:.1%}".format(
                final_detection["threshold"],
                final_detection["rejected_hollow_candidate_count"],
                final_detection["layout_quality"]["layout_area_fraction"],
            )
        )
        final_annotated = draw_white_block_detection(cv2, np, final, final_detection)
        cropped, crop_xyxy = crop_global_image(
            np, final, final_detection, int(args.global_crop_padding_px)
        )

        cropped_path = images_dir / "small_global_cropped.png"
        final_mask_path = images_dir / "small_final_mask.png"
        final_annotated_path = images_dir / "small_final_annotated.png"
        capture_pcb.write_png(cv2, cropped_path, cropped)
        capture_pcb.write_png(cv2, final_mask_path, final_mask)
        capture_pcb.write_png(cv2, final_annotated_path, final_annotated)

        print(f"Small zoom full image saved: {final_path}")
        print(f"Small zoom cropped global image saved: {cropped_path}")
        return {
            "camera": camera_record,
            "image_size_px": [int(final_width), int(final_height)],
            "alignment": {
                "before_detection": before_detection,
                "plan": alignment_plan,
                "motion": alignment_motion,
                "final_detection": final_detection,
                "residual_center_distance_px": final_detection[
                    "center_to_image_distance_px"
                ],
            },
            "top_left_block_bottom_right_px": final_detection[
                "top_left_block_bottom_right_px"
            ],
            "crop_xyxy_px": crop_xyxy,
            "before_raw_image": (
                str(before_raw_path) if before_raw_path is not None else None
            ),
            "before_undistorted_image": str(before_path),
            "before_mask_image": str(before_mask_path),
            "before_annotated_image": str(before_annotated_path),
            "raw_image": str(final_raw_path) if final_raw_path is not None else None,
            "undistorted_image": str(final_path),
            "cropped_global_image": str(cropped_path),
            "final_mask_image": str(final_mask_path),
            "final_annotated_image": str(final_annotated_path),
            "images_directory": str(images_dir),
        }
    finally:
        if camera is not None:
            camera.release()


def stage_axis_grid_calibration(stage_calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "board_to_pixel": stage_calibration["stage_to_pixel"],
        "source_file": stage_calibration["source_file"],
        "source_name": "stage_axes_used_as_pcb_axes",
        "source_label": "Automatic white-block geometry",
    }


def run(args: SimpleNamespace) -> int:
    np = capture_pcb.load_numpy()
    cv2, _ = capture_pcb.camera_tools.load_image_tools(TwoZoomCaptureError)
    summary_path = calibration_path(args)
    document = capture_pcb.load_calibration_summary_document(summary_path)

    runtime = build_capture_args(args, document, summary_path)
    small_phase = phase_values(runtime, "small")
    large_phase = phase_values(runtime, "large")
    small_calibrations = load_visual_phase_calibrations(
        np, summary_path, "small", small_phase["zoom"], document
    )
    large_calibrations = load_visual_phase_calibrations(
        np, summary_path, "large", large_phase["zoom"], document
    )
    zoom_transform = load_zoom_transform(
        np, document, summary_path, small_phase["zoom"], large_phase["zoom"]
    )
    capture_pcb.validate_args(runtime, require_pcb_dimensions=False)

    run_dir = capture_pcb.create_run_directory(runtime.output_dir)
    report_path = run_dir / "pcb_two_zoom_capture.json"
    log: dict[str, Any] = {
        "schema_version": 2,
        "type": "pcb_two_zoom_capture",
        "created_at_utc": capture_pcb.utc_timestamp(),
        "status": "starting",
        "calibration_summary": str(summary_path),
        "settings": dict(vars(args)),
        "zoom_pixel_transform": {
            key: value for key, value in zoom_transform.items() if key != "matrix"
        },
        "small": {
            "lens_zoom": small_phase["zoom"],
            "lens_focus": small_phase["focus"],
            "lens_iris": small_phase["iris"],
            "alignment_target": "image_center",
            "camera_intrinsics": capture_pcb.intrinsics_record(
                small_calibrations["intrinsics"]
            ),
        },
        "large": {
            "lens_zoom": large_phase["zoom"],
            "lens_focus": large_phase["focus"],
            "lens_iris": large_phase["iris"],
            "target_pixel": runtime.large_target_pixel,
            "camera_intrinsics": capture_pcb.intrinsics_record(
                large_calibrations["intrinsics"]
            ),
        },
    }
    capture_pcb.write_json(report_path, log)
    print(f"Output: {run_dir}")

    try:
        print("=== Small zoom white-block alignment and global image ===")
        log["small"]["lens"] = move_lens_to_phase(runtime, small_phase)
        log["small"]["capture"] = capture_small_global_image(
            cv2,
            np,
            runtime,
            small_calibrations["intrinsics"],
            small_calibrations["stage"],
            run_dir,
        )
        final_detection = log["small"]["capture"]["alignment"]["final_detection"]
        pcb_geometry = calculate_pcb_geometry_from_white_blocks(
            np,
            final_detection["inner_corners_px"],
            zoom_transform,
        )
        runtime.pcb_width_mm = pcb_geometry["physical_size_mm"]["width"]
        runtime.pcb_height_mm = pcb_geometry["physical_size_mm"]["height"]
        log["pcb_geometry"] = pcb_geometry
        print(
            "PCB size from four white blocks: {:.2f} x {:.2f} px at small "
            "zoom = {:.3f} x {:.3f} mm = {:.2f} x {:.2f} px at large zoom".format(
                pcb_geometry["small_size_px"]["width"],
                pcb_geometry["small_size_px"]["height"],
                pcb_geometry["physical_size_mm"]["width"],
                pcb_geometry["physical_size_mm"]["height"],
                pcb_geometry["large_size_px"]["width"],
                pcb_geometry["large_size_px"]["height"],
            )
        )
        log["status"] = "small_completed"
        capture_pcb.write_json(report_path, log)

        print("=== Large zoom anchor alignment and serpentine mosaic ===")
        actual_frame_width, actual_frame_height = log["small"]["capture"][
            "image_size_px"
        ]
        small_anchor = pcb_geometry["small_inner_corners_px"]["top_left"]
        large_source_pixel = pcb_geometry["large_mapped_inner_corners_px"]["top_left"]
        validate_target_pixel(
            runtime.large_target_pixel,
            int(actual_frame_width),
            int(actual_frame_height),
            bool(runtime.allow_outside_image),
            "large_target_pixel",
        )
        large_plan = calculate_pixel_alignment_plan(
            np,
            large_source_pixel,
            runtime.large_target_pixel,
            large_calibrations["stage"],
        )
        log["large"]["source_small_pixel"] = small_anchor
        log["large"]["mapped_source_pixel"] = large_source_pixel
        log["large"]["alignment_plan"] = large_plan
        print(
            "Top-left block bottom-right: small=({:.2f}, {:.2f}) px -> "
            "large=({:.2f}, {:.2f}) px; target=({:.2f}, {:.2f}) px".format(
                *small_anchor, *large_source_pixel, *runtime.large_target_pixel
            )
        )
        log["large"]["lens"] = move_lens_to_phase(runtime, large_phase)
        log["large"]["alignment_motion"] = execute_relative_stage_move(
            np,
            runtime,
            large_plan["requested_relative_stage_delta_mm"],
            "large zoom mapped-anchor alignment",
        )
        capture_cells, capture_grid = build_automatic_capture_grid(
            np,
            runtime,
            pcb_geometry,
            large_calibrations["stage"],
            int(actual_frame_width),
            int(actual_frame_height),
        )
        log["large"]["automatic_grid"] = capture_grid
        print(
            "Automatic large-zoom grid: {} columns x {} rows = {} images".format(
                capture_grid["columns"],
                capture_grid["rows"],
                len(capture_cells),
            )
        )
        grid_calibration = stage_axis_grid_calibration(large_calibrations["stage"])
        log["large"]["mosaic"] = capture_pcb.capture_mosaic_from_current_position(
            np,
            cv2,
            runtime,
            grid_calibration,
            large_calibrations["stage"],
            large_calibrations["intrinsics"],
            run_dir,
            capture_cells=capture_cells,
            capture_grid=capture_grid,
        )
        capture_pcb.write_json(run_dir / "pcb_mosaic.json", log["large"]["mosaic"])

        print("=== Restore small zoom photo position ===")
        small_photo_stage_position = log["small"]["capture"]["alignment"]["motion"][
            "actual_position_mm"
        ]
        log["return_to_small_zoom"] = {
            "target_lens": small_phase,
            "target_stage_position_mm": small_photo_stage_position,
        }
        log["return_to_small_zoom"]["lens"] = move_lens_to_phase(runtime, small_phase)
        log["return_to_small_zoom"]["stage"] = execute_absolute_stage_move(
            np,
            runtime,
            small_photo_stage_position,
            "restore small zoom photo position",
            float(runtime.max_return_mm),
        )
        log["return_to_small_zoom"]["restored"] = True
        print(
            "Restored small zoom photo position: zoom={} focus={} iris={}; "
            "stage=({:.6f}, {:.6f}) mm".format(
                small_phase["zoom"],
                small_phase["focus"],
                small_phase["iris"],
                *log["return_to_small_zoom"]["stage"]["actual_position_mm"],
            )
        )
        log["status"] = "completed"
        log["completed_at_utc"] = capture_pcb.utc_timestamp()
        capture_pcb.write_json(report_path, log)
        print("PCB two-zoom capture complete")
        print(f"Report: {report_path}")
        return 0
    except KeyboardInterrupt:
        log["status"] = "interrupted"
        log["failed_at_utc"] = capture_pcb.utc_timestamp()
        capture_pcb.write_json(report_path, log)
        raise
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = str(exc)
        log["failed_at_utc"] = capture_pcb.utc_timestamp()
        capture_pcb.write_json(report_path, log)
        if isinstance(exc, capture_pcb.PcbMosaicError):
            raise
        raise TwoZoomCaptureError(f"PCB two-zoom capture failed: {exc}") from exc


def main() -> int:
    try:
        return run(load_args())
    except capture_pcb.PcbMosaicError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; stage stop was requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
