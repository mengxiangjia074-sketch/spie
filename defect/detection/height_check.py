#!/usr/bin/env python3
"""Standalone carrier-height check via the white rectangular frame.

Captures one full (small-zoom) image and verifies that the white rectangular
frame (the fixture surrounding the PCB) is fully contained inside the field of
view.  At a fixed zoom + fixed focus the magnification is fixed, so the
apparent size of the frame is a monotonic function of the carrier height: if
the carrier is placed too high the frame grows and gets clipped by the image
edges.  This check therefore detects "the PCB carrier is not placed at the
right height" before the real two-zoom capture runs.

The check is purely geometric — no reference ratio is required.  Any placement
is accepted as long as the whole white frame stays inside the image with the
configured margin, so a small amount of up/down placement freedom and an
asymmetric fixture are both fine.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

DETECTION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DETECTION_DIR.parent
CAPTURE_DIR = DETECTION_DIR / "capture"
for module_path in (str(CAPTURE_DIR), str(DETECTION_DIR)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import capture_pcb
import capture_pcb_two_zooms as two_zoom
from LensCamera import paths
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError


DEFAULT_SETTINGS_PATH = (
    paths.PROJECT_ROOT / "config" / "height_check" / "capture.jsonnet"
)
WHITE_MAX_SATURATION = 60
WHITE_MIN_VALUE = 190
EDGE_SUPPORT_FRACTION = 0.02
EDGE_PADDING_FRACTION = 0.002


class HeightCheckError(capture_pcb.PcbMosaicError):
    pass


class WhiteFrameError(HeightCheckError):
    pass


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise HeightCheckError(
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
            raise HeightCheckError(str(exc)) from exc
    return SimpleNamespace(
        **{name: setting_value(settings, name) for name in settings}
    )


def read_image(path: Path) -> np.ndarray:
    """Read a BGR image, tolerating non-ASCII (Chinese) paths."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise HeightCheckError(f"cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HeightCheckError(f"OpenCV cannot decode image: {path}")
    return image


# ---- white rectangular frame detection ---------------------------------------


def white_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of bright, nearly achromatic pixels (the white frame)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (0, 0, WHITE_MIN_VALUE),
        (179, WHITE_MAX_SATURATION, 255),
    )
    kernel_size = max(3, int(round(min(image.shape[:2]) / 500.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    return mask


def supported_component_bbox(
    mask: np.ndarray, contour: np.ndarray
) -> tuple[int, int, int, int]:
    """Fit an outer box while ignoring short connected projections.

    A plain boundingRect is controlled by a single extreme pixel.  Here each
    horizontal/vertical edge must be supported by part of the connected white
    frame, then a small padding restores the anti-aliased outer edge.
    """
    raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(contour)
    raw_x_max = raw_x + raw_w - 1
    raw_y_max = raw_y + raw_h - 1
    component_count, labels = cv2.connectedComponents(mask, connectivity=8)
    seed_x, seed_y = (int(value) for value in contour[0, 0])
    label = int(labels[seed_y, seed_x])
    if component_count <= 1 or label == 0:
        return raw_x, raw_y, raw_w, raw_h

    region = labels[raw_y : raw_y + raw_h, raw_x : raw_x + raw_w] == label
    row_support = np.count_nonzero(region, axis=1)
    column_support = np.count_nonzero(region, axis=0)
    minimum_row_support = max(3, int(math.ceil(raw_w * EDGE_SUPPORT_FRACTION)))
    minimum_column_support = max(
        3, int(math.ceil(raw_h * EDGE_SUPPORT_FRACTION))
    )
    supported_rows = np.flatnonzero(row_support >= minimum_row_support)
    supported_columns = np.flatnonzero(column_support >= minimum_column_support)
    if supported_rows.size == 0 or supported_columns.size == 0:
        return raw_x, raw_y, raw_w, raw_h

    padding = max(1, int(round(min(raw_w, raw_h) * EDGE_PADDING_FRACTION)))
    x_min = max(raw_x, raw_x + int(supported_columns[0]) - padding)
    x_max = min(raw_x_max, raw_x + int(supported_columns[-1]) + padding)
    y_min = max(raw_y, raw_y + int(supported_rows[0]) - padding)
    y_max = min(raw_y_max, raw_y + int(supported_rows[-1]) + padding)
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def order_box_points(box: np.ndarray) -> np.ndarray:
    """Reorder a 4x2 box into TL, TR, BR, BL (sum then diff trick)."""
    points = box.astype(np.float64)
    ordered = np.zeros((4, 2), dtype=np.float64)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left
    return ordered


def detect_white_frame(image: np.ndarray) -> dict[str, Any]:
    """Detect the largest white contour (the fixture frame) in the image."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise HeightCheckError("input must be a BGR color image")
    height, width = image.shape[:2]
    mask = white_mask(image)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise WhiteFrameError("未检测到白色边框 (白色像素不足)")
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    image_area = float(width * height)
    if area < image_area * 0.005:
        raise WhiteFrameError(
            "白色边框过小 (面积占比 {:.3f}), 可能未对准或载具未放置".format(
                area / image_area
            )
        )

    raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(largest)
    x, y, w, h = supported_component_bbox(mask, largest)
    x_max = x + w - 1
    y_max = y + h - 1

    # Exclude rejected projections from the rotated fit as well.
    contour_points = largest.reshape(-1, 2)
    fitted_points = contour_points[
        (contour_points[:, 0] >= x)
        & (contour_points[:, 0] <= x_max)
        & (contour_points[:, 1] >= y)
        & (contour_points[:, 1] <= y_max)
    ]
    if len(fitted_points) < 4:
        fitted_points = contour_points
    rect = cv2.minAreaRect(fitted_points.astype(np.float32).reshape(-1, 1, 2))
    box = order_box_points(cv2.boxPoints(rect))

    return {
        "found": True,
        "contour_area_px": area,
        "area_fraction": area / image_area,
        "image_size_px": [int(width), int(height)],
        "bbox_px": [int(x), int(y), int(w), int(h)],
        "raw_bbox_px": [int(raw_x), int(raw_y), int(raw_w), int(raw_h)],
        "bbox_method": "connected_component_edge_support",
        "bbox_corners_px": {
            "x_min": int(x),
            "y_min": int(y),
            "x_max": int(x_max),
            "y_max": int(y_max),
        },
        "white_mask_hsv": {
            "maximum_saturation": WHITE_MAX_SATURATION,
            "minimum_value": WHITE_MIN_VALUE,
        },
        "rotated_box_px": box.tolist(),
        "frame_width_px": float(rect[1][0]),
        "frame_height_px": float(rect[1][1]),
        "rotation_deg": float(rect[2]),
        "ratio_x": float(w) / width,
        "ratio_y": float(h) / height,
    }


def check_height(image: np.ndarray, margin: float = 0.02) -> dict[str, Any]:
    """Return a pass/fail record for the carrier height.

    The carrier height is accepted when the whole white frame is inside the
    image, inset by ``margin`` (a fraction of the smaller image side).  Raises
    WhiteFrameError when the frame cannot be detected at all.
    """
    if not math.isfinite(float(margin)) or float(margin) < 0:
        raise HeightCheckError("margin must be a non-negative number")
    frame = detect_white_frame(image)
    height, width = image.shape[:2]
    margin_px = float(margin) * float(min(width, height))

    x_min = frame["bbox_corners_px"]["x_min"]
    y_min = frame["bbox_corners_px"]["y_min"]
    x_max = frame["bbox_corners_px"]["x_max"]
    y_max = frame["bbox_corners_px"]["y_max"]

    inside = (
        x_min >= margin_px
        and y_min >= margin_px
        and x_max <= width - margin_px
        and y_max <= height - margin_px
    )

    return {
        "passed": bool(inside),
        "margin": float(margin),
        "margin_px": float(margin_px),
        "frame": frame,
        "inside": bool(inside),
    }


def draw_check_debug(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Annotate the image with the detected frame, margin and verdict."""
    debug = image.copy()
    frame = result.get("frame")

    if frame is not None:
        box = np.asarray(frame["rotated_box_px"], dtype=np.int32)
        cv2.polylines(debug, [box], True, (0, 255, 0), 6, cv2.LINE_AA)
        x, y, w, h = frame["bbox_px"]
        cv2.rectangle(
            debug,
            (x, y),
            (x + w - 1, y + h - 1),
            (255, 160, 0),
            3,
            cv2.LINE_AA,
        )

    height, width = image.shape[:2]
    margin_px = int(round(result.get("margin_px", 0.0)))
    cv2.rectangle(
        debug,
        (margin_px, margin_px),
        (width - margin_px, height - margin_px),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    passed = bool(result.get("passed"))
    status = "PASS" if passed else "FAIL"
    color = (0, 255, 0) if passed else (0, 0, 255)
    cv2.putText(
        debug,
        status,
        (margin_px + 10, margin_px + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        color,
        5,
        cv2.LINE_AA,
    )
    error = result.get("error")
    if error:
        cv2.putText(
            debug,
            str(error)[:70],
            (margin_px + 10, margin_px + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return debug


def print_check_summary(result: dict[str, Any]) -> None:
    frame = result.get("frame")
    if result.get("passed") and frame is not None:
        print(
            "高度检测通过: 白色边框完整位于视野内 "
            "(占比 x={:.3f}, y={:.3f}; 边框={}x{} px)".format(
                frame.get("ratio_x", 0.0),
                frame.get("ratio_y", 0.0),
                frame["bbox_px"][2],
                frame["bbox_px"][3],
            )
        )
        return
    if result.get("error"):
        print(f"高度检测失败: {result['error']}")
        return
    print("高度检测失败: 白色边框超出安全边界")


# ---- hardware orchestration ---------------------------------------------------


def run_offline(args: SimpleNamespace) -> int:
    """Detect the white frame on a local image, no hardware involved."""
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise HeightCheckError(f"离线图片不存在: {image_path}")
    image = read_image(image_path)
    height, width = image.shape[:2]

    run_dir = capture_pcb.create_run_directory(args.output_dir)
    report_path = run_dir / "carrier_height_check.json"
    images_dir = run_dir / "image"
    images_dir.mkdir()

    try:
        result = check_height(image, args.margin)
    except WhiteFrameError as exc:
        result = {
            "passed": False,
            "error": str(exc),
            "reason": "white_frame_not_found",
            "margin": float(args.margin),
            "margin_px": float(args.margin) * float(min(width, height)),
            "frame": None,
        }

    debug = draw_check_debug(image, result)
    debug_path = images_dir / "height_check_debug.png"
    capture_pcb.write_png(cv2, debug_path, debug)
    source_copy = images_dir / "source.png"
    capture_pcb.write_png(cv2, source_copy, image)

    result.update(
        {
            "mode": "offline",
            "source_image": str(image_path),
            "source_copy": str(source_copy),
            "debug_image": str(debug_path),
        }
    )
    log: dict[str, Any] = {
        "schema_version": 1,
        "type": "carrier_height_check",
        "mode": "offline",
        "created_at_utc": capture_pcb.utc_timestamp(),
        "status": "passed" if result["passed"] else "failed",
        "passed": result["passed"],
        "settings": dict(vars(args)),
        "height_check": result,
        "completed_at_utc": capture_pcb.utc_timestamp(),
    }
    capture_pcb.write_json(report_path, log)

    print(f"Output: {run_dir}")
    print_check_summary(result)
    print(f"Report: {report_path}")
    return 0 if result["passed"] else 2


def run(args: SimpleNamespace) -> int:
    if getattr(args, "image", None):
        return run_offline(args)
    summary_path = two_zoom.calibration_path(args)
    document = capture_pcb.load_calibration_summary_document(summary_path)

    runtime = SimpleNamespace(**vars(args))
    runtime.small_zoom = capture_pcb.zoom_address(args.small_zoom, "small_zoom")
    lens_position = two_zoom.autofocus_position_for_zoom(
        document, summary_path, runtime.small_zoom
    )
    runtime.small_focus = lens_position["focus"]
    runtime.small_iris = lens_position["iris"]
    runtime.lens_zoom = runtime.small_zoom
    phase = two_zoom.phase_values(runtime, "small")

    calibrations = two_zoom.load_visual_phase_calibrations(
        np, summary_path, "small", phase["zoom"], document
    )

    run_dir = capture_pcb.create_run_directory(runtime.output_dir)
    report_path = run_dir / "carrier_height_check.json"
    log: dict[str, Any] = {
        "schema_version": 1,
        "type": "carrier_height_check",
        "mode": "live",
        "created_at_utc": capture_pcb.utc_timestamp(),
        "status": "starting",
        "calibration_summary": str(summary_path),
        "settings": dict(vars(args)),
        "small": {
            "lens_zoom": phase["zoom"],
            "lens_focus": phase["focus"],
            "lens_iris": phase["iris"],
            "camera_intrinsics": capture_pcb.intrinsics_record(
                calibrations["intrinsics"]
            ),
        },
    }
    capture_pcb.write_json(report_path, log)
    print(f"Output: {run_dir}")

    camera = None
    try:
        log["small"]["lens"] = two_zoom.move_lens_to_phase(runtime, phase)
        camera, camera_record = capture_pcb.open_camera(cv2, runtime)
        frame = capture_pcb.capture_frame(camera, runtime, "height-check alignment")
        height, width = frame.shape[:2]
        map1, map2 = capture_pcb.build_undistort_maps(
            cv2, calibrations["intrinsics"], width, height
        )
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        images_dir = run_dir / "image"
        images_dir.mkdir()
        alignment: dict[str, Any] = {
            "enabled": bool(args.align_frame_center),
            "target_pixel": [float(width) / 2.0, float(height) / 2.0],
        }
        if args.align_frame_center:
            before_path = images_dir / "alignment_before_undistorted.png"
            capture_pcb.write_png(cv2, before_path, undistorted)
            alignment["before_image"] = str(before_path)
            try:
                before_detection = detect_white_frame(undistorted)
            except WhiteFrameError as exc:
                alignment["performed"] = False
                alignment["error"] = str(exc)
                print(f"高度检测对齐跳过: {exc}")
            else:
                source_pixel = np.asarray(
                    before_detection["rotated_box_px"], dtype=np.float64
                ).mean(axis=0)
                plan = two_zoom.calculate_pixel_alignment_plan(
                    np,
                    source_pixel,
                    alignment["target_pixel"],
                    calibrations["stage"],
                )
                alignment["initial_frame"] = before_detection
                alignment["plan"] = plan
                print(
                    "白色边框中心: ({:.2f}, {:.2f}) px; 图像中心: "
                    "({:.2f}, {:.2f}) px; 位移台相对移动: ({:.6f}, {:.6f}) mm".format(
                        *source_pixel,
                        *alignment["target_pixel"],
                        *plan["requested_relative_stage_delta_mm"],
                    )
                )
                alignment["motion"] = two_zoom.execute_relative_stage_move(
                    np,
                    runtime,
                    plan["requested_relative_stage_delta_mm"],
                    "height-check white-frame centring",
                )
                alignment["performed"] = True
                frame = capture_pcb.capture_frame(
                    camera, runtime, "carrier height check"
                )
                final_height, final_width = frame.shape[:2]
                if (final_width, final_height) != (width, height):
                    width, height = final_width, final_height
                    map1, map2 = capture_pcb.build_undistort_maps(
                        cv2, calibrations["intrinsics"], width, height
                    )
                undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        else:
            alignment["performed"] = False
        log["small"]["alignment"] = alignment

        raw_path = None
        if args.save_raw:
            raw_path = images_dir / "raw.png"
            capture_pcb.write_png(cv2, raw_path, frame)
        undistorted_path = images_dir / "undistorted.png"
        capture_pcb.write_png(cv2, undistorted_path, undistorted)

        try:
            result = check_height(undistorted, args.margin)
        except WhiteFrameError as exc:
            result = {
                "passed": False,
                "error": str(exc),
                "reason": "white_frame_not_found",
                "margin": float(args.margin),
                "margin_px": float(args.margin) * float(min(width, height)),
                "frame": None,
            }

        debug = draw_check_debug(undistorted, result)
        debug_path = images_dir / "height_check_debug.png"
        capture_pcb.write_png(cv2, debug_path, debug)

        result.update(
            {
                "camera": camera_record,
                "raw_image": str(raw_path) if raw_path is not None else None,
                "undistorted_image": str(undistorted_path),
                "debug_image": str(debug_path),
            }
        )
        log["height_check"] = result
        log["status"] = "passed" if result["passed"] else "failed"
        log["passed"] = result["passed"]
        log["completed_at_utc"] = capture_pcb.utc_timestamp()
        capture_pcb.write_json(report_path, log)

        print_check_summary(result)
        print(f"Report: {report_path}")
        return 0 if result["passed"] else 2
    finally:
        if camera is not None:
            camera.release()


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
