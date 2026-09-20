"""Capture one camera frame, mark the calibrated optical center, and save it.

Default usage from the project root:
    python test/capture_optical_center.py

Use ``--lens-zoom`` when the lens is at another calibrated zoom position.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECTION_DIR = PROJECT_ROOT / "detection"
if str(DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_DIR))

from LensCamera import camera as camera_tools


DEFAULT_CALIBRATION = (
    DETECTION_DIR / "calibration" / "output" / "calibrate.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "working_data" / "camera_optical_center"
DEFAULT_CAMERA_NAME = "CamSPC"
DEFAULT_LENS_ZOOM = 6353
DEFAULT_FRAME_WIDTH = 3840
DEFAULT_FRAME_HEIGHT = 2160
LENS_CALIBRATION_TYPES = {
    "lens_checkerboard_camera_calibration",
    "camera_calibration_core",
    "lens_camera_calibration",
}


class OpticalCenterCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpticalCenterCalibration:
    source: Path
    lens_zoom: int
    position_name: str
    image_width: int
    image_height: int
    cx: float
    cy: float


def _zoom_from_record(record: dict[str, Any], fallback: Any = None) -> int | None:
    value = record.get("lens_zoom", fallback)
    if value is not None and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    name = str(record.get("name", ""))
    if name.startswith("zoom_"):
        try:
            return int(name[5:])
        except ValueError:
            pass
    return None


def _camera_matrix(section: dict[str, Any], source: Path) -> list[list[float]]:
    matrix = section.get("camera_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise OpticalCenterCaptureError(
            f"invalid 3x3 camera_matrix in {source}"
        )
    try:
        converted = [[float(value) for value in row] for row in matrix]
    except (TypeError, ValueError) as exc:
        raise OpticalCenterCaptureError(
            f"camera_matrix in {source} must contain numbers"
        ) from exc
    if not all(math.isfinite(value) for row in converted for value in row):
        raise OpticalCenterCaptureError(
            f"camera_matrix in {source} contains NaN or infinity"
        )
    return converted


def load_optical_center_calibration(
    source: Path | str,
    lens_zoom: int,
) -> OpticalCenterCalibration:
    source = Path(source).resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise OpticalCenterCaptureError(
            f"cannot read calibration file {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise OpticalCenterCaptureError(
            f"calibration file is not valid JSON: {source}: {exc}"
        ) from exc

    entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise OpticalCenterCaptureError(
            f"calibration file has no entries list: {source}"
        )

    requested_zoom = int(lens_zoom)
    matches: list[OpticalCenterCalibration] = []
    available_zooms: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        positions = entry.get("positions")
        if not isinstance(positions, list):
            continue
        entry_zoom = _zoom_from_record(entry)
        for position in positions:
            if not isinstance(position, dict):
                continue
            section = position.get("calibration", position)
            if not isinstance(section, dict) or "camera_matrix" not in section:
                continue
            zoom = _zoom_from_record(position, entry_zoom)
            if zoom is None:
                continue
            available_zooms.add(zoom)
            if zoom != requested_zoom:
                continue
            matrix = _camera_matrix(section, source)
            try:
                image_width = int(
                    position.get("image_width", section.get("image_width"))
                )
                image_height = int(
                    position.get("image_height", section.get("image_height"))
                )
            except (TypeError, ValueError) as exc:
                raise OpticalCenterCaptureError(
                    f"invalid calibration image size for zoom {zoom} in {source}"
                ) from exc
            if image_width <= 0 or image_height <= 0:
                raise OpticalCenterCaptureError(
                    f"calibration image size must be positive in {source}"
                )
            cx, cy = matrix[0][2], matrix[1][2]
            if not (0 <= cx < image_width and 0 <= cy < image_height):
                raise OpticalCenterCaptureError(
                    f"optical center ({cx}, {cy}) is outside the calibrated "
                    f"image {image_width}x{image_height}"
                )
            matches.append(OpticalCenterCalibration(
                source=source,
                lens_zoom=zoom,
                position_name=str(position.get("name", f"zoom_{zoom:05d}")),
                image_width=image_width,
                image_height=image_height,
                cx=cx,
                cy=cy,
            ))

    if not matches:
        available = ", ".join(str(value) for value in sorted(available_zooms))
        raise OpticalCenterCaptureError(
            f"no camera intrinsics for lens_zoom={requested_zoom} in {source}; "
            f"available zooms: {available or '<none>'}"
        )
    return matches[-1]


def scaled_optical_center(
    calibration: OpticalCenterCalibration,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise OpticalCenterCaptureError("captured frame has an invalid size")
    return (
        calibration.cx * frame_width / calibration.image_width,
        calibration.cy * frame_height / calibration.image_height,
    )


def annotate_optical_center(
    cv2: Any,
    frame: Any,
    cx: float,
    cy: float,
) -> Any:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise OpticalCenterCaptureError("camera returned an invalid image")
    height, width = frame.shape[:2]
    center = (int(round(cx)), int(round(cy)))
    if not (0 <= center[0] < width and 0 <= center[1] < height):
        raise OpticalCenterCaptureError(
            f"scaled optical center {center} is outside frame {width}x{height}"
        )

    annotated = frame.copy()
    size = max(24, min(width, height) // 30)
    thickness = max(2, min(width, height) // 700)
    black = (0, 0, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)

    for color, line_width in ((black, thickness + 4), (yellow, thickness)):
        cv2.line(
            annotated,
            (center[0] - size, center[1]),
            (center[0] + size, center[1]),
            color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.line(
            annotated,
            (center[0], center[1] - size),
            (center[0], center[1] + size),
            color,
            line_width,
            cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            center,
            max(12, size // 2),
            color,
            line_width,
            cv2.LINE_AA,
        )
    cv2.circle(annotated, center, max(3, thickness + 1), red, -1, cv2.LINE_AA)

    label = f"Optical center  cx={cx:.2f}, cy={cy:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(width, height) / 1400.0)
    text_thickness = max(1, thickness)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    label_x = min(max(8, center[0] + size + 12), max(8, width - text_width - 12))
    label_y = center[1] - size - 12
    if label_y - text_height - baseline < 8:
        label_y = min(height - baseline - 8, center[1] + size + text_height + 12)
    top_left = (label_x - 6, label_y - text_height - 6)
    bottom_right = (label_x + text_width + 6, label_y + baseline + 6)
    cv2.rectangle(annotated, top_left, bottom_right, black, -1)
    cv2.rectangle(annotated, top_left, bottom_right, yellow, thickness)
    cv2.putText(
        annotated,
        label,
        (label_x, label_y),
        font,
        font_scale,
        yellow,
        text_thickness,
        cv2.LINE_AA,
    )
    return annotated


def write_image(cv2: Any, output: Path | str, image: Any) -> Path:
    output = Path(output).resolve()
    extension = output.suffix.lower()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise OpticalCenterCaptureError(
            "output extension must be .png, .jpg, or .jpeg"
        )
    parameters = (
        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        if extension in {".jpg", ".jpeg"}
        else []
    )
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise OpticalCenterCaptureError(f"OpenCV cannot encode {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(str(output))
    except OSError as exc:
        raise OpticalCenterCaptureError(f"cannot save {output}: {exc}") from exc
    return output


def select_camera(args: argparse.Namespace) -> dict[str, Any]:
    if args.camera_index is not None:
        return {
            "index": int(args.camera_index),
            "name": None,
            "source": "command_line_index",
        }
    return camera_tools.find_camera_by_name(
        args.camera_name, OpticalCenterCaptureError
    )


def capture_annotated_photo(args: argparse.Namespace) -> dict[str, Any]:
    cv2, _np = camera_tools.load_image_tools(OpticalCenterCaptureError)
    calibration = load_optical_center_calibration(
        args.calibration, args.lens_zoom
    )
    selected = select_camera(args)
    camera = camera_tools.open_camera(cv2, camera_tools.camera_source(selected))
    if not camera.isOpened():
        camera.release()
        raise OpticalCenterCaptureError(
            f"cannot open camera index {selected['index']} ({selected.get('name')})"
        )

    try:
        settings = SimpleNamespace(
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            disable_camera_autofocus=args.disable_camera_autofocus,
        )
        camera_tools.configure_camera(cv2, camera, settings)
        for _ in range(args.startup_discard_frames):
            camera.read()
        if args.settle_seconds > 0:
            time.sleep(args.settle_seconds)
        ok = False
        frame = None
        for _ in range(args.capture_discard_frames + 1):
            ok, frame = camera.read()
        if not ok or frame is None:
            raise OpticalCenterCaptureError("camera returned no image")
    finally:
        camera.release()

    frame_height, frame_width = frame.shape[:2]
    cx, cy = scaled_optical_center(calibration, frame_width, frame_height)
    annotated = annotate_optical_center(cv2, frame, cx, cy)
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DEFAULT_OUTPUT_DIR / f"optical_center_{stamp}.png"
    saved_path = write_image(cv2, output, annotated)
    return {
        "output": saved_path,
        "camera": selected,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "calibration": calibration,
        "optical_center": (cx, cy),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one photo and mark the calibrated optical center."
    )
    parser.add_argument("--lens-zoom", type=int, default=DEFAULT_LENS_ZOOM)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--camera-index", type=int)
    camera_group.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--startup-discard-frames", type=int, default=5)
    parser.add_argument("--capture-discard-frames", type=int, default=3)
    parser.add_argument("--settle-seconds", type=float, default=0.4)
    parser.add_argument("--disable-camera-autofocus", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("frame_width", "frame_height"):
        if getattr(args, name) <= 0:
            raise OpticalCenterCaptureError(f"--{name.replace('_', '-')} must be above 0")
    for name in ("startup_discard_frames", "capture_discard_frames"):
        if getattr(args, name) < 0:
            raise OpticalCenterCaptureError(f"--{name.replace('_', '-')} cannot be negative")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        raise OpticalCenterCaptureError("--settle-seconds must be finite and non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        result = capture_annotated_photo(args)
    except OpticalCenterCaptureError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    calibration = result["calibration"]
    cx, cy = result["optical_center"]
    print(f"Camera: {result['camera'].get('name') or result['camera']['index']}")
    print(f"Lens zoom calibration: {calibration.lens_zoom} ({calibration.position_name})")
    print(f"Frame: {result['frame_width']}x{result['frame_height']}")
    print(f"Optical center: cx={cx:.2f}, cy={cy:.2f}")
    print(f"Saved: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
