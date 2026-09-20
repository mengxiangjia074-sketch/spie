"""Calibrate XY-stage commanded millimetres to image-pixel displacement.

Two configured zoom positions are calibrated in sequence. Their focus
positions are loaded from matching pcb_autofocus_lens_position entries in
calibrate.json, and each zoom uses its own X/Y command ranges. For each zoom,
the stage samples relative X/Y commands on both sides of the configured start
position and returns after every excursion. The measured image translations
are fitted to this model:

    [du_px, dv_px]^T = matrix_2x2 * [dx_command_mm, dy_command_mm]^T

The script always moves the lens and stage. On success, one standalone result
is written for each zoom below the configured output_root, and two zoom-named
entries are updated in detection/calibration/output/calibrate.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

# The LensCamera package and motion_controller.py live in the detection
# directory one level above these calibration scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import LensCamera as lens_camera
import calibration_summary
from LensCamera import camera as camera_tools
from LensCamera import paths
from LensCamera.json_io import write_json
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError


DEFAULT_SETTINGS_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "stage_to_pixel"
    / "stage2pixel.jsonnet"
)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "output" / "calibrate.json"
AUTOFOCUS_ENTRY_TYPE = "pcb_autofocus_lens_position"


class StagePixelCalibrationError(RuntimeError):
    pass


def load_settings() -> dict[str, Any]:
    try:
        return JsonnetSettings(DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)).values
    except SettingsError as exc:
        raise StagePixelCalibrationError(str(exc)) from exc


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise StagePixelCalibrationError(
            f"{DEFAULT_SETTINGS_PATH} missing required setting {name!r}"
        )
    item = settings[name]
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def load_args(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    settings = load_settings() if settings is None else settings
    return SimpleNamespace(**{name: setting_value(settings, name) for name in settings})


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def lens_address(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise StagePixelCalibrationError(
            f"{name} must be an integer address, not a boolean"
        )
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.strip():
        try:
            result = int(value.strip(), 0)
        except ValueError as exc:
            raise StagePixelCalibrationError(
                f"{name} must be an integer address"
            ) from exc
    else:
        raise StagePixelCalibrationError(f"{name} must be an integer address")
    if result < 0:
        raise StagePixelCalibrationError(f"{name} cannot be negative")
    return result


def calibration_path(args: SimpleNamespace) -> Path:
    value = getattr(args, "calibration", None)
    if value is None or not str(value).strip():
        return DEFAULT_CALIBRATION_PATH
    return paths.resolve_project_path(str(value)).resolve()


def load_calibration_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StagePixelCalibrationError(
            f"cannot read calibration summary {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StagePixelCalibrationError(
            f"calibration summary is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        raise StagePixelCalibrationError(
            f"calibration summary {path} must contain an entries list"
        )
    return document


def entry_matches_zoom(entry: dict[str, Any], zoom: int) -> bool:
    try:
        if entry.get("lens_zoom") is not None:
            return lens_address(entry["lens_zoom"], "lens_zoom") == zoom
    except StagePixelCalibrationError:
        pass
    return entry.get("name") == calibration_summary.zoom_entry_name(zoom)


def load_autofocus_focus(
    document: dict[str, Any], summary_path: Path, zoom: Any
) -> int:
    requested_zoom = lens_address(zoom, "lens_zoom")
    for entry in reversed(document["entries"]):
        if (
            isinstance(entry, dict)
            and entry.get("type") == AUTOFOCUS_ENTRY_TYPE
            and entry_matches_zoom(entry, requested_zoom)
            and entry.get("lens_focus") is not None
        ):
            return lens_address(entry["lens_focus"], "autofocus lens_focus")
    raise StagePixelCalibrationError(
        f"{summary_path} has no {AUTOFOCUS_ENTRY_TYPE} focus for zoom "
        f"{requested_zoom}; run PCB autofocus at this zoom first"
    )


def build_zoom_profiles(
    args: SimpleNamespace,
    document: dict[str, Any],
    summary_path: Path,
) -> list[dict[str, Any]]:
    profiles = []
    for index in (1, 2):
        zoom = lens_address(getattr(args, f"zoom_{index}"), f"zoom_{index}")
        profiles.append(
            {
                "index": index,
                "zoom": zoom,
                "focus": load_autofocus_focus(document, summary_path, zoom),
                "x_step_mm": normalize_step_values(
                    getattr(args, f"zoom_{index}_x_step_mm"),
                    f"zoom_{index}_x_step_mm",
                ),
                "y_step_mm": normalize_step_values(
                    getattr(args, f"zoom_{index}_y_step_mm"),
                    f"zoom_{index}_y_step_mm",
                ),
            }
        )
    if profiles[0]["zoom"] == profiles[1]["zoom"]:
        raise StagePixelCalibrationError("zoom_1 and zoom_2 must be different")
    return profiles


def build_calibration_summary_entry(
    result_document: dict[str, Any],
) -> dict[str, Any]:
    section = result_document["stage_command_to_pixel"]
    entry = {
        "type": result_document["type"],
        "calibrated_at_utc": result_document["calibrated_at_utc"],
        "stage_command_to_pixel": {
            key: section[key]
            for key in ("equation", "matrix_2x2", "input_unit", "output_unit")
        },
        "pixel_to_stage_command": section["pixel_to_stage_command"],
    }
    if result_document.get("lens_focus") is not None:
        entry["lens_focus"] = result_document["lens_focus"]
    return entry


def append_calibration_summary(
    result_document: dict[str, Any], args: SimpleNamespace, zoom: Any = None
) -> Path:
    kwargs = {"device_number": getattr(args, "lens_device", None)}
    if zoom is not None:
        kwargs["zoom"] = zoom
    return calibration_summary.append_entry(
        build_calibration_summary_entry(result_document), **kwargs
    )


def print_calibration_errors(diagnostics: dict[str, Any], zoom: Any = None) -> None:
    zoom_label = "" if zoom is None else f" Zoom {lens_address(zoom, 'lens_zoom')}:"
    print(
        "标定误差:{} 位移台拟合 RMSE {:.4f} px; 平均误差 {:.4f} px; "
        "最大误差 {:.4f} px".format(
            zoom_label,
            diagnostics["rmse_px"],
            diagnostics["mean_error_px"],
            diagnostics["max_error_px"],
        )
    )


def print_completed_calibration_errors(
    completed: Sequence[dict[str, Any]],
) -> None:
    for result in completed:
        print_calibration_errors(result["fit_quality"], zoom=result["zoom"])


def load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise StagePixelCalibrationError(
            "NumPy is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return np


def create_run_directory(root: Path, timestamp: str) -> Path:
    resolved_root = root.resolve()
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:02d}"
        candidate = resolved_root / f"{timestamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise StagePixelCalibrationError(
                f"cannot create calibration data directory {candidate}: {exc}"
            ) from exc
    raise StagePixelCalibrationError(
        f"cannot allocate a unique calibration directory below {resolved_root}"
    )


def numeric_pair(value: Any, name: str, np: Any) -> Any:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(2)
    except (TypeError, ValueError) as exc:
        raise StagePixelCalibrationError(
            f"{name} must contain two numeric values"
        ) from exc
    if not np.all(np.isfinite(result)):
        raise StagePixelCalibrationError(f"{name} contains NaN or infinity")
    return result


def parse_limits(
    values: Sequence[float] | None,
    name: str,
) -> tuple[float, float] | None:
    if values is None:
        return None
    lower, upper = float(values[0]), float(values[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise StagePixelCalibrationError(
            f"{name} must contain finite MIN MAX values with MIN < MAX"
        )
    return lower, upper


def ensure_position_in_limits(
    position: Any,
    x_limits: tuple[float, float] | None,
    y_limits: tuple[float, float] | None,
    label: str,
) -> None:
    x_mm, y_mm = float(position[0]), float(position[1])
    if x_limits and not x_limits[0] <= x_mm <= x_limits[1]:
        raise StagePixelCalibrationError(
            f"{label} X={x_mm:.6f} mm is outside limits {x_limits}"
        )
    if y_limits and not y_limits[0] <= y_mm <= y_limits[1]:
        raise StagePixelCalibrationError(
            f"{label} Y={y_mm:.6f} mm is outside limits {y_limits}"
        )


def normalize_step_values(values: Any, option: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        values = [values]
    if not values:
        raise StagePixelCalibrationError(f"{option} requires at least one value")
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise StagePixelCalibrationError(
            f"{option} must contain numeric values"
        ) from exc


def build_command_samples(np: Any, args: SimpleNamespace) -> list[dict[str, Any]]:
    """Build one pass of two-sided relative command samples.

    Every step distance is sampled on both sides of the start position: the
    stage runs out to +step and comes back, then runs out to -step and comes
    back, so each configured value moves in both the positive and the
    negative direction and the stage is back at the start position after
    every excursion.
    """
    args.x_step_mm = normalize_step_values(args.x_step_mm, "--x-step-mm")
    args.y_step_mm = normalize_step_values(args.y_step_mm, "--y-step-mm")
    samples = []
    for axis, steps in (("x", args.x_step_mm), ("y", args.y_step_mm)):
        for step_index, step_mm in enumerate(steps, start=1):
            for direction, sign, leg in (
                ("positive", 1.0, "out"),
                ("negative", -1.0, "back"),
                ("negative", -1.0, "out"),
                ("positive", 1.0, "back"),
            ):
                command = (
                    [sign * step_mm, 0.0] if axis == "x" else [0.0, sign * step_mm]
                )
                samples.append(
                    {
                        "label": (f"{axis}_{direction}_step_{step_index:02d}_{leg}"),
                        "axis": axis,
                        "direction": direction,
                        "step_index": step_index,
                        "step_mm": step_mm,
                        "leg": leg,
                        "command_delta_mm": np.asarray(command, dtype=np.float64),
                    }
                )
    return samples


def validate_args(args: SimpleNamespace) -> None:
    for option, steps in (
        ("--x-step-mm", args.x_step_mm),
        ("--y-step-mm", args.y_step_mm),
    ):
        for step in steps:
            if not math.isfinite(step) or step <= 0:
                raise StagePixelCalibrationError(
                    f"{option} values must be finite and above 0"
                )
    if args.x_slave_address == args.y_slave_address:
        raise StagePixelCalibrationError("X/Y slave addresses must be different")
    if not str(args.port).strip():
        raise StagePixelCalibrationError("--port cannot be empty")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise StagePixelCalibrationError("camera frame dimensions must be above 0")
    if args.roi is not None:
        x, y, width, height = args.roi
        if (
            min(x, y) < 0
            or width <= 0
            or height <= 0
            or x + width > args.frame_width
            or y + height > args.frame_height
        ):
            raise StagePixelCalibrationError("--roi must stay inside the camera frame")


def print_plan(
    args: SimpleNamespace,
    start_position: Any,
    samples: list[dict[str, Any]],
) -> None:
    print(f"Calibration data root: {Path(args.output_root).resolve()}")
    if start_position is None:
        print("Stage calibration start: current stage position (no initial move)")
    else:
        print("Stage calibration start: ({:.6f}, {:.6f}) mm".format(*start_position))
    print("Command samples:")
    expected_position = (
        None
        if start_position is None
        else [float(start_position[0]), float(start_position[1])]
    )
    for sample_index, sample in enumerate(samples, start=1):
        delta = sample["command_delta_mm"]
        if expected_position is None:
            suffix = ""
        else:
            expected_position[0] += float(delta[0])
            expected_position[1] += float(delta[1])
            suffix = " -> ({:.6f}, {:.6f}) mm".format(
                expected_position[0], expected_position[1]
            )
        print(
            "  sample {:02d} {:>22s}: ({:+.6f}, {:+.6f}) command mm{}".format(
                sample_index,
                sample["label"],
                delta[0],
                delta[1],
                suffix,
            )
        )


def read_stage_position(np: Any, stage: Any, label: str) -> Any:
    try:
        record = stage.read_motor_position()
        position = np.asarray(
            [float(record["x"]), float(record["y"])], dtype=np.float64
        )
    except Exception as exc:
        raise StagePixelCalibrationError(
            f"cannot read stage position for {label}: {exc}"
        ) from exc
    if not np.all(np.isfinite(position)):
        raise StagePixelCalibrationError(
            f"stage returned a non-finite position for {label}"
        )
    return position


def stop_stage(stage: Any) -> None:
    for axis in (stage.x, stage.y):
        try:
            axis.stop()
        except Exception:
            pass


def move_stage_absolute(
    np: Any,
    stage: Any,
    target: Any,
    args: SimpleNamespace,
    absolute_mode: int,
    label: str,
    maximum_distance_mm: float,
) -> dict[str, Any]:
    target_position = np.asarray(target, dtype=np.float64).reshape(2)
    ensure_position_in_limits(
        target_position,
        parse_limits(args.x_limits, "--x-limits"),
        parse_limits(args.y_limits, "--y-limits"),
        label,
    )
    before = read_stage_position(np, stage, f"before {label}")
    distance = float(np.linalg.norm(target_position - before))
    if distance > maximum_distance_mm:
        raise StagePixelCalibrationError(
            f"{label} distance {distance:.6f} mm exceeds allowed "
            f"{maximum_distance_mm:.6f} mm"
        )
    stage.move_to_position(
        x_mm=float(target_position[0]),
        y_mm=float(target_position[1]),
        speed_rpm=args.speed_rpm,
        mode=absolute_mode,
        wait=False,
    )
    x_done = bool(stage.x.wait_for_completion(timeout=args.timeout_seconds))
    y_done = bool(stage.y.wait_for_completion(timeout=args.timeout_seconds))
    if not x_done or not y_done:
        stop_stage(stage)
        raise StagePixelCalibrationError(
            f"{label} timed out: X completed={x_done}, Y completed={y_done}"
        )
    actual = read_stage_position(np, stage, f"after {label}")
    error = actual - target_position
    error_distance = float(np.linalg.norm(error))
    if error_distance > args.position_tolerance_mm:
        raise StagePixelCalibrationError(
            "{} final-position error {:.6f} mm exceeds tolerance {:.6f} mm; "
            "expected ({:.6f}, {:.6f}), actual ({:.6f}, {:.6f})".format(
                label,
                error_distance,
                args.position_tolerance_mm,
                target_position[0],
                target_position[1],
                actual[0],
                actual[1],
            )
        )
    return {
        "before_position_mm": before.tolist(),
        "requested_position_mm": target_position.tolist(),
        "actual_position_mm": actual.tolist(),
        "error_vector_mm": error.tolist(),
        "error_distance_mm": error_distance,
        "travel_distance_mm": distance,
    }


def move_stage_relative(
    np: Any,
    stage: Any,
    command_delta: Any,
    args: SimpleNamespace,
    relative_mode: int,
    label: str,
) -> dict[str, Any]:
    delta = np.asarray(command_delta, dtype=np.float64).reshape(2)
    distance = float(np.linalg.norm(delta))
    if distance > args.max_step_mm:
        raise StagePixelCalibrationError(
            f"{label} command distance {distance:.6f} mm exceeds "
            f"--max-step-mm {args.max_step_mm:.6f}"
        )
    before = read_stage_position(np, stage, f"before {label}")
    expected = before + delta
    ensure_position_in_limits(
        expected,
        parse_limits(args.x_limits, "--x-limits"),
        parse_limits(args.y_limits, "--y-limits"),
        label,
    )
    x_command = float(delta[0]) if abs(float(delta[0])) > 1e-12 else None
    y_command = float(delta[1]) if abs(float(delta[1])) > 1e-12 else None
    if x_command is not None or y_command is not None:
        stage.move_to_position(
            x_mm=x_command,
            y_mm=y_command,
            speed_rpm=args.speed_rpm,
            mode=relative_mode,
            wait=False,
        )
        x_done = (
            True
            if x_command is None
            else bool(stage.x.wait_for_completion(timeout=args.timeout_seconds))
        )
        y_done = (
            True
            if y_command is None
            else bool(stage.y.wait_for_completion(timeout=args.timeout_seconds))
        )
        if not x_done or not y_done:
            stop_stage(stage)
            raise StagePixelCalibrationError(
                f"{label} timed out: X completed={x_done}, Y completed={y_done}"
            )
    actual = read_stage_position(np, stage, f"after {label}")
    error = actual - expected
    error_distance = float(np.linalg.norm(error))
    if error_distance > args.position_tolerance_mm:
        raise StagePixelCalibrationError(
            "{} final-position error {:.6f} mm exceeds tolerance {:.6f} mm; "
            "expected ({:.6f}, {:.6f}), actual ({:.6f}, {:.6f})".format(
                label,
                error_distance,
                args.position_tolerance_mm,
                expected[0],
                expected[1],
                actual[0],
                actual[1],
            )
        )
    return {
        "before_position_mm": before.tolist(),
        "requested_relative_delta_mm": delta.tolist(),
        "expected_position_mm": expected.tolist(),
        "actual_position_mm": actual.tolist(),
        "error_vector_mm": error.tolist(),
        "error_distance_mm": error_distance,
        "command_distance_mm": distance,
    }


def open_selected_camera(
    cv2: Any,
    args: SimpleNamespace,
) -> tuple[Any, dict[str, Any]]:
    if args.camera is None:
        selected = camera_tools.find_camera_by_name(
            args.camera_name, StagePixelCalibrationError
        )
    else:
        selected = {
            "index": int(args.camera),
            "name": None,
            "source": "command_line_index",
        }
    camera = camera_tools.open_camera(cv2, camera_tools.camera_source(selected))
    if not camera.isOpened():
        raise StagePixelCalibrationError(
            f"cannot open camera index {selected['index']} ({selected.get('name')})"
        )
    settings = SimpleNamespace(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        disable_camera_autofocus=not args.keep_camera_autofocus,
    )
    camera_tools.configure_camera(cv2, camera, settings)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(args.camera_startup_discard_frames):
        camera.read()
    record = dict(selected)
    record.update(
        {
            "requested_frame_width_px": int(args.frame_width),
            "requested_frame_height_px": int(args.frame_height),
            "autofocus_kept": bool(args.keep_camera_autofocus),
        }
    )
    return camera, record


def capture_frame(
    camera: Any,
    args: SimpleNamespace,
    expected_width: int,
    expected_height: int,
    label: str,
) -> Any:
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    frame = None
    ok = False
    for _ in range(args.capture_discard_frames + 1):
        ok, frame = camera.read()
    if not ok or frame is None:
        raise StagePixelCalibrationError(f"camera returned no image for {label}")
    height, width = frame.shape[:2]
    if width != expected_width or height != expected_height:
        raise StagePixelCalibrationError(
            f"{label} image is {width}x{height}, but board calibration is "
            f"{expected_width}x{expected_height}"
        )
    return frame


def write_image(cv2: Any, path: Path, image: Any) -> None:
    extension = path.suffix.lower() or ".jpg"
    parameters = (
        [int(cv2.IMWRITE_JPEG_QUALITY), 92] if extension in (".jpg", ".jpeg") else []
    )
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise StagePixelCalibrationError(
            f"OpenCV cannot encode calibration image as {extension}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise StagePixelCalibrationError(f"cannot write image {path}: {exc}") from exc


def registration_plane(cv2: Any, image: Any, use_gradient: bool) -> Any:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plane = gray.astype("float32")
    plane = cv2.GaussianBlur(plane, (5, 5), 0)
    if use_gradient:
        gradient_x = cv2.Sobel(plane, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(plane, cv2.CV_32F, 0, 1, ksize=3)
        plane = cv2.magnitude(gradient_x, gradient_y)
    return plane


def measure_image_translation(
    cv2: Any,
    np: Any,
    reference_image: Any,
    moved_image: Any,
    args: SimpleNamespace,
) -> dict[str, Any]:
    if reference_image.shape[:2] != moved_image.shape[:2]:
        raise StagePixelCalibrationError(
            "reference and moved images have different dimensions"
        )
    if args.roi is None:
        x, y = 0, 0
        height, width = reference_image.shape[:2]
    else:
        x, y, width, height = args.roi
    reference_roi = reference_image[y : y + height, x : x + width]
    moved_roi = moved_image[y : y + height, x : x + width]

    scale = min(1.0, args.registration_max_dimension / max(width, height))
    resized_width = max(32, int(round(width * scale)))
    resized_height = max(32, int(round(height * scale)))
    if resized_width != width or resized_height != height:
        reference_roi = cv2.resize(
            reference_roi,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        moved_roi = cv2.resize(
            moved_roi,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    reference_plane = registration_plane(
        cv2, reference_roi, args.registration_use_gradient
    )
    moved_plane = registration_plane(cv2, moved_roi, args.registration_use_gradient)
    window = cv2.createHanningWindow((resized_width, resized_height), cv2.CV_32F)
    shift_scaled, response = cv2.phaseCorrelate(reference_plane, moved_plane, window)
    scale_x = resized_width / width
    scale_y = resized_height / height
    shift = np.asarray(
        [float(shift_scaled[0]) / scale_x, float(shift_scaled[1]) / scale_y],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(shift)) or not math.isfinite(float(response)):
        raise StagePixelCalibrationError("image registration returned NaN or infinity")
    if float(response) < args.min_registration_response:
        raise StagePixelCalibrationError(
            "image registration response {:.6f} is below the required {:.6f}".format(
                response, args.min_registration_response
            )
        )
    if abs(float(shift[0])) >= width / 2 or abs(float(shift[1])) >= height / 2:
        raise StagePixelCalibrationError(
            "measured translation ({:.3f}, {:.3f}) px exceeds half of the "
            "registration ROI and may have wrapped".format(*shift)
        )
    return {
        "pixel_displacement": shift,
        "response": float(response),
        "roi_xywh_px": [int(x), int(y), int(width), int(height)],
        "processing_size_px": [int(resized_width), int(resized_height)],
        "processing_scale_xy": [float(scale_x), float(scale_y)],
        "used_gradient": bool(args.registration_use_gradient),
    }


def fit_stage_command_matrix(
    np: Any,
    commands: Any,
    pixel_displacements: Any,
    responses: Any,
) -> tuple[Any, Any, dict[str, Any], Any]:
    command_array = np.asarray(commands, dtype=np.float64).reshape(-1, 2)
    pixel_array = np.asarray(pixel_displacements, dtype=np.float64).reshape(-1, 2)
    response_array = np.asarray(responses, dtype=np.float64).reshape(-1)
    if len(command_array) < 4 or len(command_array) != len(pixel_array):
        raise StagePixelCalibrationError(
            "at least four matching command/displacement samples are required"
        )
    if not np.all(np.isfinite(command_array)) or not np.all(np.isfinite(pixel_array)):
        raise StagePixelCalibrationError("fit samples contain NaN or infinity")
    if int(np.linalg.matrix_rank(command_array)) < 2:
        raise StagePixelCalibrationError("stage command samples do not span X and Y")

    weights = np.sqrt(np.clip(response_array, 0.05, 1.0))
    weighted_commands = command_array * weights[:, None]
    weighted_pixels = pixel_array * weights[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_commands, weighted_pixels, rcond=None
    )
    if int(rank) < 2:
        raise StagePixelCalibrationError("stage command matrix fit is rank deficient")
    matrix = coefficients.T
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-9:
        raise StagePixelCalibrationError("fitted stage command matrix is singular")
    inverse = np.linalg.inv(matrix)
    predicted = command_array @ coefficients
    residual_vectors = pixel_array - predicted
    residual_distances = np.linalg.norm(residual_vectors, axis=1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    diagnostics = {
        "sample_count": int(len(command_array)),
        "rmse_px": float(np.sqrt(np.mean(residual_distances**2))),
        "mean_error_px": float(np.mean(residual_distances)),
        "max_error_px": float(np.max(residual_distances)),
        "matrix_determinant_px2_per_command_mm2": determinant,
        "singular_values_px_per_command_mm": singular_values.tolist(),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "minimum_registration_response": float(np.min(response_array)),
        "mean_registration_response": float(np.mean(response_array)),
    }
    return matrix, inverse, diagnostics, residual_vectors


def calculate_equivalent_pixel_scale(np: Any, matrix: Any) -> dict[str, float]:
    matrix_array = np.asarray(matrix, dtype=np.float64).reshape(2, 2)
    axis_pixels_per_mm = np.linalg.norm(matrix_array, axis=0)
    pixels_per_mm = float(np.mean(axis_pixels_per_mm))
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise StagePixelCalibrationError(
            "cannot calculate a positive finite pixel-per-millimetre scale"
        )
    return {
        "pixels_per_stage_mm": pixels_per_mm,
        "stage_mm_per_pixel": 1.0 / pixels_per_mm,
        "x_stage_axis_pixels_per_mm": float(axis_pixels_per_mm[0]),
        "y_stage_axis_pixels_per_mm": float(axis_pixels_per_mm[1]),
    }


def collect_hardware_samples(
    cv2: Any,
    np: Any,
    args: SimpleNamespace,
    context: dict[str, Any],
    samples: list[dict[str, Any]],
    image_directory: Path | None,
    controller_class: Any | None = None,
    absolute_mode: int | None = None,
    relative_mode: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if controller_class is None or absolute_mode is None or relative_mode is None:
        try:
            from motion_controller import (
                MODE_ABSOLUTE,
                MODE_RELATIVE,
                XYMotionController,
            )
        except Exception as exc:
            raise StagePixelCalibrationError(
                f"cannot import motion controller: {exc}"
            ) from exc
        controller_class = XYMotionController
        absolute_mode = MODE_ABSOLUTE
        relative_mode = MODE_RELATIVE

    start_position = context["stage_calibration_start_position_mm"]
    stage = None
    camera = None
    collected: list[dict[str, Any]] = []
    initial_position = None
    try:
        stage = controller_class(
            args.port,
            baudrate=args.baudrate,
            x_lead_mm_per_rev=args.x_lead_mm_per_rev,
            y_lead_mm_per_rev=args.y_lead_mm_per_rev,
            x_slave_address=args.x_slave_address,
            y_slave_address=args.y_slave_address,
        )
        initial_position = read_stage_position(np, stage, "initial position")
        if start_position is None:
            # start_position_mm 为 null: 不做初始绝对移动, 以当前位置为
            # 标定起点, 并回填实际起点供结果记录使用。
            start_position = initial_position
            initial_start_move = None
        else:
            start_position = np.asarray(start_position, dtype=np.float64)
            initial_start_move = move_stage_absolute(
                np,
                stage,
                start_position,
                args,
                absolute_mode,
                "initial calibration-start move",
                args.max_reset_mm,
            )
        context["stage_calibration_start_position_mm"] = start_position
        camera, camera_record = open_selected_camera(cv2, args)
        reference_image = capture_frame(
            camera,
            args,
            args.frame_width,
            args.frame_height,
            "initial calibration reference",
        )

        for sample_index, sample in enumerate(samples, start=1):
            target_move = move_stage_relative(
                np,
                stage,
                sample["command_delta_mm"],
                args,
                relative_mode,
                f"sample {sample_index} {sample['label']}",
            )
            target = np.asarray(target_move["expected_position_mm"], dtype=np.float64)
            moved_image = capture_frame(
                camera,
                args,
                args.frame_width,
                args.frame_height,
                f"sample {sample_index} moved",
            )
            registration = measure_image_translation(
                cv2, np, reference_image, moved_image, args
            )

            reference_image_path = None
            moved_image_path = None
            if image_directory is not None:
                stem = "{:02d}_{}".format(sample_index, sample["label"])
                reference_image_path = image_directory / f"{stem}_reference.jpg"
                moved_image_path = image_directory / f"{stem}_moved.jpg"
                write_image(cv2, reference_image_path, reference_image)
                write_image(cv2, moved_image_path, moved_image)

            command_delta = sample["command_delta_mm"]
            displacement = registration["pixel_displacement"]
            collected.append(
                {
                    "index": sample_index,
                    "label": sample["label"],
                    "axis": sample["axis"],
                    "direction": sample["direction"],
                    "step_index": sample["step_index"],
                    "step_mm": sample["step_mm"],
                    "leg": sample["leg"],
                    "command_delta_mm": command_delta.tolist(),
                    "reference_position_mm": target_move["before_position_mm"],
                    "requested_target_position_mm": target.tolist(),
                    "target_move": target_move,
                    "measured_pixel_displacement": displacement.tolist(),
                    "registration_response": registration["response"],
                    "registration": {
                        key: value
                        for key, value in registration.items()
                        if key not in ("pixel_displacement", "response")
                    },
                    "reference_image": (
                        str(reference_image_path)
                        if reference_image_path is not None
                        else None
                    ),
                    "moved_image": (
                        str(moved_image_path) if moved_image_path is not None else None
                    ),
                }
            )
            print(
                "Sample {}/{} {}: command=({:+.4f}, {:+.4f}) mm, "
                "pixels=({:+.3f}, {:+.3f}), response={:.4f}".format(
                    sample_index,
                    len(samples),
                    sample["label"],
                    command_delta[0],
                    command_delta[1],
                    displacement[0],
                    displacement[1],
                    registration["response"],
                )
            )
            reference_image = moved_image

        final_return_move = move_stage_absolute(
            np,
            stage,
            start_position,
            args,
            absolute_mode,
            "final return to calibration start",
            args.max_reset_mm,
        )
        final_position = read_stage_position(np, stage, "final position")
        return (
            collected,
            camera_record,
            {
                "initial_position_mm": initial_position.tolist(),
                "initial_calibration_start_move": initial_start_move,
                "final_return_move": final_return_move,
                "final_position_mm": final_position.tolist(),
                "explicit_final_return_performed": True,
                "sampling_motion": "two_sided_relative",
            },
        )
    except KeyboardInterrupt:
        if stage is not None:
            stop_stage(stage)
        raise
    except Exception as exc:
        if stage is not None:
            stop_stage(stage)
        if isinstance(exc, StagePixelCalibrationError):
            raise
        raise StagePixelCalibrationError(f"stage calibration failed: {exc}") from exc
    finally:
        if camera is not None:
            camera.release()
        if stage is not None:
            try:
                stage.close()
            except Exception:
                pass


def connect_lens(args: SimpleNamespace) -> tuple[Any, Any, dict[str, Any]]:
    try:
        capabilities = lens_camera.connect(args.lens_device)
        ranges = lens_camera.read_ranges(capabilities)
        return lens_camera, capabilities, ranges
    except Exception as exc:
        try:
            lens_camera.close()
        except Exception:
            pass
        raise StagePixelCalibrationError(f"cannot connect LensConnect: {exc}") from exc


def move_lens(
    api: Any,
    capabilities: Any,
    ranges: dict[str, Any],
    *,
    zoom: int,
    focus: int,
    no_init: bool,
    settle_seconds: float,
) -> dict[str, Any]:
    targets = {"zoom": zoom, "focus": focus}
    try:
        api.validate_targets(targets, ranges)
        moves = {}
        for name in ("zoom", "focus"):
            api.ensure_ready(name, init_if_needed=not no_init)
            move = api.move_connected_motor(
                name,
                targets[name],
                capabilities,
                settle_seconds=0,
                init_if_needed=None,
            )
            moves[name] = {
                key: move[key]
                for key in ("target", "before", "actual", "error", "moved")
            }
            print(
                "lens {}: target {} -> actual {} (error {})".format(
                    name, move["target"], move["actual"], move["error"]
                )
            )
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        return moves
    except Exception as exc:
        raise StagePixelCalibrationError(
            f"cannot move lens to zoom={zoom}, focus={focus}: {exc}"
        ) from exc


def run_self_test() -> int:
    np = load_numpy()
    expected = np.asarray([[47.8, 1.25], [-0.45, 52.6]], dtype=np.float64)
    commands = np.asarray(
        [
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, -2.5],
            [0.0, 2.5],
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, -2.5],
            [0.0, 2.5],
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(20260818)
    displacements = commands @ expected.T
    displacements += generator.normal(0.0, 0.03, displacements.shape)
    responses = np.asarray([0.91, 0.89, 0.93, 0.92, 0.88, 0.94, 0.90, 0.95])
    matrix, inverse, diagnostics, residuals = fit_stage_command_matrix(
        np, commands, displacements, responses
    )
    if float(np.max(np.abs(matrix - expected))) > 0.03:
        raise StagePixelCalibrationError(
            f"self-test fitted matrix is inaccurate:\n{matrix - expected}"
        )
    round_trip = inverse @ matrix
    if float(np.max(np.abs(round_trip - np.eye(2)))) > 1e-12:
        raise StagePixelCalibrationError("self-test matrix inverse failed")
    if diagnostics["rmse_px"] > 0.1 or not np.all(np.isfinite(residuals)):
        raise StagePixelCalibrationError("self-test fit diagnostics failed")
    scale = calculate_equivalent_pixel_scale(np, matrix)
    expected_pixels_per_mm = float(np.mean(np.linalg.norm(expected, axis=0)))
    if abs(scale["pixels_per_stage_mm"] - expected_pixels_per_mm) > 0.03:
        raise StagePixelCalibrationError("self-test equivalent scale failed")

    traversal_args = SimpleNamespace(
        x_step_mm=[0.5, 1.5],
        y_step_mm=0.75,
    )
    traversal_samples = build_command_samples(np, traversal_args)
    traversal_commands = [
        sample["command_delta_mm"].tolist() for sample in traversal_samples
    ]
    expected_commands = [
        [0.5, 0.0],
        [-0.5, 0.0],
        [-0.5, 0.0],
        [0.5, 0.0],
        [1.5, 0.0],
        [-1.5, 0.0],
        [-1.5, 0.0],
        [1.5, 0.0],
        [0.0, 0.75],
        [0.0, -0.75],
        [0.0, -0.75],
        [0.0, 0.75],
    ]
    if traversal_commands != expected_commands or traversal_args.y_step_mm != [0.75]:
        raise StagePixelCalibrationError("self-test two-sided traversal failed")

    class FakeAxis:
        def __init__(self) -> None:
            self.stopped = False

        def wait_for_completion(self, timeout: float) -> bool:
            return timeout > 0

        def stop(self) -> None:
            self.stopped = True

    class FakeStage:
        def __init__(self) -> None:
            self.position = np.asarray([10.0, 20.0], dtype=np.float64)
            self.commands: list[dict[str, Any]] = []
            self.x = FakeAxis()
            self.y = FakeAxis()

        def read_motor_position(self) -> dict[str, float]:
            return {"x": float(self.position[0]), "y": float(self.position[1])}

        def move_to_position(
            self,
            x_mm: float | None,
            y_mm: float | None,
            speed_rpm: int,
            mode: int,
            wait: bool,
        ) -> None:
            if wait or speed_rpm <= 0:
                raise AssertionError("invalid simulated stage command")
            self.commands.append({"x": x_mm, "y": y_mm, "mode": mode})
            if mode == 0x01:
                self.position[:] = [x_mm, y_mm]
            elif mode == 0x41:
                if x_mm is not None:
                    self.position[0] += x_mm
                if y_mm is not None:
                    self.position[1] += y_mm
            else:
                raise AssertionError("unexpected simulated motion mode")

    fake_args = SimpleNamespace(
        speed_rpm=300,
        timeout_seconds=2.0,
        position_tolerance_mm=1e-9,
        max_step_mm=5.0,
        x_limits=[-100.0, 100.0],
        y_limits=[-100.0, 100.0],
    )
    fake_stage = FakeStage()
    move_stage_absolute(
        np,
        fake_stage,
        [8.0, 19.0],
        fake_args,
        0x01,
        "simulated reference reset",
        5.0,
    )
    relative_record = move_stage_relative(
        np,
        fake_stage,
        [-2.0, 2.0],
        fake_args,
        0x41,
        "simulated calibration sample",
    )
    if (
        [command["mode"] for command in fake_stage.commands] != [0x01, 0x41]
        or not np.allclose(fake_stage.position, [6.0, 21.0])
        or relative_record["requested_relative_delta_mm"] != [-2.0, 2.0]
    ):
        raise StagePixelCalibrationError("self-test stage motion sequence failed")

    print("Self-test passed")
    print("stage command mm -> pixel matrix:")
    for row in matrix:
        print("  [{: .8f}, {: .8f}]".format(*row))
    print(
        "1 mm stage command -> {:.8f} image pixels".format(scale["pixels_per_stage_mm"])
    )
    print("RMSE: {:.5f} px".format(diagnostics["rmse_px"]))
    return 0


def run_one_zoom(
    cv2: Any,
    np: Any,
    args: SimpleNamespace,
    profile: dict[str, Any],
    run_directory: Path,
    context: dict[str, Any],
    lens_moves: dict[str, Any],
) -> dict[str, Any]:
    zoom = profile["zoom"]
    focus = profile["focus"]
    samples = profile["samples"]
    zoom_directory = run_directory / calibration_summary.zoom_entry_name(zoom)
    zoom_directory.mkdir(parents=True, exist_ok=False)
    result_path = zoom_directory / "stage_command_to_pixel.json"
    log_path = zoom_directory / "stage_command_to_pixel.log"
    image_directory = zoom_directory / "images" if args.save_images else None
    observations, camera_record, motion_record = collect_hardware_samples(
        cv2,
        np,
        args,
        context,
        samples,
        image_directory,
    )
    commands = np.asarray(
        [observation["command_delta_mm"] for observation in observations],
        dtype=np.float64,
    )
    displacements = np.asarray(
        [observation["measured_pixel_displacement"] for observation in observations],
        dtype=np.float64,
    )
    responses = np.asarray(
        [observation["registration_response"] for observation in observations],
        dtype=np.float64,
    )
    matrix, inverse, diagnostics, residual_vectors = fit_stage_command_matrix(
        np, commands, displacements, responses
    )
    equivalent_scale = calculate_equivalent_pixel_scale(np, matrix)
    calibrated_at = utc_timestamp()
    for observation, residual in zip(observations, residual_vectors):
        observation["fit_residual_vector_px"] = residual.tolist()
        observation["fit_residual_distance_px"] = float(np.linalg.norm(residual))

    section = {
        "schema_version": 1,
        "calibrated_at_utc": calibrated_at,
        "equation": (
            "[du_px, dv_px]^T = matrix_2x2 * [dx_command_mm, dy_command_mm]^T"
        ),
        "matrix_2x2": matrix.tolist(),
        "input_unit": "commanded_mm",
        "output_unit": "pixel",
        "coordinate_convention": {
            "pixel_u_positive": "image right",
            "pixel_v_positive": "image down",
            "stage_x_y": "motor-controller command axes",
        },
        "pixel_to_stage_command": {
            "equation": (
                "[dx_command_mm, dy_command_mm]^T = matrix_2x2 * [du_px, dv_px]^T"
            ),
            "matrix_2x2": inverse.tolist(),
            "input_unit": "pixel",
            "output_unit": "commanded_mm",
        },
        "equivalent_scalar_scale": {
            **equivalent_scale,
            "definition": (
                "pixels_per_stage_mm is the arithmetic mean of the fitted "
                "X-axis and Y-axis image-displacement magnitudes"
            ),
        },
        "calibration_start_stage_position_mm": context[
            "stage_calibration_start_position_mm"
        ].tolist(),
        "reference_stage_position_mm": context[
            "stage_calibration_start_position_mm"
        ].tolist(),
        "fit_quality": diagnostics,
        "detail_log_file": str(log_path),
        "standalone_result_file": str(result_path),
    }
    detail_log = {
        "schema_version": 1,
        "type": "stage_command_to_pixel_calibration_log",
        "calibrated_at_utc": calibrated_at,
        "stage_command_to_pixel": section,
        "command_arguments": dict(vars(args)),
        "stage_configuration": {
            "port": args.port,
            "baudrate": args.baudrate,
            "x_lead_mm_per_rev": args.x_lead_mm_per_rev,
            "y_lead_mm_per_rev": args.y_lead_mm_per_rev,
            "x_slave_address": args.x_slave_address,
            "y_slave_address": args.y_slave_address,
            "speed_rpm": args.speed_rpm,
            "timeout_seconds": args.timeout_seconds,
            "position_tolerance_mm": args.position_tolerance_mm,
        },
        "camera": camera_record,
        "motion": motion_record,
        "observations": observations,
        "saved_images_directory": (
            str(image_directory) if image_directory is not None else None
        ),
    }
    result_document = {
        "schema_version": 1,
        "type": "stage_command_to_pixel_calibration",
        "calibrated_at_utc": calibrated_at,
        "lens_zoom": zoom,
        "lens_focus": focus,
        "stage_command_to_pixel": section,
    }
    detail_log["lens_position"] = {
        "zoom": zoom,
        "focus": focus,
        "moves": lens_moves,
    }

    # The result lives in its own standalone JSON below working_data/
    # stage_command_to_pixel.
    write_json(log_path, detail_log)
    write_json(result_path, result_document)

    print(f"Zoom {zoom} stage command-to-pixel calibration complete")
    print(f"Calibration data: {zoom_directory}")
    print(f"Standalone result: {result_path}")
    print(f"Detail log: {log_path}")
    if image_directory is not None:
        print(f"Calibration images: {image_directory}")
    print(
        "1 mm stage command -> image displacement: {:.10f} px".format(
            equivalent_scale["pixels_per_stage_mm"]
        )
    )
    print("Fit condition: {:.4f}".format(diagnostics["condition_number"]))
    return {
        "zoom": zoom,
        "focus": focus,
        "result_document": result_document,
        "result_path": result_path,
        "log_path": log_path,
        "fit_quality": diagnostics,
    }


def run(args: SimpleNamespace) -> int:
    np = load_numpy()
    args.output_root = str(paths.resolve_project_path(args.output_root))
    summary_path = calibration_path(args)
    calibration_document = load_calibration_document(summary_path)
    profiles = build_zoom_profiles(args, calibration_document, summary_path)
    if not math.isfinite(args.lens_settle_seconds) or args.lens_settle_seconds < 0:
        raise StagePixelCalibrationError("lens_settle_seconds cannot be negative")

    for profile in profiles:
        profile_args = SimpleNamespace(**vars(args))
        profile_args.x_step_mm = profile["x_step_mm"]
        profile_args.y_step_mm = profile["y_step_mm"]
        profile["args"] = profile_args
        profile["samples"] = build_command_samples(np, profile_args)
        validate_args(profile_args)

    start_position = (
        None
        if args.start_position_mm is None
        else numeric_pair(args.start_position_mm, "start_position_mm", np)
    )
    print(f"Calibration summary: {summary_path}")
    for profile in profiles:
        print(
            "Zoom {} uses autofocus Focus {}; X steps {}; Y steps {}".format(
                profile["zoom"],
                profile["focus"],
                profile["x_step_mm"],
                profile["y_step_mm"],
            )
        )

    cv2, _ = camera_tools.load_image_tools(StagePixelCalibrationError)
    run_directory = create_run_directory(Path(args.output_root), run_timestamp())
    context = {"stage_calibration_start_position_mm": start_position}
    completed: list[dict[str, Any]] = []
    api = None
    try:
        api, capabilities, ranges = connect_lens(args)
        for profile in profiles:
            print("\n=== Calibrating Zoom {} ===".format(profile["zoom"]))
            print_plan(
                profile["args"],
                context["stage_calibration_start_position_mm"],
                profile["samples"],
            )
            lens_moves = move_lens(
                api,
                capabilities,
                ranges,
                zoom=profile["zoom"],
                focus=profile["focus"],
                no_init=bool(args.lens_no_init),
                settle_seconds=float(args.lens_settle_seconds),
            )
            completed.append(
                run_one_zoom(
                    cv2,
                    np,
                    profile["args"],
                    profile,
                    run_directory,
                    context,
                    lens_moves,
                )
            )
    finally:
        if api is not None:
            try:
                api.close()
            except Exception:
                pass

    saved_summary_path = None
    for result in completed:
        try:
            saved_summary_path = append_calibration_summary(
                result["result_document"], args, zoom=result["zoom"]
            )
        except (OSError, ValueError) as exc:
            print(
                f"Warning: cannot append calibration summary: {exc}",
                file=sys.stderr,
            )

    batch_path = run_directory / "stage_command_to_pixel_batch.json"
    write_json(
        batch_path,
        {
            "schema_version": 1,
            "type": "dual_zoom_stage_command_to_pixel_calibration",
            "completed_at_utc": utc_timestamp(),
            "calibration_summary": str(summary_path),
            "results": [
                {
                    "lens_zoom": result["zoom"],
                    "lens_focus": result["focus"],
                    "standalone_result_file": str(result["result_path"]),
                    "detail_log_file": str(result["log_path"]),
                    "fit_quality": result["fit_quality"],
                }
                for result in completed
            ],
        },
    )
    print("\nDual-Zoom stage command-to-pixel calibration complete")
    print(f"Calibration data: {run_directory}")
    print(f"Batch result: {batch_path}")
    if saved_summary_path is not None:
        print(f"Calibration summary updated: {saved_summary_path}")
    print_completed_calibration_errors(completed)
    return 0


def main() -> int:
    try:
        return run(load_args())
    except StagePixelCalibrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; stage stop was requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
