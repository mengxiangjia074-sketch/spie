"""Capture a PCB as a serpentine image grid from the current stage position.

Board, stage, and camera-intrinsics calibrations are selected from
detection/calibration/output/calibrate.json using the configured lens_zoom.
This script does not move the lens or align a physical point to a pixel; the
two-zoom orchestration is implemented by capture_pcb_two_zooms.py.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Capture helpers live in this directory; LensCamera and motion_controller.py
# live one level up in detection, and calibration scripts live alongside them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))

from LensCamera import camera as camera_tools
from LensCamera import paths
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError

from calibrate_stage_to_pixel import (
    move_stage_absolute,
    read_stage_position,
    stop_stage,
)


DEFAULT_SETTINGS_PATH = (
    paths.PROJECT_ROOT / "config" / "pcb_capture" / "capture.jsonnet"
)
DEFAULT_CALIBRATION_SUMMARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "calibration"
    / "output"
    / "calibrate.json"
)
LENS_CALIBRATION_ENTRY_TYPES = (
    "lens_checkerboard_camera_calibration",
    "full_electric_lens_checkerboard_calibration",
)


class PcbMosaicError(RuntimeError):
    pass


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        raise PcbMosaicError(f"cannot read calibration JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PcbMosaicError(f"calibration JSON root must be an object: {path}")
    return value


def _matrix(np: Any, value: Any, shape: tuple[int, int], name: str) -> Any:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as exc:
        raise PcbMosaicError(
            f"{name} must be a numeric {shape[0]}x{shape[1]} matrix"
        ) from exc
    if not np.all(np.isfinite(matrix)):
        raise PcbMosaicError(f"{name} contains NaN or infinity")
    if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-12:
        raise PcbMosaicError(f"{name} is singular")
    return matrix


def parse_board_calibration(
    np: Any, document: dict[str, Any]
) -> dict[str, Any]:
    """Extract the board-to-pixel matrix and saved stage configuration."""
    section = document.get("board_to_pixel")
    if not isinstance(section, dict) or "matrix_2x3" not in section:
        raise PcbMosaicError(
            "calibration JSON has no board_to_pixel.matrix_2x3 field"
        )
    board_to_pixel = _matrix(
        np, section["matrix_2x3"], (2, 3), "board_to_pixel.matrix_2x3"
    )

    stage_reference = document.get("stage_reference")
    if not isinstance(stage_reference, dict):
        stage_reference = {}
    configuration = stage_reference.get("configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise PcbMosaicError("stage_reference.configuration must be an object")

    return {
        "board_to_pixel": board_to_pixel,
        "stage_reference_configuration": configuration,
    }


def parse_stage_calibration(
    np: Any, document: dict[str, Any]
) -> dict[str, Any]:
    """Extract mutually consistent stage/pixel displacement matrices."""
    section = document.get("stage_command_to_pixel")
    if not isinstance(section, dict) or "matrix_2x2" not in section:
        raise PcbMosaicError(
            "stage calibration JSON has no stage_command_to_pixel.matrix_2x2"
        )
    stage_to_pixel = _matrix(
        np,
        section["matrix_2x2"],
        (2, 2),
        "stage_command_to_pixel.matrix_2x2",
    )

    inverse_section = section.get("pixel_to_stage_command")
    if not isinstance(inverse_section, dict) or "matrix_2x2" not in inverse_section:
        inverse_section = document.get("pixel_to_stage_command")
    if isinstance(inverse_section, dict) and "matrix_2x2" in inverse_section:
        pixel_to_stage = _matrix(
            np,
            inverse_section["matrix_2x2"],
            (2, 2),
            "pixel_to_stage_command.matrix_2x2",
        )
    else:
        pixel_to_stage = np.linalg.inv(stage_to_pixel)

    inverse_error = float(
        np.max(np.abs(pixel_to_stage @ stage_to_pixel - np.eye(2)))
    )
    if inverse_error > 1e-6:
        raise PcbMosaicError(
            "stage command forward/inverse matrices are inconsistent; "
            f"maximum inverse error is {inverse_error:.6g}"
        )

    return {
        "stage_to_pixel": stage_to_pixel,
        "pixel_to_stage": pixel_to_stage,
        "source_board_calibration_json": document.get(
            "source_board_calibration_json"
        ),
    }


def load_calibration_summary_document(
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Load a calibrate.json summary and validate its top-level entries."""
    resolved_path = (
        Path(path) if path is not None else DEFAULT_CALIBRATION_SUMMARY_PATH
    )
    document = _load_json_object(resolved_path)
    if not isinstance(document.get("entries"), list):
        raise PcbMosaicError(
            "calibration file {} has no 'entries' list; expected a "
            "calibrate.json-style summary with all calibration results".format(
                resolved_path
            )
        )
    return document


def load_settings() -> dict[str, Any]:
    try:
        return JsonnetSettings(DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)).values
    except SettingsError as exc:
        raise PcbMosaicError(str(exc)) from exc


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise PcbMosaicError(
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


def load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise PcbMosaicError(
            "NumPy is required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return np


def parse_intrinsics_section(
    np: Any,
    path: Path,
    section: Any,
    position_name: str | None,
) -> dict[str, Any]:
    if not isinstance(section, dict) or "camera_matrix" not in section:
        raise PcbMosaicError(
            f"{path} has no camera_matrix/distortion_coefficients section"
        )
    try:
        matrix = np.asarray(section["camera_matrix"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(
            section["distortion_coefficients"], dtype=np.float64
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PcbMosaicError(f"{path} intrinsics must be numeric") from exc
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise PcbMosaicError(f"{path} intrinsics contain NaN or infinity")
    width = int(section["image_width"])
    height = int(section["image_height"])
    if width <= 0 or height <= 0:
        raise PcbMosaicError(f"{path} image_width/image_height must be above 0")
    return {
        "file": str(path),
        "position_name": position_name,
        "camera_matrix": matrix,
        "distortion_coefficients": distortion,
        "image_width_px": width,
        "image_height_px": height,
    }


def zoom_address(value: Any, name: str = "lens_zoom") -> int:
    if isinstance(value, bool):
        raise PcbMosaicError(f"{name} must be an integer address, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PcbMosaicError(f"{name} must not be empty")
        try:
            return int(text, 0)
        except ValueError as exc:
            raise PcbMosaicError(f"{name} must be an integer address") from exc
    raise PcbMosaicError(f"{name} must be an integer address")


def zoom_entry_name(value: Any) -> str:
    return "zoom_{:05d}".format(zoom_address(value))


def newest_summary_entry_by_lens_zoom(
    document: dict[str, Any],
    entry_type: str | tuple[str, ...],
    lens_zoom: Any,
    source_path: Path | str,
) -> dict[str, Any]:
    requested_zoom = zoom_address(lens_zoom)
    requested_name = zoom_entry_name(requested_zoom)
    entry_types = (entry_type,) if isinstance(entry_type, str) else entry_type
    matches = []
    for entry in document["entries"]:
        if not isinstance(entry, dict) or entry.get("type") not in entry_types:
            continue
        matched = False
        try:
            entry_zoom = zoom_address(entry.get("lens_zoom"))
        except PcbMosaicError:
            entry_zoom = None
        if entry_zoom == requested_zoom:
            matched = True
        elif entry.get("name") == requested_name:
            matched = True
        else:
            positions = entry.get("positions")
            if isinstance(positions, list):
                for item in positions:
                    if not isinstance(item, dict):
                        continue
                    try:
                        item_zoom = zoom_address(item.get("lens_zoom"))
                    except PcbMosaicError:
                        item_zoom = None
                    if (
                        item_zoom == requested_zoom
                        or item.get("name") == requested_name
                    ):
                        matched = True
                        break
        if matched:
            matches.append(entry)
    if not matches:
        raise PcbMosaicError(
            "calibration summary {} has no {!r} result with lens_zoom={}; "
            "run the corresponding calibration at this zoom first".format(
                source_path, entry_type, requested_zoom
            )
        )
    return matches[-1]


def select_intrinsics_position_for_zoom(
    entry: dict[str, Any],
    path: Path | str,
    lens_zoom: Any,
) -> dict[str, Any]:
    positions = entry.get("positions")
    if not isinstance(positions, list) or not positions:
        raise PcbMosaicError(f"lens calibration entry in {path} has no positions")

    requested_zoom = zoom_address(lens_zoom)
    requested_name = zoom_entry_name(requested_zoom)
    selected = None
    for item in positions:
        if not isinstance(item, dict):
            continue
        item_zoom = item.get("lens_zoom", entry.get("lens_zoom"))
        try:
            if item_zoom is not None and zoom_address(item_zoom) == requested_zoom:
                selected = item
                break
        except PcbMosaicError:
            pass
        if item.get("name") == requested_name:
            selected = item
            break
    if selected is None and len(positions) == 1:
        selected = positions[0]
    if selected is None:
        raise PcbMosaicError(
            "lens calibration entry in {} has no position for lens_zoom={}".format(
                path, requested_zoom
            )
        )
    return selected


def load_intrinsics_from_summary(
    np: Any,
    path: Path | str | None = None,
    *,
    lens_zoom: Any,
    document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load camera intrinsics matching ``lens_zoom`` from calibrate.json."""
    if path is None:
        path = DEFAULT_CALIBRATION_SUMMARY_PATH
    if document is None:
        document = load_calibration_summary_document(path)
    entry = newest_summary_entry_by_lens_zoom(
        document,
        LENS_CALIBRATION_ENTRY_TYPES,
        lens_zoom,
        path,
    )
    selected = select_intrinsics_position_for_zoom(entry, path, lens_zoom)
    selected_name = selected.get("name") if isinstance(selected, dict) else None
    return parse_intrinsics_section(np, path, selected, selected_name)


def load_board_calibration_for_zoom(
    np: Any,
    document: dict[str, Any],
    path: Path | str,
    lens_zoom: Any,
) -> dict[str, Any]:
    entry = newest_summary_entry_by_lens_zoom(
        document,
        "calibration_board_to_pixel_affine_core",
        lens_zoom,
        path,
    )
    calibration = parse_board_calibration(np, entry)
    calibration["source_file"] = str(path)
    calibration["source_saved_at_utc"] = entry.get("saved_at_utc")
    calibration["source_name"] = entry.get("name")
    calibration["lens_zoom"] = zoom_address(lens_zoom)
    return calibration


def load_stage_calibration_for_zoom(
    np: Any,
    document: dict[str, Any],
    path: Path | str,
    lens_zoom: Any,
) -> dict[str, Any]:
    entry = newest_summary_entry_by_lens_zoom(
        document,
        "stage_command_to_pixel_calibration",
        lens_zoom,
        path,
    )
    stage = parse_stage_calibration(np, entry)
    stage["source_file"] = str(path)
    stage["source_saved_at_utc"] = entry.get("saved_at_utc")
    stage["source_name"] = entry.get("name")
    stage["lens_zoom"] = zoom_address(lens_zoom)
    return stage


def load_phase_calibrations(
    np: Any,
    calibration_path: Path,
    phase_name: str,
    lens_zoom: Any,
    document: dict[str, Any],
) -> dict[str, Any]:
    zoom = zoom_address(lens_zoom, f"{phase_name}_zoom")
    board = load_board_calibration_for_zoom(np, document, calibration_path, zoom)
    stage = load_stage_calibration_for_zoom(np, document, calibration_path, zoom)
    intrinsics = load_intrinsics_from_summary(
        np,
        calibration_path,
        lens_zoom=zoom,
        document=document,
    )
    return {
        "phase": phase_name,
        "lens_zoom": zoom,
        "board": board,
        "stage": stage,
        "intrinsics": intrinsics,
    }


def resolve_stage_configuration(
    args: SimpleNamespace,
    calibration: dict[str, Any],
) -> None:
    saved = calibration.get("stage_reference_configuration") or {}
    default_port = "/dev/lensdetect-stage" if platform.system() == "Linux" else "COM3"
    for name, key, converter, fallback in (
        ("port", "port", str, default_port),
        ("baudrate", "baudrate", int, 38400),
        ("x_lead_mm_per_rev", "x_lead_mm_per_rev", float, 1.0),
        ("y_lead_mm_per_rev", "y_lead_mm_per_rev", float, 1.0),
        ("x_slave_address", "x_slave_address", int, 1),
        ("y_slave_address", "y_slave_address", int, 2),
    ):
        value = getattr(args, name)
        if value is None:
            value = saved.get(key, fallback)
        try:
            setattr(args, name, converter(value))
        except (TypeError, ValueError) as exc:
            raise PcbMosaicError(
                f"invalid stage configuration {name}={value!r}"
            ) from exc


def validate_args(
    args: SimpleNamespace, *, require_pcb_dimensions: bool = True
) -> None:
    zoom_address(args.lens_zoom)
    if require_pcb_dimensions:
        if not math.isfinite(args.pcb_width_mm) or args.pcb_width_mm <= 0:
            raise PcbMosaicError("pcb_width_mm must be finite and above 0")
        if not math.isfinite(args.pcb_height_mm) or args.pcb_height_mm <= 0:
            raise PcbMosaicError("pcb_height_mm must be finite and above 0")
    if not math.isfinite(args.capture_margin_mm) or args.capture_margin_mm < 0:
        raise PcbMosaicError("capture_margin_mm must be finite and at least 0")
    if not 0.2 <= args.overlap_fraction < 0.95:
        raise PcbMosaicError("overlap_fraction must be at least 0.2 and below 0.95")
    if args.max_step_mm <= 0 or args.max_return_mm <= 0:
        raise PcbMosaicError("max_step_mm and max_return_mm must be above 0")
    if args.frame_width <= 0 or args.frame_height <= 0:
        raise PcbMosaicError("camera frame dimensions must be above 0")
    if not str(args.port).strip():
        raise PcbMosaicError("port cannot be empty")
    if args.x_slave_address == args.y_slave_address:
        raise PcbMosaicError("X/Y slave addresses must be different")
    if args.camera is None and not str(args.camera_name).strip():
        raise PcbMosaicError("camera_name cannot be empty")


def grid_axis(
    span_mm: float,
    fov_mm: float,
    overlap_fraction: float,
    capture_margin_mm: float,
) -> tuple[int, float, float]:
    """Return a grid anchored at zero and expanded only at the far edge."""
    capture_span_mm = span_mm + capture_margin_mm
    if capture_span_mm <= fov_mm + 1e-9:
        return 1, 0.0, 0.0
    step = fov_mm * (1.0 - overlap_fraction)
    count = int(math.ceil((capture_span_mm - fov_mm) / step - 1e-12)) + 1
    spacing = (capture_span_mm - fov_mm) / (count - 1)
    return count, spacing, 0.0


def build_capture_cells(
    np: Any,
    calibration: dict[str, Any],
    stage_calibration: dict[str, Any],
    args: SimpleNamespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    board_to_pixel = calibration["board_to_pixel"]
    pixel_to_stage = stage_calibration["pixel_to_stage"]
    scale_x = float(np.linalg.norm(board_to_pixel[:, 0]))
    scale_y = float(np.linalg.norm(board_to_pixel[:, 1]))
    fov_width_mm = args.frame_width / scale_x
    fov_height_mm = args.frame_height / scale_y
    columns, spacing_x_mm, start_x_mm = grid_axis(
        args.pcb_width_mm,
        fov_width_mm,
        args.overlap_fraction,
        args.capture_margin_mm,
    )
    rows, spacing_y_mm, start_y_mm = grid_axis(
        args.pcb_height_mm,
        fov_height_mm,
        args.overlap_fraction,
        args.capture_margin_mm,
    )

    cells = []
    for row in range(rows):
        column_order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        for col in column_order:
            board_delta = np.asarray(
                [
                    start_x_mm + col * spacing_x_mm,
                    start_y_mm + row * spacing_y_mm,
                ],
                dtype=np.float64,
            )
            # 相机随位移台移动时画面内容反向位移: 要让视野沿板 +X/+Y
            # 前进, 画面内容需要向相反方向移动, 因此取负号。
            stage_offset = -(pixel_to_stage @ (board_to_pixel[:, :2] @ board_delta))
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "board_offset_mm": board_delta.tolist(),
                    "stage_offset_mm": stage_offset.tolist(),
                }
            )

    grid = {
        "pcb_width_mm": float(args.pcb_width_mm),
        "pcb_height_mm": float(args.pcb_height_mm),
        "capture_margin_mm": float(args.capture_margin_mm),
        "overlap_fraction": float(args.overlap_fraction),
        "fov_width_mm": fov_width_mm,
        "fov_height_mm": fov_height_mm,
        "columns": columns,
        "rows": rows,
        "spacing_x_mm": spacing_x_mm,
        "spacing_y_mm": spacing_y_mm,
        "coverage_bounds_board_mm": {
            "x_min": start_x_mm,
            "x_max": start_x_mm + (columns - 1) * spacing_x_mm + fov_width_mm,
            "y_min": start_y_mm,
            "y_max": start_y_mm + (rows - 1) * spacing_y_mm + fov_height_mm,
        },
        "minimum_overlap_x": (
            1.0 - spacing_x_mm / fov_width_mm if columns > 1 else 0.0
        ),
        "minimum_overlap_y": (1.0 - spacing_y_mm / fov_height_mm if rows > 1 else 0.0),
        "capture_order": "serpentine: left-to-right, one row down, right-to-left",
        "frames_inside_pcb": bool(args.capture_margin_mm <= 1e-9),
    }
    return cells, grid


def create_run_directory(base_dir_text: str) -> Path:
    # Relative output roots are anchored to the project root (never the
    # current directory), and all output must stay inside working_data/.
    base_dir = paths.resolve_project_path(base_dir_text)
    working_root = (paths.PROJECT_ROOT / "working_data").resolve()
    if not base_dir.resolve().is_relative_to(working_root):
        raise PcbMosaicError(
            f"output_dir {base_dir} is outside {working_root}; output must "
            "stay inside working_data/"
        )
    run_dir = base_dir / run_timestamp()
    suffix = 2
    while run_dir.exists():
        run_dir = base_dir / "{}_{:02d}".format(run_timestamp(), suffix)
        suffix += 1
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PcbMosaicError(
            f"cannot create output directory {run_dir}: {exc}"
        ) from exc
    return run_dir


def open_camera(cv2: Any, args: SimpleNamespace) -> tuple[Any, dict[str, Any]]:
    if args.camera is None:
        selected = camera_tools.find_camera_by_name(
            args.camera_name, PcbMosaicError
        )
    else:
        selected = {
            "index": int(args.camera),
            "name": None,
            "source": "config_index",
        }
    camera = camera_tools.open_camera(cv2, camera_tools.camera_source(selected))
    if not camera.isOpened():
        raise PcbMosaicError(
            f"cannot open camera index {selected['index']} ({selected.get('name')})"
        )
    settings = SimpleNamespace(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        disable_camera_autofocus=not args.keep_camera_autofocus,
    )
    camera_tools.configure_camera(cv2, camera, settings)
    buffer_size_requested = False
    buffer_size_actual = None
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        try:
            buffer_size_requested = bool(camera.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            buffer_size_actual = float(camera.get(cv2.CAP_PROP_BUFFERSIZE))
        except Exception:
            buffer_size_requested = False
    args._capture_buffer_size_actual = buffer_size_actual
    for _ in range(args.camera_startup_discard_frames):
        camera.read()
    record = dict(selected)
    record.update(
        {
            "requested_frame_width_px": int(args.frame_width),
            "requested_frame_height_px": int(args.frame_height),
            "autofocus_kept": bool(args.keep_camera_autofocus),
            "capture_buffer_size_requested": 1,
            "capture_buffer_size_set": buffer_size_requested,
            "capture_buffer_size_actual": buffer_size_actual,
        }
    )
    return camera, record


def capture_frame(
    camera: Any,
    args: SimpleNamespace,
    label: str,
) -> Any:
    if args.capture_delay_seconds > 0:
        time.sleep(args.capture_delay_seconds)
    frame = None
    ok = False
    buffer_size = getattr(args, "_capture_buffer_size_actual", None)
    required_discards = int(buffer_size) if buffer_size and buffer_size > 0 else 0
    discard_frames = max(int(args.capture_discard_frames), required_discards)
    for _ in range(discard_frames + 1):
        ok, frame = camera.read()
    if not ok or frame is None:
        raise PcbMosaicError(f"camera returned no image for {label}")
    return frame


def build_undistort_maps(
    cv2: Any,
    intrinsics: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[Any, Any]:
    matrix = intrinsics["camera_matrix"].copy()
    matrix[0, :] *= frame_width / intrinsics["image_width_px"]
    matrix[1, :] *= frame_height / intrinsics["image_height_px"]
    map1, map2 = cv2.initUndistortRectifyMap(
        matrix,
        intrinsics["distortion_coefficients"],
        None,
        matrix,
        (frame_width, frame_height),
        cv2.CV_32FC1,
    )
    return map1, map2


def write_png(cv2: Any, path: Path, image: Any) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise PcbMosaicError(f"OpenCV cannot encode image as {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise PcbMosaicError(f"cannot write image {path}: {exc}") from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise PcbMosaicError(f"cannot write JSON {path}: {exc}") from exc


def intrinsics_record(intrinsics: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": intrinsics["file"],
        "position_name": intrinsics["position_name"],
        "image_width_px": intrinsics["image_width_px"],
        "image_height_px": intrinsics["image_height_px"],
        "camera_matrix": intrinsics["camera_matrix"].tolist(),
        "distortion_coefficients": intrinsics["distortion_coefficients"].tolist(),
    }


def warn_stage_source(
    board_path: str,
    stage_calibration: dict[str, Any],
) -> None:
    source_board = stage_calibration.get("source_board_calibration_json")
    if (
        source_board
        and str(Path(source_board).resolve()).casefold()
        != str(Path(board_path).resolve()).casefold()
    ):
        print(
            "Warning: the stage calibration was recorded with a different "
            f"board calibration ({source_board}); planned overlap may be "
            "inaccurate. Recalibrate the stage matrix if the magnification "
            "changed.",
            file=sys.stderr,
        )


def capture_mosaic_from_current_position(
    np: Any,
    cv2: Any,
    args: SimpleNamespace,
    calibration: dict[str, Any],
    stage_calibration: dict[str, Any],
    intrinsics: dict[str, Any],
    run_dir: Path,
    *,
    capture_cells: list[dict[str, Any]] | None = None,
    capture_grid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    board_path = calibration["source_file"]
    stage_calibration_path = stage_calibration["source_file"]
    if (capture_cells is None) != (capture_grid is None):
        raise PcbMosaicError(
            "capture_cells and capture_grid must either both be provided or both be null"
        )
    if capture_cells is None:
        cells, grid = build_capture_cells(np, calibration, stage_calibration, args)
    else:
        cells = list(capture_cells)
        grid = dict(capture_grid)
        if not cells:
            raise PcbMosaicError("custom capture grid contains no cells")
    images_dir = run_dir / "images"
    images_dir.mkdir()

    print(
        "{}: {}".format(
            calibration.get("source_label", "Board calibration"), board_path
        )
    )
    print(f"Stage calibration: {stage_calibration_path}")
    warn_stage_source(board_path, stage_calibration)
    if args.capture_margin_mm <= 1e-9 and (
        grid["fov_width_mm"] > grid["pcb_width_mm"] + 1e-9
        or grid["fov_height_mm"] > grid["pcb_height_mm"] + 1e-9
    ):
        print(
            "Warning: frame FOV {:.2f} x {:.2f} mm exceeds the PCB "
            "{:.2f} x {:.2f} mm, so single frames capture beyond the "
            "PCB bounds.".format(
                grid["fov_width_mm"],
                grid["fov_height_mm"],
                grid["pcb_width_mm"],
                grid["pcb_height_mm"],
            ),
            file=sys.stderr,
        )
    print(f"Camera intrinsics: {intrinsics['file']} ({intrinsics['position_name']})")
    print(
        "PCB {:.3f} x {:.3f} mm; margin {:.3f} mm; frame FOV {:.3f} x {:.3f} mm".format(
            grid["pcb_width_mm"],
            grid["pcb_height_mm"],
            grid["capture_margin_mm"],
            grid["fov_width_mm"],
            grid["fov_height_mm"],
        )
    )
    bounds = grid["coverage_bounds_board_mm"]
    print(
        "Coverage: X [{:.3f}, {:.3f}] mm; Y [{:.3f}, {:.3f}] mm".format(
            bounds["x_min"], bounds["x_max"], bounds["y_min"], bounds["y_max"]
        )
    )
    print(
        "Grid: {} columns x {} rows = {} cells; spacing {:.3f} x {:.3f} mm; "
        "min overlap {:.1%} x {:.1%}".format(
            grid["columns"],
            grid["rows"],
            len(cells),
            grid["spacing_x_mm"],
            grid["spacing_y_mm"],
            grid["minimum_overlap_x"],
            grid["minimum_overlap_y"],
        )
    )
    print("Order: {}".format(grid["capture_order"]))

    from motion_controller import MODE_ABSOLUTE, XYMotionController

    stage = None
    camera = None
    start_position = None
    observations: list[dict[str, Any]] = []
    try:
        stage = XYMotionController(
            args.port,
            baudrate=args.baudrate,
            x_lead_mm_per_rev=args.x_lead_mm_per_rev,
            y_lead_mm_per_rev=args.y_lead_mm_per_rev,
            x_slave_address=args.x_slave_address,
            y_slave_address=args.y_slave_address,
        )
        start_position = read_stage_position(np, stage, "initial sweep position")
        camera, camera_record = open_camera(cv2, args)
        undistort_maps = None
        frame_width = 0
        frame_height = 0

        for index, cell in enumerate(cells, start=1):
            label = f"cell r{cell['row']:02d} c{cell['col']:02d}"
            target = start_position + np.asarray(
                cell["stage_offset_mm"], dtype=np.float64
            )
            move = move_stage_absolute(
                np,
                stage,
                target,
                args,
                MODE_ABSOLUTE,
                label,
                args.max_step_mm,
            )
            frame = capture_frame(camera, args, label)
            height, width = frame.shape[:2]
            if width != frame_width or height != frame_height:
                frame_width, frame_height = width, height
                undistort_maps = (
                    build_undistort_maps(cv2, intrinsics, frame_width, frame_height)
                    if args.undistort
                    else None
                )

            stem = "r{:02d}_c{:02d}".format(cell["row"], cell["col"])
            raw_path = None
            undistorted_path = None
            if args.save_raw:
                raw_path = images_dir / f"{stem}_raw.png"
                write_png(cv2, raw_path, frame)
            if args.undistort:
                undistorted = cv2.remap(
                    frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR
                )
                undistorted_path = images_dir / f"{stem}_undistorted.png"
                write_png(cv2, undistorted_path, undistorted)

            observations.append(
                {
                    "index": index,
                    "row": cell["row"],
                    "col": cell["col"],
                    "board_offset_mm": cell["board_offset_mm"],
                    "requested_stage_offset_mm": cell["stage_offset_mm"],
                    "requested_target_position_mm": target.tolist(),
                    "actual_position_mm": move["actual_position_mm"],
                    "position_error_mm": move["error_distance_mm"],
                    "travel_distance_mm": move["travel_distance_mm"],
                    "image_size_px": [int(width), int(height)],
                    "raw_image": str(raw_path) if raw_path is not None else None,
                    "undistorted_image": (
                        str(undistorted_path) if undistorted_path is not None else None
                    ),
                    **(
                        {"pixel_offset_px": cell["pixel_offset_px"]}
                        if "pixel_offset_px" in cell
                        else {}
                    ),
                }
            )
            print(
                "[{}/{}] {}: target=({:.4f}, {:.4f}) mm, error={:.4f} mm".format(
                    index,
                    len(cells),
                    label,
                    target[0],
                    target[1],
                    move["error_distance_mm"],
                )
            )

        return_move = move_stage_absolute(
            np,
            stage,
            start_position,
            args,
            MODE_ABSOLUTE,
            "return to initial sweep position",
            args.max_return_mm,
        )
        final_position = read_stage_position(np, stage, "final position")

        print("PCB serpentine capture complete")
        print(f"Captured {len(observations)} images: {images_dir}")
        print(
            "Returned to local sweep start: ({:.6f}, {:.6f}) mm".format(
                final_position[0], final_position[1]
            )
        )
        return {
            "board_calibration": str(board_path),
            "stage_calibration": str(stage_calibration_path),
            "camera_intrinsics": intrinsics_record(intrinsics),
            "grid": grid,
            "camera": camera_record,
            "motion": {
                "initial_position_mm": start_position.tolist(),
                "return_to_initial": return_move,
                "final_position_mm": final_position.tolist(),
                "returned_to_initial": bool(
                    np.linalg.norm(final_position - start_position)
                    <= args.position_tolerance_mm
                ),
            },
            "observations": observations,
            "images_directory": str(images_dir),
        }
    except KeyboardInterrupt:
        if stage is not None:
            stop_stage(stage)
            try_return_home(np, stage, start_position, args)
        raise
    except Exception as exc:
        if stage is not None:
            stop_stage(stage)
            try_return_home(np, stage, start_position, args)
        if isinstance(exc, PcbMosaicError):
            raise
        raise PcbMosaicError(f"PCB serpentine capture failed: {exc}") from exc
    finally:
        if camera is not None:
            camera.release()
        if stage is not None:
            try:
                stage.close()
            except Exception:
                pass


def run(args: SimpleNamespace) -> int:
    np = load_numpy()
    cv2, _ = camera_tools.load_image_tools(PcbMosaicError)
    lens_zoom = zoom_address(args.lens_zoom)
    calibration_path = (
        Path(args.calibration).resolve()
        if args.calibration
        else DEFAULT_CALIBRATION_SUMMARY_PATH
    )
    document = load_calibration_summary_document(calibration_path)
    calibrations = load_phase_calibrations(
        np,
        calibration_path,
        "mosaic",
        lens_zoom,
        document,
    )
    resolve_stage_configuration(args, calibrations["board"])
    validate_args(args)

    run_dir = create_run_directory(args.output_dir)
    report_path = run_dir / "pcb_mosaic.json"
    log: dict[str, Any] = {
        "schema_version": 1,
        "type": "pcb_serpentine_capture",
        "created_at_utc": utc_timestamp(),
        "status": "starting",
        "calibration_summary": str(calibration_path),
        "settings": dict(vars(args)),
        "lens_zoom": lens_zoom,
        "board_calibration": calibrations["board"]["source_file"],
        "stage_calibration": calibrations["stage"]["source_file"],
        "camera_intrinsics": intrinsics_record(calibrations["intrinsics"]),
    }
    write_json(report_path, log)
    print(f"Output: {run_dir}")

    try:
        log["mosaic"] = capture_mosaic_from_current_position(
            np,
            cv2,
            args,
            calibrations["board"],
            calibrations["stage"],
            calibrations["intrinsics"],
            run_dir,
        )
        log["status"] = "completed"
        log["completed_at_utc"] = utc_timestamp()
        write_json(report_path, log)

        print("PCB serpentine capture complete")
        print(f"Report: {report_path}")
        return 0
    except KeyboardInterrupt:
        log["status"] = "interrupted"
        log["failed_at_utc"] = utc_timestamp()
        write_json(report_path, log)
        raise
    except Exception as exc:
        log["status"] = "failed"
        log["error"] = str(exc)
        log["failed_at_utc"] = utc_timestamp()
        write_json(report_path, log)
        if isinstance(exc, PcbMosaicError):
            raise
        raise PcbMosaicError(f"PCB full/local capture failed: {exc}") from exc


def try_return_home(
    np: Any, stage: Any, start_position: Any, args: SimpleNamespace
) -> None:
    if start_position is None:
        return
    try:
        from motion_controller import MODE_ABSOLUTE

        move_stage_absolute(
            np,
            stage,
            start_position,
            args,
            MODE_ABSOLUTE,
            "emergency return to initial position",
            args.max_return_mm,
        )
    except Exception:
        pass


def main() -> int:
    try:
        return run(load_args())
    except PcbMosaicError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; stage stop was requested", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
