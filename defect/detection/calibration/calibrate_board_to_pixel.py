"""Fit a calibration-board-to-pixel affine transform from a small field of view.

This program is intended for a fixed telephoto camera looking at a planar
checkerboard.  Only a rectangular, locally visible group of inner corners is
required; the entire physical board does not need to fit in the image.

The coordinate convention is explicit because an ordinary checkerboard has no
way to encode the absolute row and column of a partial view.  By default, the
detected corners are reordered so the visible upper-left inner corner is board
coordinate (0, 0), image-right columns point along board +X, and image-down
rows point along board +Y.  Use the origin and axis settings when the local
group belongs to a larger board.

All parameters are read from config/board_to_pixel/settings.jsonnet.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# The LensCamera package and motion_controller.py live in the detection
# directory one level above these calibration scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration_summary
from LensCamera import camera as camera_tools
from LensCamera import paths
from LensCamera.json_io import write_json
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError


DEFAULT_OUTPUT_ROOT = paths.PROJECT_ROOT / "working_data" / "board_pixel_affine"
DEFAULT_SETTINGS_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "board_to_pixel"
    / "board2pixel.jsonnet"
)
AXIS_VECTORS = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}


class AffineCalibrationError(RuntimeError):
    pass


def load_settings() -> dict[str, Any]:
    try:
        return JsonnetSettings(
            DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)
        ).values
    except SettingsError as exc:
        raise AffineCalibrationError(str(exc)) from exc


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise AffineCalibrationError(
            f"{DEFAULT_SETTINGS_PATH} missing required setting {name!r}"
        )
    item = settings[name]
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def load_args(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    settings = load_settings() if settings is None else settings
    return SimpleNamespace(
        **{name: setting_value(settings, name) for name in settings}
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def append_calibration_summary(
    core_result: dict[str, Any],
    stage_reference: dict[str, Any],
    args: SimpleNamespace,
) -> Path:
    lens_position = calibration_summary.current_lens_position(
        device_number=getattr(args, "lens_device", None)
    )
    return calibration_summary.append_entry(
        {
            "type": core_result["type"],
            "created_at_utc": core_result["created_at_utc"],
            "lens_focus": lens_position["focus"],
            "board_to_pixel": core_result["board_to_pixel"],
            "pixel_to_board": core_result["pixel_to_board"],
            "image": core_result["image"],
            "stage_reference": {
                key: value
                for key, value in stage_reference.items()
                if key not in ("coordinate_source", "configuration")
            },
        },
        zoom=lens_position["zoom"],
    )


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_tools() -> tuple[Any, Any]:
    return camera_tools.load_image_tools(AffineCalibrationError)


def read_stage_reference(
    args: SimpleNamespace,
    controller_class: Any | None = None,
) -> dict[str, Any]:
    if controller_class is None:
        try:
            from motion_controller import XYMotionController
        except Exception as exc:
            raise AffineCalibrationError(
                f"cannot load the XY-stage controller: {exc}"
            ) from exc
        controller_class = XYMotionController

    stage = None
    try:
        stage = controller_class(
            args.port,
            baudrate=args.baudrate,
            x_lead_mm_per_rev=args.x_lead_mm_per_rev,
            y_lead_mm_per_rev=args.y_lead_mm_per_rev,
            x_slave_address=args.x_slave_address,
            y_slave_address=args.y_slave_address,
        )
        raw_position = stage.read_motor_position()
        x_mm = float(raw_position["x"])
        y_mm = float(raw_position["y"])
        if not math.isfinite(x_mm) or not math.isfinite(y_mm):
            raise AffineCalibrationError(
                "XY-stage returned a non-finite current position"
            )
        return {
            "recorded": True,
            "captured_at_utc": utc_timestamp(),
            "position_mm": {"x": x_mm, "y": y_mm},
            "coordinate_source": "XYMotionController absolute motor position",
            "configuration": {
                "port": str(args.port),
                "baudrate": int(args.baudrate),
                "x_lead_mm_per_rev": float(args.x_lead_mm_per_rev),
                "y_lead_mm_per_rev": float(args.y_lead_mm_per_rev),
                "x_slave_address": int(args.x_slave_address),
                "y_slave_address": int(args.y_slave_address),
            },
        }
    except AffineCalibrationError:
        raise
    except Exception as exc:
        raise AffineCalibrationError(
            f"cannot read current XY-stage position on {args.port}: {exc}"
        ) from exc
    finally:
        if stage is not None:
            try:
                stage.close()
            except Exception:
                pass


def validate_args(args: SimpleNamespace) -> None:
    if args.board_cols < 3 or args.board_rows < 3:
        raise AffineCalibrationError(
            "--board-cols and --board-rows must both be at least 3"
        )
    if not math.isfinite(args.square_size) or args.square_size <= 0:
        raise AffineCalibrationError("--square-size must be a finite value above 0")
    if not math.isfinite(args.origin_x_mm) or not math.isfinite(args.origin_y_mm):
        raise AffineCalibrationError("--origin-x-mm and --origin-y-mm must be finite")
    if args.ransac_threshold_px <= 0 or not math.isfinite(
        args.ransac_threshold_px
    ):
        raise AffineCalibrationError("--ransac-threshold-px must be above 0")
    if args.ransac_iterations <= 0:
        raise AffineCalibrationError("--ransac-iterations must be above 0")
    if not 0 < args.min_inlier_ratio <= 1:
        raise AffineCalibrationError("--min-inlier-ratio must be in (0, 1]")
    if args.startup_discard_frames < 0:
        raise AffineCalibrationError("--startup-discard-frames cannot be negative")
    if args.record_stage_position:
        if not str(args.port).strip():
            raise AffineCalibrationError("--port cannot be empty")
        if args.baudrate <= 0:
            raise AffineCalibrationError("--baudrate must be above 0")
        if (
            not math.isfinite(args.x_lead_mm_per_rev)
            or args.x_lead_mm_per_rev <= 0
            or not math.isfinite(args.y_lead_mm_per_rev)
            or args.y_lead_mm_per_rev <= 0
        ):
            raise AffineCalibrationError(
                "--x-lead-mm-per-rev and --y-lead-mm-per-rev must be finite "
                "values above 0"
            )
        for option, address in (
            ("--x-slave-address", args.x_slave_address),
            ("--y-slave-address", args.y_slave_address),
        ):
            if not 1 <= address <= 247:
                raise AffineCalibrationError(f"{option} must be in [1, 247]")
        if args.x_slave_address == args.y_slave_address:
            raise AffineCalibrationError(
                "--x-slave-address and --y-slave-address must be different"
            )

    column_axis = AXIS_VECTORS[args.column_axis]
    row_axis = AXIS_VECTORS[args.row_axis]
    dot = column_axis[0] * row_axis[0] + column_axis[1] * row_axis[1]
    if abs(dot) > 1e-12:
        raise AffineCalibrationError(
            "--column-axis and --row-axis must be perpendicular"
        )


def create_output_paths(
    output_text: str | None,
    log_text: str | None,
    annotated_text: str | None,
) -> tuple[Path, Path, Path]:
    if output_text:
        # Relative output paths are anchored to the project root, so the save
        # location does not change with the directory the script is run from.
        output_path = paths.resolve_project_path(output_text).resolve()
    else:
        run_dir = (DEFAULT_OUTPUT_ROOT / run_timestamp()).resolve()
        suffix = 1
        while run_dir.exists():
            run_dir = (
                DEFAULT_OUTPUT_ROOT / f"{run_timestamp()}_{suffix:02d}"
            ).resolve()
            suffix += 1
        output_path = run_dir / "board_to_pixel_affine.json"

    log_path = (
        paths.resolve_project_path(log_text).resolve()
        if log_text
        else output_path.with_suffix(".log")
    )
    annotated_path = (
        paths.resolve_project_path(annotated_text).resolve()
        if annotated_text
        else output_path.with_name(f"{output_path.stem}_annotated.png")
    )
    paths = (output_path, log_path, annotated_path)
    normalized = [str(path).casefold() for path in paths]
    if len(set(normalized)) != len(normalized):
        raise AffineCalibrationError(
            "--output, --log-output, and --annotated-output must be different files"
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def write_image(cv2: Any, path: Path, image: Any) -> None:
    extension = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise AffineCalibrationError(f"OpenCV cannot encode image as {extension}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise AffineCalibrationError(f"cannot write annotated image {path}: {exc}") from exc


def open_selected_camera(
    cv2: Any, args: SimpleNamespace
) -> tuple[Any, dict[str, Any]]:
    if args.camera is None:
        selected = camera_tools.find_camera_by_name(
            args.camera_name, AffineCalibrationError
        )
    else:
        selected = {
            "index": int(args.camera),
            "name": None,
            "source": "command_line_index",
        }

    camera = camera_tools.open_camera(cv2, camera_tools.camera_source(selected))
    if not camera.isOpened():
        raise AffineCalibrationError(
            f"cannot open camera index {selected['index']} ({selected.get('name')})"
        )

    settings = SimpleNamespace(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        disable_camera_autofocus=not args.keep_camera_autofocus,
    )
    camera_tools.configure_camera(cv2, camera, settings)
    for _ in range(args.startup_discard_frames):
        camera.read()

    record = dict(selected)
    record.update(
        {
            "requested_frame_width": args.frame_width,
            "requested_frame_height": args.frame_height,
            "autofocus_kept": bool(args.keep_camera_autofocus),
        }
    )
    return camera, record


def detect_checkerboard(
    cv2: Any,
    image: Any,
    board_size: tuple[int, int],
) -> tuple[Any | None, str]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
        if found:
            return corners.reshape(-1, 2).astype("float64"), "findChessboardCornersSB"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return None, "not_found"
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.0005,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype("float64"), "findChessboardCorners"


def order_corners_from_image_top_left(
    np: Any,
    corners: Any,
    board_size: tuple[int, int],
) -> Any:
    """Make local columns point image-right and local rows point image-down."""
    columns, rows = board_size
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, columns, 2).copy()

    column_steps = grid[:, 1:, :] - grid[:, :-1, :]
    mean_column_step = np.median(column_steps.reshape(-1, 2), axis=0)
    if float(mean_column_step[0]) < 0:
        grid = grid[:, ::-1, :]

    row_steps = grid[1:, :, :] - grid[:-1, :, :]
    mean_row_step = np.median(row_steps.reshape(-1, 2), axis=0)
    if float(mean_row_step[1]) < 0:
        grid = grid[::-1, :, :]

    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float64)


def acquire_camera_image(
    cv2: Any,
    camera: Any,
    board_size: tuple[int, int],
    capture_now: bool,
) -> tuple[Any, Any, str]:
    if capture_now:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise AffineCalibrationError("camera returned no image")
        corners, detector = detect_checkerboard(cv2, frame, board_size)
        if corners is None:
            raise AffineCalibrationError(
                f"no complete local {board_size[0]}x{board_size[1]} inner-corner "
                "grid was detected"
            )
        return frame, corners, detector

    window = "board-to-pixel affine calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    message = "SPACE/ENTER: detect and accept   Q/ESC: cancel"
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise AffineCalibrationError("camera returned no image")
            display = frame.copy()
            cv2.putText(
                display,
                message,
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                raise AffineCalibrationError("calibration cancelled by user")
            if key not in (ord(" "), 10, 13):
                continue
            corners, detector = detect_checkerboard(cv2, frame, board_size)
            if corners is None:
                message = (
                    f"No complete {board_size[0]}x{board_size[1]} local grid; "
                    "adjust board and retry"
                )
                continue
            return frame, corners, detector
    finally:
        cv2.destroyWindow(window)


def build_board_points(np: Any, args: SimpleNamespace) -> Any:
    column_axis = np.asarray(AXIS_VECTORS[args.column_axis], dtype=np.float64)
    row_axis = np.asarray(AXIS_VECTORS[args.row_axis], dtype=np.float64)
    origin = np.asarray(
        [args.origin_x_mm, args.origin_y_mm], dtype=np.float64
    )
    points = []
    for row in range(args.board_rows):
        for column in range(args.board_cols):
            points.append(
                origin
                + args.square_size * (column * column_axis + row * row_axis)
            )
    return np.asarray(points, dtype=np.float64)


def fit_affine_least_squares(
    np: Any, source_points: Any, target_points: Any
) -> Any:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if len(source) != len(target) or len(source) < 3:
        raise AffineCalibrationError(
            "an affine fit needs at least three matching source and target points"
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise AffineCalibrationError("calibration points contain NaN or infinity")

    design = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < 3:
        raise AffineCalibrationError("calibration board points are collinear")
    matrix = coefficients.T
    if abs(float(np.linalg.det(matrix[:, :2]))) < 1e-12:
        raise AffineCalibrationError("fitted affine transform is singular")
    return matrix.astype(np.float64)


def transform_points(np: Any, points: Any, matrix_2x3: Any) -> Any:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix_2x3, dtype=np.float64).reshape(2, 3)
    return values @ matrix[:, :2].T + matrix[:, 2]


def triangle_area2(np: Any, points: Any) -> float:
    first, second, third = np.asarray(points, dtype=np.float64).reshape(3, 2)
    edge_a = second - first
    edge_b = third - first
    return abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))


def estimate_affine_ransac(
    np: Any,
    source_points: Any,
    target_points: Any,
    threshold_px: float,
    max_iterations: int,
    min_inlier_ratio: float,
    random_seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    count = len(source)
    if count < 4:
        matrix = fit_affine_least_squares(np, source, target)
        return matrix, np.ones(count, dtype=bool), {"iterations": 0}

    source_span = float(np.linalg.norm(np.ptp(source, axis=0)))
    min_sample_area2 = max(1e-12, source_span * source_span * 1e-8)
    generator = np.random.default_rng(random_seed)
    best_mask = None
    best_count = -1
    best_squared_error = float("inf")
    required_count = max(3, int(math.ceil(count * min_inlier_ratio)))
    trials = 0

    for _ in range(max_iterations):
        trials += 1
        indices = generator.choice(count, size=3, replace=False)
        if triangle_area2(np, source[indices]) <= min_sample_area2:
            continue
        try:
            candidate = fit_affine_least_squares(
                np, source[indices], target[indices]
            )
        except AffineCalibrationError:
            continue
        errors = np.linalg.norm(
            transform_points(np, source, candidate) - target, axis=1
        )
        mask = errors <= threshold_px
        inlier_count = int(mask.sum())
        squared_error = float(np.sum(errors[mask] ** 2))
        if inlier_count > best_count or (
            inlier_count == best_count and squared_error < best_squared_error
        ):
            best_mask = mask
            best_count = inlier_count
            best_squared_error = squared_error
            if inlier_count == count:
                break

    if best_mask is None or best_count < required_count:
        raise AffineCalibrationError(
            f"RANSAC found only {max(0, best_count)}/{count} inliers; "
            f"at least {required_count} are required. Inspect corner detection, "
            "board planarity, and --ransac-threshold-px."
        )

    # Refit and reclassify until the inlier set is stable.
    mask = best_mask
    for _ in range(4):
        matrix = fit_affine_least_squares(np, source[mask], target[mask])
        errors = np.linalg.norm(
            transform_points(np, source, matrix) - target, axis=1
        )
        new_mask = errors <= threshold_px
        if int(new_mask.sum()) < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    matrix = fit_affine_least_squares(np, source[mask], target[mask])
    return matrix, mask, {
        "iterations": trials,
        "threshold_px": float(threshold_px),
        "minimum_inlier_ratio": float(min_inlier_ratio),
        "minimum_inlier_count": required_count,
        "random_seed": int(random_seed),
    }


def invert_affine(np: Any, matrix_2x3: Any) -> Any:
    matrix = np.asarray(matrix_2x3, dtype=np.float64).reshape(2, 3)
    linear_inverse = np.linalg.inv(matrix[:, :2])
    translation_inverse = -linear_inverse @ matrix[:, 2]
    return np.column_stack([linear_inverse, translation_inverse])


def calculate_pixel_physical_scale(np: Any, pixel_to_board_matrix: Any) -> dict[str, Any]:
    """Describe the physical displacement produced by one image pixel."""
    inverse = np.asarray(pixel_to_board_matrix, dtype=np.float64).reshape(2, 3)
    linear = inverse[:, :2]
    u_displacement = linear[:, 0]
    v_displacement = linear[:, 1]
    u_distance = float(np.linalg.norm(u_displacement))
    v_distance = float(np.linalg.norm(v_displacement))
    if u_distance <= 0 or v_distance <= 0:
        raise AffineCalibrationError("pixel-to-board scale is zero")

    area = abs(float(np.linalg.det(linear)))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    cosine = float(
        np.dot(u_displacement, v_displacement) / (u_distance * v_distance)
    )
    cosine = max(-1.0, min(1.0, cosine))
    return {
        "model": "constant_affine_pixel_scale",
        "distance_unit": "mm/pixel",
        "image_u_axis": {
            "direction": "one pixel to image right",
            "board_displacement_mm": u_displacement.tolist(),
            "distance_mm_per_pixel": u_distance,
        },
        "image_v_axis": {
            "direction": "one pixel to image down",
            "board_displacement_mm": v_displacement.tolist(),
            "distance_mm_per_pixel": v_distance,
        },
        "mean_axis_distance_mm_per_pixel": 0.5 * (u_distance + v_distance),
        "equivalent_area_scale_mm_per_pixel": math.sqrt(area),
        "physical_area_mm2_per_pixel2": area,
        "principal_distance_mm_per_pixel": {
            "maximum": float(singular_values[0]),
            "minimum": float(singular_values[-1]),
        },
        "anisotropy_ratio": float(singular_values[0] / singular_values[-1]),
        "physical_angle_between_image_axes_degrees": math.degrees(
            math.acos(cosine)
        ),
        "note": (
            "A full affine transform can have different horizontal and vertical "
            "scales and shear, so the two axis distances are authoritative."
        ),
    }


def fit_diagnostics(
    np: Any,
    source: Any,
    target: Any,
    matrix: Any,
    inlier_mask: Any,
) -> tuple[dict[str, Any], Any, Any]:
    predicted = transform_points(np, source, matrix)
    residual_vectors = predicted - target
    distances = np.linalg.norm(residual_vectors, axis=1)
    effective = distances[inlier_mask]
    linear = matrix[:, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    diagnostics = {
        "point_count": int(len(source)),
        "inlier_count": int(inlier_mask.sum()),
        "outlier_count": int(len(source) - inlier_mask.sum()),
        "rmse_px": float(np.sqrt(np.mean(effective**2))),
        "mean_error_px": float(np.mean(effective)),
        "max_error_px": float(np.max(effective)),
        "all_points_rmse_px": float(np.sqrt(np.mean(distances**2))),
        "linear_determinant": float(np.linalg.det(linear)),
        "linear_singular_values_px_per_mm": singular_values.tolist(),
        "linear_condition_number": float(np.linalg.cond(linear)),
    }
    return diagnostics, predicted, residual_vectors


def format_physical_coordinate(point: Any) -> str:
    def format_value(value: float) -> str:
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text

    return f"({format_value(point[0])},{format_value(point[1])}) mm"


def annotate_result(
    cv2: Any,
    np: Any,
    image: Any,
    board_points: Any,
    corners: Any,
    predicted: Any,
    inlier_mask: Any,
    args: SimpleNamespace,
    diagnostics: dict[str, Any],
) -> Any:
    display = image.copy()
    image_height, image_width = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    for board_point, observed, fitted, is_inlier in zip(
        board_points, corners, predicted, inlier_mask
    ):
        observed_point = tuple(np.rint(observed).astype(int))
        fitted_point = tuple(np.rint(fitted).astype(int))
        color = (40, 210, 40) if bool(is_inlier) else (0, 0, 255)
        cv2.line(display, observed_point, fitted_point, (255, 80, 0), 1, cv2.LINE_AA)
        cv2.circle(display, observed_point, 6, color, 2, cv2.LINE_AA)
        label = format_physical_coordinate(board_point)
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)
        text_x = min(
            max(2, observed_point[0] + 8), max(2, image_width - text_width - 2)
        )
        text_y = min(
            max(text_height + 2, observed_point[1] - 8), image_height - 2
        )
        text_origin = (text_x, text_y)
        cv2.putText(
            display,
            label,
            text_origin,
            font,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            label,
            text_origin,
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    origin = tuple(np.rint(corners[0]).astype(int))
    if args.board_cols > 1:
        column_point = tuple(np.rint(corners[1]).astype(int))
        cv2.arrowedLine(display, origin, column_point, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(
            display,
            f"columns {args.column_axis}",
            column_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if args.board_rows > 1:
        row_point = tuple(np.rint(corners[args.board_cols]).astype(int))
        cv2.arrowedLine(display, origin, row_point, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(
            display,
            f"rows {args.row_axis}",
            row_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        display,
        "origin=({:.3f}, {:.3f}) mm  RMSE={:.3f} px  inliers={}/{}".format(
            args.origin_x_mm,
            args.origin_y_mm,
            diagnostics["rmse_px"],
            diagnostics["inlier_count"],
            diagnostics["point_count"],
        ),
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def make_observations(
    args: SimpleNamespace,
    board_points: Any,
    corners: Any,
    predicted: Any,
    residual_vectors: Any,
    inlier_mask: Any,
) -> list[dict[str, Any]]:
    observations = []
    for index in range(len(board_points)):
        row, column = divmod(index, args.board_cols)
        residual = residual_vectors[index]
        observations.append(
            {
                "index": index,
                "detected_grid_column": column,
                "detected_grid_row": row,
                "board_coordinate_mm": board_points[index].tolist(),
                "observed_pixel": corners[index].tolist(),
                "fitted_pixel": predicted[index].tolist(),
                "residual_vector_px": residual.tolist(),
                "residual_distance_px": float(
                    math.hypot(float(residual[0]), float(residual[1]))
                ),
                "inlier": bool(inlier_mask[index]),
            }
        )
    return observations


def run_self_test() -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise AffineCalibrationError("NumPy is required for the self-test") from exc

    columns, rows = 7, 5
    ordered_grid = np.asarray(
        [
            [(100.0 + 20.0 * column + 2.0 * row, 50.0 + column + 18.0 * row)
             for column in range(columns)]
            for row in range(rows)
        ],
        dtype=np.float64,
    )
    for raw_grid in (
        ordered_grid,
        ordered_grid[:, ::-1, :],
        ordered_grid[::-1, :, :],
        ordered_grid[::-1, ::-1, :],
    ):
        reordered = order_corners_from_image_top_left(
            np, raw_grid.reshape(-1, 2), (columns, rows)
        )
        if not np.allclose(reordered, ordered_grid.reshape(-1, 2)):
            raise AffineCalibrationError("self-test corner-order normalization failed")

    source = np.asarray(
        [(column * 2.5, row * 2.5) for row in range(rows) for column in range(columns)],
        dtype=np.float64,
    )
    expected = np.asarray(
        [[82.4, -3.1, 1860.0], [2.7, 81.8, 920.0]], dtype=np.float64
    )
    generator = np.random.default_rng(12345)
    target = transform_points(np, source, expected)
    target += generator.normal(0.0, 0.035, target.shape)
    target[[3, 27]] += np.asarray([[18.0, -12.0], [-15.0, 20.0]])

    fitted, mask, _ = estimate_affine_ransac(
        np,
        source,
        target,
        threshold_px=0.25,
        max_iterations=1000,
        min_inlier_ratio=0.7,
        random_seed=7,
    )
    diagnostics, _, _ = fit_diagnostics(np, source, target, fitted, mask)
    if int(mask.sum()) != len(source) - 2:
        raise AffineCalibrationError("self-test failed to reject the injected outliers")
    if float(np.max(np.abs(fitted - expected))) > 0.08:
        raise AffineCalibrationError(
            f"self-test matrix error is too high:\n{fitted - expected}"
        )
    inverse = invert_affine(np, fitted)
    round_trip = transform_points(
        np, transform_points(np, source, fitted), inverse
    )
    if float(np.max(np.abs(round_trip - source))) > 1e-9:
        raise AffineCalibrationError("self-test inverse round trip failed")
    pixel_scale = calculate_pixel_physical_scale(np, inverse)
    test_pixel = np.asarray([[1234.5, 678.25]], dtype=np.float64)
    test_board = transform_points(np, test_pixel, inverse)[0]
    u_board = transform_points(np, test_pixel + [1.0, 0.0], inverse)[0]
    v_board = transform_points(np, test_pixel + [0.0, 1.0], inverse)[0]
    measured_u = float(np.linalg.norm(u_board - test_board))
    measured_v = float(np.linalg.norm(v_board - test_board))
    if not math.isclose(
        measured_u,
        pixel_scale["image_u_axis"]["distance_mm_per_pixel"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        measured_v,
        pixel_scale["image_v_axis"]["distance_mm_per_pixel"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AffineCalibrationError("self-test pixel physical scale failed")

    class FakeStage:
        instance = None

        def __init__(self, *constructor_args: Any, **constructor_kwargs: Any):
            self.constructor_args = constructor_args
            self.constructor_kwargs = constructor_kwargs
            self.closed = False
            FakeStage.instance = self

        def read_motor_position(self) -> dict[str, float]:
            return {"x": 12.345, "y": -6.789}

        def close(self) -> None:
            self.closed = True

    stage_args = SimpleNamespace(
        port="COM_TEST",
        baudrate=38400,
        x_lead_mm_per_rev=1.0,
        y_lead_mm_per_rev=2.0,
        x_slave_address=1,
        y_slave_address=2,
    )
    stage_reference = read_stage_reference(stage_args, FakeStage)
    if stage_reference["position_mm"] != {"x": 12.345, "y": -6.789}:
        raise AffineCalibrationError("self-test stage position serialization failed")
    if FakeStage.instance is None or not FakeStage.instance.closed:
        raise AffineCalibrationError("self-test stage connection was not closed")
    if FakeStage.instance.constructor_kwargs["y_lead_mm_per_rev"] != 2.0:
        raise AffineCalibrationError("self-test stage configuration failed")

    print(
        "Self-test passed: RMSE={:.4f} px, inliers={}/{}, "
        "pixel scale=({:.6f}, {:.6f}) mm/px, stage=({:.3f}, {:.3f}) mm".format(
            diagnostics["rmse_px"],
            int(mask.sum()),
            len(source),
            measured_u,
            measured_v,
            stage_reference["position_mm"]["x"],
            stage_reference["position_mm"]["y"],
        )
    )
    return 0


def run_calibration(args: SimpleNamespace) -> int:
    cv2, np = load_tools()
    board_size = (args.board_cols, args.board_rows)
    camera = None
    camera_record = None

    try:
        camera, camera_record = open_selected_camera(cv2, args)
        image, corners, detector = acquire_camera_image(
            cv2, camera, board_size, args.capture_now
        )
        source_record = {"kind": "camera", "camera": camera_record}

        expected_count = args.board_cols * args.board_rows
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if len(corners) != expected_count:
            raise AffineCalibrationError(
                f"detector returned {len(corners)} corners, expected {expected_count}"
            )
        corners = order_corners_from_image_top_left(np, corners, board_size)
        stage_reference = (
            read_stage_reference(args)
            if args.record_stage_position
            else {"recorded": False}
        )
        board_points = build_board_points(np, args)
        matrix, inlier_mask, robust = estimate_affine_ransac(
            np,
            board_points,
            corners,
            args.ransac_threshold_px,
            args.ransac_iterations,
            args.min_inlier_ratio,
            args.random_seed,
        )
        inverse = invert_affine(np, matrix)
        pixel_scale = calculate_pixel_physical_scale(np, inverse)
        diagnostics, predicted, residual_vectors = fit_diagnostics(
            np, board_points, corners, matrix, inlier_mask
        )
        output_path, log_path, annotated_path = create_output_paths(
            args.output, args.log_output, args.annotated_output
        )
        annotated = annotate_result(
            cv2,
            np,
            image,
            board_points,
            corners,
            predicted,
            inlier_mask,
            args,
            diagnostics,
        )
        write_image(cv2, annotated_path, annotated)

        height, width = image.shape[:2]
        created_at = utc_timestamp()
        coordinate_definition = (
            "Board coordinates are planar millimetres. Pixel coordinates use "
            "OpenCV convention: u increases right, v increases down. The affine "
            "model is valid only while camera pose, board plane, zoom, focus, and "
            "image resolution remain unchanged."
        )
        core_result = {
            "schema_version": 2,
            "type": "calibration_board_to_pixel_affine_core",
            "created_at_utc": created_at,
            "image": {"width_px": int(width), "height_px": int(height)},
            "board_to_pixel": {
                "equation": "[u, v]^T = matrix_2x3 * [X_mm, Y_mm, 1]^T",
                "matrix_2x3": matrix.tolist(),
            },
            "pixel_to_board": {
                "equation": "[X_mm, Y_mm]^T = matrix_2x3 * [u, v, 1]^T",
                "matrix_2x3": inverse.tolist(),
            },
            "pixel_physical_scale": pixel_scale,
            "stage_reference": stage_reference,
            "detail_log_file": str(log_path),
        }
        detail_log = {
            "schema_version": 2,
            "type": "calibration_board_to_pixel_affine_log",
            "created_at_utc": created_at,
            "calibration_json_file": str(output_path),
            "calibration_fields_saved_in_json": [
                "image",
                "board_to_pixel",
                "pixel_to_board",
                "pixel_physical_scale",
                "stage_reference",
            ],
            "coordinate_definition": coordinate_definition,
            "image": {"width_px": int(width), "height_px": int(height)},
            "command_arguments": dict(vars(args)),
            "source": source_record,
            "detector": detector,
            "board": {
                "pattern": "checkerboard_local_visible_rectangle",
                "visible_inner_corners": {
                    "columns": args.board_cols,
                    "rows": args.board_rows,
                },
                "square_size_mm": float(args.square_size),
                "corner_order_in_image": (
                    "visible upper-left inner corner first; columns increase "
                    "toward image right; rows increase toward image down"
                ),
                "detected_first_corner_coordinate_mm": [
                    float(args.origin_x_mm),
                    float(args.origin_y_mm),
                ],
                "detected_column_axis_in_board_coordinates": args.column_axis,
                "detected_row_axis_in_board_coordinates": args.row_axis,
            },
            "robust_fit": robust,
            "fit_quality": diagnostics,
            "observations": make_observations(
                args,
                board_points,
                corners,
                predicted,
                residual_vectors,
                inlier_mask,
            ),
            "annotated_image": str(annotated_path),
        }
        write_json(output_path, core_result)
        write_json(log_path, detail_log)

        summary_path = None
        try:
            summary_path = append_calibration_summary(
                core_result,
                stage_reference,
                args,
            )
        except (OSError, ValueError) as exc:
            print(
                f"Warning: cannot append calibration summary: {exc}",
                file=sys.stderr,
            )

        print("Calibration complete")
        print(f"Output: {output_path}")
        print(f"Detail log: {log_path}")
        print(f"Annotated image: {annotated_path}")
        if summary_path is not None:
            print(f"Calibration summary: {summary_path}")
        if stage_reference["recorded"]:
            print(
                "Stage reference: x={:.6f} mm, y={:.6f} mm".format(
                    stage_reference["position_mm"]["x"],
                    stage_reference["position_mm"]["y"],
                )
            )
        else:
            print("Stage reference: not recorded")
        print("board -> pixel matrix:")
        for row in matrix:
            print("  [{: .10f}, {: .10f}, {: .10f}]".format(*row))
        print(
            "Pixel scale: u={:.10f} mm/px, v={:.10f} mm/px, mean={:.10f} mm/px".format(
                pixel_scale["image_u_axis"]["distance_mm_per_pixel"],
                pixel_scale["image_v_axis"]["distance_mm_per_pixel"],
                pixel_scale["mean_axis_distance_mm_per_pixel"],
            )
        )
        print(
            "RMSE: {:.4f} px; max: {:.4f} px; inliers: {}/{}".format(
                diagnostics["rmse_px"],
                diagnostics["max_error_px"],
                diagnostics["inlier_count"],
                diagnostics["point_count"],
            )
        )
        if diagnostics["max_error_px"] > args.ransac_threshold_px:
            print(
                "Warning: inlier maximum error exceeds the requested threshold; "
                "inspect the annotated image and residuals.",
                file=sys.stderr,
            )
        return 0
    finally:
        if camera is not None:
            camera.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> int:
    try:
        args = load_args()
        validate_args(args)
        return run_calibration(args)
    except AffineCalibrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

