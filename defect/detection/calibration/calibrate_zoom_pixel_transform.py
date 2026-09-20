"""Calibrate the pixel-coordinate transform between two lens zoom positions.

The camera and checkerboard must remain fixed while this program captures one
image at each zoom.  Corners are undistorted with the intrinsics belonging to
their own zoom before a robust 3x3 homography is fitted.  The resulting matrix
therefore maps *undistorted pixel coordinates*, not raw distorted pixels.
The known checkerboard square size is also used to estimate millimetres per
pixel independently for both zoom positions.

Run artifacts are written below working_data/zoom_pixel_transform.  A compact
matrix entry is also written to detection/calibration/output/calibrate.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import LensCamera as lens_camera
import calibration_summary
from LensCamera import camera as camera_tools
from LensCamera import paths
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError


DEFAULT_SETTINGS_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "zoom_pixel_transform"
    / "zoom_pixel_transform.jsonnet"
)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "output" / "calibrate.json"
DEFAULT_OUTPUT_ROOT = paths.PROJECT_ROOT / "working_data" / "zoom_pixel_transform"
LENS_CALIBRATION_TYPES = (
    "lens_checkerboard_camera_calibration",
    "full_electric_lens_checkerboard_calibration",
)
AUTOFOCUS_ENTRY_TYPE = "pcb_autofocus_lens_position"
RESULT_TYPE = "zoom_pixel_homography_calibration"
SUMMARY_OMITTED_FIELDS = frozenset(
    (
        "coordinate_space",
        "coordinate_convention",
        "source_image",
        "target_image",
        "fit_quality",
        "detail_report",
        "saved_at_utc",
    )
)


class ZoomTransformCalibrationError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_tools() -> tuple[Any, Any]:
    return camera_tools.load_image_tools(ZoomTransformCalibrationError)


def load_settings() -> dict[str, Any]:
    try:
        return JsonnetSettings(DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)).values
    except SettingsError as exc:
        raise ZoomTransformCalibrationError(str(exc)) from exc


def setting_value(settings: dict[str, Any], name: str) -> Any:
    if name not in settings:
        raise ZoomTransformCalibrationError(
            f"{DEFAULT_SETTINGS_PATH} missing required setting {name!r}"
        )
    value = settings[name]
    return value["value"] if isinstance(value, dict) and "value" in value else value


def load_args(settings: dict[str, Any] | None = None) -> SimpleNamespace:
    settings = load_settings() if settings is None else settings
    return SimpleNamespace(**{name: setting_value(settings, name) for name in settings})


def zoom_address(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ZoomTransformCalibrationError(
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
            raise ZoomTransformCalibrationError(
                f"{name} must be an integer address"
            ) from exc
    else:
        raise ZoomTransformCalibrationError(f"{name} must be an integer address")
    if result < 0:
        raise ZoomTransformCalibrationError(f"{name} cannot be negative")
    return result


def zoom_name(zoom: Any) -> str:
    return "zoom_{:05d}".format(zoom_address(zoom, "zoom"))


def pair_name(source_zoom: Any, target_zoom: Any) -> str:
    return "{}_to_{}".format(zoom_name(source_zoom), zoom_name(target_zoom))


def resolve_project_path(path_text: Any) -> Path:
    return paths.resolve_project_path(str(path_text)).resolve()


def calibration_path(args: SimpleNamespace) -> Path:
    value = getattr(args, "calibration", None)
    return DEFAULT_CALIBRATION_PATH if value is None else resolve_project_path(value)


def optional_image_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_project_path(value)


def validate_args(args: SimpleNamespace) -> None:
    source_zoom = zoom_address(args.source_zoom, "source_zoom")
    target_zoom = zoom_address(args.target_zoom, "target_zoom")
    if source_zoom == target_zoom:
        raise ZoomTransformCalibrationError(
            "source_zoom and target_zoom must be different"
        )
    if args.board_cols < 3 or args.board_rows < 3:
        raise ZoomTransformCalibrationError(
            "board_cols and board_rows must both be at least 3"
        )
    try:
        square_size_mm = float(args.square_size_mm)
    except (TypeError, ValueError) as exc:
        raise ZoomTransformCalibrationError(
            "square_size_mm must be a number above 0"
        ) from exc
    if not math.isfinite(square_size_mm) or square_size_mm <= 0:
        raise ZoomTransformCalibrationError("square_size_mm must be above 0")
    if not math.isfinite(args.ransac_threshold_px) or args.ransac_threshold_px <= 0:
        raise ZoomTransformCalibrationError("ransac_threshold_px must be above 0")
    if args.ransac_iterations <= 0:
        raise ZoomTransformCalibrationError("ransac_iterations must be above 0")
    if not 0 < args.ransac_confidence < 1:
        raise ZoomTransformCalibrationError("ransac_confidence must be in (0, 1)")
    if not 0 < args.min_inlier_ratio <= 1:
        raise ZoomTransformCalibrationError("min_inlier_ratio must be in (0, 1]")
    if not math.isfinite(args.max_inlier_rmse_px) or args.max_inlier_rmse_px <= 0:
        raise ZoomTransformCalibrationError("max_inlier_rmse_px must be above 0")
    if args.camera_startup_discard_frames < 0 or args.capture_discard_frames < 0:
        raise ZoomTransformCalibrationError("discard frame counts cannot be negative")
    if not math.isfinite(args.lens_settle_seconds) or args.lens_settle_seconds < 0:
        raise ZoomTransformCalibrationError("lens_settle_seconds cannot be negative")
    for name in ("frame_width", "frame_height"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ZoomTransformCalibrationError(f"{name} must be above 0 or null")

    source_image = optional_image_path(args.source_image)
    target_image = optional_image_path(args.target_image)
    if (source_image is None) != (target_image is None):
        raise ZoomTransformCalibrationError(
            "source_image and target_image must either both be set or both be null"
        )
    if (
        source_image is None
        and args.camera is None
        and not str(args.camera_name).strip()
    ):
        raise ZoomTransformCalibrationError(
            "camera_name cannot be empty when camera is null"
        )


def load_summary_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ZoomTransformCalibrationError(
            f"cannot read calibration summary {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ZoomTransformCalibrationError(
            f"calibration summary is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        raise ZoomTransformCalibrationError(
            f"calibration summary {path} must contain an entries list"
        )
    return document


def entry_matches_zoom(entry: dict[str, Any], zoom: int) -> bool:
    try:
        if entry.get("lens_zoom") is not None:
            return zoom_address(entry["lens_zoom"], "lens_zoom") == zoom
    except ZoomTransformCalibrationError:
        pass
    return entry.get("name") == zoom_name(zoom)


def _position_for_zoom(entry: dict[str, Any], zoom: int) -> dict[str, Any] | None:
    positions = entry.get("positions")
    if not isinstance(positions, list):
        return None
    requested_name = zoom_name(zoom)
    for position in reversed(positions):
        if not isinstance(position, dict):
            continue
        try:
            if (
                position.get("lens_zoom") is not None
                and zoom_address(position["lens_zoom"], "position lens_zoom") == zoom
            ):
                return position
        except ZoomTransformCalibrationError:
            pass
        if position.get("name") == requested_name:
            return position
    if len(positions) == 1 and entry_matches_zoom(entry, zoom):
        return positions[0] if isinstance(positions[0], dict) else None
    return None


def load_intrinsics(
    np: Any,
    document: dict[str, Any],
    summary_path: Path,
    zoom: Any,
) -> dict[str, Any]:
    requested_zoom = zoom_address(zoom, "lens_zoom")
    selected = None
    selected_entry = None
    for entry in reversed(document["entries"]):
        if (
            not isinstance(entry, dict)
            or entry.get("type") not in LENS_CALIBRATION_TYPES
        ):
            continue
        position = _position_for_zoom(entry, requested_zoom)
        if position is not None:
            selected = position.get("calibration") or position
            selected_entry = entry
            break
    if not isinstance(selected, dict):
        raise ZoomTransformCalibrationError(
            f"{summary_path} has no camera intrinsics for zoom {requested_zoom}"
        )
    try:
        matrix = np.asarray(selected["camera_matrix"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(
            selected["distortion_coefficients"], dtype=np.float64
        ).reshape(-1)
        width = int(selected["image_width"])
        height = int(selected["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZoomTransformCalibrationError(
            f"invalid camera intrinsics for zoom {requested_zoom} in {summary_path}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ZoomTransformCalibrationError(
            f"intrinsics image dimensions for zoom {requested_zoom} must be positive"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
        raise ZoomTransformCalibrationError(
            f"camera intrinsics for zoom {requested_zoom} contain NaN or infinity"
        )
    return {
        "lens_zoom": requested_zoom,
        "position_name": selected.get("name") or selected_entry.get("name"),
        "image_width_px": width,
        "image_height_px": height,
        "camera_matrix": matrix,
        "distortion_coefficients": distortion,
    }


def load_focus(
    document: dict[str, Any],
    summary_path: Path,
    zoom: Any,
) -> int:
    requested_zoom = zoom_address(zoom, "lens_zoom")
    for entry in reversed(document["entries"]):
        if (
            isinstance(entry, dict)
            and entry.get("type") == AUTOFOCUS_ENTRY_TYPE
            and entry_matches_zoom(entry, requested_zoom)
            and entry.get("lens_focus") is not None
        ):
            return zoom_address(entry["lens_focus"], "autofocus lens_focus")
    raise ZoomTransformCalibrationError(
        f"{summary_path} has no {AUTOFOCUS_ENTRY_TYPE} focus for zoom "
        f"{requested_zoom}; run PCB autofocus at this zoom first"
    )


def scaled_camera_matrix(
    np: Any, intrinsics: dict[str, Any], frame_width: int, frame_height: int
) -> Any:
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64).copy()
    matrix[0, :] *= frame_width / float(intrinsics["image_width_px"])
    matrix[1, :] *= frame_height / float(intrinsics["image_height_px"])
    return matrix


def undistort_points(
    cv2: Any,
    np: Any,
    points: Any,
    intrinsics: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[Any, Any]:
    matrix = scaled_camera_matrix(np, intrinsics, frame_width, frame_height)
    corrected = cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        matrix,
        intrinsics["distortion_coefficients"],
        P=matrix,
    )
    return corrected.reshape(-1, 2), matrix


def _spacing_scale_statistics(
    np: Any, distances_px: Any, square_size_mm: float
) -> dict[str, Any]:
    distances = np.asarray(distances_px, dtype=np.float64).reshape(-1)
    if len(distances) == 0 or not np.all(np.isfinite(distances)):
        raise ZoomTransformCalibrationError(
            "checkerboard pixel spacing contains no finite samples"
        )
    if np.any(distances <= 1e-9):
        raise ZoomTransformCalibrationError(
            "checkerboard has a zero or negative adjacent-corner distance"
        )
    median_spacing = float(np.median(distances))
    mm_per_pixel = float(square_size_mm / median_spacing)
    return {
        "sample_count": int(len(distances)),
        "standard_deviation_px": float(np.std(distances)),
        "mm_per_pixel": mm_per_pixel,
        "pixels_per_mm": float(1.0 / mm_per_pixel),
    }


def calculate_checkerboard_pixel_scale(
    np: Any,
    undistorted_points: Any,
    board_size: tuple[int, int],
    square_size_mm: float,
) -> dict[str, Any]:
    columns, rows = board_size
    try:
        grid = np.asarray(undistorted_points, dtype=np.float64).reshape(
            rows, columns, 2
        )
    except (TypeError, ValueError) as exc:
        raise ZoomTransformCalibrationError(
            "checkerboard points do not match board_cols x board_rows"
        ) from exc
    if not np.all(np.isfinite(grid)):
        raise ZoomTransformCalibrationError(
            "checkerboard points contain NaN or infinity"
        )
    square_size = float(square_size_mm)
    horizontal_distances = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2)
    vertical_distances = np.linalg.norm(grid[1:] - grid[:-1], axis=2)
    horizontal = _spacing_scale_statistics(np, horizontal_distances, square_size)
    vertical = _spacing_scale_statistics(np, vertical_distances, square_size)
    representative_mm_per_pixel = 0.5 * (
        horizontal["mm_per_pixel"] + vertical["mm_per_pixel"]
    )
    return {
        "method": "median_adjacent_checkerboard_corner_spacing",
        "square_size_mm": square_size,
        "mm_per_pixel": float(representative_mm_per_pixel),
        "pixels_per_mm": float(1.0 / representative_mm_per_pixel),
        "horizontal": horizontal,
        "vertical": vertical,
    }


def print_pixel_scale(label: str, zoom: int, scale: dict[str, Any]) -> None:
    print(
        "{} zoom={} pixel scale: {:.9f} mm/px ({:.3f} px/mm); "
        "horizontal {:.9f} mm/px, vertical {:.9f} mm/px".format(
            label,
            zoom,
            scale["mm_per_pixel"],
            scale["pixels_per_mm"],
            scale["horizontal"]["mm_per_pixel"],
            scale["vertical"]["mm_per_pixel"],
        )
    )


def project_points(np: Any, points: Any, matrix_3x3: Any) -> Any:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = (np.asarray(matrix_3x3, dtype=np.float64) @ homogeneous.T).T
    denominators = projected[:, 2]
    if np.any(np.abs(denominators) < 1e-12):
        raise ZoomTransformCalibrationError(
            "homography maps one or more calibration points to infinity"
        )
    return projected[:, :2] / denominators[:, None]


def _error_statistics(np: Any, errors: Any) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "mean_error_px": float(np.mean(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors * errors))),
        "max_error_px": float(np.max(errors)),
    }


def print_calibration_errors(quality: dict[str, Any]) -> None:
    forward = quality["forward_inliers"]
    reverse = quality["reverse_inliers"]
    print(
        "标定误差: 源到目标内点 RMSE {:.4f} px, 平均 {:.4f} px, 最大 {:.4f} px; "
        "目标到源内点 RMSE {:.4f} px, 平均 {:.4f} px, 最大 {:.4f} px".format(
            forward["rmse_px"],
            forward["mean_error_px"],
            forward["max_error_px"],
            reverse["rmse_px"],
            reverse["mean_error_px"],
            reverse["max_error_px"],
        )
    )


def estimate_homography(
    cv2: Any,
    np: Any,
    source_points: Any,
    target_points: Any,
    *,
    threshold_px: float,
    max_iterations: int,
    confidence: float,
    min_inlier_ratio: float,
    max_inlier_rmse_px: float,
) -> tuple[Any, Any, Any, dict[str, Any], Any]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or len(source) < 4:
        raise ZoomTransformCalibrationError(
            "source and target must contain the same number of at least four points"
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ZoomTransformCalibrationError(
            "calibration points contain NaN or infinity"
        )
    matrix, mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        float(threshold_px),
        maxIters=int(max_iterations),
        confidence=float(confidence),
    )
    if matrix is None or mask is None:
        raise ZoomTransformCalibrationError("OpenCV could not estimate a homography")
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-15:
        raise ZoomTransformCalibrationError("estimated homography is singular")
    if abs(float(matrix[2, 2])) > 1e-12:
        matrix /= matrix[2, 2]
    inliers = np.asarray(mask, dtype=bool).reshape(-1)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = inlier_count / len(source)
    if inlier_count < 4 or inlier_ratio < min_inlier_ratio:
        raise ZoomTransformCalibrationError(
            "homography has too few inliers: {}/{} ({:.1%}), required {:.1%}".format(
                inlier_count, len(source), inlier_ratio, min_inlier_ratio
            )
        )
    predicted = project_points(np, source, matrix)
    errors = np.linalg.norm(predicted - target, axis=1)
    all_stats = _error_statistics(np, errors)
    inlier_stats = _error_statistics(np, errors[inliers])
    if inlier_stats["rmse_px"] > max_inlier_rmse_px:
        raise ZoomTransformCalibrationError(
            "homography inlier RMSE {:.4f} px exceeds limit {:.4f} px".format(
                inlier_stats["rmse_px"], max_inlier_rmse_px
            )
        )
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    reverse_predicted = project_points(np, target, inverse)
    reverse_errors = np.linalg.norm(reverse_predicted - source, axis=1)
    quality = {
        "point_count": int(len(source)),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "ransac_threshold_px": float(threshold_px),
        "forward_all_points": all_stats,
        "forward_inliers": inlier_stats,
        "reverse_all_points": _error_statistics(np, reverse_errors),
        "reverse_inliers": _error_statistics(np, reverse_errors[inliers]),
    }
    return matrix, inverse, inliers, quality, predicted


def detect_checkerboard(
    cv2: Any, np: Any, image: Any, board_size: tuple[int, int]
) -> tuple[Any | None, str]:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)
        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
        if found:
            return order_corners(np, corners, board_size), "findChessboardCornersSB"
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
    return order_corners(np, refined, board_size), "findChessboardCorners"


def order_corners(np: Any, corners: Any, board_size: tuple[int, int]) -> Any:
    """Normalize detector orientation to image-right columns/image-down rows."""
    columns, rows = board_size
    grid = np.asarray(corners, dtype=np.float64).reshape(rows, columns, 2).copy()
    column_step = np.median((grid[:, 1:] - grid[:, :-1]).reshape(-1, 2), axis=0)
    if float(column_step[0]) < 0:
        grid = grid[:, ::-1]
    row_step = np.median((grid[1:] - grid[:-1]).reshape(-1, 2), axis=0)
    if float(row_step[1]) < 0:
        grid = grid[::-1]
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float64)


def read_image(cv2: Any, np: Any, path: Path) -> Any:
    if not path.is_file():
        raise ZoomTransformCalibrationError(f"image does not exist: {path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ZoomTransformCalibrationError(f"cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ZoomTransformCalibrationError(f"OpenCV cannot decode image {path}")
    return image


def write_image(cv2: Any, path: Path, image: Any) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ZoomTransformCalibrationError(f"OpenCV cannot encode image {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise ZoomTransformCalibrationError(
            f"cannot write image {path}: {exc}"
        ) from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, indent=2, ensure_ascii=False)
            file_obj.write("\n")
    except OSError as exc:
        raise ZoomTransformCalibrationError(f"cannot write JSON {path}: {exc}") from exc


def allocate_run_directory(output_dir: Any) -> Path:
    root = resolve_project_path(output_dir)
    working_root = (paths.PROJECT_ROOT / "working_data").resolve()
    try:
        root.relative_to(working_root)
    except ValueError as exc:
        raise ZoomTransformCalibrationError(
            f"output_dir {root} must be inside {working_root}"
        ) from exc
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / run_timestamp()
    suffix = 2
    while candidate.exists():
        candidate = root / "{}_{:02d}".format(run_timestamp(), suffix)
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def open_camera(cv2: Any, args: SimpleNamespace) -> tuple[Any, dict[str, Any]]:
    if args.camera is None:
        selected = camera_tools.find_camera_by_name(
            args.camera_name, ZoomTransformCalibrationError
        )
    else:
        selected = {
            "index": int(args.camera),
            "name": None,
            "source": "config_index",
        }
    camera = camera_tools.open_camera(cv2, camera_tools.camera_source(selected))
    if not camera.isOpened():
        raise ZoomTransformCalibrationError(
            "cannot open camera index {} ({})".format(
                selected["index"], selected.get("name")
            )
        )
    settings = SimpleNamespace(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        disable_camera_autofocus=not bool(args.keep_camera_autofocus),
    )
    camera_tools.configure_camera(cv2, camera, settings)
    for _ in range(args.camera_startup_discard_frames):
        camera.read()
    return camera, dict(selected)


def capture_frame(camera: Any, discard_frames: int, label: str) -> Any:
    ok = False
    frame = None
    for _ in range(discard_frames + 1):
        ok, frame = camera.read()
    if not ok or frame is None:
        raise ZoomTransformCalibrationError(f"camera returned no image for {label}")
    return frame


def get_screen_size() -> tuple[int, int] | None:
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return None


def configure_preview_window(cv2: Any, window_name: str) -> None:
    flags = getattr(cv2, "WINDOW_NORMAL", 0)
    flags |= getattr(cv2, "WINDOW_KEEPRATIO", 0)
    cv2.namedWindow(window_name, flags)

    screen_size = get_screen_size()
    if screen_size is None:
        return

    screen_width, screen_height = screen_size
    window_width = max(320, int(round(screen_width * 2.0 / 3.0)))
    window_height = max(240, int(round(screen_height * 2.0 / 3.0)))
    x = max(0, int(round((screen_width - window_width) / 2.0)))
    y = max(0, int(round((screen_height - window_height) / 2.0)))
    cv2.resizeWindow(window_name, window_width, window_height)
    cv2.moveWindow(window_name, x, y)


def acquire_image(
    cv2: Any,
    np: Any,
    camera: Any,
    board_size: tuple[int, int],
    label: str,
    capture_now: bool,
    discard_frames: int,
) -> tuple[Any, Any, str]:
    if capture_now:
        image = capture_frame(camera, discard_frames, label)
        corners, detector = detect_checkerboard(cv2, np, image, board_size)
        if corners is None:
            raise ZoomTransformCalibrationError(
                f"no complete {board_size[0]}x{board_size[1]} checkerboard in {label} image"
            )
        return image, corners, detector

    window = "zoom pixel transform - " + label
    message = "SPACE/ENTER: capture   Q/ESC: cancel"
    try:
        configure_preview_window(cv2, window)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise ZoomTransformCalibrationError(
                    f"camera returned no preview image for {label}"
                )
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
                raise ZoomTransformCalibrationError("calibration cancelled by user")
            if key not in (ord(" "), 10, 13):
                continue
            image = capture_frame(camera, discard_frames, label)
            corners, detector = detect_checkerboard(cv2, np, image, board_size)
            if corners is not None:
                return image, corners, detector
            message = (
                f"No complete {board_size[0]}x{board_size[1]} grid; adjust and retry"
            )
    finally:
        try:
            cv2.destroyWindow(window)
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
        raise ZoomTransformCalibrationError(
            f"cannot connect LensConnect: {exc}"
        ) from exc


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
        raise ZoomTransformCalibrationError(
            f"cannot move lens to zoom={zoom}, focus={focus}: {exc}"
        ) from exc


def annotate_corners(
    cv2: Any, np: Any, image: Any, corners: Any, board_size: tuple[int, int]
) -> Any:
    annotated = image.copy()
    cv2.drawChessboardCorners(
        annotated,
        board_size,
        np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
        True,
    )
    return annotated


def validation_overlay(
    cv2: Any,
    np: Any,
    source_image: Any,
    target_image: Any,
    source_matrix: Any,
    target_matrix: Any,
    source_intrinsics: dict[str, Any],
    target_intrinsics: dict[str, Any],
    homography: Any,
    target_points: Any,
    predicted_points: Any,
    inliers: Any,
) -> Any:
    source_undistorted = cv2.undistort(
        source_image,
        source_matrix,
        source_intrinsics["distortion_coefficients"],
        None,
        source_matrix,
    )
    target_undistorted = cv2.undistort(
        target_image,
        target_matrix,
        target_intrinsics["distortion_coefficients"],
        None,
        target_matrix,
    )
    target_height, target_width = target_image.shape[:2]
    warped = cv2.warpPerspective(
        source_undistorted, homography, (target_width, target_height)
    )
    overlay = cv2.addWeighted(target_undistorted, 0.65, warped, 0.35, 0)
    for observed, predicted, inlier in zip(target_points, predicted_points, inliers):
        observed_point = tuple(np.rint(observed).astype(int))
        predicted_point = tuple(np.rint(predicted).astype(int))
        color = (40, 220, 40) if bool(inlier) else (40, 40, 230)
        cv2.circle(overlay, observed_point, 5, color, 2, cv2.LINE_AA)
        cv2.drawMarker(
            overlay,
            predicted_point,
            (230, 60, 230),
            cv2.MARKER_CROSS,
            11,
            2,
            cv2.LINE_AA,
        )
        cv2.line(overlay, observed_point, predicted_point, color, 1, cv2.LINE_AA)
    return source_undistorted, target_undistorted, overlay


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def intrinsics_record(intrinsics: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_name": intrinsics["position_name"],
        "image_width_px": intrinsics["image_width_px"],
        "image_height_px": intrinsics["image_height_px"],
        "camera_matrix": intrinsics["camera_matrix"].tolist(),
        "distortion_coefficients": intrinsics["distortion_coefficients"].tolist(),
    }


def build_summary_entry(
    *,
    created_at: str,
    source_zoom: int,
    target_zoom: int,
    source_focus: int,
    target_focus: int,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    source_pixel_scale: dict[str, Any],
    target_pixel_scale: dict[str, Any],
    forward: Any,
    inverse: Any,
    quality: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    return {
        "type": RESULT_TYPE,
        "created_at_utc": created_at,
        "source_zoom": source_zoom,
        "target_zoom": target_zoom,
        "source_focus": source_focus,
        "target_focus": target_focus,
        "coordinate_space": "undistorted_pixel",
        "coordinate_convention": "OpenCV pixels: u right, v down",
        "source_image": {"width_px": source_size[0], "height_px": source_size[1]},
        "target_image": {"width_px": target_size[0], "height_px": target_size[1]},
        "source_pixel_scale": source_pixel_scale,
        "target_pixel_scale": target_pixel_scale,
        "source_to_target": {
            "equation": "q = H * [u, v, 1]^T; [u2, v2] = [q0/q2, q1/q2]",
            "matrix_3x3": forward.tolist(),
        },
        "target_to_source": {
            "equation": "q = H_inv * [u2, v2, 1]^T; [u, v] = [q0/q2, q1/q2]",
            "matrix_3x3": inverse.tolist(),
        },
        "fit_quality": quality,
        "detail_report": relative_path(report_path),
    }


def append_calibration_summary(
    entry: dict[str, Any], source_zoom: int, target_zoom: int
) -> Path:
    summary_entry = {
        key: value for key, value in entry.items() if key not in SUMMARY_OMITTED_FIELDS
    }
    return calibration_summary.append_entry(
        summary_entry,
        name=pair_name(source_zoom, target_zoom),
        include_saved_at_utc=False,
    )


def run_calibration(args: SimpleNamespace) -> int:
    validate_args(args)
    cv2, np = load_tools()
    summary_path = calibration_path(args)
    document = load_summary_document(summary_path)
    source_zoom = zoom_address(args.source_zoom, "source_zoom")
    target_zoom = zoom_address(args.target_zoom, "target_zoom")
    source_focus = load_focus(document, summary_path, source_zoom)
    target_focus = load_focus(document, summary_path, target_zoom)
    source_intrinsics = load_intrinsics(np, document, summary_path, source_zoom)
    target_intrinsics = load_intrinsics(np, document, summary_path, target_zoom)
    board_size = (int(args.board_cols), int(args.board_rows))
    run_dir = allocate_run_directory(args.output_dir)

    camera = None
    api = None
    camera_record: dict[str, Any]
    lens_moves: dict[str, Any] = {}
    source_path = optional_image_path(args.source_image)
    target_path = optional_image_path(args.target_image)
    try:
        if source_path is not None and target_path is not None:
            source_image = read_image(cv2, np, source_path)
            target_image = read_image(cv2, np, target_path)
            source_corners, source_detector = detect_checkerboard(
                cv2, np, source_image, board_size
            )
            target_corners, target_detector = detect_checkerboard(
                cv2, np, target_image, board_size
            )
            if source_corners is None or target_corners is None:
                missing = []
                if source_corners is None:
                    missing.append("source")
                if target_corners is None:
                    missing.append("target")
                raise ZoomTransformCalibrationError(
                    "checkerboard not found in offline image(s): " + ", ".join(missing)
                )
            camera_record = {
                "mode": "offline_images",
                "source_image": str(source_path),
                "target_image": str(target_path),
            }
        else:
            camera, selected_camera = open_camera(cv2, args)
            api, capabilities, ranges = connect_lens(args)
            lens_moves["source"] = move_lens(
                api,
                capabilities,
                ranges,
                zoom=source_zoom,
                focus=source_focus,
                no_init=bool(args.lens_no_init),
                settle_seconds=float(args.lens_settle_seconds),
            )
            print(
                f"Capture source zoom={source_zoom}, focus={source_focus}; "
                "keep the camera and checkerboard fixed"
            )
            source_image, source_corners, source_detector = acquire_image(
                cv2,
                np,
                camera,
                board_size,
                f"source zoom {source_zoom}",
                bool(args.capture_now),
                int(args.capture_discard_frames),
            )
            lens_moves["target"] = move_lens(
                api,
                capabilities,
                ranges,
                zoom=target_zoom,
                focus=target_focus,
                no_init=bool(args.lens_no_init),
                settle_seconds=float(args.lens_settle_seconds),
            )
            print(
                f"Capture target zoom={target_zoom}, focus={target_focus}; "
                "do not move the camera or checkerboard"
            )
            target_image, target_corners, target_detector = acquire_image(
                cv2,
                np,
                camera,
                board_size,
                f"target zoom {target_zoom}",
                bool(args.capture_now),
                int(args.capture_discard_frames),
            )
            camera_record = {
                "mode": "live_camera",
                **selected_camera,
                "requested_frame_width_px": args.frame_width,
                "requested_frame_height_px": args.frame_height,
                "autofocus_kept": bool(args.keep_camera_autofocus),
            }

        source_height, source_width = source_image.shape[:2]
        target_height, target_width = target_image.shape[:2]
        source_undistorted_points, source_matrix = undistort_points(
            cv2,
            np,
            source_corners,
            source_intrinsics,
            source_width,
            source_height,
        )
        target_undistorted_points, target_matrix = undistort_points(
            cv2,
            np,
            target_corners,
            target_intrinsics,
            target_width,
            target_height,
        )
        source_pixel_scale = calculate_checkerboard_pixel_scale(
            np,
            source_undistorted_points,
            board_size,
            float(args.square_size_mm),
        )
        target_pixel_scale = calculate_checkerboard_pixel_scale(
            np,
            target_undistorted_points,
            board_size,
            float(args.square_size_mm),
        )
        forward, inverse, inliers, quality, predicted = estimate_homography(
            cv2,
            np,
            source_undistorted_points,
            target_undistorted_points,
            threshold_px=float(args.ransac_threshold_px),
            max_iterations=int(args.ransac_iterations),
            confidence=float(args.ransac_confidence),
            min_inlier_ratio=float(args.min_inlier_ratio),
            max_inlier_rmse_px=float(args.max_inlier_rmse_px),
        )

        image_paths = {
            "source_raw": run_dir / "source_raw.png",
            "target_raw": run_dir / "target_raw.png",
            "source_corners": run_dir / "source_corners.png",
            "target_corners": run_dir / "target_corners.png",
            "source_undistorted": run_dir / "source_undistorted.png",
            "target_undistorted": run_dir / "target_undistorted.png",
            "validation_overlay": run_dir / "validation_overlay.png",
        }
        source_undistorted, target_undistorted, overlay = validation_overlay(
            cv2,
            np,
            source_image,
            target_image,
            source_matrix,
            target_matrix,
            source_intrinsics,
            target_intrinsics,
            forward,
            target_undistorted_points,
            predicted,
            inliers,
        )
        write_image(cv2, image_paths["source_raw"], source_image)
        write_image(cv2, image_paths["target_raw"], target_image)
        write_image(
            cv2,
            image_paths["source_corners"],
            annotate_corners(cv2, np, source_image, source_corners, board_size),
        )
        write_image(
            cv2,
            image_paths["target_corners"],
            annotate_corners(cv2, np, target_image, target_corners, board_size),
        )
        write_image(cv2, image_paths["source_undistorted"], source_undistorted)
        write_image(cv2, image_paths["target_undistorted"], target_undistorted)
        write_image(cv2, image_paths["validation_overlay"], overlay)

        created_at = utc_timestamp()
        report_path = run_dir / "zoom_pixel_transform.json"
        core_entry = build_summary_entry(
            created_at=created_at,
            source_zoom=source_zoom,
            target_zoom=target_zoom,
            source_focus=source_focus,
            target_focus=target_focus,
            source_size=(source_width, source_height),
            target_size=(target_width, target_height),
            source_pixel_scale=source_pixel_scale,
            target_pixel_scale=target_pixel_scale,
            forward=forward,
            inverse=inverse,
            quality=quality,
            report_path=report_path,
        )
        report = {
            "schema_version": 1,
            **core_entry,
            "calibration_summary": relative_path(summary_path),
            "board": {
                "inner_corner_columns": board_size[0],
                "inner_corner_rows": board_size[1],
                "corner_count": board_size[0] * board_size[1],
                "square_size_mm": float(args.square_size_mm),
                "constraint": "camera and checkerboard remain fixed between captures",
            },
            "camera": camera_record,
            "lens_moves": lens_moves,
            "source_intrinsics": intrinsics_record(source_intrinsics),
            "target_intrinsics": intrinsics_record(target_intrinsics),
            "detectors": {
                "source": source_detector,
                "target": target_detector,
            },
            "artifacts": {
                key: relative_path(value) for key, value in image_paths.items()
            },
            "observations": [
                {
                    "index": index,
                    "source_raw_pixel": source_corners[index].tolist(),
                    "target_raw_pixel": target_corners[index].tolist(),
                    "source_undistorted_pixel": source_undistorted_points[
                        index
                    ].tolist(),
                    "target_undistorted_pixel": target_undistorted_points[
                        index
                    ].tolist(),
                    "predicted_target_pixel": predicted[index].tolist(),
                    "forward_error_px": float(
                        np.linalg.norm(
                            predicted[index] - target_undistorted_points[index]
                        )
                    ),
                    "inlier": bool(inliers[index]),
                }
                for index in range(len(source_corners))
            ],
        }
        write_json(report_path, report)
        try:
            saved_summary_path = append_calibration_summary(
                core_entry, source_zoom, target_zoom
            )
        except (OSError, ValueError) as exc:
            raise ZoomTransformCalibrationError(
                f"result report was saved, but calibrate.json update failed: {exc}"
            ) from exc

        print("Zoom pixel transform calibration complete")
        print(f"Working data: {run_dir}")
        print(f"Detailed report: {report_path}")
        print(f"Calibration summary: {saved_summary_path}")
        print("source -> target homography (undistorted pixels):")
        for row in forward:
            print("  [{: .12g}, {: .12g}, {: .12g}]".format(*row))
        print(
            "Inliers: {}/{} ({:.1%})".format(
                quality["inlier_count"],
                quality["point_count"],
                quality["inlier_ratio"],
            )
        )
        print_pixel_scale("Source", source_zoom, source_pixel_scale)
        print_pixel_scale("Target", target_zoom, target_pixel_scale)
        print_calibration_errors(quality)
        return 0
    finally:
        if camera is not None:
            camera.release()
        if api is not None:
            try:
                api.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> int:
    try:
        return run_calibration(load_args())
    except ZoomTransformCalibrationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
