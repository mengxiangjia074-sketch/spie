import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SETTINGS_FILE = "calibrate_lens.jsonnet"
SETTINGS = None
CALIBRATION_WINDOW_NAME = "Lens board calibration"
FOCUS_SELECTION_RULE = (
    "prefer more frames with detected checkerboard; use score as tiebreaker"
)
CORE_CALIBRATION_KEYS = (
    "rms_reprojection_error",
    "mean_reprojection_error",
    "image_width",
    "image_height",
    "camera_matrix",
    "distortion_coefficients",
    "views",
)

# The LensCamera package lives in the detection directory one level above
# these calibration scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import LensCamera as lens_camera
import autofocus_pcb
import calibration_summary
from LensCamera import paths
from LensCamera import camera as camera_tools
from LensCamera.json_io import load_json_object as shared_load_json_object
from LensCamera.lens_config import (
    read_device_number as shared_read_device_number,
    read_lens_targets as shared_read_lens_targets,
)
from LensCamera.sampling import build_sample_points as shared_build_sample_points
from LensCamera.settings import (
    Settings as JsonnetSettings,
    SettingsError,
    load_jsonnet_settings as shared_load_jsonnet_settings,
    load_jsonnet_settings_fallback as shared_load_jsonnet_settings_fallback,
    remove_trailing_commas as shared_remove_trailing_commas,
    strip_jsonnet_comments as shared_strip_jsonnet_comments,
)
from LensCamera.timestamps import run_timestamp as shared_run_timestamp
from LensCamera.timestamps import utc_timestamp as shared_utc_timestamp


class CalibrationError(Exception):
    pass


class NoCheckerboardFocusError(CalibrationError):
    def __init__(self, message, autofocus):
        super().__init__(message)
        self.autofocus = autofocus


def utc_timestamp():
    return shared_utc_timestamp()


def run_timestamp():
    return shared_run_timestamp()


def load_settings():
    BASE_DIR = Path(__file__).resolve().parent
    settings_path = BASE_DIR / "configs" / "lens" / SETTINGS_FILE
    try:
        return JsonnetSettings(settings_path, SETTINGS_FILE).values
    except SettingsError as exc:
        raise CalibrationError(str(exc)) from exc


def load_args():
    global SETTINGS
    SETTINGS = load_settings()
    args = SimpleNamespace(
        **{name: setting_value(name) for name in SETTINGS}
    )
    args.output = setting_value("default_output_file")
    args.log_output = setting_value("default_log_file")
    return enforce_online_capture(args)


def enforce_online_capture(args):
    """Keep camera-intrinsics calibration on its sole supported GUI workflow."""
    args.image_dir = None
    args.skip_lens = True
    args.full_workflow = False
    return args


def load_jsonnet_settings(settings_path):
    try:
        return shared_load_jsonnet_settings(settings_path, SETTINGS_FILE)
    except SettingsError as exc:
        raise CalibrationError(str(exc)) from exc


def load_jsonnet_settings_fallback(settings_path):
    try:
        return shared_load_jsonnet_settings_fallback(settings_path, SETTINGS_FILE)
    except SettingsError as exc:
        raise CalibrationError(str(exc)) from exc


def strip_jsonnet_comments(text):
    return shared_strip_jsonnet_comments(text)


def remove_trailing_commas(text):
    return shared_remove_trailing_commas(text)


def setting_value(name):
    if SETTINGS is None:
        raise CalibrationError("{} has not been loaded".format(SETTINGS_FILE))

    if name not in SETTINGS:
        raise CalibrationError("{} missing required setting {!r}".format(SETTINGS_FILE, name))

    item = SETTINGS[name]
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def load_image_tools():
    return camera_tools.load_image_tools(CalibrationError)


def resolve_path(path_text, default_to_script=False):
    return paths.resolve_path(path_text, default_to_project=default_to_script)


def resolve_output_path(path_text):
    return resolve_path(
        path_text, default_to_script=(path_text == setting_value("default_output_file"))
    )


def resolve_log_path(path_text, output_path):
    if path_text:
        default_log_file = setting_value("default_log_file")
        return resolve_path(
            path_text,
            default_to_script=(
                default_log_file is not None and path_text == default_log_file
            ),
        )
    return output_path.with_suffix(".log")


def create_run_directory(debug_dir_text):
    if not debug_dir_text:
        raise CalibrationError("--debug-dir is required for timestamped run output")

    # The run directory is anchored to the project root, so results always land
    # in working_data no matter which directory the script is run from.
    base_dir = resolve_path(debug_dir_text, default_to_script=True)
    timestamp = run_timestamp()
    run_dir = base_dir / timestamp
    suffix = 2
    while run_dir.exists():
        run_dir = base_dir / "{}_{:02d}".format(timestamp, suffix)
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_run_directory(args):
    return getattr(args, "run_dir", None)


def path_name(path_text, fallback_name):
    if path_text:
        name = Path(path_text).name
        if name:
            return name
    return fallback_name


def resolve_run_output_path(args, path_text):
    run_dir = get_run_directory(args)
    if run_dir is None:
        return resolve_output_path(path_text)
    return run_dir / path_name(path_text, setting_value("default_output_file"))


def resolve_run_log_path(args, path_text, output_path):
    run_dir = get_run_directory(args)
    if run_dir is None:
        return resolve_log_path(path_text, output_path)
    if path_text:
        return run_dir / path_name(path_text, output_path.with_suffix(".log").name)
    return output_path.with_suffix(".log")


def debug_image_base_dir(args):
    run_dir = get_run_directory(args)
    if run_dir is not None:
        return run_dir / "corner_images"
    if args.debug_dir:
        return resolve_path(args.debug_dir, default_to_script=True)
    return None


def capture_image_base_dir(args):
    if getattr(args, "image_dir", None):
        return None
    if not args.capture_dir:
        return None

    run_dir = get_run_directory(args)
    if run_dir is not None:
        return run_dir / "capture_images"
    return resolve_path(args.capture_dir, default_to_script=True)


def build_sample_points(min_addr, max_addr, steps):
    return shared_build_sample_points(min_addr, max_addr, steps)


def parse_int_value(name, value):
    if isinstance(value, bool):
        raise ValueError("{} must be an integer, not a boolean".format(name))

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("{} must not contain an empty value".format(name))
        try:
            return int(text, 0)
        except ValueError as exc:
            raise ValueError("{} must contain integer values".format(name)) from exc

    raise ValueError("{} must contain integer values".format(name))


def parse_zoom_positions(value):
    if value is None:
        return []

    items = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("none", "null", "auto"):
            return []
        items = [item.strip() for item in text.split(",")]
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        items = [value]

    if not isinstance(items, list):
        raise ValueError("zoom_positions must be null, an integer, or a list")

    zoom_positions = []
    for item in items:
        zoom = parse_int_value("zoom_positions", item)
        if zoom not in zoom_positions:
            zoom_positions.append(zoom)
    return zoom_positions


def select_zoom_points(args, zoom_min, zoom_max):
    if args.zoom_positions:
        out_of_range = [
            zoom for zoom in args.zoom_positions if zoom < zoom_min or zoom > zoom_max
        ]
        if out_of_range:
            raise CalibrationError(
                "manual zoom position(s) {} outside allowed range {}..{}".format(
                    out_of_range, zoom_min, zoom_max
                )
            )
        return list(args.zoom_positions), "manual"

    return build_sample_points(zoom_min, zoom_max, args.zoom_steps), "automatic"


def read_device_number(config):
    return shared_read_device_number(config)


def read_lens_targets(config):
    return shared_read_lens_targets(config)


def load_lens_positions():
    # 镜头目标已合并进 autofocus 配置文件 (原 lens_target.json)。
    merged = Path(autofocus_pcb.DEFAULT_SETTINGS_PATH)
    config = load_jsonnet_settings(merged)
    targets = read_lens_targets(config)
    if not targets:
        return []
    return [
        {
            "name": "merged_autofocus_settings",
            "device": read_device_number(config),
            "targets": targets,
            "source": str(merged),
        }
    ]


def read_current(name, motor):
    return lens_camera.read_current(name)


def read_lens_snapshot(capabilities):
    return lens_camera.read_lens_snapshot(capabilities)


def move_connected_motor(name, target, capabilities, settle_seconds):
    try:
        move = lens_camera.move_connected_motor(
            name,
            target,
            capabilities,
            settle_seconds=settle_seconds,
        )
    except lens_camera.LensControlError as exc:
        raise CalibrationError(str(exc)) from exc

    print(
        "{}: target {} -> actual {} (error {})".format(
            name, move["target"], move["actual"], move["error"]
        )
    )
    return {
        "target": move["target"],
        "before": move["before"],
        "actual": move["actual"],
        "error": move["error"],
    }


def move_lens_position(position, device_override, init_if_needed, settle_seconds):
    targets = position["targets"]
    device_number = device_override
    if device_number is None:
        device_number = position["device"]

    capabilities = lens_camera.connect(device_number)
    device_number = lens_camera.get_last_connected_device_number(device_number)
    try:
        ranges = lens_camera.read_ranges(capabilities)
        lens_camera.print_ranges(ranges, targets)
        lens_camera.validate_targets(targets, ranges)
        lens_camera.ensure_targets_ready(targets, init_if_needed)

        moves = {}
        for name in lens_camera.MOTOR_ORDER:
            if name not in targets:
                continue

            moves[name] = move_connected_motor(
                name, targets[name], capabilities, settle_seconds
            )

        return {
            "enabled": True,
            "device": device_number,
            "source": position["source"],
            "name": position["name"],
            "targets": targets,
            "moves": moves,
            "snapshot": read_lens_snapshot(capabilities),
        }
    finally:
        lens_camera.close()


def build_object_points(np, board_cols, board_rows, square_size):
    object_points = np.zeros((board_cols * board_rows, 3), np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * square_size
    return object_points


def detect_checkerboard(cv2, image, board_size):
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, board_size, sb_flags)
        if found:
            return True, corners, gray.shape[::-1]

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return False, None, gray.shape[::-1]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined, gray.shape[::-1]


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    files = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in setting_value("image_suffixes")
    ]
    if not files:
        raise CalibrationError("no calibration images found in {}".format(image_dir))
    return files


def position_image_dir(base_dir, position_name):
    base_path = Path(base_dir)
    subdir = base_path / position_name
    if subdir.is_dir():
        return subdir
    return base_path


def save_debug_image(cv2, debug_dir, label, image_name, image, board_size, corners, found):
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    display = image.copy()
    if corners is not None:
        cv2.drawChessboardCorners(display, board_size, corners, found)
    cv2.imwrite(str(debug_dir / image_name), display)


def save_capture_image(cv2, capture_dir, image_name, image):
    if capture_dir is None:
        return None

    capture_dir.mkdir(parents=True, exist_ok=True)
    image_path = capture_dir / image_name
    cv2.imwrite(str(image_path), image)
    return str(image_path)


def get_screen_size():
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


def configure_video_window(cv2, window_name):
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


def build_capture_display(
    cv2, frame, board_size, corners, found, position_name, accepted_count, total_count, note
):
    display = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(display, board_size, corners, found)

    status = "{} valid frames: {}/{}".format(
        position_name, accepted_count, total_count
    )
    if found is None:
        color = (255, 255, 255)
    else:
        color = (0, 255, 0) if found else (0, 0, 255)
    cv2.putText(
        display,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        note,
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return display


def wait_for_enter_or_quit(cv2):
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (10, 13):
            return key
        if key in (27, ord("q"), ord("Q")):
            return key


def record_accepted_frame(
    cv2,
    object_template,
    object_points,
    image_points,
    frames,
    frame,
    corners,
    position_name,
    board_size,
    debug_dir,
    capture_dir,
    args,
):
    frame_index = len(image_points) + 1
    object_points.append(object_template.copy())
    image_points.append(corners)

    frame_name = "{:03d}.png".format(frame_index)
    capture_path = save_capture_image(cv2, capture_dir, frame_name, frame)
    save_debug_image(
        cv2,
        debug_dir,
        position_name,
        frame_name,
        frame,
        board_size,
        corners,
        True,
    )
    frames.append(
        {
            "frame": frame_index,
            "accepted": True,
            "corners": args.board_cols * args.board_rows,
            "capture_file": capture_path,
        }
    )
    print(
        "{}: accepted frame {}/{}".format(
            position_name, frame_index, args.frames
        )
    )
    return frame_index


def capture_single_shot_frames(
    cv2,
    camera,
    object_template,
    object_points,
    image_points,
    frames,
    args,
    position_name,
    board_size,
    debug_dir,
    capture_dir,
):
    print(
        "{}: single-shot mode; Space/Enter saves, r retakes, q/Esc finishes".format(
            position_name
        )
    )

    while len(image_points) < args.frames:
        ok, frame = camera.read()
        if not ok:
            raise CalibrationError("camera frame read failed")

        found, corners, detected_size = detect_checkerboard(cv2, frame, board_size)
        note = "Space/Enter save, r retake, q finish"
        if not found:
            note = "checkerboard not found; r retake, q finish"

        display = build_capture_display(
            cv2,
            frame,
            board_size,
            corners,
            found,
            position_name,
            len(image_points),
            args.frames,
            note,
        )
        cv2.imshow(CALIBRATION_WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key in (13, 32):
            if not found:
                print("{}: checkerboard not found; retake".format(position_name))
                continue

            record_accepted_frame(
                cv2,
                object_template,
                object_points,
                image_points,
                frames,
                frame,
                corners,
                position_name,
                board_size,
                debug_dir,
                capture_dir,
                args,
            )
        elif key in (ord("r"), ord("R")):
            continue

    return detected_size if "detected_size" in locals() else None


def collect_from_images(cv2, np, args, position_name):
    board_size = (args.board_cols, args.board_rows)
    object_template = build_object_points(
        np, args.board_cols, args.board_rows, args.square_size
    )
    base_dir = resolve_path(args.image_dir)
    image_dir = position_image_dir(base_dir, position_name)
    debug_dir = None
    debug_base_dir = debug_image_base_dir(args)
    if debug_base_dir is not None:
        debug_dir = debug_base_dir / position_name

    object_points = []
    image_points = []
    frames = []
    image_size = None

    for image_path in list_image_files(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            frames.append({"file": str(image_path), "accepted": False, "error": "read_failed"})
            continue

        found, corners, detected_size = detect_checkerboard(cv2, image, board_size)
        image_size = detected_size
        frames.append(
            {
                "file": str(image_path),
                "accepted": bool(found),
                "corners": args.board_cols * args.board_rows if found else 0,
            }
        )

        debug_name = image_path.name
        save_debug_image(
            cv2, debug_dir, position_name, debug_name, image, board_size, corners, found
        )

        if found:
            object_points.append(object_template.copy())
            image_points.append(corners)
            print("{}: accepted {}".format(position_name, image_path.name))
        else:
            print("{}: checkerboard not found in {}".format(position_name, image_path.name))

    if len(image_points) < args.min_frames:
        raise CalibrationError(
            "{}: only {} valid image(s), but --min-frames is {}".format(
                position_name, len(image_points), args.min_frames
            )
        )

    return object_points, image_points, image_size, frames


def select_named_camera(args):
    selected = getattr(args, "selected_camera", None)
    if selected is None:
        selected = camera_tools.find_camera_by_name(
            args.camera_name, CalibrationError
        )
        args.selected_camera = selected
    return selected


def open_camera(cv2, args):
    selected = select_named_camera(args)
    return camera_tools.open_camera(cv2, camera_tools.camera_source(selected)), selected


def configure_camera(cv2, camera, args):
    return camera_tools.configure_camera(cv2, camera, args)


def build_pcb_autofocus_args(args, autofocus_label):
    return SimpleNamespace(
        metric=args.focus_metric,
        optimizer="golden-section",
        fine_steps=args.focus_golden_final_steps,
        score_max_side=args.focus_score_max_side,
        score_frames=args.focus_score_frames,
        aggregate=args.focus_score_aggregate,
        discard_frames=args.focus_discard_frames,
        startup_discard_frames=args.focus_startup_discard_frames,
        settle=args.focus_settle,
        roi=args.focus_roi,
        roi_scale=args.focus_roi_scale,
        blur_ksize=args.focus_blur_ksize,
        golden_tolerance=args.focus_golden_tolerance,
        golden_bracket_steps=args.focus_golden_bracket_steps,
        golden_tie_relative_margin=args.focus_golden_tie_relative_margin,
        golden_highres_auto=args.focus_golden_highres_auto,
        golden_max_iter=args.focus_golden_max_iter,
        autofocus_label=autofocus_label,
    )


def autofocus_focus(cv2, np, args, capabilities, autofocus_label):
    focus_range = lens_camera.get_motor_range("focus")
    lens_focus_range = dict(focus_range)
    if args.focus_min is not None:
        focus_range["min"] = args.focus_min
    if args.focus_max is not None:
        focus_range["max"] = args.focus_max
    if (
        focus_range["min"] < lens_focus_range["min"]
        or focus_range["max"] > lens_focus_range["max"]
    ):
        raise CalibrationError(
            "requested focus range {}..{} is outside lens range {}..{}".format(
                focus_range["min"],
                focus_range["max"],
                lens_focus_range["min"],
                lens_focus_range["max"],
            )
        )
    if focus_range["min"] > focus_range["max"]:
        raise CalibrationError("--focus-min must be <= --focus-max")

    autofocus_args = build_pcb_autofocus_args(args, autofocus_label)

    camera, selected_camera = open_camera(cv2, args)
    if not camera.isOpened():
        raise CalibrationError(
            "cannot open camera {!r} at index {} for autofocus".format(
                selected_camera["name"], selected_camera["index"]
            )
        )

    configure_camera(cv2, camera, args)
    autofocus_pcb.apply_golden_highres_profile(cv2, camera, autofocus_args)
    autofocus_pcb.discard_frames(camera, autofocus_args.startup_discard_frames)

    print("")
    try:
        evaluator = autofocus_pcb.FocusEvaluator(
            cv2, np, camera, capabilities, autofocus_args
        )
        optimizer_record = autofocus_pcb.run_golden_section_search(
            evaluator, focus_range, autofocus_args
        )
    except autofocus_pcb.AutofocusError as exc:
        raise CalibrationError(str(exc)) from exc
    finally:
        camera.release()

    if evaluator.best is None:
        raise CalibrationError("autofocus did not produce any focus score")

    final_move = move_connected_motor(
        "focus", evaluator.best["actual"], capabilities, autofocus_args.settle
    )
    print(
        "autofocus: best focus {} with score {:.3f}".format(
            final_move["actual"], evaluator.best["score"]
        )
    )

    scans = {}
    for record in evaluator.records:
        scans.setdefault(record["stage"], []).append(record)

    return {
        "method": "autofocus_pcb",
        "metric": args.focus_metric,
        "metric_note": (
            "Uses the same PCB autofocus sharpness metrics as autofocus_pcb.py. "
            "Default tenengrad is recommended for PCB traces, pads, and silkscreen."
        ),
        "optimizer": optimizer_record,
        "focus_range": focus_range,
        "optimizer_name": "golden-section",
        "lens_focus_range": lens_focus_range,
        "golden_final_steps": autofocus_args.fine_steps,
        "score_frames": autofocus_args.score_frames,
        "score_max_side": autofocus_args.score_max_side,
        "score_aggregate": args.focus_score_aggregate,
        "discard_frames": autofocus_args.discard_frames,
        "startup_discard_frames": autofocus_args.startup_discard_frames,
        "settle_seconds": autofocus_args.settle,
        "roi": args.focus_roi,
        "roi_scale": autofocus_args.roi_scale,
        "blur_ksize": autofocus_args.blur_ksize,
        "golden_bracket_steps": autofocus_args.golden_bracket_steps,
        "golden_tie_relative_margin": autofocus_args.golden_tie_relative_margin,
        "golden_highres_profile": getattr(autofocus_args, "golden_highres_record", None),
        "scan": scans,
        "best": {
            "target": evaluator.best["target"],
            "actual": final_move["actual"],
            "score": evaluator.best["score"],
            "stage": evaluator.best["stage"],
            "roi": evaluator.best.get("roi"),
        },
        "label": autofocus_label,
    }


def capture_from_camera(cv2, np, args, position_name):
    board_size = (args.board_cols, args.board_rows)
    object_template = build_object_points(
        np, args.board_cols, args.board_rows, args.square_size
    )
    debug_base_dir = debug_image_base_dir(args)
    capture_base_dir = capture_image_base_dir(args)
    debug_dir = debug_base_dir / position_name if debug_base_dir is not None else None
    capture_dir = (
        capture_base_dir / position_name if capture_base_dir is not None else None
    )

    camera, selected_camera = open_camera(cv2, args)
    if not camera.isOpened():
        raise CalibrationError(
            "cannot open camera {!r} at index {}".format(
                selected_camera["name"], selected_camera["index"]
            )
        )

    configure_camera(cv2, camera, args)

    object_points = []
    image_points = []
    frames = []
    image_size = None

    print("")
    print("{}: show the checkerboard to the camera".format(position_name))
    if args.single_shot_capture:
        print(
            "{}: each prompt captures one still image for review".format(
                position_name
            )
        )
    elif args.live_preview_detect_on_key:
        print(
            "{}: live preview only; press Space/Enter to detect and save, q to finish".format(
                position_name
            )
        )
    else:
        print(
            "{}: live preview with continuous checkerboard detection; press Space/Enter to save, q to finish".format(
                position_name
            )
        )

    try:
        if args.single_shot_capture:
            image_size = capture_single_shot_frames(
                cv2,
                camera,
                object_template,
                object_points,
                image_points,
                frames,
                args,
                position_name,
                board_size,
                debug_dir,
                capture_dir,
            )
        else:
            configure_video_window(cv2, CALIBRATION_WINDOW_NAME)
            while len(image_points) < args.frames:
                ok, frame = camera.read()
                if not ok:
                    raise CalibrationError("camera frame read failed")

                if args.live_preview_detect_on_key:
                    display = build_capture_display(
                        cv2,
                        frame,
                        board_size,
                        None,
                        None,
                        position_name,
                        len(image_points),
                        args.frames,
                        "Space/Enter detect/save, q finish",
                    )
                    cv2.imshow(CALIBRATION_WINDOW_NAME, display)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (27, ord("q"), ord("Q")):
                        break

                    if key not in (13, 32):
                        continue

                    found, corners, detected_size = detect_checkerboard(
                        cv2, frame, board_size
                    )
                    image_size = detected_size
                    if not found:
                        print("{}: checkerboard not found; try another frame".format(position_name))
                        display = build_capture_display(
                            cv2,
                            frame,
                            board_size,
                            corners,
                            found,
                            position_name,
                            len(image_points),
                            args.frames,
                            "checkerboard not found",
                        )
                        cv2.imshow(CALIBRATION_WINDOW_NAME, display)
                        cv2.waitKey(250)
                        continue

                    record_accepted_frame(
                        cv2,
                        object_template,
                        object_points,
                        image_points,
                        frames,
                        frame,
                        corners,
                        position_name,
                        board_size,
                        debug_dir,
                        capture_dir,
                        args,
                    )
                    display = build_capture_display(
                        cv2,
                        frame,
                        board_size,
                        corners,
                        found,
                        position_name,
                        len(image_points),
                        args.frames,
                        "saved; Enter next, q finish",
                    )
                    cv2.imshow(CALIBRATION_WINDOW_NAME, display)
                    key = wait_for_enter_or_quit(cv2)
                    if key in (27, ord("q"), ord("Q")):
                        break
                    continue

                found, corners, detected_size = detect_checkerboard(cv2, frame, board_size)
                image_size = detected_size

                display = build_capture_display(
                    cv2,
                    frame,
                    board_size,
                    corners,
                    found,
                    position_name,
                    len(image_points),
                    args.frames,
                    "Space/Enter save, q finish",
                )

                cv2.imshow(CALIBRATION_WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    break

                if found and key in (13, 32):
                    record_accepted_frame(
                        cv2,
                        object_template,
                        object_points,
                        image_points,
                        frames,
                        frame,
                        corners,
                        position_name,
                        board_size,
                        debug_dir,
                        capture_dir,
                        args,
                    )
                    display = build_capture_display(
                        cv2,
                        frame,
                        board_size,
                        corners,
                        found,
                        position_name,
                        len(image_points),
                        args.frames,
                        "saved; Enter next, q finish",
                    )
                    cv2.imshow(CALIBRATION_WINDOW_NAME, display)
                    key = wait_for_enter_or_quit(cv2)
                    if key in (27, ord("q"), ord("Q")):
                        break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(image_points) < args.min_frames:
        raise CalibrationError(
            "{}: only {} valid frame(s), but --min-frames is {}".format(
                position_name, len(image_points), args.min_frames
            )
        )

    return object_points, image_points, image_size, frames


def calculate_reprojection_errors(
    cv2, object_points, image_points, camera_matrix, dist_coeffs, rvecs, tvecs
):
    errors = []
    total_error = 0.0
    for index, obj_points in enumerate(object_points):
        projected, _jacobian = cv2.projectPoints(
            obj_points, rvecs[index], tvecs[index], camera_matrix, dist_coeffs
        )
        error = cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
        total_error += error

    return float(total_error / len(object_points)), errors


def print_calibration_errors(output):
    for position in output.get("positions", []):
        calibration = position.get("calibration") or {}
        rms = calibration.get("rms_reprojection_error")
        if rms is None:
            continue
        print(
            "标定误差: {} RMS 重投影误差 {:.4f} px".format(
                position.get("name", "camera"), float(rms)
            )
        )


def calibrate_camera(cv2, object_points, image_points, image_size):
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    mean_error, per_view_errors = calculate_reprojection_errors(
        cv2,
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    )

    return {
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": mean_error,
        "per_view_reprojection_errors": per_view_errors,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "views": len(image_points),
        "rvecs": [rvec.reshape(-1).tolist() for rvec in rvecs],
        "tvecs": [tvec.reshape(-1).tolist() for tvec in tvecs],
    }


def write_json(output_path, data):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, sort_keys=False)
        file_obj.write("\n")


def extract_core_lens_position(lens):
    if not lens or not lens.get("enabled"):
        return None

    moves = lens.get("moves") or {}
    lens_position = {}
    for name, move in moves.items():
        if isinstance(move, dict) and "actual" in move:
            lens_position[name] = move["actual"]

    return lens_position or None


def extract_lens_zoom(lens):
    if not lens:
        return None

    moves = lens.get("moves") or {}
    zoom_move = moves.get("zoom")
    if isinstance(zoom_move, dict) and zoom_move.get("actual") is not None:
        return zoom_move["actual"]

    snapshot = lens.get("snapshot") or {}
    zoom_snapshot = snapshot.get("zoom")
    if isinstance(zoom_snapshot, dict) and zoom_snapshot.get("current") is not None:
        return zoom_snapshot["current"]

    targets = lens.get("targets") or {}
    if targets.get("zoom") is not None:
        return targets["zoom"]
    return None


def build_core_calibration_result(calibration):
    return {
        key: calibration[key] for key in CORE_CALIBRATION_KEYS if key in calibration
    }


def build_core_output(output):
    core = {
        "schema_version": output["schema_version"],
        "type": "camera_calibration_core",
        "source_type": output["type"],
        "created_at_utc": output["created_at_utc"],
        "board": output["board"],
        "positions": [],
    }

    for position in output.get("positions", []):
        item = {
            "name": position["name"],
            "calibration": build_core_calibration_result(position["calibration"]),
        }
        lens_position = extract_core_lens_position(position.get("lens"))
        if lens_position is not None:
            item["lens_position"] = lens_position
        core["positions"].append(item)

    return core


def build_log_output(output, core_output_path):
    log_output = copy.deepcopy(output)
    log_output["type"] = output["type"] + "_log"
    log_output["calibration_json_file"] = str(core_output_path)
    log_output["calibration_fields_saved_in_json"] = list(CORE_CALIBRATION_KEYS)

    for position in log_output.get("positions", []):
        calibration = position.pop("calibration", None)
        if calibration:
            details = {
                key: value
                for key, value in calibration.items()
                if key not in CORE_CALIBRATION_KEYS
            }
            if details:
                position["calibration_details"] = details

    return log_output


def append_calibration_summary(output):
    """Append the camera intrinsics and distortion coefficients of this run
    to the shared per-day summary JSON below detection/calibration/output."""
    summary_path = None
    for position in output.get("positions", []):
        calibration = position.get("calibration")
        if not calibration:
            continue
        zoom = extract_lens_zoom(position.get("lens"))
        name = None if zoom is None else calibration_summary.zoom_entry_name(zoom)
        summary_path = calibration_summary.append_entry(
            {
                "type": output["type"],
                "created_at_utc": output["created_at_utc"],
                "positions": [
                    {
                        "name": name or position["name"],
                        "original_position_name": position["name"],
                        "image_width": calibration["image_width"],
                        "image_height": calibration["image_height"],
                        "camera_matrix": calibration["camera_matrix"],
                        "distortion_coefficients": calibration[
                            "distortion_coefficients"
                        ],
                        "rms_reprojection_error": calibration[
                            "rms_reprojection_error"
                        ],
                        "mean_reprojection_error": calibration[
                            "mean_reprojection_error"
                        ],
                        "views": calibration["views"],
                    }
                ],
            },
            zoom=zoom,
        )

    return summary_path


def write_calibration_outputs(args, output):
    output_path = resolve_run_output_path(args, args.output)
    log_path = resolve_run_log_path(args, args.log_output, output_path)
    if str(output_path.resolve()).lower() == str(log_path.resolve()).lower():
        raise CalibrationError("--output and --log-output must not be the same file")

    write_json(output_path, build_core_output(output))
    write_json(log_path, build_log_output(output, output_path))

    summary_path = None
    try:
        summary_path = append_calibration_summary(output)
    except (OSError, ValueError) as exc:
        print(
            "warning: cannot append calibration summary: {}".format(exc),
            file=sys.stderr,
        )
    if summary_path is not None:
        print("calibration summary saved: {}".format(summary_path))
    print_calibration_errors(output)
    return output_path, log_path


def build_position_list(args):
    if args.skip_lens:
        return [
            {
                "name": "current_lens",
                "device": None,
                "targets": {},
                "source": None,
            }
        ]

    positions = load_lens_positions()
    if positions:
        return positions

    return [
        {
            "name": "current_lens",
            "device": args.device,
            "targets": {},
            "source": None,
        }
    ]


def collect_board_points(cv2, np, args, position_name):
    if args.image_dir:
        return collect_from_images(cv2, np, args, position_name)
    return capture_from_camera(cv2, np, args, position_name)


def build_output_header(args, calibration_type):
    run_dir = get_run_directory(args)
    debug_base_dir = debug_image_base_dir(args)
    capture_base_dir = capture_image_base_dir(args)
    selected_camera = None if args.image_dir else select_named_camera(args)
    return {
        "schema_version": 2,
        "type": calibration_type,
        "created_at_utc": utc_timestamp(),
        "run": {
            "directory": str(run_dir) if run_dir is not None else None,
            "corner_images_dir": str(debug_base_dir)
            if debug_base_dir is not None
            else None,
            "capture_images_dir": str(capture_base_dir)
            if capture_base_dir is not None
            else None,
        },
        "board": {
            "pattern": "checkerboard",
            "inner_corners": {
                "columns": args.board_cols,
                "rows": args.board_rows,
            },
            "square_size": args.square_size,
            "square_unit": args.square_unit,
        },
        "image_source": {
            "mode": "images" if args.image_dir else "camera",
            "camera": (
                None if selected_camera is None else selected_camera["index"]
            ),
            "camera_name": (
                None if selected_camera is None else selected_camera["name"]
            ),
            "camera_name_match": (
                None if selected_camera is None else args.camera_name
            ),
            "camera_source": (
                None if selected_camera is None else selected_camera.get("source")
            ),
            "image_dir": str(resolve_path(args.image_dir)) if args.image_dir else None,
            "capture_dir": str(capture_base_dir)
            if capture_base_dir is not None
            else None,
            "single_shot_capture": args.single_shot_capture,
            "live_preview_detect_on_key": args.live_preview_detect_on_key,
            "frames_requested": args.frames,
            "min_frames": args.min_frames,
        },
        "positions": [],
    }


def require_full_workflow_lens(capabilities):
    missing = []
    for name in ("zoom", "focus"):
        if not lens_camera.motor_supported(name, capabilities):
            missing.append(name)

    if missing:
        raise CalibrationError(
            "full workflow requires supported motor(s): {}".format(
                ", ".join(missing)
            )
        )


def run_full_workflow(args, cv2, np):
    if args.skip_lens:
        raise CalibrationError("--full-workflow cannot be used with --skip-lens")
    if args.image_dir:
        raise CalibrationError("--full-workflow requires a live camera, not --image-dir")

    device_number = args.device
    output = build_output_header(args, "full_electric_lens_checkerboard_calibration")
    output["workflow"] = {
        "mode": "full",
        "zoom_steps": args.zoom_steps,
        "zoom_positions": args.zoom_positions,
        "focus_autofocus_method": "autofocus_pcb",
        "focus_metric": args.focus_metric,
        "focus_optimizer": "golden-section",
        "focus_min": args.focus_min,
        "focus_max": args.focus_max,
        "focus_golden_final_steps": args.focus_golden_final_steps,
        "focus_score_max_side": args.focus_score_max_side,
        "focus_score_frames": args.focus_score_frames,
        "focus_score_aggregate": args.focus_score_aggregate,
        "focus_discard_frames": args.focus_discard_frames,
        "focus_startup_discard_frames": args.focus_startup_discard_frames,
        "focus_settle_seconds": args.focus_settle,
        "focus_roi": args.focus_roi,
        "focus_roi_scale": args.focus_roi_scale,
        "focus_blur_ksize": args.focus_blur_ksize,
        "focus_golden_tolerance": args.focus_golden_tolerance,
        "focus_golden_bracket_steps": args.focus_golden_bracket_steps,
        "focus_golden_tie_relative_margin": args.focus_golden_tie_relative_margin,
        "focus_golden_highres_auto": args.focus_golden_highres_auto,
        "focus_golden_max_iter": args.focus_golden_max_iter,
        "iris_position": args.iris_position,
    }

    capabilities = lens_camera.connect(device_number)
    device_number = lens_camera.get_last_connected_device_number(device_number)
    try:
        ranges = lens_camera.read_ranges(capabilities)
        lens_camera.print_ranges(ranges)
        require_full_workflow_lens(capabilities)

        lens_camera.ensure_ready("zoom", init_if_needed=not args.no_init)
        lens_camera.ensure_ready("focus", init_if_needed=not args.no_init)

        if args.iris_position is not None:
            if not lens_camera.motor_supported("iris", capabilities):
                raise CalibrationError(
                    "--iris-position was set, but connected lens does not support iris"
                )
            lens_camera.ensure_ready("iris", init_if_needed=not args.no_init)

        zoom_range = lens_camera.get_motor_range("zoom")
        zoom_min = zoom_range["min"]
        zoom_max = zoom_range["max"]
        zoom_points, zoom_source = select_zoom_points(args, zoom_min, zoom_max)
        output["lens_range"] = read_lens_snapshot(capabilities)
        output["zoom_source"] = zoom_source
        output["zoom_points"] = zoom_points
        output["skipped_positions"] = []

        for index, zoom_target in enumerate(zoom_points, start=1):
            print("")
            print(
                "=== Full workflow position {}/{}: zoom {} ===".format(
                    index, len(zoom_points), zoom_target
                )
            )

            moves = {
                "zoom": move_connected_motor(
                    "zoom", zoom_target, capabilities, args.settle
                )
            }
            if args.iris_position is not None:
                moves["iris"] = move_connected_motor(
                    "iris", args.iris_position, capabilities, args.settle
                )

            autofocus_label = "zoom_{:05d}".format(moves["zoom"]["actual"])
            try:
                autofocus = autofocus_focus(cv2, np, args, capabilities, autofocus_label)
            except NoCheckerboardFocusError as exc:
                position_name = "zoom_{:05d}_skipped_no_checkerboard".format(
                    moves["zoom"]["actual"]
                )
                print(
                    "{}: skipped because focus scan did not detect checkerboard".format(
                        position_name
                    )
                )
                output["skipped_positions"].append(
                    {
                        "name": position_name,
                        "reason": "no_checkerboard_detected_during_focus_scan",
                        "lens": {
                            "enabled": True,
                            "device": device_number,
                            "targets": {
                                name: move["target"] for name, move in moves.items()
                            },
                            "moves": moves,
                            "snapshot": read_lens_snapshot(capabilities),
                            "autofocus": exc.autofocus,
                        },
                    }
                )
                continue
            moves["focus"] = {
                "target": autofocus["best"]["target"],
                "actual": autofocus["best"]["actual"],
                "error": autofocus["best"]["actual"] - autofocus["best"]["target"],
            }

            position_name = "zoom_{:05d}_focus_{:05d}".format(
                moves["zoom"]["actual"], moves["focus"]["actual"]
            )
            object_points, image_points, image_size, frames = collect_board_points(
                cv2, np, args, position_name
            )
            camera_result = calibrate_camera(
                cv2, object_points, image_points, image_size
            )

            output["positions"].append(
                {
                    "name": position_name,
                    "lens": {
                        "enabled": True,
                        "device": device_number,
                        "targets": {
                            name: move["target"] for name, move in moves.items()
                        },
                        "moves": moves,
                        "snapshot": read_lens_snapshot(capabilities),
                        "autofocus": autofocus,
                    },
                    "frames": frames,
                    "calibration": camera_result,
                }
            )
    finally:
        lens_camera.close()

    output_path, log_path = write_calibration_outputs(args, output)
    print("")
    if output.get("skipped_positions"):
        print(
            "full workflow skipped {} zoom position(s) without checkerboard focus score".format(
                len(output["skipped_positions"])
            )
        )
    print("core camera calibration saved: {}".format(output_path))
    print("calibration log saved: {}".format(log_path))


def run_calibration(args):
    enforce_online_capture(args)
    cv2, np = load_image_tools()
    output = build_output_header(args, "lens_checkerboard_camera_calibration")
    position_name = "online_capture"
    print("")
    print("=== Online camera calibration ===")
    object_points, image_points, image_size, frames = capture_from_camera(
        cv2, np, args, position_name
    )
    camera_result = calibrate_camera(cv2, object_points, image_points, image_size)
    output["positions"].append(
        {
            "name": position_name,
            "frames": frames,
            "calibration": camera_result,
        }
    )

    output_path, log_path = write_calibration_outputs(args, output)
    print("")
    print("core camera calibration saved: {}".format(output_path))
    print("calibration log saved: {}".format(log_path))


def main():
    try:
        args = load_args()
        args.selected_camera = None

        if args.board_cols < 2 or args.board_rows < 2:
            raise ValueError("--board-cols and --board-rows must be at least 2")
        if args.square_size <= 0:
            raise ValueError("--square-size must be greater than 0")
        if args.frames < 1:
            raise ValueError("--frames must be at least 1")
        if args.min_frames < 1:
            raise ValueError("--min-frames must be at least 1")
        if args.min_frames > args.frames:
            raise ValueError("--min-frames must not be greater than --frames")
        selected_camera = select_named_camera(args)
        print(
            "selected camera: [{}] {} ({})".format(
                selected_camera["index"],
                selected_camera["name"],
                selected_camera.get("source", "unknown"),
            )
        )

        args.run_dir = create_run_directory(args.debug_dir)
        print("run directory: {}".format(args.run_dir))

        run_calibration(args)
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        CalibrationError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
