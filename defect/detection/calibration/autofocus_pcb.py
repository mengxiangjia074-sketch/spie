# -*- coding: utf-8 -*-
"""
Find the sharpest focus position for a PCB at the zoom/iris from the settings file.

The script first moves LensConnect zoom and iris to the addresses configured in
the lens JSON file, then moves only focus during autofocus, scores live camera
frames, moves back to the best focus address, and saves the best frame plus a
JSON log.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

# The LensCamera package lives in the detection directory one level above
# these calibration scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import LensCamera as lens_camera
import calibration_summary
from LensCamera import paths
from LensCamera import camera as camera_tools
from LensCamera.sampling import build_sample_points as shared_build_sample_points
from LensCamera.settings import Settings as JsonnetSettings
from LensCamera.settings import SettingsError
from LensCamera.timestamps import local_timestamp, run_timestamp, utc_timestamp


DEFAULT_CAMERA_NAME = "CamSPC"
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SETTINGS_PATH = BASE_DIR / "configs" / "autofocus" / "autofocus.jsonnet"
SUMMARY_ENTRY_TYPE = "pcb_autofocus_lens_position"

class AutofocusError(Exception):
    pass


def load_settings() -> dict:
    try:
        return JsonnetSettings(
            DEFAULT_SETTINGS_PATH, str(DEFAULT_SETTINGS_PATH)
        ).values
    except SettingsError as exc:
        raise AutofocusError(str(exc)) from exc


def setting_value(settings, name):
    if name not in settings:
        raise AutofocusError(
            "{} missing required setting {!r}".format(DEFAULT_SETTINGS_PATH, name)
        )
    item = settings[name]
    if isinstance(item, dict) and "value" in item:
        return item["value"]
    return item


def load_args():
    settings = load_settings()
    args = SimpleNamespace(
        **{name: setting_value(settings, name) for name in settings}
    )
    args.device = lens_camera.parse_device_number(args.device)
    args.zoom = (
        lens_camera.parse_position("zoom", args.zoom)
        if args.zoom is not None
        else None
    )
    args.iris = (
        lens_camera.parse_position("iris", args.iris)
        if args.iris is not None
        else None
    )
    return args


def timestamp_folder_name():
    return run_timestamp()


def resolve_path(path_text):
    # Output paths are anchored to the project root, so results always land in
    # working_data no matter which directory the script is run from.
    return paths.resolve_project_path(path_text)


def load_image_tools():
    return camera_tools.load_image_tools(AutofocusError)


def open_camera(cv2, camera_index):
    return camera_tools.open_camera(cv2, camera_index)


def configure_camera(cv2, camera, args):
    return camera_tools.configure_camera(cv2, camera, args)


def camera_frame_size(cv2, camera):
    return camera_tools.frame_size(cv2, camera)


def resolve_camera_selection(args):
    selected = camera_tools.find_camera_by_name(
        args.camera_name, AutofocusError
    )
    print(
        "camera: selected [{index}] {name} by name match {match!r}".format(
            **selected
        )
    )
    return camera_tools.camera_source(selected), selected


def is_4k_frame(width, height):
    return width >= 3840 and height >= 2160


def ensure_min_arg(args, name, minimum, changes):
    value = getattr(args, name, None)
    if value is None or value < minimum:
        setattr(args, name, minimum)
        changes[name] = minimum


def ensure_max_arg(args, name, maximum, changes):
    value = getattr(args, name, None)
    if value is None:
        return
    if value > maximum:
        setattr(args, name, maximum)
        changes[name] = maximum


def apply_golden_highres_profile(cv2, camera, args):
    width, height = camera_frame_size(cv2, camera)
    record = {
        "enabled": bool(getattr(args, "golden_highres_auto", True)),
        "applied": False,
        "frame_width": width,
        "frame_height": height,
        "changes": {},
    }
    setattr(args, "golden_highres_record", record)

    if not record["enabled"]:
        return record
    if getattr(args, "optimizer", "golden-section") != "golden-section":
        return record
    if not is_4k_frame(width, height):
        return record

    changes = record["changes"]
    ensure_min_arg(args, "score_frames", 5, changes)
    ensure_min_arg(args, "discard_frames", 5, changes)
    ensure_min_arg(args, "startup_discard_frames", 10, changes)
    ensure_min_arg(args, "settle", 0.5, changes)
    ensure_min_arg(args, "blur_ksize", 5, changes)
    ensure_min_arg(args, "fine_steps", 15, changes)
    ensure_min_arg(args, "golden_bracket_steps", 7, changes)
    ensure_min_arg(args, "score_max_side", 1600, changes)
    ensure_min_arg(args, "golden_tie_relative_margin", 0.005, changes)
    if getattr(args, "roi", None) is None and not getattr(args, "interactive_roi", False):
        ensure_max_arg(args, "roi_scale", 0.5, changes)

    record["applied"] = bool(changes)
    if record["applied"]:
        print(
            "autofocus: applied 4K golden-section profile for {}x{}: {}".format(
                width, height, changes
            )
        )
    return record


def clamp(value, low, high):
    return max(low, min(high, value))


def parse_roi(roi_text):
    if not roi_text:
        return None
    parts = [item.strip() for item in roi_text.split(",")]
    if len(parts) != 4:
        raise AutofocusError("--roi must use x,y,width,height")
    try:
        x, y, width, height = [int(item) for item in parts]
    except ValueError as exc:
        raise AutofocusError("--roi values must be integers") from exc
    if width <= 0 or height <= 0:
        raise AutofocusError("--roi width and height must be positive")
    return x, y, width, height


def normalize_roi_rect(rect, image_width, image_height):
    x, y, roi_width, roi_height = [int(value) for value in rect]
    x0 = clamp(x, 0, image_width - 1)
    y0 = clamp(y, 0, image_height - 1)
    x1 = clamp(x + roi_width, x0 + 1, image_width)
    y1 = clamp(y + roi_height, y0 + 1, image_height)
    return {
        "x": x0,
        "y": y0,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def selected_interactive_roi_record(args, image_width, image_height):
    selected_rois = getattr(args, "selected_rois", None)
    if not selected_rois:
        return None

    regions = []
    for index, rect in enumerate(selected_rois, start=1):
        region = normalize_roi_rect(rect, image_width, image_height)
        region["index"] = index
        regions.append(region)

    return {
        "mode": "interactive_multi" if len(regions) > 1 else "interactive",
        "count": len(regions),
        "regions": regions,
    }


def select_roi_regions(gray, args):
    height, width = gray.shape[:2]
    explicit_roi = parse_roi(args.roi)
    if explicit_roi is not None:
        roi_record = normalize_roi_rect(explicit_roi, width, height)
        roi_record["mode"] = "explicit"
        return [roi_record], roi_record

    interactive_record = selected_interactive_roi_record(args, width, height)
    if interactive_record is not None:
        return interactive_record["regions"], interactive_record

    scale = args.roi_scale
    if scale <= 0.0 or scale > 1.0:
        raise AutofocusError("--roi-scale must be in (0, 1]")
    roi_width = max(1, int(round(width * scale)))
    roi_height = max(1, int(round(height * scale)))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    roi_record = {
        "mode": "center_scale",
        "scale": scale,
        "x": x0,
        "y": y0,
        "width": roi_width,
        "height": roi_height,
    }
    return [roi_record], roi_record


def resize_roi_for_scoring(cv2, roi, roi_record, args):
    max_side = getattr(args, "score_max_side", None)
    roi_height, roi_width = roi.shape[:2]
    roi_record["score_width"] = roi_width
    roi_record["score_height"] = roi_height
    roi_record["score_scale"] = 1.0

    if max_side is None or max_side <= 0:
        return roi, roi_record

    longest_side = max(roi_width, roi_height)
    if longest_side <= max_side:
        return roi, roi_record

    scale = max_side / float(longest_side)
    score_width = max(1, int(round(roi_width * scale)))
    score_height = max(1, int(round(roi_height * scale)))
    resized = cv2.resize(roi, (score_width, score_height), interpolation=cv2.INTER_AREA)
    roi_record["score_width"] = score_width
    roi_record["score_height"] = score_height
    roi_record["score_scale"] = scale
    return resized, roi_record


def preprocess_roi_regions(cv2, frame, args):
    if len(frame.shape) == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    regions, roi_record = select_roi_regions(gray, args)
    processed = []
    scored_regions = []
    for region in regions:
        x0 = region["x"]
        y0 = region["y"]
        x1 = x0 + region["width"]
        y1 = y0 + region["height"]
        roi = gray[y0:y1, x0:x1]
        region_record = dict(region)
        roi, region_record = resize_roi_for_scoring(cv2, roi, region_record, args)
        if args.blur_ksize > 1:
            kernel = args.blur_ksize
            if kernel % 2 == 0:
                kernel += 1
            roi = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        processed.append((roi, region_record))
        scored_regions.append(region_record)

    if len(scored_regions) == 1 and roi_record.get("mode") != "interactive":
        roi_record = dict(scored_regions[0])
        roi_record["mode"] = regions[0].get("mode", roi_record.get("mode"))
        if roi_record["mode"] == "center_scale":
            roi_record["scale"] = args.roi_scale
    else:
        roi_record = dict(roi_record)
        roi_record["regions"] = scored_regions

    return processed, roi_record


# 清晰度评价算法说明:
# Tenengrad, 即 Sobel 梯度能量, 是工业视觉自动对焦最常用的方法之一。
# 优点: 对 PCB 走线、焊盘、丝印等边缘响应强, 比 Laplacian 方差更抗随机噪声。
# 缺点: 受照明方向、反光和高对比度脏点影响, 画面纹理过少时峰值会不明显。
def score_tenengrad(cv2, np, roi):
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


# 清晰度评价算法说明:
# Laplacian 方差是最常见、实现最简单的清晰度指标, 直接衡量二阶高频变化。
# 优点: 计算快, 对虚焦非常敏感, 常作为默认 baseline。
# 缺点: 对传感器噪声、反光高亮点、压缩伪影也很敏感, PCB 金属反光强时可能误判。
def score_laplacian_var(cv2, np, roi):
    del np
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


# 清晰度评价算法说明:
# Brenner 梯度使用相隔两个像素的灰度差平方, 是传统快速自动对焦算法。
# 优点: 非常快, 对清晰边缘有较好响应, 适合算力有限的场景。
# 缺点: 有方向偏置, 对 PCB 上细小多方向结构不如 Tenengrad 稳定。
def score_brenner(cv2, np, roi):
    del cv2
    if roi.shape[0] < 3 and roi.shape[1] < 3:
        return 0.0
    data = roi.astype("float64")
    score = 0.0
    if data.shape[1] >= 3:
        dx = data[:, 2:] - data[:, :-2]
        score += float(np.mean(dx * dx))
    if data.shape[0] >= 3:
        dy = data[2:, :] - data[:-2, :]
        score += float(np.mean(dy * dy))
    return score


# 清晰度评价算法说明:
# 高频 FFT 能量统计频域高频分量, 理论上能直接反映失焦造成的高频衰减。
# 优点: 对整体频谱变化敏感, 可用于纹理丰富或周期结构明显的 PCB。
# 缺点: 计算量较大, 对周期纹理、噪声和 ROI 边界敏感, 通常不作为实时首选。
def score_fft_highfreq(cv2, np, roi):
    del cv2
    data = roi.astype("float64")
    data -= data.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    magnitude = np.abs(spectrum)
    height, width = data.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    cy = height / 2.0
    cx = width / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = 0.18 * min(height, width)
    mask = radius >= cutoff
    if not mask.any():
        return 0.0
    return float(np.mean(magnitude[mask]))


SHARPNESS_METRICS = {
    "tenengrad": score_tenengrad,
    "laplacian": score_laplacian_var,
    "brenner": score_brenner,
    "fft": score_fft_highfreq,
}


def build_sample_points(min_addr, max_addr, steps):
    return shared_build_sample_points(
        min_addr, max_addr, steps, AutofocusError, range_name="focus"
    )


def read_current(name):
    return lens_camera.read_current(name)


def focus_range_from_lens(args):
    lens_range = lens_camera.get_motor_range("focus")
    lens_min = lens_range["min"]
    lens_max = lens_range["max"]
    range_min = lens_min if args.focus_min is None else args.focus_min
    range_max = lens_max if args.focus_max is None else args.focus_max
    if range_min < lens_min or range_max > lens_max:
        raise AutofocusError(
            "requested focus range {}..{} is outside lens range {}..{}".format(
                range_min, range_max, lens_min, lens_max
            )
        )
    if range_min > range_max:
        raise AutofocusError("--focus-min must be <= --focus-max")
    return {"min": range_min, "max": range_max, "lens_min": lens_min, "lens_max": lens_max}


def ensure_lens_motor_ready(name, capabilities, init_if_needed):
    try:
        lens_camera.require_motor_supported(name, capabilities)
        lens_camera.ensure_ready(name, init_if_needed=init_if_needed)
    except lens_camera.LensControlError as exc:
        raise AutofocusError(str(exc)) from exc


def move_lens_motor(name, target, capabilities, settle_seconds):
    try:
        return lens_camera.move_connected_motor(
            name,
            target,
            capabilities,
            settle_seconds=settle_seconds,
            clamp_target=True,
        )
    except lens_camera.LensControlError as exc:
        raise AutofocusError(str(exc)) from exc


def move_focus(target, capabilities, settle_seconds):
    return move_lens_motor("focus", target, capabilities, settle_seconds)


def move_zoom(target, capabilities, settle_seconds):
    return move_lens_motor("zoom", target, capabilities, settle_seconds)


def move_iris(target, capabilities, settle_seconds):
    return move_lens_motor("iris", target, capabilities, settle_seconds)


def discard_frames(camera, count):
    for _index in range(max(0, count)):
        camera.read()


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def score_statistics(np, scores, score_value):
    if not scores:
        return {
            "score_min": None,
            "score_max": None,
            "score_std": None,
            "score_cv": None,
        }

    score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
    score_cv = None
    if score_value and score_value != 0:
        score_cv = abs(score_std / float(score_value))
    return {
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
        "score_std": score_std,
        "score_cv": score_cv,
    }


def save_image(cv2, path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise AutofocusError("failed to save image: {}".format(path))
    return str(path)


def score_roi_regions(cv2, np, metric_func, processed_regions):
    region_scores = []
    for roi, region_record in processed_regions:
        score = metric_func(cv2, np, roi)
        scored_region = dict(region_record)
        scored_region["score"] = score
        region_scores.append(scored_region)

    if not region_scores:
        raise AutofocusError("no ROI regions available for scoring")
    score = sum(item["score"] for item in region_scores) / float(len(region_scores))
    return float(score), region_scores


def measure_focus(cv2, np, camera, args, focus_actual, save_dir=None):
    metric_func = SHARPNESS_METRICS[args.metric]
    scores = []
    frames = []
    saved_frames = []
    roi_record = None
    roi_score_frames = []

    discard_frames(camera, args.discard_frames)
    for frame_index in range(args.score_frames):
        ok, frame = camera.read()
        if not ok:
            raise AutofocusError("camera frame read failed")

        processed_regions, roi_record = preprocess_roi_regions(cv2, frame, args)
        score, region_scores = score_roi_regions(
            cv2, np, metric_func, processed_regions
        )
        scores.append(score)
        frames.append({"index": frame_index + 1, "score": score, "frame": frame})
        roi_score_frames.append(
            {
                "index": frame_index + 1,
                "score": score,
                "regions": region_scores,
            }
        )
        if save_dir is not None:
            image_name = "{:03d}_score_{:.3f}.png".format(frame_index + 1, score)
            saved_file = save_image(cv2, save_dir / image_name, frame)
            saved_frames.append(
                {"index": frame_index + 1, "score": score, "file": saved_file}
            )

    best_frame = max(frames, key=lambda item: item["score"])
    score_value = median(scores) if args.aggregate == "median" else float(sum(scores) / len(scores))
    stats = score_statistics(np, scores, score_value)
    return {
        "focus": focus_actual,
        "score": score_value,
        "scores": scores,
        "aggregate": args.aggregate,
        "roi": roi_record,
        "roi_score_frames": roi_score_frames,
        "best_frame": best_frame["frame"],
        "best_frame_score": best_frame["score"],
        "saved_frames": saved_frames,
        **stats,
    }


def is_better(candidate, current_best):
    if candidate is None:
        return False
    if current_best is None:
        return True
    return candidate["score"] > current_best["score"]


def update_consecutive_drop(previous_record, current_record, current_drop_count):
    if previous_record is None:
        return 0
    if current_record["score"] < previous_record["score"]:
        return current_drop_count + 1
    return 0


def build_early_stop_config(args):
    return {
        "enabled": bool(args.early_stop),
        "mode": "consecutive_score_drop_after_peak",
        "drop_count": args.early_stop_drop_count,
        "min_points": args.early_stop_min_points,
        "min_relative_drop": args.early_stop_min_relative_drop,
    }


def maybe_build_early_stop_reason(
    args,
    stage,
    records,
    planned_count,
    best,
    latest,
    consecutive_drop_count,
):
    if not args.early_stop:
        return None
    if best is None or latest is None:
        return None
    if len(records) < args.early_stop_min_points:
        return None
    if latest is best:
        return None
    if consecutive_drop_count < args.early_stop_drop_count:
        return None

    if best["score"] > 0:
        relative_drop = (best["score"] - latest["score"]) / best["score"]
    else:
        relative_drop = 0.0
    if relative_drop < args.early_stop_min_relative_drop:
        return None

    best_index = None
    for index, record in enumerate(records, start=1):
        if record is best:
            best_index = index
            break

    return {
        "stage": stage,
        "mode": "consecutive_score_drop_after_peak",
        "scanned_positions": len(records),
        "planned_positions": planned_count,
        "best_scan_index": best_index,
        "best_focus": best["actual"],
        "best_score": best["score"],
        "trigger_focus": latest["actual"],
        "trigger_score": latest["score"],
        "score_drop_count": consecutive_drop_count,
        "drop_count_threshold": args.early_stop_drop_count,
        "relative_drop": relative_drop,
        "relative_drop_threshold": args.early_stop_min_relative_drop,
        "min_points": args.early_stop_min_points,
    }


def print_early_stop_reason(reason):
    print(
        (
            "autofocus: {stage} early stop after {scanned}/{planned} positions; "
            "score dropped {drops} times and is {relative_drop:.1%} below best"
        ).format(
            stage=reason["stage"],
            scanned=reason["scanned_positions"],
            planned=reason["planned_positions"],
            drops=reason["score_drop_count"],
            relative_drop=reason["relative_drop"],
        )
    )


class FocusEvaluator:
    def __init__(self, cv2, np, camera, capabilities, args):
        self.cv2 = cv2
        self.np = np
        self.camera = camera
        self.capabilities = capabilities
        self.args = args
        self.records = []
        self.cache = {}
        self.best = None

    def evaluate(self, target, stage):
        target = int(target)
        cache_key = (target, stage)
        if cache_key in self.cache:
            return self.cache[cache_key]

        move = move_focus(target, self.capabilities, self.args.settle)
        save_dir = None
        if getattr(self.args, "save_focus_scan_images", False):
            base_dir = Path(getattr(self.args, "focus_scan_capture_dir"))
            autofocus_label = getattr(self.args, "autofocus_label", "autofocus")
            save_dir = (
                base_dir
                / autofocus_label
                / stage
                / "focus_{:05d}".format(move["actual"])
            )
        measurement = measure_focus(
            self.cv2,
            self.np,
            self.camera,
            self.args,
            move["actual"],
            save_dir=save_dir,
        )
        record = {
            "stage": stage,
            "target": move["target"],
            "before": move["before"],
            "actual": move["actual"],
            "error": move["error"],
            "score": measurement["score"],
            "frame_scores": measurement["scores"],
            "best_frame_score": measurement["best_frame_score"],
            "score_min": measurement["score_min"],
            "score_max": measurement["score_max"],
            "score_std": measurement["score_std"],
            "score_cv": measurement["score_cv"],
            "aggregate": measurement["aggregate"],
            "roi": measurement["roi"],
            "roi_score_frames": measurement["roi_score_frames"],
            "saved_frames": measurement["saved_frames"],
        }
        self.records.append(record)
        self.cache[cache_key] = record
        if is_better(record, self.best):
            self.best = record
            self.best_frame = measurement["best_frame"]

        print(
            "autofocus: {} focus {} score {:.3f}".format(
                stage, record["actual"], record["score"]
            )
        )
        return record


# 优化算法说明:
# 粗扫加局部细扫是机器视觉自动对焦中最稳妥的常用方案。
# 优点: 先覆盖完整 focus 范围, 不容易被局部峰值误导, 对 PCB 反光和评分噪声更稳。
# 缺点: 需要拍摄的点位较多, 速度比金分搜索慢, 最终精度取决于细扫步数和半径。
def run_coarse_to_fine_search(evaluator, focus_range, args):
    coarse_points = build_sample_points(
        focus_range["min"], focus_range["max"], args.coarse_steps
    )
    coarse_best = None
    coarse_records = []
    coarse_previous = None
    coarse_drop_count = 0
    coarse_stop_reason = None
    print("autofocus: coarse scan {} positions".format(len(coarse_points)))
    for point in coarse_points:
        record = evaluator.evaluate(point, "coarse")
        coarse_records.append(record)
        coarse_drop_count = update_consecutive_drop(
            coarse_previous, record, coarse_drop_count
        )
        if is_better(record, coarse_best):
            coarse_best = record
        coarse_stop_reason = maybe_build_early_stop_reason(
            args,
            "coarse",
            coarse_records,
            len(coarse_points),
            coarse_best,
            record,
            coarse_drop_count,
        )
        if coarse_stop_reason is not None:
            print_early_stop_reason(coarse_stop_reason)
            break
        coarse_previous = record

    if coarse_best is None:
        raise AutofocusError("coarse scan did not produce a valid focus score")

    if args.fine_steps <= 0:
        return {
            "optimizer": "coarse-to-fine",
            "early_stop": build_early_stop_config(args),
            "coarse_points": coarse_points,
            "fine_points": [],
            "fine_radius": 0,
            "stop_reasons": {"coarse": coarse_stop_reason, "fine": None},
        }

    if len(coarse_points) > 1:
        coarse_step = min(
            abs(coarse_points[index + 1] - coarse_points[index])
            for index in range(len(coarse_points) - 1)
            if coarse_points[index + 1] != coarse_points[index]
        )
    else:
        coarse_step = focus_range["max"] - focus_range["min"]

    fine_radius = coarse_step if args.fine_radius is None else args.fine_radius
    fine_min = max(focus_range["min"], coarse_best["actual"] - fine_radius)
    fine_max = min(focus_range["max"], coarse_best["actual"] + fine_radius)
    fine_points = build_sample_points(fine_min, fine_max, args.fine_steps)

    print(
        "autofocus: fine scan {} positions around focus {}".format(
            len(fine_points), coarse_best["actual"]
        )
    )
    fine_records = []
    fine_previous = None
    fine_best = None
    fine_drop_count = 0
    fine_stop_reason = None
    for point in fine_points:
        record = evaluator.evaluate(point, "fine")
        fine_records.append(record)
        fine_drop_count = update_consecutive_drop(fine_previous, record, fine_drop_count)
        if is_better(record, fine_best):
            fine_best = record
        fine_stop_reason = maybe_build_early_stop_reason(
            args,
            "fine",
            fine_records,
            len(fine_points),
            fine_best,
            record,
            fine_drop_count,
        )
        if fine_stop_reason is not None:
            print_early_stop_reason(fine_stop_reason)
            break
        fine_previous = record

    return {
        "optimizer": "coarse-to-fine",
        "early_stop": build_early_stop_config(args),
        "coarse_points": coarse_points,
        "fine_points": fine_points,
        "fine_radius": fine_radius,
        "stop_reasons": {"coarse": coarse_stop_reason, "fine": fine_stop_reason},
    }


# 优化算法说明:
# 金分搜索是常用的一维连续优化算法, 用较少评估次数逼近峰值。
# 优点: 对单峰清晰度曲线收敛快, 比全范围细扫节省时间。
# 缺点: 假设曲线近似单峰, 如果 PCB 局部反光、噪声或纹理导致多峰, 可能收敛到非全局最优。
def golden_scores_are_tied(left, right, args):
    margin = float(getattr(args, "golden_tie_relative_margin", 0.0) or 0.0)
    if margin <= 0.0:
        return False

    left_score = float(left["score"])
    right_score = float(right["score"])
    scale = max(abs(left_score), abs(right_score), 1.0)
    return abs(left_score - right_score) <= scale * margin


def golden_probe_bracket(evaluator, focus_range, args):
    lo = int(focus_range["min"])
    hi = int(focus_range["max"])
    steps = int(getattr(args, "golden_bracket_steps", 0) or 0)
    if steps < 3 or lo >= hi:
        return lo, hi, []

    points = build_sample_points(lo, hi, steps)
    records = [evaluator.evaluate(point, "golden_bracket") for point in points]
    best_index = max(range(len(records)), key=lambda index: records[index]["score"])
    left_index = max(0, best_index - 1)
    right_index = min(len(points) - 1, best_index + 1)
    if left_index == right_index:
        return lo, hi, records

    bracket_lo = int(points[left_index])
    bracket_hi = int(points[right_index])
    print(
        "autofocus: golden bracket {} positions, best focus {}, search {}..{}".format(
            len(points), records[best_index]["actual"], bracket_lo, bracket_hi
        )
    )
    return bracket_lo, bracket_hi, records


def run_golden_section_search(evaluator, focus_range, args):
    lo, hi, bracket_records = golden_probe_bracket(evaluator, focus_range, args)
    bracket_range = {"min": lo, "max": hi} if bracket_records else None
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    tolerance = max(1, args.golden_tolerance)
    iterations = 0

    c = int(round(hi - ratio * (hi - lo)))
    d = int(round(lo + ratio * (hi - lo)))
    if c == d:
        c = max(lo, min(hi, c - 1))
        d = max(lo, min(hi, d + 1))
    fc = evaluator.evaluate(c, "golden")
    fd = evaluator.evaluate(d, "golden")

    while hi - lo > tolerance and iterations < args.golden_max_iter:
        iterations += 1
        if golden_scores_are_tied(fc, fd, args):
            lo = min(c, d)
            hi = max(c, d)
            c = int(round(hi - ratio * (hi - lo)))
            d = int(round(lo + ratio * (hi - lo)))
            if c == d:
                break
            fc = evaluator.evaluate(c, "golden")
            fd = evaluator.evaluate(d, "golden")
        elif fc["score"] < fd["score"]:
            lo = c
            c = d
            fc = fd
            d = int(round(lo + ratio * (hi - lo)))
            fd = evaluator.evaluate(d, "golden")
        else:
            hi = d
            d = c
            fd = fc
            c = int(round(hi - ratio * (hi - lo)))
            fc = evaluator.evaluate(c, "golden")

        if c == d:
            break

    final_count = args.fine_steps if args.fine_steps > 0 else 1
    final_points = build_sample_points(lo, hi, min(final_count, max(1, hi - lo + 1)))
    print(
        "autofocus: golden final scan {} positions in {}..{}".format(
            len(final_points), lo, hi
        )
    )
    for point in final_points:
        evaluator.evaluate(point, "golden_final")

    return {
        "optimizer": "golden-section",
        "initial_range": {"min": focus_range["min"], "max": focus_range["max"]},
        "bracket_points": [record["actual"] for record in bracket_records],
        "bracket_range": bracket_range,
        "final_range": {"min": lo, "max": hi},
        "final_points": final_points,
        "iterations": iterations,
        "tolerance": tolerance,
        "tie_relative_margin": getattr(args, "golden_tie_relative_margin", 0.0),
        "highres_profile": getattr(args, "golden_highres_record", None),
    }


def read_lens_snapshot(capabilities):
    return lens_camera.read_lens_snapshot(capabilities)


def current_lens_position(snapshot):
    position = {}
    for name in lens_camera.MOTOR_ORDER:
        item = snapshot.get(name) or {}
        position[name] = item.get("current") if item.get("supported") else None
    return position


def capture_final_frame(camera, args):
    discard_frames(camera, args.discard_frames)
    ok, frame = camera.read()
    if not ok:
        raise AutofocusError("camera frame read failed after final focus move")
    return frame


def display_scale_for_frame(frame, max_side):
    height, width = frame.shape[:2]
    if max_side is None or max_side <= 0:
        return 1.0
    return min(1.0, float(max_side) / float(max(width, height)))


def scale_point_to_image(point, scale, image_width, image_height):
    x, y = point
    if scale <= 0:
        scale = 1.0
    return (
        clamp(int(round(x / scale)), 0, image_width - 1),
        clamp(int(round(y / scale)), 0, image_height - 1),
    )


def roi_from_display_rect(start, end, scale, image_width, image_height):
    sx, sy = start
    ex, ey = end
    if sx == ex or sy == ey:
        return None
    x0, y0 = scale_point_to_image((min(sx, ex), min(sy, ey)), scale, image_width, image_height)
    x1, y1 = scale_point_to_image((max(sx, ex), max(sy, ey)), scale, image_width, image_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def draw_display_roi(cv2, display, rect, scale, color, thickness=2, label=None):
    x, y, width, height = rect
    p0 = (int(round(x * scale)), int(round(y * scale)))
    p1 = (int(round((x + width) * scale)), int(round((y + height) * scale)))
    cv2.rectangle(display, p0, p1, color, thickness)
    if label is not None:
        cv2.putText(
            display,
            str(label),
            (p0[0] + 4, max(16, p0[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def build_interactive_roi_display(cv2, frame, scale, rois, drag_start, drag_current):
    if len(frame.shape) == 2:
        base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        base = frame.copy()

    if scale < 1.0:
        height, width = frame.shape[:2]
        display_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        display = cv2.resize(base, display_size, interpolation=cv2.INTER_AREA)
    else:
        display = base

    for index, rect in enumerate(rois, start=1):
        draw_display_roi(cv2, display, rect, scale, (0, 255, 0), label=index)

    if drag_start is not None and drag_current is not None:
        height, width = frame.shape[:2]
        rect = roi_from_display_rect(drag_start, drag_current, scale, width, height)
        if rect is not None:
            draw_display_roi(cv2, display, rect, scale, (0, 255, 255), thickness=1)
    return display


def select_interactive_rois(cv2, frame, args):
    height, width = frame.shape[:2]
    scale = display_scale_for_frame(frame, args.interactive_roi_max_side)
    window_name = "autofocus ROI selection"
    state = {
        "drag_start": None,
        "drag_current": None,
        "rois": [],
    }

    def mouse_callback(event, x, y, flags, userdata):
        del flags, userdata
        display_height = max(1, int(round(height * scale)))
        display_width = max(1, int(round(width * scale)))
        point = (
            clamp(int(x), 0, display_width - 1),
            clamp(int(y), 0, display_height - 1),
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag_start"] = point
            state["drag_current"] = point
        elif event == cv2.EVENT_MOUSEMOVE and state["drag_start"] is not None:
            state["drag_current"] = point
        elif event == cv2.EVENT_LBUTTONUP and state["drag_start"] is not None:
            rect = roi_from_display_rect(
                state["drag_start"], point, scale, width, height
            )
            if rect is not None:
                state["rois"].append(rect)
            state["drag_start"] = None
            state["drag_current"] = None

    print(
        "autofocus: drag one or more ROI boxes; Enter starts autofocus, "
        "Backspace removes last, C clears, Esc cancels"
    )
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            display = build_interactive_roi_display(
                cv2,
                frame,
                scale,
                state["rois"],
                state["drag_start"],
                state["drag_current"],
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20)
            if key < 0:
                continue
            key &= 0xFF
            if key in (13, 10):
                if state["rois"]:
                    break
                print("autofocus: select at least one ROI before pressing Enter")
            elif key in (8, 127):
                if state["rois"]:
                    state["rois"].pop()
            elif key in (ord("c"), ord("C")):
                state["rois"] = []
            elif key in (27, ord("q"), ord("Q")):
                raise AutofocusError("interactive ROI selection cancelled")
    except AutofocusError:
        raise
    except Exception as exc:
        raise AutofocusError(
            "interactive ROI selection requires an OpenCV display window: {}".format(
                exc
            )
        ) from exc
    finally:
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    return state["rois"]


def prepare_interactive_rois(cv2, camera, args):
    if not getattr(args, "interactive_roi", False):
        return None
    ok, frame = camera.read()
    if not ok:
        raise AutofocusError("camera frame read failed before ROI selection")
    rois = select_interactive_rois(cv2, frame, args)
    setattr(args, "selected_rois", rois)
    print(
        "autofocus: selected {} ROI region(s): {}".format(
            len(rois), "; ".join("{},{},{},{}".format(*roi) for roi in rois)
        )
    )
    return {
        "mode": "interactive_multi" if len(rois) > 1 else "interactive",
        "count": len(rois),
        "regions": [
            {"index": index, "x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
            for index, roi in enumerate(rois, start=1)
        ],
    }


def create_run_output_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    name = timestamp_folder_name()
    run_dir = base_dir / name
    if not run_dir.exists():
        run_dir.mkdir()
        return run_dir

    # If two runs start in the same second, keep both results instead of overwriting.
    for index in range(1, 100):
        candidate = base_dir / "{}_{:02d}".format(name, index)
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise AutofocusError("cannot create a unique output directory under {}".format(base_dir))


def build_core_result(result, image_path, run_dir):
    optimizer = result["optimizer"]
    lens_position = current_lens_position(result["lens_after"])
    core = {
        "schema_version": 1,
        "type": "pcb_autofocus_core",
        "created_at_local": result["created_at_local"],
        "created_at_utc": result["created_at_utc"],
        "run_dir": str(run_dir),
        "device": result["device"],
        "lens_targets_source": result["lens_targets_source"],
        "lens_position": lens_position,
        "method": {
            "sharpness_metric": result["metric"],
            "optimizer": optimizer["optimizer"],
            "aggregate": result["aggregate"],
            "score_frames": result["score_frames"],
            "score_max_side": result.get("score_max_side"),
            "roi": result["best"].get("roi"),
            "early_stop": optimizer.get("early_stop"),
            "golden_highres_profile": optimizer.get("highres_profile"),
        },
        "best": {
            "focus": result["best"]["actual"],
            "score": result["best"]["score"],
            "stage": result["best"]["stage"],
            "image": str(image_path),
            "original_image": str(image_path),
        },
    }
    return core


def append_autofocus_summary(result):
    position = current_lens_position(result["lens_after"])
    required = ("zoom", "focus", "iris")
    missing = [name for name in required if position.get(name) is None]
    if missing:
        raise AutofocusError(
            "cannot save autofocus lens position: missing current {} value(s)".format(
                ", ".join(missing)
            )
        )

    lens_position = {name: int(position[name]) for name in required}
    entry = {
        "type": SUMMARY_ENTRY_TYPE,
        "created_at_utc": result["created_at_utc"],
        "device": result["device"],
        "lens_zoom": lens_position["zoom"],
        "lens_focus": lens_position["focus"],
        "lens_iris": lens_position["iris"],
        "lens_position": lens_position,
    }
    return calibration_summary.append_entry(
        entry,
        zoom=lens_position["zoom"],
        include_saved_at_utc=False,
    )


def write_result(args, result, focused_frame, cv2):
    output_dir = create_run_output_dir(resolve_path(args.output_dir))

    best_focus = result["best"]["actual"]
    image_path = output_dir / "pcb_best_focus_{:05d}.png".format(best_focus)
    json_path = output_dir / "pcb_autofocus_core.json"
    log_path = output_dir / "pcb_autofocus.log"
    save_image(cv2, image_path, focused_frame)

    result["best_image"] = str(image_path)
    result["best_original_image"] = str(image_path)
    result["core_json_file"] = str(json_path)
    result["log_file"] = str(log_path)
    result["run_dir"] = str(output_dir)
    core_result = build_core_result(result, image_path, output_dir)

    with open(str(json_path), "w", encoding="utf-8") as file_obj:
        json.dump(core_result, file_obj, indent=2, sort_keys=False)
        file_obj.write("\n")

    with open(str(log_path), "w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2, sort_keys=False)
        file_obj.write("\n")

    return image_path, json_path, log_path, output_dir


def run(args):
    cv2, np = load_image_tools()

    zoom_target = None if args.skip_json_zoom else args.zoom
    iris_target = None if args.skip_json_iris else args.iris
    if zoom_target is None and not args.skip_json_zoom:
        raise AutofocusError(
            "{} must contain a zoom value, or set skip_json_zoom to true".format(
                DEFAULT_SETTINGS_PATH
            )
        )
    if iris_target is None and not args.skip_json_iris:
        raise AutofocusError(
            "{} must contain an iris value, or set skip_json_iris to true".format(
                DEFAULT_SETTINGS_PATH
            )
        )

    device_number = args.device

    capabilities = lens_camera.connect(device_number)
    device_number = lens_camera.get_last_connected_device_number(device_number)
    camera = None
    try:
        zoom_move = None
        iris_move = None
        if zoom_target is not None:
            ensure_lens_motor_ready("zoom", capabilities, init_if_needed=not args.no_init)
            zoom_move = move_zoom(zoom_target, capabilities, args.settle)
            print(
                "autofocus: zoom target {} -> actual {}".format(
                    zoom_move["target"], zoom_move["actual"]
                )
            )

        if iris_target is not None:
            ensure_lens_motor_ready("iris", capabilities, init_if_needed=not args.no_init)
            iris_move = move_iris(iris_target, capabilities, args.settle)
            print(
                "autofocus: iris target {} -> actual {}".format(
                    iris_move["target"], iris_move["actual"]
                )
            )

        ensure_lens_motor_ready("focus", capabilities, init_if_needed=not args.no_init)
        focus_range = focus_range_from_lens(args)
        lens_snapshot_before = read_lens_snapshot(capabilities)

        camera_index, camera_selection = resolve_camera_selection(args)
        camera = open_camera(cv2, camera_index)
        if not camera.isOpened():
            raise AutofocusError("cannot open camera {}".format(camera_index))
        configure_camera(cv2, camera, args)
        apply_golden_highres_profile(cv2, camera, args)
        discard_frames(camera, args.startup_discard_frames)
        skip_autofocus = bool(getattr(args, "skip_autofocus", False))
        roi_selection = None
        if skip_autofocus:
            # 跳过 focus 清晰度搜索与移动: 在当前实际位置直接拍照。
            optimizer_record = {
                "optimizer": "none",
                "note": (
                    "skip_autofocus is true: no focus search or move was run; "
                    "the photo was captured directly at the current focus position."
                ),
            }
            scan_records = []
            lens_snapshot_after = read_lens_snapshot(capabilities)
            focused_frame = capture_final_frame(camera, args)
            current_focus = current_lens_position(lens_snapshot_after).get("focus")
            if current_focus is None:
                raise AutofocusError(
                    "skip_autofocus is true: cannot read the current focus position"
                )
            best_record = {
                "target": current_focus,
                "actual": current_focus,
                "score": None,
                "stage": "current",
                "move_error": 0,
                "roi": None,
            }
        else:
            roi_selection = prepare_interactive_rois(cv2, camera, args)
            evaluator = FocusEvaluator(cv2, np, camera, capabilities, args)
            if args.optimizer == "coarse-to-fine":
                optimizer_record = run_coarse_to_fine_search(evaluator, focus_range, args)
            else:
                optimizer_record = run_golden_section_search(evaluator, focus_range, args)

            if evaluator.best is None:
                raise AutofocusError("autofocus did not produce a valid score")

            focus_move = move_focus(evaluator.best["actual"], capabilities, args.settle)
            lens_snapshot_after = read_lens_snapshot(capabilities)
            focused_frame = capture_final_frame(camera, args)
            scan_records = evaluator.records
            best_record = {
                "target": evaluator.best["target"],
                "actual": focus_move["actual"],
                "score": evaluator.best["score"],
                "stage": evaluator.best["stage"],
                "move_error": focus_move["error"],
                "roi": evaluator.best.get("roi"),
            }

        result = {
            "schema_version": 1,
            "type": "pcb_target_lens_autofocus",
            "created_at_local": local_timestamp(),
            "created_at_utc": utc_timestamp(),
            "device": device_number,
            "lens_targets_source": str(DEFAULT_SETTINGS_PATH),
            "zoom_target_from_settings": zoom_target,
            "iris_target_from_settings": iris_target,
            "zoom_move": zoom_move,
            "iris_move": iris_move,
            "camera": {
                "index": camera_index,
                "name": camera_selection.get("name"),
                "source": camera_selection.get("source"),
                "name_match": camera_selection.get("match"),
                "frame_width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "camera_autofocus_disabled": bool(args.disable_camera_autofocus),
            },
            "metric": args.metric,
            "metric_note": (
                "Default tenengrad is recommended for PCB edges, pads, traces, "
                "and silkscreen; use laplacian only after checking noise/reflection."
            ),
            "optimizer": optimizer_record,
            "autofocus_skipped": skip_autofocus,
            "focus_range": focus_range,
            "score_frames": args.score_frames,
            "score_max_side": args.score_max_side,
            "aggregate": args.aggregate,
            "roi_selection": roi_selection,
            "settle_seconds": args.settle,
            "golden_highres_profile": getattr(args, "golden_highres_record", None),
            "lens_before": lens_snapshot_before,
            "lens_after": lens_snapshot_after,
            "scan": scan_records,
            "best": best_record,
        }
        image_path, json_path, log_path, output_dir = write_result(
            args, result, focused_frame, cv2
        )
        summary_path = append_autofocus_summary(result)

        print("")
        if skip_autofocus:
            print(
                "autofocus: skipped, captured directly at focus {}".format(
                    best_record["actual"]
                )
            )
        else:
            print(
                "autofocus: best focus {} score {:.3f}".format(
                    focus_move["actual"], evaluator.best["score"]
                )
            )
        print("run directory: {}".format(output_dir))
        print("focused image saved: {}".format(image_path))
        print("core json saved: {}".format(json_path))
        print("detail log saved: {}".format(log_path))
        print("calibration summary saved: {}".format(summary_path))
        return 0
    finally:
        if camera is not None:
            camera.release()
        lens_camera.close()


def validate_args(args):
    if args.coarse_steps < 1:
        raise AutofocusError("--coarse-steps must be at least 1")
    if args.fine_steps < 0:
        raise AutofocusError("--fine-steps must not be negative")
    if args.score_max_side < 0:
        raise AutofocusError("--score-max-side must not be negative")
    if args.roi and args.interactive_roi:
        raise AutofocusError("--roi and --interactive-roi cannot be used together")
    if args.interactive_roi_max_side < 0:
        raise AutofocusError("--interactive-roi-max-side must not be negative")
    if args.fine_radius is not None and args.fine_radius < 0:
        raise AutofocusError("--fine-radius must not be negative")
    if args.early_stop_drop_count < 1:
        raise AutofocusError("--early-stop-drop-count must be at least 1")
    if args.early_stop_min_points < 1:
        raise AutofocusError("--early-stop-min-points must be at least 1")
    if args.early_stop_min_relative_drop < 0.0:
        raise AutofocusError("--early-stop-min-relative-drop must not be negative")
    if args.early_stop_min_relative_drop >= 1.0:
        raise AutofocusError("--early-stop-min-relative-drop must be less than 1")
    if args.golden_tolerance < 1:
        raise AutofocusError("--golden-tolerance must be at least 1")
    if args.golden_bracket_steps < 0:
        raise AutofocusError("--golden-bracket-steps must not be negative")
    if args.golden_bracket_steps in (1, 2):
        raise AutofocusError("--golden-bracket-steps must be 0 or at least 3")
    if args.golden_tie_relative_margin < 0.0:
        raise AutofocusError("--golden-tie-relative-margin must not be negative")
    if args.golden_tie_relative_margin >= 1.0:
        raise AutofocusError("--golden-tie-relative-margin must be less than 1")
    if args.golden_max_iter < 1:
        raise AutofocusError("--golden-max-iter must be at least 1")
    if args.score_frames < 1:
        raise AutofocusError("--score-frames must be at least 1")
    if args.discard_frames < 0:
        raise AutofocusError("--discard-frames must not be negative")
    if args.startup_discard_frames < 0:
        raise AutofocusError("--startup-discard-frames must not be negative")
    if args.settle < 0:
        raise AutofocusError("--settle must not be negative")
    if args.blur_ksize < 1:
        raise AutofocusError("--blur-ksize must be at least 1")
    if args.roi_scale <= 0.0 or args.roi_scale > 1.0:
        raise AutofocusError("--roi-scale must be in (0, 1]")


def main():
    try:
        args = load_args()
        if args.list_cameras:
            devices = camera_tools.enumerate_cameras(AutofocusError)
            print("Camera devices:")
            print(camera_tools.format_camera_devices(devices))
            return 0
        validate_args(args)
        return run(args)
    except (OSError, ValueError, lens_camera.LensControlError, AutofocusError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
