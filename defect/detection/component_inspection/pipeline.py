"""End-to-end SAM3, classifier, RoMaV2 and missing-component pipeline."""

from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .classifier import ComponentClassifier
from .config import (
    InspectionConfig,
    WORKING_DATA,
    collect_inspection_images,
    resolve_classifier_checkpoint,
    resolve_roma_checkpoint,
    resolve_sam3_checkpoint,
)
from .geometry import (
    compare_components,
    map_detection,
    suppress_duplicate_mapped_detections,
)
from .ground_truth import GroundTruth, load_ground_truth
from .registration import RegistrationError, RomaRegistrar
from .segmentation import SamPromptSegmenter


ProgressCallback = Callable[[str], None]
STATUS_COLORS = {
    "present": (55, 205, 70),
    "missing": (45, 45, 235),
    "uninspected": (145, 145, 145),
}


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
    else:
        print(message, flush=True)


def default_output_directory() -> Path:
    return (
        WORKING_DATA
        / "component_inspection"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_stem(path: Path, used: set[str]) -> str:
    base = "".join(char if char.isalnum() or char in "-_" else "_" for char in path.stem)
    base = base.strip("_") or "image"
    value = base
    suffix = 2
    while value in used:
        value = f"{base}_{suffix}"
        suffix += 1
    used.add(value)
    return value


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _write_status_csv(comparison: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "id",
                "designator",
                "category",
                "center_x",
                "center_y",
                "detection_id",
                "source_image_id",
                "positive_prob",
                "polygon_iou",
                "center_distance_px",
            ],
        )
        writer.writeheader()
        for status in ("present", "missing", "uninspected"):
            for item in comparison[status]:
                match = item.get("match") or {}
                writer.writerow(
                    {
                        "status": status,
                        "id": item["id"],
                        "designator": item["designator"],
                        "category": item["category"],
                        "center_x": f"{item['center_xy'][0]:.3f}",
                        "center_y": f"{item['center_xy'][1]:.3f}",
                        "detection_id": match.get("detection_id", ""),
                        "source_image_id": match.get("source_image_id", ""),
                        "positive_prob": match.get("positive_prob", ""),
                        "polygon_iou": match.get("polygon_iou", ""),
                        "center_distance_px": match.get("center_distance_px", ""),
                    }
                )


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def draw_result_overlay(
    ground_truth: GroundTruth,
    comparison: dict,
    footprints: list[np.ndarray],
    mapped_detections: list[dict],
    output_path: Path,
) -> None:
    canvas = _read_bgr(ground_truth.mask_overlay_image)
    for footprint in footprints:
        polygon = np.rint(footprint).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [polygon], True, (235, 180, 30), 3, cv2.LINE_AA)

    missing_layer = canvas.copy()
    for item in comparison["missing"]:
        polygon = np.rint(item["polygon_xy"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(missing_layer, [polygon], STATUS_COLORS["missing"])
    canvas = cv2.addWeighted(missing_layer, 0.30, canvas, 0.70, 0)

    for status in ("uninspected", "present", "missing"):
        thickness = 5 if status == "missing" else 3
        for item in comparison[status]:
            polygon = np.rint(item["polygon_xy"]).astype(np.int32).reshape(-1, 1, 2)
            color = STATUS_COLORS[status]
            cv2.polylines(canvas, [polygon], True, color, thickness, cv2.LINE_AA)
            center = tuple(map(int, np.rint(item["center_xy"])))
            cv2.putText(
                canvas,
                item["designator"],
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                2 if status == "missing" else 1,
                cv2.LINE_AA,
            )
    for detection in mapped_detections:
        center = tuple(map(int, np.rint(detection["target_center_xy"])))
        cv2.drawMarker(canvas, center, (255, 220, 50), cv2.MARKER_CROSS, 10, 2)

    status_text = {
        "complete": "COMPLETE",
        "incomplete": "INCOMPLETE",
        "coverage_incomplete": "COVERAGE INCOMPLETE",
    }[comparison["status"]]
    summary = (
        f"{status_text}  present={len(comparison['present'])}  "
        f"missing={len(comparison['missing'])}  "
        f"uninspected={len(comparison['uninspected'])}"
    )
    (text_width, text_height), baseline = cv2.getTextSize(
        summary, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(
        canvas,
        (12, 12),
        (28 + text_width, 28 + text_height + baseline),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        canvas,
        summary,
        (20, 22 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"cannot encode result overlay: {output_path}")
    encoded.tofile(str(output_path))


def run_component_inspection(
    ground_truth_metadata: str | Path,
    inspection_source: str | Path,
    output_directory: str | Path | None = None,
    config: InspectionConfig | None = None,
    classifier_checkpoint: str | Path | None = None,
    sam3_checkpoint: str | Path | None = None,
    roma_checkpoint: str | Path | None = None,
    device: str = "auto",
    progress: ProgressCallback | None = None,
) -> dict:
    config = config or InspectionConfig()
    config.validate()
    output_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else default_output_directory().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = load_ground_truth(ground_truth_metadata)
    inspection_images = collect_inspection_images(inspection_source)
    classifier_path = resolve_classifier_checkpoint(classifier_checkpoint)
    sam3_path = resolve_sam3_checkpoint(sam3_checkpoint)
    roma_path = resolve_roma_checkpoint(roma_checkpoint)
    _emit(progress, f"真值元器件: {len(ground_truth.components)} 个")
    _emit(progress, f"高倍检查图: {len(inspection_images)} 张")

    used_ids: set[str] = set()
    image_records = [
        {"id": _safe_stem(path, used_ids), "path": path} for path in inspection_images
    ]
    segmenter = None
    classifier = None
    try:
        segmenter = SamPromptSegmenter(sam3_path, config, device=device, progress=progress)
        classifier = ComponentClassifier(
            classifier_path, config, device=device, progress=progress
        )
        for number, image_record in enumerate(image_records, start=1):
            path = image_record["path"]
            _emit(progress, f"[{number}/{len(image_records)}] 检测 {path.name}")
            with Image.open(path) as pil_image:
                rgb_pil = pil_image.convert("RGB")
                candidates = segmenter.segment(rgb_pil, image_name=path.name)
                image_rgb = np.asarray(rgb_pil)
            detection_output = output_dir / "detections" / image_record["id"]
            result = classifier.classify(
                image_rgb,
                candidates,
                image_name=path.name,
                output_dir=detection_output,
            )
            image_record["detection"] = result
            _emit(
                progress,
                f"{path.name}: 分类通过 {result['accepted_count']}/{result['candidate_count']} 个",
            )
    finally:
        if classifier is not None:
            classifier.close()
        if segmenter is not None:
            segmenter.close()
        del classifier, segmenter
        _release_cuda()

    registrar = None
    footprints: list[np.ndarray] = []
    mapped_detections: list[dict] = []
    registration_failures = []
    try:
        registrar = RomaRegistrar(roma_path, config, progress=progress)
        for number, image_record in enumerate(image_records, start=1):
            path = image_record["path"]
            _emit(progress, f"[{number}/{len(image_records)}] 配准 {path.name}")
            visualization = output_dir / "registrations" / f"{image_record['id']}.png"
            try:
                registration = registrar.register(
                    ground_truth.registration_image, path, visualization
                )
            except RegistrationError as exc:
                failure = {"image_id": image_record["id"], "image": str(path), "error": str(exc)}
                registration_failures.append(failure)
                image_record["registration_error"] = str(exc)
                _emit(progress, f"配准失败，跳过 {path.name}: {exc}")
                continue
            image_record["registration"] = registration.serializable()
            footprints.append(registration.test_footprint_in_truth)
            for detection in image_record["detection"]["accepted_detections"]:
                mapped_detections.append(
                    map_detection(
                        detection,
                        registration.homography_test_to_truth,
                        image_record["id"],
                    )
                )
    finally:
        if registrar is not None:
            registrar.close()
        del registrar
        _release_cuda()
    if not footprints:
        details = "; ".join(item["error"] for item in registration_failures)
        raise RegistrationError(f"all inspection images failed registration: {details}")

    deduplicated = suppress_duplicate_mapped_detections(
        mapped_detections,
        config.mapped_detection_nms_iou,
        config.mapped_detection_nms_center_ratio,
    )
    comparison = compare_components(
        ground_truth.components,
        deduplicated,
        footprints,
        config.match_polygon_iou_threshold,
        config.match_center_ratio,
        config.coverage_margin_ratio,
    )
    overlay_path = output_dir / "component_completeness_overlay.png"
    status_csv = output_dir / "component_status.csv"
    report_path = output_dir / "inspection_report.json"
    draw_result_overlay(ground_truth, comparison, footprints, deduplicated, overlay_path)
    _write_status_csv(comparison, status_csv)

    serialized_images = []
    for image_record in image_records:
        serialized = {
            "id": image_record["id"],
            "path": str(image_record["path"]),
            "candidate_count": image_record["detection"]["candidate_count"],
            "accepted_count": image_record["detection"]["accepted_count"],
            "detection_outputs": image_record["detection"]["outputs"],
        }
        if "registration" in image_record:
            serialized["registration"] = image_record["registration"]
        else:
            serialized["registration_error"] = image_record["registration_error"]
        serialized_images.append(serialized)

    payload = {
        "schema_version": 1,
        "type": "pcb_component_completeness_inspection",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": comparison["status"],
        "complete": comparison["complete"],
        "ground_truth": {
            "component_masks_json": str(ground_truth.metadata_path),
            "registration_image": str(ground_truth.registration_image),
            "coordinate_mask_overlay": str(ground_truth.mask_overlay_image),
            "component_count": len(ground_truth.components),
            "image_size_px": list(ground_truth.image_size),
        },
        "inspection_source": str(Path(inspection_source).expanduser().resolve()),
        "inspection_image_count": len(image_records),
        "registered_image_count": len(footprints),
        "registration_failures": registration_failures,
        "counts": {
            "present": len(comparison["present"]),
            "missing": len(comparison["missing"]),
            "uninspected": len(comparison["uninspected"]),
            "mapped_detections_before_deduplication": len(mapped_detections),
            "mapped_detections": len(deduplicated),
        },
        "missing_ids": [item["id"] for item in comparison["missing"]],
        "present_components": comparison["present"],
        "missing_components": comparison["missing"],
        "uninspected_components": comparison["uninspected"],
        "mapped_detections": deduplicated,
        "images": serialized_images,
        "models": {
            "sam3_checkpoint": str(sam3_path),
            "classifier_checkpoint": str(classifier_path),
            "romav2_checkpoint": str(roma_path),
        },
        "settings": config.to_dict(),
        "outputs": {
            "report_json": str(report_path),
            "component_status_csv": str(status_csv),
            "component_completeness_overlay": str(overlay_path),
        },
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _emit(
        progress,
        f"检测完成: 存在 {len(comparison['present'])}，缺失 {len(comparison['missing'])}，"
        f"未覆盖 {len(comparison['uninspected'])}",
    )
    return payload
