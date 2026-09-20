"""Geometry helpers for registration, coverage and one-to-one comparison."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .ground_truth import GroundTruthComponent


def polygon_bbox(polygon: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    return float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])


def bbox_diagonal(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return float(np.hypot(x2 - x1, y2 - y1))


def transform_points(points_xy, homography) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)


def polygon_iou(polygon_a, polygon_b) -> float:
    a = cv2.convexHull(np.asarray(polygon_a, dtype=np.float32).reshape(-1, 1, 2))
    b = cv2.convexHull(np.asarray(polygon_b, dtype=np.float32).reshape(-1, 1, 2))
    area_a = float(abs(cv2.contourArea(a)))
    area_b = float(abs(cv2.contourArea(b)))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def signed_polygon_distance(point, polygon) -> float:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def component_is_covered(
    component: GroundTruthComponent,
    footprints: list[np.ndarray],
    margin_ratio: float,
) -> bool:
    margin = bbox_diagonal(component.bbox_xyxy) * float(margin_ratio)
    return any(
        signed_polygon_distance(component.center_xy, footprint) >= -margin
        for footprint in footprints
    )


def map_detection(detection: dict, homography, image_id: str) -> dict:
    polygon = np.asarray(detection["rotated_box_points"], dtype=np.float32)
    mapped_polygon = transform_points(polygon, homography)
    mapped_center = transform_points([detection["center_xy"]], homography)[0]
    return {
        "id": f"{image_id}:{detection['id']}",
        "source_image_id": image_id,
        "source_detection_id": detection["id"],
        "positive_prob": float(detection["positive_prob"]),
        "sam_score": float(detection["sam_score"]),
        "source_center_xy": list(map(float, detection["center_xy"])),
        "source_polygon_xy": polygon.tolist(),
        "target_center_xy": [float(mapped_center[0]), float(mapped_center[1])],
        "target_polygon_xy": mapped_polygon.tolist(),
        "target_bbox_xyxy": list(polygon_bbox(mapped_polygon)),
    }


def suppress_duplicate_mapped_detections(
    detections: list[dict],
    iou_threshold: float,
    center_ratio: float,
) -> list[dict]:
    kept: list[dict] = []
    ordered = sorted(
        detections,
        key=lambda item: (item["positive_prob"], item["sam_score"]),
        reverse=True,
    )
    for detection in ordered:
        polygon = np.asarray(detection["target_polygon_xy"], dtype=np.float32)
        center = np.asarray(detection["target_center_xy"], dtype=np.float32)
        diagonal = bbox_diagonal(detection["target_bbox_xyxy"])
        duplicate = False
        for existing in kept:
            existing_polygon = np.asarray(existing["target_polygon_xy"], dtype=np.float32)
            existing_center = np.asarray(existing["target_center_xy"], dtype=np.float32)
            existing_diagonal = bbox_diagonal(existing["target_bbox_xyxy"])
            close = float(np.linalg.norm(center - existing_center)) <= (
                min(diagonal, existing_diagonal) * center_ratio
            )
            if close or polygon_iou(polygon, existing_polygon) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def compare_components(
    components: tuple[GroundTruthComponent, ...],
    detections: list[dict],
    footprints: list[np.ndarray],
    polygon_iou_threshold: float,
    center_ratio: float,
    coverage_margin_ratio: float,
) -> dict:
    """Assign detections to truth components once, then classify each truth item."""
    covered_indices = {
        index
        for index, component in enumerate(components)
        if component_is_covered(component, footprints, coverage_margin_ratio)
    }
    possible_matches: list[tuple[float, int, int, float, float]] = []
    for component_index in covered_indices:
        component = components[component_index]
        truth_polygon = np.asarray(component.polygon_xy, dtype=np.float32)
        truth_center = np.asarray(component.center_xy, dtype=np.float32)
        center_threshold = max(1e-6, bbox_diagonal(component.bbox_xyxy) * center_ratio)
        for detection_index, detection in enumerate(detections):
            detected_polygon = np.asarray(detection["target_polygon_xy"], dtype=np.float32)
            detected_center = np.asarray(detection["target_center_xy"], dtype=np.float32)
            iou = polygon_iou(truth_polygon, detected_polygon)
            center_distance = float(np.linalg.norm(truth_center - detected_center))
            if iou < polygon_iou_threshold and center_distance > center_threshold:
                continue
            center_score = max(0.0, 1.0 - center_distance / center_threshold)
            geometry_score = max(iou, center_score)
            score = geometry_score * (0.5 + 0.5 * float(detection["positive_prob"]))
            possible_matches.append(
                (score, component_index, detection_index, iou, center_distance)
            )

    assigned_components: set[int] = set()
    assigned_detections: set[int] = set()
    assignments: dict[int, dict] = {}
    for score, component_index, detection_index, iou, center_distance in sorted(
        possible_matches, reverse=True
    ):
        if component_index in assigned_components or detection_index in assigned_detections:
            continue
        assigned_components.add(component_index)
        assigned_detections.add(detection_index)
        detection = detections[detection_index]
        assignments[component_index] = {
            "detection_id": detection["id"],
            "source_image_id": detection["source_image_id"],
            "score": float(score),
            "polygon_iou": float(iou),
            "center_distance_px": float(center_distance),
            "positive_prob": float(detection["positive_prob"]),
            "target_center_xy": detection["target_center_xy"],
            "target_polygon_xy": detection["target_polygon_xy"],
        }

    present = []
    missing = []
    uninspected = []
    for index, component in enumerate(components):
        record = {
            "id": component.component_id,
            "designator": component.designator,
            "category": component.category,
            "center_xy": list(component.center_xy),
            "polygon_xy": [list(point) for point in component.polygon_xy],
            "bbox_xyxy": list(component.bbox_xyxy),
        }
        if index not in covered_indices:
            record["match"] = None
            uninspected.append(record)
        elif index in assignments:
            record["match"] = assignments[index]
            present.append(record)
        else:
            record["match"] = None
            missing.append(record)

    status = "complete"
    if missing:
        status = "incomplete"
    elif uninspected:
        status = "coverage_incomplete"
    return {
        "status": status,
        "complete": status == "complete",
        "present": present,
        "missing": missing,
        "uninspected": uninspected,
        "unmatched_detections": [
            detection
            for index, detection in enumerate(detections)
            if index not in assigned_detections
        ],
    }
