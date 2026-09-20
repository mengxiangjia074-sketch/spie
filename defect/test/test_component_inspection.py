import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from component_inspection.config import (  # noqa: E402
    InspectionConfig,
    InspectionConfigError,
    collect_inspection_images,
)
from component_inspection.geometry import (  # noqa: E402
    compare_components,
    map_detection,
    suppress_duplicate_mapped_detections,
)
from component_inspection.ground_truth import (  # noqa: E402
    GroundTruthComponent,
    GroundTruthError,
    load_ground_truth,
)
from component_inspection.segmentation import split_mask_components  # noqa: E402


def _write_png(path: Path, size=(120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _truth_component(name, polygon):
    points = np.asarray(polygon, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = points.mean(axis=0)
    return GroundTruthComponent(
        component_id=name,
        designator=name,
        category="resistor",
        polygon_xy=tuple(map(tuple, points.tolist())),
        center_xy=tuple(center.tolist()),
        bbox_xyxy=(minimum[0], minimum[1], maximum[0], maximum[1]),
        source={},
    )


def _mapped_detection(name, polygon, probability=0.9):
    points = np.asarray(polygon, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = points.mean(axis=0)
    return {
        "id": name,
        "source_image_id": "tile",
        "positive_prob": probability,
        "sam_score": 0.9,
        "target_polygon_xy": points.tolist(),
        "target_center_xy": center.tolist(),
        "target_bbox_xyxy": [minimum[0], minimum[1], maximum[0], maximum[1]],
    }


def test_ground_truth_loads_coordinate_mask_artifacts(tmp_path):
    run_dir = tmp_path / "capture"
    mask_dir = run_dir / "mask_alignment"
    stitched = run_dir / "pcb_stitched.png"
    overlay = mask_dir / "pcb_with_component_masks.png"
    _write_png(stitched)
    _write_png(overlay)
    metadata = mask_dir / "component_masks.json"
    metadata.write_text(
        json.dumps(
            {
                "type": "pcb_component_instance_masks",
                "stitched_image": str(stitched),
                "image_size_px": [120, 80],
                "components": [
                    {
                        "instance_id": 1,
                        "designator": "R1",
                        "category": "resistor",
                        "target_polygon_px": [[10, 10], [30, 10], [30, 20], [10, 20]],
                        "visible": True,
                    },
                    {
                        "instance_id": 2,
                        "designator": "R2",
                        "category": "resistor",
                        "target_polygon_px": [[-30, 10], [-20, 10], [-20, 20], [-30, 20]],
                        "visible": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    truth = load_ground_truth(metadata)

    assert truth.registration_image == stitched.resolve()
    assert truth.mask_overlay_image == overlay.resolve()
    assert truth.image_size == (120, 80)
    assert [component.component_id for component in truth.components] == ["R1"]
    assert truth.components[0].center_xy == (20.0, 15.0)


def test_ground_truth_rejects_old_box_annotation_json(tmp_path):
    metadata = tmp_path / "boxes.json"
    metadata.write_text('{"annotations": []}', encoding="utf-8")
    with pytest.raises(GroundTruthError, match="coordinate Mask alignment"):
        load_ground_truth(metadata)


def test_capture_run_resolves_only_undistorted_grid_images(tmp_path):
    images = tmp_path / "run" / "images"
    _write_png(images / "r00_c01_undistorted.png")
    _write_png(images / "r00_c00_undistorted.png")
    _write_png(images / "r00_c00_mask.png")

    result = collect_inspection_images(tmp_path / "run")

    assert [path.name for path in result] == [
        "r00_c00_undistorted.png",
        "r00_c01_undistorted.png",
    ]


def test_config_rejects_tile_overlap_equal_to_tile_size():
    with pytest.raises(InspectionConfigError, match="tile_overlap"):
        InspectionConfig(tile_size=256, tile_overlap=256).validate()


def test_disconnected_sam_mask_becomes_independent_candidates():
    mask = np.zeros((80, 100), dtype=bool)
    mask[5:25, 8:38] = True
    mask[45:60, 65:85] = True

    components = split_mask_components(mask, 20, 0.20, 1)

    assert [item["mask_box"] for item in components] == [
        (8, 5, 38, 25),
        (65, 45, 85, 60),
    ]


def test_one_detection_cannot_mark_two_truth_components_present():
    components = (
        _truth_component("R1", [[10, 10], [20, 10], [20, 20], [10, 20]]),
        _truth_component("R2", [[22, 10], [32, 10], [32, 20], [22, 20]]),
    )
    detection = _mapped_detection(
        "tile:d1", [[11, 10], [29, 10], [29, 20], [11, 20]]
    )
    footprint = np.asarray([[0, 0], [50, 0], [50, 40], [0, 40]], dtype=np.float32)

    result = compare_components(
        components,
        [detection],
        [footprint],
        polygon_iou_threshold=0.10,
        center_ratio=0.65,
        coverage_margin_ratio=0.10,
    )

    assert len(result["present"]) == 1
    assert len(result["missing"]) == 1
    assert result["status"] == "incomplete"


def test_uncovered_truth_prevents_complete_verdict_without_calling_it_missing():
    components = (
        _truth_component("R1", [[10, 10], [20, 10], [20, 20], [10, 20]]),
        _truth_component("R2", [[80, 10], [90, 10], [90, 20], [80, 20]]),
    )
    detection = _mapped_detection(
        "tile:d1", [[10, 10], [20, 10], [20, 20], [10, 20]]
    )
    footprint = np.asarray([[0, 0], [50, 0], [50, 40], [0, 40]], dtype=np.float32)

    result = compare_components(
        components,
        [detection],
        [footprint],
        polygon_iou_threshold=0.10,
        center_ratio=0.65,
        coverage_margin_ratio=0.0,
    )

    assert [item["id"] for item in result["present"]] == ["R1"]
    assert result["missing"] == []
    assert [item["id"] for item in result["uninspected"]] == ["R2"]
    assert result["status"] == "coverage_incomplete"
    assert result["complete"] is False


def test_overlapping_detections_from_neighbor_tiles_are_deduplicated():
    first = _mapped_detection("a:d1", [[10, 10], [20, 10], [20, 20], [10, 20]], 0.95)
    second = _mapped_detection("b:d4", [[11, 10], [21, 10], [21, 20], [11, 20]], 0.80)

    kept = suppress_duplicate_mapped_detections([second, first], 0.35, 0.20)

    assert [item["id"] for item in kept] == ["a:d1"]


def test_map_detection_applies_test_to_truth_homography():
    detection = {
        "id": "component_1",
        "positive_prob": 0.9,
        "sam_score": 0.8,
        "center_xy": [5.0, 5.0],
        "rotated_box_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
    }
    homography = np.asarray([[2, 0, 100], [0, 3, 200], [0, 0, 1]], dtype=np.float64)

    mapped = map_detection(detection, homography, "tile")

    assert mapped["target_center_xy"] == pytest.approx([110.0, 215.0])
    assert mapped["target_bbox_xyxy"] == pytest.approx([100.0, 200.0, 120.0, 230.0])
