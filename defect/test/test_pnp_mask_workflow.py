import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from pnp_mask_workflow import (  # noqa: E402
    PnpMaskError,
    build_pnp_preview,
    export_masks,
    fit_affine,
    fit_alignment,
    latest_small_global_capture,
    render_target_masks,
    stitch_capture,
    transform_points,
)


def _write_image(path: Path, value: tuple[int, int, int], size=(20, 20)) -> None:
    image = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_latest_small_global_capture_uses_newest_dated_run(tmp_path):
    older = tmp_path / "20260901_100000" / "small_zoom_image"
    newer = tmp_path / "20260901_110000" / "small_zoom_image"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older_image = older / "small_global_cropped.png"
    expected = newer / "small_global_cropped.png"
    older_image.write_bytes(b"older")
    expected.write_bytes(b"newer")
    older.parent.touch()

    run_dir, image_path = latest_small_global_capture(tmp_path)

    assert run_dir == newer.parent
    assert image_path == expected


def test_latest_small_global_capture_rejects_missing_image_in_newest_run(tmp_path):
    older = tmp_path / "20260901_100000" / "small_zoom_image"
    older.mkdir(parents=True)
    (older / "small_global_cropped.png").write_bytes(b"older")
    (tmp_path / "20260901_110000").mkdir()

    with pytest.raises(PnpMaskError, match="20260901_110000"):
        latest_small_global_capture(tmp_path)


def test_build_pnp_preview_uses_parser_bundled_with_defect(tmp_path):
    pnp_path = tmp_path / "board.txt"
    pnp_path.write_text(
        "Designator Comment Layer Footprint Center-X(mm) Center-Y(mm) Rotation Description\n"
        "R1 10k TopLayer 0603R 0.0 0.0 0 resistor\n"
        "C1 1uF TopLayer 0603C 10.0 5.0 90 capacitor\n",
        encoding="gb18030",
    )

    preview, metadata = build_pnp_preview(pnp_path, "top", tmp_path / "preview")
    document = json.loads(metadata.read_text(encoding="utf-8"))

    assert preview.is_file()
    assert document["component_count"] == 2
    assert [item["designator"] for item in document["components"]] == ["R1", "C1"]


def test_bottom_preview_is_mirrored_for_back_side_view(tmp_path):
    pnp_path = tmp_path / "board.txt"
    pnp_path.write_text(
        "Designator Comment Layer Footprint Center-X(mm) Center-Y(mm) Rotation Description\n"
        "R1 10k BottomLayer 0603R 0.0 0.0 0 resistor\n"
        "C1 1uF BottomLayer 0603C 10.0 5.0 90 capacitor\n",
        encoding="gb18030",
    )

    _preview, metadata = build_pnp_preview(
        pnp_path, "bottom", tmp_path / "back"
    )
    document = json.loads(metadata.read_text(encoding="utf-8"))
    centers = {
        item["designator"]: item["source_center_px"]
        for item in document["components"]
    }

    assert document["board_side"] == "back"
    assert document["mirrored_horizontally"] is True
    assert centers["R1"][0] > centers["C1"][0]


def test_fit_affine_recovers_rotation_scale_and_translation():
    source = [[0, 0], [20, 0], [0, 10], [20, 10]]
    expected = np.asarray([[1.5, -0.2, 31.0], [0.3, 2.0, -7.0]])
    source_array = np.asarray(source, dtype=np.float64)
    target = source_array @ expected[:, :2].T + expected[:, 2]

    actual = fit_affine(source, target.tolist())

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_fit_affine_rejects_collinear_points():
    with pytest.raises(PnpMaskError, match="collinear"):
        fit_affine([[0, 0], [1, 1], [2, 2]], [[3, 4], [5, 6], [7, 8]])


def test_local_similarity_handles_regions_without_shearing_components():
    source = np.asarray(
        [[0, 0], [20, 0], [0, 20], [200, 0], [220, 0], [200, 20]],
        dtype=np.float64,
    )
    target = source * 2.0 + np.asarray([100.0, 80.0])
    target[3:] += np.asarray([45.0, 25.0])
    global_model = fit_alignment(source.tolist(), target.tolist(), "global_similarity")
    local_model = fit_alignment(source.tolist(), target.tolist(), "local_similarity")
    right_center = np.asarray([207.0, 8.0])
    expected_center = right_center * 2.0 + np.asarray([145.0, 105.0])
    global_center = transform_points([right_center], global_model)[0]
    local_center = transform_points([right_center], local_model)[0]

    assert np.linalg.norm(local_center - expected_center) < np.linalg.norm(
        global_center - expected_center
    )

    rectangle = np.asarray([[202, 4], [212, 4], [212, 12], [202, 12]], dtype=float)
    transformed = transform_points(rectangle, local_model, anchor=right_center)
    first_edge = transformed[1] - transformed[0]
    second_edge = transformed[2] - transformed[1]
    assert abs(float(np.dot(first_edge, second_edge))) < 1e-6
    assert np.isclose(
        np.linalg.norm(first_edge) / 10.0,
        np.linalg.norm(second_edge) / 8.0,
        atol=1e-8,
    )


def test_export_masks_writes_overlay_binary_and_uint16_instances(tmp_path):
    image_path = tmp_path / "stitched.png"
    _write_image(image_path, (40, 80, 120), size=(100, 80))
    components_path = tmp_path / "components.json"
    components_path.write_text(
        json.dumps(
            {
                "components": [
                    {
                        "instance_id": 1,
                        "designator": "R1",
                        "category": "resistor",
                        "source_polygon_px": [[5, 5], [20, 5], [20, 15], [5, 15]],
                    },
                    {
                        "instance_id": 2,
                        "designator": "C1",
                        "category": "capacitor",
                        "source_polygon_px": [[30, 25], [42, 25], [42, 40], [30, 40]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = export_masks(
        image_path,
        components_path,
        np.asarray([[1.0, 0.0, 10.0], [0.0, 1.0, 8.0]]),
        tmp_path / "output",
    )

    overlay = cv2.imread(str(outputs["overlay"]), cv2.IMREAD_COLOR)
    binary = cv2.imread(str(outputs["binary_mask"]), cv2.IMREAD_UNCHANGED)
    instances = cv2.imread(str(outputs["instance_mask"]), cv2.IMREAD_UNCHANGED)
    assert overlay.shape == (80, 100, 3)
    assert set(np.unique(binary)) == {0, 255}
    assert instances.dtype == np.uint16
    assert set(np.unique(instances)) == {0, 1, 2}


def test_render_target_masks_preserves_manual_edits_and_renumbers_instances(tmp_path):
    image_path = tmp_path / "stitched.png"
    _write_image(image_path, (20, 30, 40), size=(100, 80))
    records = [
        {
            "instance_id": 8,
            "designator": "R8",
            "category": "resistor",
            "target_polygon_px": [[15, 10], [25, 10], [25, 20], [15, 20]],
            "manually_edited": True,
        },
        {
            "instance_id": 99,
            "designator": "MANUAL_001",
            "category": "other",
            "target_polygon_px": [[50, 30], [70, 30], [70, 45], [50, 45]],
            "manually_added": True,
        },
    ]

    outputs = render_target_masks(
        image_path,
        records,
        tmp_path / "edited",
        manually_edited=True,
    )
    instances = cv2.imread(str(outputs["instance_mask"]), cv2.IMREAD_UNCHANGED)
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))

    assert set(np.unique(instances)) == {0, 1, 2}
    assert instances[15, 20] == 1
    assert instances[35, 60] == 2
    assert metadata["manually_edited"] is True
    assert [item["source_instance_id"] for item in metadata["components"]] == [8, 99]


def test_stitch_capture_uses_measured_stage_positions(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = run_dir / "first.png"
    second = run_dir / "second.png"
    _write_image(first, (0, 0, 255))
    _write_image(second, (0, 255, 0))
    calibration = tmp_path / "calibrate.json"
    calibration.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "type": "stage_command_to_pixel_calibration",
                        "lens_zoom": 100,
                        "stage_command_to_pixel": {
                            "matrix_2x2": [[10.0, 0.0], [0.0, 10.0]]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pcb_two_zoom_capture.json").write_text(
        json.dumps({"large": {"lens_zoom": 100}}), encoding="utf-8"
    )
    (run_dir / "pcb_mosaic.json").write_text(
        json.dumps(
            {
                "stage_calibration": str(calibration),
                "motion": {"initial_position_mm": [0.0, 0.0]},
                "observations": [
                    {
                        "row": 0,
                        "col": 0,
                        "actual_position_mm": [0.0, 0.0],
                        "raw_image": str(first),
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "actual_position_mm": [1.0, 0.0],
                        "raw_image": str(second),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output = stitch_capture(run_dir)
    image = cv2.imread(str(output))
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert image.shape == (20, 30, 3)
    assert metadata["placements"][0]["canvas_xy"] == [10, 0]
    assert metadata["placements"][1]["canvas_xy"] == [0, 0]


def test_stitch_capture_refines_calibration_with_image_content(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rng = np.random.default_rng(42)
    panorama = rng.integers(0, 256, (180, 450, 3), dtype=np.uint8)
    first = run_dir / "first.png"
    second = run_dir / "second.png"
    assert cv2.imwrite(str(first), panorama[:, :300])
    assert cv2.imwrite(str(second), panorama[:, 150:450])
    calibration = tmp_path / "calibrate.json"
    calibration.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "type": "stage_command_to_pixel_calibration",
                        "lens_zoom": 100,
                        "stage_command_to_pixel": {
                            "matrix_2x2": [[100.0, 0.0], [0.0, 100.0]]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pcb_two_zoom_capture.json").write_text(
        json.dumps({"large": {"lens_zoom": 100}}), encoding="utf-8"
    )
    (run_dir / "pcb_mosaic.json").write_text(
        json.dumps(
            {
                "stage_calibration": str(calibration),
                "motion": {"initial_position_mm": [0.0, 0.0]},
                "observations": [
                    {
                        "row": 0,
                        "col": 0,
                        "actual_position_mm": [0.0, 0.0],
                        "raw_image": str(first),
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "actual_position_mm": [-1.4, 0.0],
                        "raw_image": str(second),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    output = stitch_capture(run_dir)
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert metadata["placements"][1]["canvas_xy"] == [150, 0]
    registration = metadata["placements"][1]["registration"]
    assert registration["status"] == "registered"
    assert registration["measured_delta_px"] == pytest.approx([150.0, 0.0], abs=0.5)


def test_stitch_capture_rejects_stale_buffer_frame(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (180, 300, 3), dtype=np.uint8)
    first = run_dir / "first.png"
    duplicate = run_dir / "duplicate.png"
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(duplicate), image)
    calibration = tmp_path / "calibrate.json"
    calibration.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "type": "stage_command_to_pixel_calibration",
                        "lens_zoom": 100,
                        "stage_command_to_pixel": {
                            "matrix_2x2": [[100.0, 0.0], [0.0, 100.0]]
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pcb_two_zoom_capture.json").write_text(
        json.dumps({"large": {"lens_zoom": 100}}), encoding="utf-8"
    )
    (run_dir / "pcb_mosaic.json").write_text(
        json.dumps(
            {
                "stage_calibration": str(calibration),
                "motion": {"initial_position_mm": [0.0, 0.0]},
                "observations": [
                    {
                        "row": 0,
                        "col": 0,
                        "actual_position_mm": [0.0, 0.0],
                        "raw_image": str(first),
                    },
                    {
                        "row": 0,
                        "col": 1,
                        "actual_position_mm": [-2.5, 0.0],
                        "raw_image": str(duplicate),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PnpMaskError, match="stale V4L2 buffer frame"):
        stitch_capture(run_dir)
