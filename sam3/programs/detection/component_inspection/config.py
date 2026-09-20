"""Configuration and local model/input discovery for component inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DETECTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DETECTION_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "models"
WORKING_DATA = PROJECT_ROOT / "working_data"

LOCAL_CLASSIFIER = (
    MODELS_ROOT
    / "component_classifier"
    / "best_resnet18_component_classifier.pt"
)
LOCAL_SAM3_CHECKPOINT = MODELS_ROOT / "sam3" / "sam3.pt"
LOCAL_ROMA_CHECKPOINT = MODELS_ROOT / "romav2" / "romav2.0.1.pt"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class InspectionConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class InspectionConfig:
    prompt: str = "component"
    sam_confidence_threshold: float = 0.50
    classifier_threshold: float = 0.50
    min_mask_area: int = 60
    max_mask_area_ratio: float = 0.25
    min_mask_fill_ratio: float = 0.05
    min_component_area_ratio: float = 0.05
    mask_open_kernel: int = 3
    edge_margin: int = 5
    mutual_overlap_threshold: float = 0.60
    padding_ratio: float = 0.15
    min_patch_size: int = 160
    tile_size: int = 1600
    tile_overlap: int = 256
    classifier_batch_size: int = 32
    roma_setting: str = "precise"
    roma_sample_count: int = 5000
    roma_top_k: int = 100
    min_match_confidence: float = 0.0
    ransac_reproj_threshold: float = 5.0
    min_homography_inliers: int = 8
    max_median_reprojection_error: float = 8.0
    match_polygon_iou_threshold: float = 0.10
    match_center_ratio: float = 0.65
    coverage_margin_ratio: float = 0.10
    mapped_detection_nms_iou: float = 0.35
    mapped_detection_nms_center_ratio: float = 0.20
    save_candidate_patches: bool = True

    def validate(self) -> None:
        unit_values = {
            "sam_confidence_threshold": self.sam_confidence_threshold,
            "classifier_threshold": self.classifier_threshold,
            "max_mask_area_ratio": self.max_mask_area_ratio,
            "min_mask_fill_ratio": self.min_mask_fill_ratio,
            "min_component_area_ratio": self.min_component_area_ratio,
            "mutual_overlap_threshold": self.mutual_overlap_threshold,
            "match_polygon_iou_threshold": self.match_polygon_iou_threshold,
            "coverage_margin_ratio": self.coverage_margin_ratio,
            "mapped_detection_nms_iou": self.mapped_detection_nms_iou,
        }
        for name, value in unit_values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise InspectionConfigError(f"{name} must be between 0 and 1")
        if self.min_mask_area < 1:
            raise InspectionConfigError("min_mask_area must be positive")
        if self.mask_open_kernel < 1:
            raise InspectionConfigError("mask_open_kernel must be positive")
        if self.tile_size < 64:
            raise InspectionConfigError("tile_size must be at least 64 pixels")
        if not 0 <= self.tile_overlap < self.tile_size:
            raise InspectionConfigError("tile_overlap must be smaller than tile_size")
        if self.classifier_batch_size < 1:
            raise InspectionConfigError("classifier_batch_size must be positive")
        if self.roma_top_k < 4:
            raise InspectionConfigError("roma_top_k must be at least 4")
        if self.roma_sample_count < self.roma_top_k:
            raise InspectionConfigError(
                "roma_sample_count must be greater than or equal to roma_top_k"
            )
        if self.min_homography_inliers < 4:
            raise InspectionConfigError("min_homography_inliers must be at least 4")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_required_path(
    explicit: str | Path | None,
    local_path: Path,
    description: str,
) -> Path:
    # Keep the default runtime self-contained. An explicit CLI/API path is still
    # accepted for controlled experiments, but no user cache is searched.
    candidate = Path(explicit).expanduser() if explicit else local_path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"{description} not found: {candidate}\n"
        f"Place the file at {local_path} or pass an explicit checkpoint path."
    )


def resolve_classifier_checkpoint(explicit: str | Path | None = None) -> Path:
    return _resolve_required_path(
        explicit,
        LOCAL_CLASSIFIER,
        "Component classifier checkpoint",
    )


def resolve_sam3_checkpoint(explicit: str | Path | None = None) -> Path:
    return _resolve_required_path(
        explicit,
        LOCAL_SAM3_CHECKPOINT,
        "SAM3 checkpoint",
    )


def resolve_roma_checkpoint(explicit: str | Path | None = None) -> Path:
    return _resolve_required_path(
        explicit,
        LOCAL_ROMA_CHECKPOINT,
        "RoMaV2 checkpoint",
    )


def latest_ground_truth() -> Path | None:
    root = WORKING_DATA / "pcb_two_zoom_capture"
    candidates = list(root.glob("*/mask_alignment/component_masks.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def latest_capture_source() -> Path | None:
    root = WORKING_DATA / "pcb_two_zoom_capture"
    candidates = [path.parent for path in root.glob("*/pcb_two_zoom_capture.json")]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def collect_inspection_images(source: str | Path) -> list[Path]:
    """Resolve one image, an images directory, or a two-zoom capture run."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InspectionConfigError(f"unsupported inspection image: {source_path}")
        return [source_path]
    if not source_path.is_dir():
        raise FileNotFoundError(f"inspection input not found: {source_path}")

    search_root = source_path / "images" if (source_path / "images").is_dir() else source_path
    all_images = sorted(
        path.resolve()
        for path in search_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    undistorted = [path for path in all_images if path.stem.endswith("_undistorted")]
    images = undistorted or [
        path
        for path in all_images
        if not any(
            marker in path.stem.lower()
            for marker in ("mask", "annotated", "overlay", "stitched")
        )
    ]
    if not images:
        raise InspectionConfigError(f"no inspection images found under {search_root}")
    return images
