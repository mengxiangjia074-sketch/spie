"""RoMaV2 dense matching and validated image-to-ground-truth registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .config import InspectionConfig
from .geometry import transform_points


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RegistrationResult:
    homography_test_to_truth: np.ndarray
    test_footprint_in_truth: np.ndarray
    truth_points: np.ndarray
    test_points: np.ndarray
    confidences: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    median_reprojection_error: float
    mean_reprojection_error: float
    visualization: Path

    def serializable(self) -> dict:
        return {
            "homography_test_to_truth": self.homography_test_to_truth.tolist(),
            "test_footprint_in_truth": self.test_footprint_in_truth.tolist(),
            "sampled_match_count": int(len(self.truth_points)),
            "homography_inliers": int(self.inlier_count),
            "median_reprojection_error_px": float(self.median_reprojection_error),
            "mean_reprojection_error_px": float(self.mean_reprojection_error),
            "confidence": {
                "minimum": float(np.min(self.confidences)),
                "maximum": float(np.max(self.confidences)),
                "mean": float(np.mean(self.confidences)),
            },
            "feature_match_overlay": str(self.visualization.resolve()),
        }


class RegistrationError(RuntimeError):
    pass


class RomaRegistrar:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config: InspectionConfig,
        progress: ProgressCallback | None = None,
    ):
        import torch
        from romav2 import RoMaV2

        if not torch.cuda.is_available():
            raise RegistrationError("RoMaV2 registration requires an available CUDA GPU")
        self.torch = torch
        self.config = config
        self.progress = progress
        if progress is not None:
            progress("正在加载 RoMaV2 图像配准模型…")
        torch.set_float32_matmul_precision("highest")
        self.model = RoMaV2(weights_path=Path(checkpoint_path).resolve())
        self.model.apply_setting(config.roma_setting)

    def _sample_top_matches(self, truth_image: Path, test_image: Path):
        predictions = self.model.match(str(truth_image), str(test_image))
        matches, confidences, _, _ = self.model.sample(
            predictions, self.config.roma_sample_count
        )
        match_height = int(self.model.H_hr or self.model.H_lr)
        match_width = int(self.model.W_hr or self.model.W_lr)
        truth_points, test_points = self.model.to_pixel_coordinates(
            matches,
            match_height,
            match_width,
            match_height,
            match_width,
        )
        truth_size = Image.open(truth_image).size
        test_size = Image.open(test_image).size
        truth_np = truth_points.detach().cpu().numpy().astype(np.float32)
        test_np = test_points.detach().cpu().numpy().astype(np.float32)
        confidence_np = confidences.detach().cpu().numpy().astype(np.float32).reshape(-1)
        truth_np[:, 0] *= float(truth_size[0]) / match_width
        truth_np[:, 1] *= float(truth_size[1]) / match_height
        test_np[:, 0] *= float(test_size[0]) / match_width
        test_np[:, 1] *= float(test_size[1]) / match_height

        valid = np.isfinite(confidence_np)
        valid &= confidence_np >= self.config.min_match_confidence
        valid &= np.all(np.isfinite(truth_np), axis=1)
        valid &= np.all(np.isfinite(test_np), axis=1)
        valid &= (truth_np[:, 0] >= 0) & (truth_np[:, 0] < truth_size[0])
        valid &= (truth_np[:, 1] >= 0) & (truth_np[:, 1] < truth_size[1])
        valid &= (test_np[:, 0] >= 0) & (test_np[:, 0] < test_size[0])
        valid &= (test_np[:, 1] >= 0) & (test_np[:, 1] < test_size[1])
        valid_indices = np.where(valid)[0]
        if len(valid_indices) < 4:
            raise RegistrationError("RoMaV2 returned fewer than four valid matches")
        ordered = valid_indices[np.argsort(confidence_np[valid_indices])[::-1]]
        selected = ordered[: min(self.config.roma_top_k, len(ordered))]
        del predictions, matches, confidences
        return truth_np[selected], test_np[selected], confidence_np[selected]

    def register(
        self,
        truth_image: str | Path,
        test_image: str | Path,
        visualization_path: str | Path,
    ) -> RegistrationResult:
        truth_path = Path(truth_image).resolve()
        test_path = Path(test_image).resolve()
        if self.progress is not None:
            self.progress(f"{test_path.name}: RoMaV2 匹配到真值大图")
        truth_points, test_points, confidences = self._sample_top_matches(
            truth_path, test_path
        )
        homography, raw_inliers = cv2.findHomography(
            test_points.reshape(-1, 1, 2),
            truth_points.reshape(-1, 1, 2),
            cv2.RANSAC,
            self.config.ransac_reproj_threshold,
        )
        if homography is None or not np.all(np.isfinite(homography)):
            raise RegistrationError(f"{test_path.name}: cannot estimate a valid homography")
        inliers = (
            raw_inliers.reshape(-1).astype(bool)
            if raw_inliers is not None
            else np.zeros(len(test_points), dtype=bool)
        )
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < self.config.min_homography_inliers:
            raise RegistrationError(
                f"{test_path.name}: homography has {inlier_count} inliers; "
                f"need {self.config.min_homography_inliers}"
            )
        projected = transform_points(test_points, homography)
        errors = np.linalg.norm(projected - truth_points, axis=1)
        inlier_errors = errors[inliers]
        median_error = float(np.median(inlier_errors))
        mean_error = float(np.mean(inlier_errors))
        if median_error > self.config.max_median_reprojection_error:
            raise RegistrationError(
                f"{test_path.name}: median registration error {median_error:.2f}px exceeds "
                f"{self.config.max_median_reprojection_error:.2f}px"
            )

        with Image.open(test_path) as test_pil:
            test_width, test_height = test_pil.size
        test_corners = np.asarray(
            [
                [0, 0],
                [test_width - 1, 0],
                [test_width - 1, test_height - 1],
                [0, test_height - 1],
            ],
            dtype=np.float32,
        )
        footprint = transform_points(test_corners, homography)
        if not cv2.isContourConvex(np.rint(footprint).astype(np.int32)):
            raise RegistrationError(f"{test_path.name}: registered footprint is not convex")
        footprint_area = float(abs(cv2.contourArea(footprint.astype(np.float32))))
        with Image.open(truth_path) as truth_pil:
            truth_area = float(truth_pil.width * truth_pil.height)
        if footprint_area <= truth_area * 1e-4 or footprint_area > truth_area * 4.0:
            raise RegistrationError(
                f"{test_path.name}: implausible registered footprint area {footprint_area:.0f}px"
            )

        visualization = Path(visualization_path).resolve()
        save_match_visualization(
            truth_path,
            test_path,
            truth_points,
            test_points,
            confidences,
            inliers,
            footprint,
            visualization,
        )
        if self.progress is not None:
            self.progress(
                f"{test_path.name}: 配准内点 {inlier_count}/{len(inliers)}，"
                f"中位误差 {median_error:.2f}px"
            )
        return RegistrationResult(
            homography_test_to_truth=homography,
            test_footprint_in_truth=footprint,
            truth_points=truth_points,
            test_points=test_points,
            confidences=confidences,
            inlier_mask=inliers,
            inlier_count=inlier_count,
            median_reprojection_error=median_error,
            mean_reprojection_error=mean_error,
            visualization=visualization,
        )

    def close(self) -> None:
        self.model = None


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RegistrationError(f"cannot read image for visualization: {path}")
    return image


def _resize_height(image: np.ndarray, target_height: int):
    height, width = image.shape[:2]
    scale = float(target_height) / float(height)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), target_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    return resized, scale


def save_match_visualization(
    truth_image: Path,
    test_image: Path,
    truth_points: np.ndarray,
    test_points: np.ndarray,
    confidences: np.ndarray,
    inliers: np.ndarray,
    footprint: np.ndarray,
    output_path: Path,
) -> None:
    target_height = 800
    gap = 28
    truth_view, truth_scale = _resize_height(_read_image(truth_image), target_height)
    test_view, test_scale = _resize_height(_read_image(test_image), target_height)
    canvas = np.full(
        (target_height, truth_view.shape[1] + gap + test_view.shape[1], 3),
        20,
        dtype=np.uint8,
    )
    canvas[:, : truth_view.shape[1]] = truth_view
    test_offset = truth_view.shape[1] + gap
    canvas[:, test_offset:] = test_view

    footprint_scaled = np.rint(footprint * truth_scale).astype(np.int32).reshape(-1, 1, 2)
    layer = canvas.copy()
    cv2.fillPoly(layer, [footprint_scaled], (0, 190, 255))
    canvas = cv2.addWeighted(layer, 0.15, canvas, 0.85, 0)
    cv2.polylines(canvas, [footprint_scaled], True, (0, 220, 255), 3, cv2.LINE_AA)

    valid_indices = np.where(inliers)[0]
    if len(valid_indices) > 60:
        order = np.argsort(confidences[valid_indices])[::-1][:60]
        valid_indices = valid_indices[order]
    hsv = np.zeros((max(1, len(valid_indices)), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, len(hsv), endpoint=False, dtype=np.uint8)
    hsv[:, 0, 1:] = (210, 255)
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
    for color, match_index in zip(colors, valid_indices):
        color_value = tuple(map(int, color))
        truth_point = tuple(np.rint(truth_points[match_index] * truth_scale).astype(int))
        test_point = tuple(
            np.rint(test_points[match_index] * test_scale).astype(int)
            + np.asarray([test_offset, 0])
        )
        cv2.line(canvas, truth_point, test_point, color_value, 1, cv2.LINE_AA)
        cv2.circle(canvas, truth_point, 4, color_value, -1, cv2.LINE_AA)
        cv2.circle(canvas, test_point, 4, color_value, -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RegistrationError(f"cannot encode visualization: {output_path}")
    encoded.tofile(str(output_path))
