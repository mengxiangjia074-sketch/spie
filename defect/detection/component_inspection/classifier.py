"""Trained ResNet classification head for SAM component candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .config import InspectionConfig
from .segmentation import SegmentCandidate, extract_rectified_patch, rect_corners


ProgressCallback = Callable[[str], None]
POSITIVE_COLOR = (60, 205, 70)
NEGATIVE_COLOR = (45, 55, 225)
SAM_COLOR = (220, 165, 40)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"cannot encode image: {path}")
    encoded.tofile(str(path))


def build_resnet18_classifier(num_classes: int):
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ComponentClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config: InspectionConfig,
        device: str = "auto",
        progress: ProgressCallback | None = None,
    ):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.config = config
        self.progress = progress
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if progress is not None:
            progress("正在加载元器件分类头…")
        checkpoint = torch.load(
            Path(checkpoint_path).resolve(), map_location=self.device, weights_only=True
        )
        self.class_names = list(checkpoint.get("class_names") or ["negative", "positive"])
        if "positive" not in self.class_names:
            raise RuntimeError("classifier checkpoint does not define a positive class")
        image_size = checkpoint.get("image_size") or [96, 224]
        if len(image_size) != 2:
            raise RuntimeError(f"invalid classifier image_size: {image_size}")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.positive_index = self.class_names.index("positive")
        self.negative_index = (
            self.class_names.index("negative") if "negative" in self.class_names else 0
        )
        self.model = build_resnet18_classifier(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def classify(
        self,
        image_rgb: np.ndarray,
        candidates: list[SegmentCandidate],
        image_name: str,
        output_dir: Path,
    ) -> dict:
        patches = [
            extract_rectified_patch(
                image_rgb,
                candidate,
                self.config.padding_ratio,
                self.config.min_patch_size,
            )
            for candidate in candidates
        ]
        if self.progress is not None:
            self.progress(f"{image_name}: 分类 {len(patches)} 个候选")
        probability_batches = []
        batch_size = self.config.classifier_batch_size
        for start in range(0, len(patches), batch_size):
            tensors = [
                self.transform(Image.fromarray(patch.astype(np.uint8), mode="RGB"))
                for patch in patches[start : start + batch_size]
            ]
            if not tensors:
                continue
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(batch)
                probability_batches.append(
                    self.torch.softmax(logits, dim=1).detach().cpu().numpy()
                )
        probabilities = (
            np.concatenate(probability_batches, axis=0)
            if probability_batches
            else np.zeros((0, len(self.class_names)), dtype=np.float32)
        )

        patch_root = output_dir / "patches"
        records = []
        accepted = []
        for index, (candidate, patch, probs) in enumerate(
            zip(candidates, patches, probabilities), start=1
        ):
            candidate_id = f"component_{index:04d}"
            positive_prob = float(probs[self.positive_index])
            negative_prob = float(probs[self.negative_index])
            predicted_index = int(np.argmax(probs))
            is_positive = (
                predicted_index == self.positive_index
                and positive_prob >= self.config.classifier_threshold
            )
            patch_path = None
            if is_positive or self.config.save_candidate_patches:
                label = "positive" if is_positive else "negative"
                patch_path = patch_root / label / (
                    f"{candidate_id}_sam_{candidate.sam_score:.3f}_pos_{positive_prob:.3f}.png"
                )
                _write_image(patch_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            record = {
                "id": candidate_id,
                "is_positive": bool(is_positive),
                "predicted_class": self.class_names[predicted_index],
                "sam_score": float(candidate.sam_score),
                "positive_prob": positive_prob,
                "negative_prob": negative_prob,
                "probabilities": {
                    name: float(probs[class_index])
                    for class_index, name in enumerate(self.class_names)
                },
                "sam_bbox_xyxy": list(candidate.sam_bbox_xyxy),
                "mask_bbox_xyxy": list(candidate.mask_bbox_xyxy),
                "center_xy": list(candidate.center_xy),
                "inner_point_xy": list(candidate.inner_point_xy),
                "rotated_box_points": rect_corners(candidate.rotated_rect).tolist(),
                "mask_area_px": candidate.mask_area,
                "mask_fill_ratio": candidate.mask_fill_ratio,
                "patch_size_px": [int(patch.shape[1]), int(patch.shape[0])],
                "patch_path": str(patch_path.resolve()) if patch_path is not None else None,
            }
            records.append(record)
            if is_positive:
                accepted.append(record)

        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_overlay = output_dir / "candidate_overlay.png"
        accepted_overlay = output_dir / "accepted_overlay.png"
        mask_overlay = output_dir / "sam_mask_overlay.png"
        draw_classification_overlay(image_rgb, records, candidate_overlay, positives_only=False)
        draw_classification_overlay(image_rgb, records, accepted_overlay, positives_only=True)
        draw_mask_overlay(image_rgb, candidates, mask_overlay)
        result_json = output_dir / "detections.json"
        payload = {
            "image_name": image_name,
            "candidate_count": len(records),
            "accepted_count": len(accepted),
            "classifier": {
                "class_names": self.class_names,
                "image_size": list(self.image_size),
                "positive_threshold": self.config.classifier_threshold,
            },
            "outputs": {
                "candidate_overlay": str(candidate_overlay.resolve()),
                "accepted_overlay": str(accepted_overlay.resolve()),
                "sam_mask_overlay": str(mask_overlay.resolve()),
            },
            "accepted_detections": accepted,
            "candidates": records,
        }
        payload["outputs"]["detections_json"] = str(result_json.resolve())
        result_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    def close(self) -> None:
        self.model = None


def draw_classification_overlay(
    image_rgb: np.ndarray,
    records: list[dict],
    output_path: Path,
    positives_only: bool,
) -> None:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for record in records:
        if positives_only and not record["is_positive"]:
            continue
        color = POSITIVE_COLOR if record["is_positive"] else NEGATIVE_COLOR
        polygon = np.rint(record["rotated_box_points"]).astype(np.int32)
        cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True, color, 3, cv2.LINE_AA)
        center = tuple(np.rint(record["center_xy"]).astype(int))
        cv2.drawMarker(canvas, center, color, cv2.MARKER_CROSS, 12, 2)
        label = f"{record['id']} p={record['positive_prob']:.2f}"
        label_xy = (int(polygon[:, 0].min()), max(16, int(polygon[:, 1].min()) - 5))
        cv2.putText(
            canvas,
            label,
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    _write_image(output_path, canvas)


def draw_mask_overlay(
    image_rgb: np.ndarray,
    candidates: list[SegmentCandidate],
    output_path: Path,
) -> None:
    overlay = image_rgb.copy()
    colors = (
        np.asarray([255, 120, 80], dtype=np.float32),
        np.asarray([80, 220, 120], dtype=np.float32),
        np.asarray([80, 170, 255], dtype=np.float32),
        np.asarray([255, 210, 80], dtype=np.float32),
    )
    for index, candidate in enumerate(candidates):
        x1, y1, x2, y2 = candidate.mask_bbox_xyxy
        mask = candidate.mask_crop
        region = overlay[y1:y2, x1:x2]
        if region.shape[:2] != mask.shape:
            continue
        color = colors[index % len(colors)]
        blended = region.astype(np.float32)
        blended[mask] = blended[mask] * 0.65 + color * 0.35
        region[:] = np.clip(blended, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(region, contours, -1, color.astype(np.uint8).tolist(), 2)
    _write_image(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
