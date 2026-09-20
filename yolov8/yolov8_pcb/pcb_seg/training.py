"""Ultralytics YOLOv8-Seg training wrapper."""

from __future__ import annotations

from pathlib import Path


def _yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed in this Python environment; "
            "install requirements.txt first"
        ) from exc
    return YOLO


def train_model(
    model_path: Path,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    image_size: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    amp: bool = False,
    resume: bool = False,
) -> Path:
    YOLO = _yolo_class()
    if not model_path.is_file():
        raise FileNotFoundError(f"initial YOLOv8-Seg weights not found: {model_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml.resolve()),
        task="segment",
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(project.resolve()),
        name=name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        plots=True,
        amp=amp,
        resume=resume,
    )
    best = project.resolve() / name / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training finished but best weights were not found: {best}")
    return best
