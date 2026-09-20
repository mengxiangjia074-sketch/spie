import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROMPTS = (
    "electronic component",
    "integrated circuit",
    "chip",
    "resistor",
    "capacitor",
    "connector",
    "inductor",
    "diode",
    "transistor",
)


@dataclass
class Instance:
    prompt: str
    score: float
    mask: np.ndarray
    bbox_xywh_norm: list[float]

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment PCB components in data/ images with local SAM3.1 weights."
    )
    parser.add_argument(
        "--input",
        default="images/img3.png",
        help="Image file or directory containing images to segment.",
    )
    parser.add_argument(
        "--checkpoint",
        default="pretrained/sam3.pt",
        help="Path to the local SAM3.1 multiplex checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="outputs/img3",
        help="Directory for overlays, masks, and JSON metadata.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help=(
            "Text prompt to run. Repeat this option to use multiple prompts. "
            "Defaults cover common PCB components."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.45,
        help="Keep instances with model score at least this value.",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.85,
        help="Suppress lower-score masks whose IoU with a kept mask exceeds this value.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=80,
        help="Discard masks with fewer foreground pixels than this.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.25,
        help="Discard masks larger than this fraction of the image area.",
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=128,
        help="Maximum number of objects the SAM3.1 predictor may track per image.",
    )
    parser.add_argument(
        "--multiplex-count",
        type=int,
        default=16,
        help="SAM3.1 object multiplex bucket size.",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3 kernels. Leave off unless the environment has FA3.",
    )
    parser.add_argument(
        "--use-perflib",
        action="store_true",
        help="Enable SAM3 optional perflib accelerations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and arguments without loading the model.",
    )
    return parser.parse_args()


def image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    paths = [
        p
        for p in sorted(input_path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise RuntimeError(f"No images found in {input_path}")
    return paths


def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "prompt"


def build_predictor(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise SystemExit(
            "SAM3.1 multiplex inference in this repository requires CUDA, but "
            "torch.cuda.is_available() is False in the current environment."
        )

    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache_sam3")
    os.environ["USE_PERFLIB"] = "1" if args.use_perflib else "0"

    from sam3 import build_sam3_predictor

    return build_sam3_predictor(
        checkpoint_path=str(Path(args.checkpoint)),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        use_fa3=args.use_fa3,
        use_rope_real=True,
        async_loading_frames=False,
    )


def run_prompt(
    predictor,
    session_id: str,
    prompt: str,
    score_threshold: float,
) -> list[Instance]:
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
            "output_prob_thresh": score_threshold,
        }
    )
    output = response["outputs"]

    masks = np.asarray(output.get("out_binary_masks", []), dtype=bool)
    scores = np.asarray(output.get("out_probs", []), dtype=np.float32)
    boxes = np.asarray(output.get("out_boxes_xywh", []), dtype=np.float32)

    if masks.size == 0:
        return []

    instances = []
    for mask, score, box in zip(masks, scores, boxes):
        if float(score) < score_threshold:
            continue
        instances.append(
            Instance(
                prompt=prompt,
                score=float(score),
                mask=mask,
                bbox_xywh_norm=[float(v) for v in box.tolist()],
            )
        )
    return instances


def filter_instances(
    instances: list[Instance],
    image_shape: tuple[int, int],
    min_area: int,
    max_area_ratio: float,
) -> list[Instance]:
    image_area = image_shape[0] * image_shape[1]
    max_area = image_area * max_area_ratio
    return [
        inst
        for inst in instances
        if inst.area >= min_area and inst.area <= max_area
    ]


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / max(union, 1))


def dedupe_instances(instances: list[Instance], iou_threshold: float) -> list[Instance]:
    kept: list[Instance] = []
    for inst in sorted(instances, key=lambda x: x.score, reverse=True):
        if all(mask_iou(inst.mask, prev.mask) <= iou_threshold for prev in kept):
            kept.append(inst)
    return kept


def bbox_xyxy_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (230, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
    )
    return palette[index % len(palette)]


def save_results(
    image_path: Path,
    instances: list[Instance],
    output_dir: Path,
    prompts: list[str],
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    stem = image_path.stem

    overlay = np.asarray(image).astype(np.float32)
    labels = np.zeros((height, width), dtype=np.uint16)
    mask_dir = output_dir / "masks" / stem
    mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, inst in enumerate(instances, start=1):
        color = np.asarray(color_for_index(idx - 1), dtype=np.float32)
        overlay[inst.mask] = overlay[inst.mask] * 0.55 + color * 0.45
        labels[inst.mask] = idx
        mask = Image.fromarray((inst.mask.astype(np.uint8) * 255), mode="L")
        mask.save(mask_dir / f"{stem}_{idx:03d}_{safe_name(inst.prompt)}.png")

    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_img)
    font = ImageFont.load_default()

    metadata = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "prompts": prompts,
        "num_instances": len(instances),
        "instances": [],
    }

    for idx, inst in enumerate(instances, start=1):
        x0, y0, x1, y1 = bbox_xyxy_from_mask(inst.mask)
        color = color_for_index(idx - 1)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = f"{idx}:{inst.score:.2f}"
        tx, ty = x0, max(0, y0 - 12)
        draw.rectangle((tx, ty, tx + 48, ty + 12), fill=color)
        draw.text((tx + 2, ty), label, fill=(255, 255, 255), font=font)
        metadata["instances"].append(
            {
                "id": idx,
                "prompt": inst.prompt,
                "score": inst.score,
                "area": inst.area,
                "bbox_xyxy": [x0, y0, x1, y1],
                "bbox_xywh_norm": inst.bbox_xywh_norm,
                "mask": str(mask_dir / f"{stem}_{idx:03d}_{safe_name(inst.prompt)}.png"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_img.save(output_dir / f"{stem}_overlay.png")
    Image.fromarray(labels).save(output_dir / f"{stem}_labels.png")
    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def segment_image(
    predictor,
    image_path: Path,
    prompts: list[str],
    args: argparse.Namespace,
) -> int:
    with Image.open(image_path) as image:
        image_shape = (image.height, image.width)

    collected: list[Instance] = []
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(image_path)}
    )
    session_id = response["session_id"]
    try:
        for prompt in prompts:
            print(f"[{image_path.name}] prompt: {prompt}")
            collected.extend(
                run_prompt(
                    predictor=predictor,
                    session_id=session_id,
                    prompt=prompt,
                    score_threshold=args.score_threshold,
                )
            )
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": True}
        )

    filtered = filter_instances(
        instances=collected,
        image_shape=image_shape,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
    )
    deduped = dedupe_instances(filtered, args.dedupe_iou)
    save_results(image_path, deduped, Path(args.output), prompts)
    print(f"[{image_path.name}] saved {len(deduped)} component masks")
    return len(deduped)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    checkpoint = Path(args.checkpoint)
    prompts = args.prompts or list(DEFAULT_PROMPTS)
    paths = image_paths(input_path)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.dry_run:
        print(f"Images: {len(paths)}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Output: {Path(args.output)}")
        print(f"Prompts: {', '.join(prompts)}")
        print("Dry run complete; model was not loaded.")
        return

    predictor = build_predictor(args)
    total = 0
    for path in paths:
        total += segment_image(predictor, path, prompts, args)
    print(f"Done. Segmented {total} component instances from {len(paths)} image(s).")


if __name__ == "__main__":
    main()
