# PCB component segmentation with YOLOv8-Seg

This program fine-tunes YOLOv8-Seg to recognize and segment the PCB components
in the four images under `../../data`. Because the bundled
`../yolov8l-seg.pt` is trained on COCO and has no PCB component class, the
program bootstraps a one-class dataset from the latest accepted SAM3 results.
SAM3 is only needed to create the initial labels; training and inference use
YOLOv8-Seg.

The existing SAM3 JSON stores rotated mask bounds rather than full mask
contours. The generated labels are therefore rotated-polygon approximations.
Reviewing/correcting the labels before a production training run is strongly
recommended. Four source images are enough to reproduce these boards, but not
enough to guarantee generalization to different boards or lighting.

## Environment

Use a CUDA-enabled Python environment and install the dependencies:

```bash
cd /home/vision/Users/DSC/yolov8/yolov8_pcb
python -m pip install -r requirements.txt
```

## Run

Create a tiled dataset. The newest complete run under
`../../sam3/results/component_inspection` is selected automatically:

```bash
python main.py prepare
# Rebuild an existing dataset directory with: python main.py prepare --force
```

Train from the bundled YOLOv8 large segmentation checkpoint:

```bash
python main.py train --epochs 100 --device 0
# AMP is off by default for offline-safe startup; add --amp when desired
```

Run tiled inference over all four original images:

```bash
python main.py predict --device 0
```

Or execute all three stages:

```bash
python main.py all --epochs 100 --device 0
```

Useful overrides:

```bash
python main.py prepare --sam-results /path/to/a/sam3/run --validation-image r01_c01_undistorted.png
python main.py predict --weights /path/to/best.pt --source /path/to/images --conf 0.35
```

Outputs:

- `dataset/`: 1024 px tiled images, YOLO polygon labels, `data.yaml`, and split manifest.
- `runs/train/pcb_components/weights/best.pt`: trained segmentation weights.
- `runs/predict/<image>/yolo_mask_overlay.png`: recognition and mask visualization.
- `runs/predict/<image>/detections.json`: confidence, box, center and polygon for each component.

The default split keeps all tiles from one source image in validation, avoiding
train/validation leakage from overlapping tiles of the same original image.

## FPIC s1 training

Convert the local FPIC sample in `../../data/s1` and train the five-class model:

```bash
python main.py prepare-fpic --validation-group Microscope_s1_p1_1x_40_ring --force
python main.py train --data ground_truth/fpic_yolo/data.yaml --project runs/fpic_train --name yolov8l_seg_s1 --epochs 100 --imgsz 1024 --batch 4 --device 0 --amp --patience 20
```

Run full-board inference:

```bash
python main.py predict --weights runs/fpic_train/yolov8l_seg_s1/weights/best.pt --source ../../data/s1/DSLR/img/s1_front.tif --output runs/fpic_predict --device 0
```

The local s1 annotations contain 435 rectangles and two polygons. Rectangles are converted to four-point polygons, so most labels are weak segmentation labels rather than pixel-accurate masks. The generated manifest records this limitation and the group-level split.

## Detailed evaluation

Evaluate the held-out `val` group as the test split and save AP, boundary, F1, small-object recall, and center-error metrics:

```bash
python main.py evaluate --weights runs/fpic_train/yolov8l_seg_s1/weights/best.pt --data ground_truth/fpic_yolo/data.yaml --output runs/fpic_evaluation --imgsz 1024 --batch 4 --device 0 --conf 0.25 --boundary-tolerance 2 --small-area 4096
```

The report uses IoU 0.5 for instance matching, a 2-pixel boundary tolerance, and a 4096 px2 small-object threshold. `val` is a held-out capture group, not an independent PCB board.
