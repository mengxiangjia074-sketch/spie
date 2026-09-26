# YOLOv8-Seg PCB Image Segmentation Results

- Weights: `results/runs/yolov8_seg_1000/yolov8l_seg_s1/weights/best.pt`
- Input: `pcb_image/`
- Output: `results/yolov8_seg_1000_pcb_image/`
- Confidence threshold: 0.25
- Tile size / overlap: 1024 / 256 px

| Image | Resistors | Capacitors | ICs | Inductors | Diodes | Total |
|---|---:|---:|---:|---:|---:|---:|
| r00_c00_undistorted.png | 13 | 24 | 1 | 2 | 0 | 40 |
| r00_c01_undistorted.png | 10 | 12 | 1 | 2 | 0 | 25 |
| r01_c00_undistorted.png | 13 | 45 | 1 | 4 | 0 | 63 |
| r01_c01_undistorted.png | 7 | 22 | 0 | 6 | 0 | 35 |
| **Total** | **43** | **103** | **3** | **14** | **0** | **163** |

Each image output directory contains `yolo_mask_overlay.png` and
`detections.json`. The overlay follows the reference format with translucent
colored masks, contours, instance identifiers, and confidence values.

These four images do not include matching ground-truth labels in `pcb_image/`,
so the table reports prediction counts rather than accuracy metrics.
