# Mask R-CNN PCB Image Segmentation Results

- Weights: `results/runs/maskrcnn_s1_1000/best.pth`
- Best checkpoint epoch: 15
- Input: `pcb_image/`
- Output: `results/maskrcnn_s1_1000_pcb_image/`
- Confidence threshold: 0.25
- Tile size / overlap: 1024 / 256 px
- Maximum mask area: 20,000 px2 for small-component classes and 100,000 px2 for ICs

| Image | Resistors | Capacitors | ICs | Inductors | Diodes | Total |
|---|---:|---:|---:|---:|---:|---:|
| r00_c00_undistorted.png | 8 | 26 | 4 | 1 | 0 | 39 |
| r00_c01_undistorted.png | 9 | 14 | 1 | 1 | 0 | 25 |
| r01_c00_undistorted.png | 6 | 51 | 3 | 0 | 0 | 60 |
| r01_c01_undistorted.png | 3 | 29 | 1 | 0 | 0 | 33 |
| **Total** | **26** | **120** | **9** | **2** | **0** | **157** |

Each image output directory contains `maskrcnn_mask_overlay.png` and
`detections.json`. The overlay follows the reference format with translucent
class-colored masks, contours, instance identifiers, class names, and confidence
values.

These four images do not include matching ground-truth labels in `pcb_image/`,
so the table reports prediction counts rather than accuracy metrics.
