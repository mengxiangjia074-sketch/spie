# SAM3 + Mask Refiner S1 Final Evaluation

- Quality head: `results/runs/sam3_quality_1000/classifier_quality_best.pt`
- Quality-head best epoch: 309
- Mask refiner: `results/runs/sam3_refiner_1000/mask_refiner_best.pt`
- Mask-refiner best epoch: 134
- Test split: `yolov8/yolov8_pcb/ground_truth/fpic_yolo/images/val`
- Test images: 32
- Ground-truth instances: 93

| Metric | Value |
|---|---:|
| Joint-F1 | 0.8923 |
| Macro-F1 | 0.8492 |
| Mask AP50:95 | 0.6986 |
| Boundary F1 | 0.7276 |
| Small Component Recall | 0.9841 |

The evaluation uses a SAM threshold of 0.25, foreground threshold of 0.50,
quality threshold of 0.30, refined-mask threshold of 0.55, class-aware matching
at Mask IoU 0.50, a 2-pixel boundary tolerance, and a 4096 px2
small-component threshold.
