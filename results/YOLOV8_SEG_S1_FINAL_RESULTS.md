# YOLOv8-Seg S1 Final Evaluation

- Weights: `results/runs/yolov8_seg_1000/yolov8l_seg_s1/weights/best.pt`
- Test split: `yolov8/yolov8_pcb/ground_truth/fpic_yolo/images/val`
- Test images: 32
- Ground-truth instances: 93

| Metric | Value |
|---|---:|
| Joint-F1 | 0.7750 |
| Macro-F1 | 0.8274 |
| Mask AP50:95 | 0.7834 |
| Boundary F1 | 0.7774 |
| Small Component Recall | 1.0000 |

The evaluation uses class-aware one-to-one matching at Mask IoU 0.50, a
2-pixel boundary tolerance, and a 4096 px2 small-component threshold.
