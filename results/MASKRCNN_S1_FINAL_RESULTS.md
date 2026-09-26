# Mask R-CNN S1 Final Evaluation

- Model: Mask R-CNN ResNet-50 FPN v2
- Weights: `results/runs/maskrcnn_s1_1000/best.pth`
- Best checkpoint epoch: 15
- Test split: `yolov8/yolov8_pcb/ground_truth/fpic_yolo/images/val`
- Test images: 32
- Ground-truth instances: 93

| Metric | Value |
|---|---:|
| Joint-F1 | 0.8683 |
| Macro-F1 | 0.7926 |
| Mask AP50:95 | 0.7792 |
| Boundary F1 | 0.7131 |
| Small Component Recall | 1.0000 |

The detailed instance metrics use confidence 0.25, class-aware matching at Mask
IoU 0.50, a 2-pixel boundary tolerance, and a 4096 px2 small-component
threshold. Mask AP50:95 is computed from the complete confidence ranking.
