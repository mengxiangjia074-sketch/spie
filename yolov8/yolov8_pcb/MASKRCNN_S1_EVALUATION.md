# Mask R-CNN S1 Evaluation

## Evaluation Setup

- Evaluated: 2026-09-20T23:55:29+08:00
- Model: torchvision Mask R-CNN ResNet-50 FPN v2
- Checkpoint: `/home/vision/Users/DSC/runs/maskrcnn_s1_100/best.pth`
- Best checkpoint epoch: 8
- Dataset split: `/home/vision/Users/DSC/yolov8/yolov8_pcb/ground_truth/fpic_yolo/images/val` (used as the test set because no separate test split exists)
- Test images: 32
- Ground-truth instances: 93
- Classes: 5
- Device: NVIDIA GeForce RTX 4080 SUPER
- Mean inference time: 65.36 ms/image

Precision and recall use class-aware one-to-one matching at IoU 0.50. The confidence threshold is selected by the highest mean F1 across classes. AP follows COCO's 101-point interpolation; mAP50:95 averages IoU thresholds 0.50 through 0.95 in steps of 0.05.

## Overall Metrics

| Type | Precision | Recall | F1 | mAP50 | mAP75 | mAP50:95 | mAR100 | Best confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Box | 0.6189 | 0.8000 | 0.6979 | 0.7588 | 0.7302 | 0.5972 | 0.7711 | 0.169 |
| Mask | 0.6189 | 0.8000 | 0.6979 | 0.7634 | 0.7349 | 0.6120 | 0.7694 | 0.169 |

## Per-Class Metrics

| Class | Instances | Mask P | Mask R | Mask F1 | Mask mAP50:95 | Mask mAR100 | Box mAP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resistors | 53 | 0.9138 | 1.0000 | 0.9550 | 0.7757 | 0.8189 | 0.7504 |
| capacitors | 24 | 0.6667 | 1.0000 | 0.8000 | 0.7178 | 0.8583 | 0.6735 |
| ICs | 10 | 0.7143 | 1.0000 | 0.8333 | 0.7344 | 0.8700 | 0.7193 |
| inductors | 4 | 0.8000 | 1.0000 | 0.8889 | 0.7713 | 0.8500 | 0.7713 |
| diodes | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0607 | 0.4500 | 0.0714 |

## Notes

- These metrics are directly comparable in structure to the Box and Mask metrics reported by YOLOv8-Seg, while the evaluated model is Mask R-CNN.
- The held-out split is small and imbalanced. In particular, inductors and diodes have very few instances, so their per-class metrics have high uncertainty.
- The report evaluates the best early-stopping checkpoint, not the final checkpoint.

## Reproduction

```bash
/home/vision/miniconda3/envs/sam3/bin/python yolov8/yolov8_pcb/evaluate_maskrcnn_s1.py
```

Raw metrics: `/home/vision/Users/DSC/runs/maskrcnn_s1_100/evaluation/metrics.json`
