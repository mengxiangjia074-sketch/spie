# SAM3 Quality Head + Mask Refiner S1 Evaluation

## Setup

- Dataset: FPIC S1
- Train candidates: frozen SAM3 with `component` prompt
- Internal development group: `Microscope_s1_p1_2x_40_ring`
- Held-out test group: `Microscope_s1_p1_1x_40_ring`
- Test images: 32
- Test instances: 93
- Quality head: ResNet18 with background/five-class classification and mask-IoU regression
- Mask refiner: lightweight U-Net using RGB, raw SAM mask, and candidate-box prior
- Refiner checkpoint: epoch 27, development soft Dice 0.9333
- Mask threshold: 0.55, selected only on the internal development group

The training process starts from random classifier/refiner weights and does not read previous training logs. The original SAM3 remains frozen.

## Held-Out Results

| Method | Mask AP50 | Mask AP75 | Mask AP50:95 | Boundary F1 | Macro-F1 | Joint-F1 | Predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen SAM3 + five-class head (100 epoch) | 0.3170 | 0.1883 | 0.2025 | 0.4287 | 0.1565 | 0.2357 | 484 |
| Quality-aware SAM3 head | 0.8241 | 0.4733 | 0.4314 | 0.4205 | 0.8168 | 0.8265 | 103 |
| Quality head + mask refiner (balanced thresholds) | 0.8895 | 0.8455 | 0.6762 | 0.7195 | 0.8659 | 0.8980 | 103 |
| Quality head + mask refiner (AP-calibrated thresholds) | 0.8965 | 0.8455 | 0.6780 | 0.7175 | 0.8517 | 0.8641 | 113 |
| Mask R-CNN baseline | 0.7634 | 0.7349 | 0.6120 | N/A | N/A | N/A | N/A |
| YOLOv8l-Seg baseline | 0.9703 | 0.9643 | 0.7842 | 0.7522 | 0.8538 | 0.7949 | 141 |

The balanced-threshold refiner is the preferred overall configuration because it has higher Macro-F1 and Joint-F1 with fewer predictions. The AP-calibrated configuration uses foreground/quality thresholds of 0.20/0.20 and gives the highest Mask AP50:95.

## Outcome

The refiner exceeds Mask R-CNN on Mask AP50, AP75, and AP50:95. Compared with YOLOv8l-Seg, it exceeds Macro-F1 and Joint-F1 but remains lower on mask AP and boundary metrics. The remaining limitation is candidate recall and the frozen SAM3 mask decoder.

Most S1 masks are rectangle-derived weak labels (435 rectangles and 2 polygons). These results measure agreement with that annotation format and should not be treated as pixel-accurate physical component boundaries.

## Artifacts

- Quality-head checkpoint: `runs/sam3_fpic_scheme_b_quality/classifier_quality_best.pt`
- Refiner checkpoint: `runs/sam3_fpic_mask_refiner/mask_refiner_best.pt`
- Balanced evaluation: `runs/sam3_fpic_mask_refiner/evaluation/metrics.json`
- AP-calibrated evaluation: `runs/sam3_fpic_mask_refiner/evaluation_calibrated/metrics.json`
