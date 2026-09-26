# Final S1 Experiment Evidence

This directory contains the final paper-facing outputs for the FPIC S1
experiments.

Included evidence:

- five-metric evaluation summaries for YOLOv8-Seg, Mask R-CNN, and SAM3 + Mask
  Refiner;
- raw evaluation JSON files used to produce the summary tables;
- per-image segmentation overlays and detection JSON files for the four
  `pcb_image` inputs;
- training configuration, curves, histories, and validation visualizations
  that are small enough for normal Git storage.

The five reported metrics are Joint-F1, Macro-F1, Mask AP50:95, Boundary F1,
and Small Component Recall. All three final evaluations use the same 32-image
S1 held-out split with 93 ground-truth instances.

Model checkpoints and serialized candidate caches are intentionally excluded
from Git because the Mask R-CNN checkpoints are approximately 368 MB each and
exceed GitHub's 100 MB per-file limit. Their original local paths are retained
in the evaluation JSON metadata. The metrics, configurations, and rendered
segmentation evidence are versioned here.
