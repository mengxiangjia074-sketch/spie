# PCB component completeness pipeline

The local pipeline is implemented in four stages:

1. `segmentation.py` runs SAM3 text-prompt segmentation and cleans, splits and
   de-duplicates candidate masks.
2. `classifier.py` rectifies each mask patch and runs the trained ResNet18
   positive/negative classification head.
3. `registration.py` uses the bundled RoMaV2 and DINOv3 runtime to estimate and
   validate each high-magnification image-to-truth homography.
4. `geometry.py` maps detections into the large image, removes overlap between
   neighboring camera frames and performs one-to-one truth assignment.

`pipeline.py` aggregates all high-magnification images and writes the final
JSON, CSV and overlay. Ground truth is accepted only as `component_masks.json`
exported by the GUI's coordinate Mask alignment page. The clean stitched image
is used for feature registration, while the final result is rendered on
`pcb_with_component_masks.png`.

Command-line entry point:

```bash
micromamba run -n sam python detection/inspect_components.py \
  --ground-truth working_data/pcb_two_zoom_capture/<run>/mask_alignment/component_masks.json \
  --inspection working_data/pcb_two_zoom_capture/<inspection-run>
```
