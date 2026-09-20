# Component inspection models

The trained project classifier is stored locally at:

`component_classifier/best_resnet18_component_classifier.pt`

The SAM3 and RoMaV2 weights are also stored locally at:

- `sam3/sam3.pt`
- `romav2/romav2.0.1.pt`

The component inspection runtime uses these internal paths by default and does
not search ModelScope, Torch, or other user cache directories. The command-line
programs still accept an explicit checkpoint argument for controlled testing.
