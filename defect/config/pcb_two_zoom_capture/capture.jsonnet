// capture_pcb_two_zooms.py configuration.
// Focus/iris are read from the matching pcb_autofocus_lens_position entry.
local stage = import "../../detection/calibration/configs/stage_to_pixel/stage2pixel.jsonnet";

{
  // null uses detection/calibration/output/calibrate.json.
  "calibration": null,

  // Small zoom: detect four corner markers, centre them, then crop the PCB
  // between the markers' inward-facing corners.
  "small_zoom": 2800,
  "white_block_threshold": 210,
  "white_block_min_area_px": 200.0,
  "white_block_max_area_fraction": 0.08,
  "white_block_min_square_ratio": 0.70,
  "white_block_min_fill_ratio": 0.70,
  "white_block_morph_kernel": 5,
  // Positive expands beyond the inward marker corners; negative contracts inward.
  "global_crop_padding_px": -5,

  // Large zoom: calculate PCB dimensions from the four markers, map the
  // top-left marker's bottom-right corner from small zoom, place it at this
  // target pixel, then automatically capture enough images to cover the PCB.
  "large_zoom": 6353,
  "large_target_pixel": [50.0, 50.0],

  // Extra coverage on the PCB right and bottom edges only.
  "capture_margin_mm": 1.0,
  "overlap_fraction": 0.2,

  // LensConnect.
  "lens_device": stage.lens_device,
  "lens_no_init": stage.lens_no_init,
  "lens_settle_seconds": stage.lens_settle_seconds,

  // Camera.
  "camera": stage.camera,
  "camera_name": stage.camera_name,
  "frame_width": stage.frame_width,
  "frame_height": stage.frame_height,
  "keep_camera_autofocus": stage.keep_camera_autofocus,
  "camera_startup_discard_frames": stage.camera_startup_discard_frames,
  "capture_discard_frames": stage.capture_discard_frames,
  "capture_delay_seconds": stage.settle_seconds,
  "save_raw": false,

  // XY stage and point-alignment safety limits.
  "port": stage.port,
  "baudrate": stage.baudrate,
  "x_lead_mm_per_rev": stage.x_lead_mm_per_rev,
  "y_lead_mm_per_rev": stage.y_lead_mm_per_rev,
  "x_slave_address": stage.x_slave_address,
  "y_slave_address": stage.y_slave_address,
  "speed_rpm": stage.speed_rpm,
  "timeout_seconds": stage.timeout_seconds,
  "position_tolerance_mm": stage.position_tolerance_mm,
  "max_move_mm": stage.max_reset_mm,
  "max_step_mm": 80.0,
  "max_return_mm": 200.0,
  "x_limits": stage.x_limits,
  "y_limits": stage.y_limits,
  "allow_outside_image": false,

  // Every run creates a timestamped child directory below this path.
  "output_dir": "working_data/pcb_two_zoom_capture",
}
