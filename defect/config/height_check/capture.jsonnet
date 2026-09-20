// Carrier-height check configuration.
// Focus and iris are read from the matching pcb_autofocus_lens_position entry.
local stage = import "../../detection/calibration/configs/stage_to_pixel/stage2pixel.jsonnet";

{
  // Set this to an image path to run offline without accessing hardware.
  "image": null,

  // Required clear border as a fraction of the shorter image side.
  "margin": 0.045,

  // null uses detection/calibration/output/calibrate.json.
  "calibration": null,
  "small_zoom": 2800,
  "align_frame_center": true,

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

  // XY stage, used to centre the detected white frame before the final shot.
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
  "x_limits": stage.x_limits,
  "y_limits": stage.y_limits,

  // Every run creates a timestamped child directory below this path.
  "output_dir": "working_data/height_check",
}
