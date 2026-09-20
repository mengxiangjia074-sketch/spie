// Shared checkerboard and camera settings come from the lens calibration.
local lens = import "../lens/calibrate_lens.jsonnet";

{
  // The transform direction saved as the primary 3x3 matrix.
  "source_zoom": 2800,
  "target_zoom": 6353,

  // Focus is read from the matching pcb_autofocus_lens_position entries in
  // detection/calibration/output/calibrate.json.
  "lens_device": null,
  "lens_no_init": false,
  "lens_settle_seconds": 0.5,

  "board_cols": lens.board_cols,
  "board_rows": lens.board_rows,
  // 棋盘格相邻内角点的实际距离，用于计算每个 Zoom 的像素物理长度。
  "square_size_mm": lens.square_size,

  // Live camera settings. Set camera to an integer to override camera_name.
  "camera_name": lens.camera_name,
  "camera": null,
  "frame_width": lens.frame_width,
  "frame_height": lens.frame_height,
  "camera_startup_discard_frames": 5,
  "capture_discard_frames": 3,
  "keep_camera_autofocus": false,

  // false shows a preview for each zoom; press Space/Enter to accept it.
  "capture_now": false,

  // Set both paths to reuse two existing images without camera/lens control.
  // Their zoom comes from the fields above; focus comes from calibrate.json.
  "source_image": null,
  "target_image": null,

  // null uses detection/calibration/output/calibrate.json.
  "calibration": null,

  // Robust source-to-target homography fitting in undistorted pixel space.
  "ransac_threshold_px": 1.5,
  "ransac_iterations": 3000,
  "ransac_confidence": 0.995,
  "min_inlier_ratio": 0.8,
  "max_inlier_rmse_px": 1.5,

  // Each run creates a timestamped child directory here.
  "output_dir": "working_data/zoom_pixel_transform",
}
