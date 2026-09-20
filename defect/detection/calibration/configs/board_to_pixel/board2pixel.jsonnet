// 共享的标定板几何与相机参数 (board_cols、board_rows、square_size、
// camera_name、frame_width、frame_height) 直接引用 lens 标定配置,
// 在此不再重复定义; 修改 lens 配置后两个标定会自动保持一致。
local lens = import "../lens/calibrate_lens.jsonnet";

{
  // 标定板几何与相机参数, 继承自 calibrate_lens.jsonnet。
  "board_cols": lens.board_cols,
  "board_rows": lens.board_rows,
  "square_size": lens.square_size,
  "camera_name": lens.camera_name,
  "frame_width": lens.frame_width,
  "frame_height": lens.frame_height,
  "startup_discard_frames": 5,
  "lens_device": null,

  // 标定板原点 (板坐标系) 与坐标轴方向。
  "origin_x_mm": 0.0,
  "origin_y_mm": 0.0,
  "column_axis": "+x",
  "row_axis": "+y",

  // Robust affine fitting.
  "ransac_threshold_px": 1.5,
  "ransac_iterations": 2000,
  "min_inlier_ratio": 0.7,
  "random_seed": 20260817,

  // Camera. Set camera to an integer to override camera_name.
  "camera": null,
  "keep_camera_autofocus": false,
  "capture_now": false,

  // XY-stage position recorded with the board calibration.
  "record_stage_position": true,
  "port": "/dev/lensdetect-stage",
  "baudrate": 38400,
  "x_lead_mm_per_rev": 1.0,
  "y_lead_mm_per_rev": 1.0,
  "x_slave_address": 1,
  "y_slave_address": 2,

  // null creates timestamped files below working_data/board_pixel_affine.
  "output": null,
  "log_output": null,
  "annotated_output": null,
}
