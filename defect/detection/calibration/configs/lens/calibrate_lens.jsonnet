{
  // 标定结果 JSON 的默认输出文件名。85×48
  // 相对路径以项目根目录为基准, 与运行时所在目录无关。
  "default_output_file": "working_data/lens_board_calibration.json",

  // 标定详细日志的默认输出文件名。null 表示自动使用 default_output_file 的同名 .log 文件。
  "default_log_file": null,

  // 棋盘格横向内角点数量，不是黑白方格数量。
  "board_cols": 11,

  // 棋盘格纵向内角点数量，不是黑白方格数量。
  "board_rows": 8,

  // 单个棋盘格方格的实际边长。设为 null 时，运行程序必须传入 --square-size。
  "square_size": 3.0,

  // 方格边长单位，仅用于记录到输出 JSON 中，不参与数值计算。
  "square_unit": "mm",

  // 相机设备名称的唯一子串，程序自动查找 Windows DirectShow 或 Linux V4L2 设备。
  "camera_name": "CamSPC",

  // 请求相机输出的图像宽度。null 表示使用相机默认宽度。
  "frame_width": 3840,

  // 请求相机输出的图像高度。null 表示使用相机默认高度。
  "frame_height": 2160,

  // 需要在线采集的有效标定板图像数量。
  "frames": 25,

  // 最少需要的有效标定板图像数量，少于该数量会报错退出。
  "min_frames": 10,

  // 手动采集时是否使用单张拍摄确认模式。true 表示每次只拍一张静态图显示出来，不显示实时视频流。
  "single_shot_capture": false,

  // 实时预览模式下是否只在按 Space/Enter 后才检测棋盘格。true 可减少角点检测导致的视频卡顿。
  "live_preview_detect_on_key": true,

  // 角点检测调试图片输出目录。null 表示不保存调试图片。
  // 相对路径以项目根目录为基准, 与运行时所在目录无关。
  "debug_dir": "working_data/calibrate",

  // 实时采集时保存原始标定板图像的目录。null 表示不保存原始图像。
  "capture_dir": null,

  // 关闭相机自身自动对焦，保证在线采集期间焦距不变。
  "disable_camera_autofocus": true,
}
