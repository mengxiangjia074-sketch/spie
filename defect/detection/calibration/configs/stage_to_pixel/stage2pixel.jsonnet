// 相机名称和图像尺寸继续引用镜头标定配置。
// 相机选择、自动对焦和位移台通信参数在本文件中直接设置。
local lens = import "../lens/calibrate_lens.jsonnet";

{
  // ---- 相机参数 ----

  // Camera. Set camera to an integer to override camera_name.
  "camera": null,
  "camera_name": lens.camera_name,
  // 图像尺寸, 继承自镜头标定配置。
  "frame_width": lens.frame_width,
  "frame_height": lens.frame_height,
  "camera_startup_discard_frames": 5,
  "capture_discard_frames": 3,
  "settle_seconds": 0.4,
  "keep_camera_autofocus": false,
  "lens_device": null,
  "lens_no_init": false,
  "lens_settle_seconds": 0.5,

  // 从标定汇总文件中精确查找两个 zoom 对应的
  // pcb_autofocus_lens_position.focus。null 使用默认 calibrate.json。
  "calibration": null,

  // ---- 位移台通信参数 ----
  "port": "/dev/lensdetect-stage",
  "baudrate": 38400,
  "x_lead_mm_per_rev": 1.0,
  "y_lead_mm_per_rev": 1.0,
  "x_slave_address": 1,
  "y_slave_address": 2,

  // ---- 本标定独有参数 ----

  // 输出根目录。相对路径以项目根目录为基准, 与运行时所在目录无关。
  "output_root": "working_data/stage_command_to_pixel",

  // 标定开始前先绝对移动到的位置 (mm)。
  // null 表示不先移动, 直接以当前位移台位置为起点开始标定。
  "start_position_mm": null,

  // 两档 zoom 分别使用独立的 X/Y 采样范围。每个步距都会在起点两侧
  // 做 +step/返回、-step/返回的往返采样。
  "zoom_1": 2800,
  "zoom_1_x_step_mm": [2.0, 4.0, 6.0],
  "zoom_1_y_step_mm": [2.0, 4.0, 6.0],
  "zoom_2": 6353,
  "zoom_2_x_step_mm": [2.0, 4.0, 6.0],
  "zoom_2_y_step_mm": [2.0, 4.0],

  // Image-translation registration.
  // roi format: [x, y, width, height]; null uses the full image.
  "roi": null,
  "registration_max_dimension": 1920,
  "registration_use_gradient": true,
  "min_registration_response": 0.1,
  "save_images": true,

  // 位移台运行参数。
  "speed_rpm": 350,
  "timeout_seconds": 30.0,
  "position_tolerance_mm": 0.02,
  "max_step_mm": 10.0,
  "max_reset_mm": 50.0,
  // Optional absolute soft limits: [minimum_mm, maximum_mm].
  "x_limits": null,
  "y_limits": null,
}
