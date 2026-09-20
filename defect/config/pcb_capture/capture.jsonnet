// capture_pcb.py 单 Zoom PCB 网格采集配置:
// 继承配置放前面, 相机、位移台与运动公共参数引用 stage_to_pixel 标定配置;
// 修改标定配置后自动保持一致, 在此不再重复定义。
local stage = import "../../detection/calibration/configs/stage_to_pixel/stage2pixel.jsonnet";

{
  // ---- 继承自 stage2pixel.jsonnet 的相机参数 ----
  // camera 为整数时覆盖 camera_name。
  "camera": stage.camera,
  "camera_name": stage.camera_name,
  "frame_width": stage.frame_width,
  "frame_height": stage.frame_height,
  "camera_startup_discard_frames": stage.camera_startup_discard_frames,
  "keep_camera_autofocus": stage.keep_camera_autofocus,
  // 拍照前等待相机画面稳定(对焦/曝光/振动衰减)的时间。
  // 每次采集前的丢弃帧数与稳定等待时间。
  "capture_discard_frames": stage.capture_discard_frames,
  "capture_delay_seconds": stage.settle_seconds,

  // ---- 继承自 stage2pixel.jsonnet 的位移台参数 ----
  "port": stage.port,
  "baudrate": stage.baudrate,
  "x_lead_mm_per_rev": stage.x_lead_mm_per_rev,
  "y_lead_mm_per_rev": stage.y_lead_mm_per_rev,
  "x_slave_address": stage.x_slave_address,
  "y_slave_address": stage.y_slave_address,
  "speed_rpm": stage.speed_rpm,
  "timeout_seconds": stage.timeout_seconds,
  "position_tolerance_mm": stage.position_tolerance_mm,
  // ---- 标定参数 ----
  // null 直接从 detection/calibration/output/calibrate.json 读取与
  // lens_zoom 匹配的 board_to_pixel、stage_command_to_pixel 与相机内参标定结果；
  // 显式指定路径时三者也都从该文件（calibrate.json 格式的汇总文件）读取。
  "calibration": null,

  // ---- capture_pcb.py 标定选择 ----
  // 镜头需已位于该 zoom；程序只按此值选择蛇形拍摄所需的三类标定。
  "lens_zoom": 6353,

  // ---- capture_pcb.py PCB 网格拍摄参数 ----
  // PCB 长和宽, 单位 mm, 沿标定板 X/Y 方向。
  "pcb_width_mm": 129,
  "pcb_height_mm": 66,
  // 只在 PCB 右边和下边额外拍摄的安全余量。
  "capture_margin_mm": 5.0,

  // 相邻图像最小重叠比例 (0.2 = 20%)。首尾帧与扩展后的覆盖边界对齐,
  // 中间帧等距分布, 保证重叠不低于该比例。
  "overlap_fraction": 0.2,

  // 输出根目录; 每次运行在其中新建时间戳子目录。必须位于 working_data 内。
  "output_dir": "working_data/pcb_picture",

  // 保存原始图与去畸变图。
  "save_raw": false,
  "undistort": true,

  // 单次网格移动允许的最大距离。相邻帧移动约 0.8 倍视野, 需大于该值。
  "max_step_mm": 80.0,
  // 返回初始位置允许的最大距离。需大于整块 PCB 对角线的位移台行程。
  "max_return_mm": 200.0,
}
