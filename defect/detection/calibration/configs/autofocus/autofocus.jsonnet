{
  // LensConnect 设备编号。0 = 0 号设备, null = 自动选择。
  "device": 0,

  // 镜头目标位置。程序会先移动到 zoom/iris, 然后自动搜索 PCB 最清晰的 focus。
  "zoom": 3000,
  "iris": 400,

  // 跳过 lens JSON 中的 zoom/iris, 在当前实际位置搜索 focus。
  "skip_json_zoom": false,
  "skip_json_iris": true,

  // true 表示跳过 focus 清晰度搜索与移动, 在当前实际位置直接拍照;
  // false 才执行自动对焦搜索 (默认)。
  "skip_autofocus": true,

  // 相机。list_cameras 为 true 时只列出当前系统相机设备并退出。
  "camera_name": "CamSPC",
  "list_cameras": false,
  "frame_width": 3840,
  "frame_height": 2160,
  // true 表示尝试关闭相机自身自动对焦, 避免与电动镜头调焦互相干扰。
  "disable_camera_autofocus": true,

  // true 表示禁止自动初始化 LensConnect zoom/iris/focus 电机。
  "no_init": false,

  // focus 搜索上下边界。null 表示使用镜头支持的最小/最大 focus 地址。
  "focus_min": null,
  "focus_max": null,

  // 清晰度评价算法。PCB 推荐 tenengrad。
  "metric": "tenengrad",

  // focus 搜索优化算法: coarse-to-fine 或 golden-section。
  "optimizer": "golden-section",

  // 粗扫点数（coarse-to-fine）。点数越多越不容易漏峰, 但耗时越长。
  "coarse_steps": 15,
  // 细扫点数。coarse-to-fine 下 0 表示跳过细扫。
  "fine_steps": 15,
  // 评分 ROI 缩放后最长边像素上限。0 表示不缩放。
  "score_max_side": 1600,
  // 细扫半径。null 表示自动使用粗扫步距。
  "fine_radius": null,

  // coarse-to-fine 粗扫/细扫提前停止。
  "early_stop": true,
  "early_stop_drop_count": 3,
  "early_stop_min_points": 6,
  "early_stop_min_relative_drop": 0.03,

  // golden-section 搜索参数。
  "golden_tolerance": 8,
  "golden_bracket_steps": 7,
  "golden_tie_relative_margin": 0.002,
  "golden_highres_auto": true,
  "golden_max_iter": 24,

  // 每个 focus 位置采集并评分的帧数。
  "score_frames": 3,
  // 多帧评分聚合方式: median 抗反光/闪烁, mean 适合稳定光源。
  "aggregate": "mean",
  // 每次 focus 移动后丢弃的帧数, 避开相机缓冲旧帧。
  "discard_frames": 5,
  // 打开相机后先丢弃的帧数, 等待曝光/增益/白平衡稳定。
  "startup_discard_frames": 5,
  // focus 移动完成后的等待时间, 单位秒。
  "settle": 0.5,

  // 评分 ROI, 格式 "x,y,width,height"。null 表示使用画面中心 roi_scale 区域。
  "roi": null,
  // true 表示运行前用鼠标交互选择评分 ROI。
  "interactive_roi": false,
  // 交互 ROI 预览缩放后最长边像素上限。0 表示全尺寸。
  "interactive_roi_max_side": 1600,
  // 未指定 roi 时使用画面中心区域的比例。0.8 表示中心 80%。
  "roi_scale": 0.8,
  // 评分前的高斯模糊核大小。1 表示不模糊。
  "blur_ksize": 3,

  // 输出基础目录。每次运行在其下新建当前时间命名的子目录。
  // 相对路径以项目根目录为基准, 与运行时所在目录无关。
  "output_dir": "working_data/pcb_autofocus",
}
