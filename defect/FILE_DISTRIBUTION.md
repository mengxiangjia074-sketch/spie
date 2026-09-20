# `defect` 文件夹文件分布与用途

> 统计范围：`defect/` 当前工作区（2026-09-12）。
> 说明：`__pycache__`、`.pytest_cache`、`.ruff_cache` 属于解释器/工具缓存，不纳入源码职责说明；
> `working_data` 中按时间戳生成的大量图片和日志按文件模式归纳。

## 1. 总体分布

`defect` 共约 **3,638 个文件**：

| 目录/文件组 | 文件数 | 内容与作用 |
|---|---:|---|
| `detection/` | 325 | 检测程序、GUI、硬件适配、标定脚本及内置 DINOv3/RoMaV2/SAM3 运行时 |
| `working_data/` | 2,891 | 采集、拼图、Mask 对齐、缺陷模拟、元件检测的运行产物 |
| `test/` | 32 | 单元测试、GUI 测试和硬件流程测试 |
| `models/` | 4 | 3 个模型权重和模型说明 |
| `config/` | 3 | PCB 拍摄/高度检查的 Jsonnet 配置 |
| `linux/` | 3 | udev 规则、安装脚本和 GUI 启动脚本 |
| `third_party_licenses/` | 3 | DINOv3、RoMaV2、SAM3 许可证 |
| 根目录其他文件 | 4 | 依赖清单、示例图片、本说明文档和本地 Claude 设置 |
| 缓存（上述目录中的 `__pycache__` 等） | 约 373 | Python 字节码及测试/格式化缓存，可删除后自动生成 |

非缓存文件按类型统计：313 个 `.py`、109 个 `.json`、11 个 `.yaml`、11 个 `.jsonnet`、2,750 个图像/模型等二进制或其他资源。

## 2. 顶层文件与配置

| 路径 | 作用 |
|---|---|
| `requirements-linux.txt` | Linux/SAM 环境依赖：PySide6、OpenCV、PyTorch、Jsonnet、串口/Modbus、图像模型依赖等。 |
| `image.png` | 项目/界面使用的 PNG 图片资源（1370×879 RGBA）。 |
| `.claude/settings.local.json` | 本地 Claude 工具设置，不参与业务运行。 |
| `config/height_check/capture.jsonnet` | 白色矩形框高度检查的相机、镜头、位移台和采集参数。 |
| `config/pcb_capture/capture.jsonnet` | 单倍率/传统 PCB 拍摄流程参数。 |
| `config/pcb_two_zoom_capture/capture.jsonnet` | 双倍率 PCB 全图与局部图拍摄、拼接、边距和输出参数。 |
| `linux/99-lensdetect.rules` | udev 规则：授予 LensConnect HID 和 FT232 位移台权限，并创建 `/dev/lensdetect-stage` 别名。 |
| `linux/install_udev_rules.sh` | 安装并重新加载上述 udev 规则。 |
| `linux/run_gui.sh` | 进入项目目录，在 `sam` micromamba 环境启动 `detection/GUI/main.py`。 |
| `third_party_licenses/DINOv3-LICENSE` | DINOv3 第三方许可证。 |
| `third_party_licenses/RoMaV2-LICENSE` | RoMaV2 第三方许可证。 |
| `third_party_licenses/SAM3-LICENSE` | SAM3 第三方许可证。 |

## 3. 模型权重

| 路径 | 大小（约） | 作用 |
|---|---:|---|
| `models/component_classifier/best_resnet18_component_classifier.pt` | 134 MB | 元件候选 Patch 的 ResNet18 正/负分类器。 |
| `models/romav2/romav2.0.1.pt` | 1.10 GB | RoMaV2 稠密特征匹配和图像配准权重。 |
| `models/sam3/sam3.pt` | 3.45 GB | SAM3 文本提示分割模型权重。 |
| `models/README.md` | — | 说明权重位置及运行时默认加载路径。 |

## 4. `detection/` 业务代码

### 4.1 顶层脚本

| 文件 | 作用 |
|---|---|
| `defect_sim.py` | 缺陷模拟：在 PCB 局部图中框选元件并移除；支持 Codex 图像生成、Images/Responses API 和本地 OpenCV inpaint 后端。 |
| `inspect_components.py` | 命令行元件完整性检测入口，串联 SAM3 分割、ResNet 分类、RoMaV2 配准和缺件报告。 |
| `height_check.py` | 采集小倍率全图，检测白色矩形载具框是否完整可见，以判断 PCB 载台高度。 |
| `hardware_check.py` | 启动前检查相机、LensConnect 镜头、位移台串口和模型/依赖是否可用。 |
| `motion_controller.py` | 双轴 Modbus RTU 位移台底层控制：连接、状态读取、绝对/相对移动、点动、回零、急停和报警。 |
| `pnp_layout.py` | 解析 Altium PnP 坐标文件，按 Top/Bottom 层生成元件布局及坐标变换。 |
| `pnp_mask_workflow.py` | PnP 坐标布局与实拍 PCB 拼图的 Mask 对齐工作流：拼接、对应点拟合、编辑和导出。 |
| `E720_RFID_Tool.py` | E720 RFID 协议实现及命令行/基础 GUI 工具：寻卡、连续扫描、EPC/User 区读写、功率和信道设置。 |

### 4.2 GUI（`detection/GUI/`，37 个非缓存文件）

GUI 是 PySide6 控制台。`main.py` 启动应用，`app.py` 管理模式和硬件互斥，页面负责具体功能，`core/` 负责后台线程/配置/任务，`widgets/` 提供通用控件，`images/` 是导航图标。

| 文件 | 作用 |
|---|---|
| `GUI/main.py` | GUI 进程入口。 |
| `GUI/app.py` | 主窗口、模式栈、导航、硬件互斥和运行控制台 Dock。 |
| `GUI/README.md` | GUI 启动、页面功能、输出目录和扩展方式说明。 |
| `GUI/__init__.py` | GUI 包标记。 |
| `GUI/core/camera_worker.py` | 相机后台线程：V4L2/UVC 连接、预览、拍照、录像和参数控制。 |
| `GUI/core/lens_worker.py` | LensConnect 阻塞式 USB 调用线程，串行执行移动、初始化、状态轮询。 |
| `GUI/core/motion_worker.py` | 位移台 Modbus 工作线程，执行移动/状态命令并发出通讯日志。 |
| `GUI/core/rfid_worker.py` | E720 RFID 工作线程，复用协议实现并记录收发日志。 |
| `GUI/core/runner.py` | 以子进程运行标定/采集脚本，实时转发 stdout/stderr，限制任务互斥。 |
| `GUI/core/jsonnet_store.py` | Jsonnet“手术式”单键编辑、校验、失败回滚和备份。 |
| `GUI/core/paths.py` | GUI、项目根目录、脚本、配置和运行数据路径常量。 |
| `GUI/core/theme.py` | 深色/浅色主题和 token 化 QSS。 |
| `GUI/core/__init__.py` | core 包标记。 |
| `GUI/pages/home.py` | 主界面六张功能卡片及入口。 |
| `GUI/pages/camera_control.py` | 相机连接、预览、曝光/白平衡/颜色参数、拍照和录像页面。 |
| `GUI/pages/lens_control.py` | 镜头扫描、Zoom/Focus/Iris/滤镜、预设位和镜头信息页面。 |
| `GUI/pages/motion_control.py` | 双轴位移台连接、定位、点动、回零、急停和状态页面。 |
| `GUI/pages/rfid_control.py` | RFID 寻卡、读写、功率/地区/信道及十六进制调试页面。 |
| `GUI/pages/task_page.py` | 通用“配置编辑 + 保存并运行 + 控制台”任务页面基类。 |
| `GUI/pages/component_inspection_page.py` | 元件完整性检测任务页面。 |
| `GUI/pages/defect_sim_page.py` | 缺陷模拟任务页面。 |
| `GUI/pages/pnp_mask_page.py` | PnP 坐标 Mask 对齐和预览导出页面。 |
| `GUI/pages/results.py` | 采集/标定结果目录浏览和结果预览。 |
| `GUI/pages/settings_page.py` | 主题、目录和关于设置页面。 |
| `GUI/pages/dashboard.py` | 旧版总览页面，当前导航未引用但保留兼容。 |
| `GUI/pages/registry.py` | 任务规格（脚本、配置、字段）注册表。 |
| `GUI/pages/__init__.py` | pages 包标记。 |
| `GUI/widgets/forms.py` | Jsonnet 字段的布尔、数值、枚举、向量等表单控件。 |
| `GUI/widgets/console.py` | 彩色运行控制台和日志 Dock 控件。 |
| `GUI/widgets/resultview.py` | 图片、JSON/CSV 等采集结果查看控件。 |
| `GUI/widgets/__init__.py` | widgets 包标记。 |
| `GUI/images/RFID.svg`、`图像采集.svg`、`相机控制.svg`、`标定.svg`、`镜头.svg`、`位移台控制.svg` | 主界面功能卡片图标。 |

### 4.3 相机与电动镜头（`detection/LensCamera/`）

| 文件/目录 | 作用 |
|---|---|
| `LensCamera/__init__.py` | 对外暴露镜头连接、运动、状态和相机辅助 API。 |
| `camera.py` | OpenCV 相机工具加载、设备枚举、帧读取和图像处理辅助。 |
| `lensconnect_adapter.py` | LensConnect SDK 适配器，实现连接重试、能力检测、初始化、读写位置。 |
| `lens_ports.py` | 镜头控制端口抽象接口和异常类型。 |
| `lens_config.py` | 镜头电机顺序、默认配置文件和设备编号解析。 |
| `settings.py` | Jsonnet 设置加载、类型校验和错误封装。 |
| `json_io.py` | JSON 对象读写和结构校验小工具。 |
| `paths.py` | LensCamera 包、默认配置和运行数据路径常量。 |
| `sampling.py` | 在电机地址范围内生成标定采样点。 |
| `timestamps.py` | UTC、本地和运行批次时间戳格式化。 |
| `LensConnect_Controller/*.py` | 厂商 LensConnect Python SDK：`LensCtrl.py` 电机控制，`LensInfo.py` 信息/寄存器，`LensSetup.py` 初始化和能力，`LensAccess.py` 高层访问，`UsbCtrl.py` USB 通讯，`ConfigVal.py`/`DefVal.py`/`DevAddr.py` 常量和寄存器地址，`LensConnect_Controller.py` 控制器封装，`SLABHIDDevice.py`/`SLABHIDtoSMBUS.py` Silicon Labs HID-SMBus 绑定。 |
| `LensConnect_Controller/Arm7_lib/` | ARM/通用 Linux 版本的 HID-SMBus Python 绑定及 `libslabhid*.so.1.0` 动态库。 |
| `LensConnect_Controller/x86_lib/` | Windows/x86 兼容版本的 Python 绑定及 `.dll`/`.lib` 库。 |
| `LensConnect_Controller/*.dll`、`*.so.1.0` | 厂商 USB HID-SMBus 原生驱动库，运行时按平台加载。 |

### 4.4 标定与拍摄

| 文件 | 作用 |
|---|---|
| `calibration/autofocus_pcb.py` | 在指定 Zoom 下扫描 Focus，计算清晰度并保存 PCB 自动对焦位置。 |
| `calibration/calibrate_lens.py` | 在线棋盘格相机内参/畸变标定，按镜头 Zoom 保存相机矩阵。 |
| `calibration/calibrate_zoom_pixel_transform.py` | 两个 Zoom 的棋盘格图像配准，拟合像素坐标单应矩阵和毫米/像素比例。 |
| `calibration/calibrate_stage_to_pixel.py` | 采样位移台运动与图像位移，拟合 stage 命令到像素的变换。 |
| `calibration/calibrate_board_to_pixel.py` | 用 PCB 板角点建立物理板坐标到像素坐标映射。 |
| `calibration/calibration_summary.py` | 将各标定脚本结果按类型/Zoom 汇总到 `calibration/output/calibrate.json`。 |
| `calibration/run_height_check.py` | 标定工作区调用顶层 `height_check.py` 的入口。 |
| `calibration/output/calibrate.json` | 当前共享标定摘要（schema_version=1），供拍摄和检测流程读取。 |
| `calibration/configs/*/*.jsonnet` | 各标定任务参数：`autofocus.jsonnet`、`calibrate_lens.jsonnet`、`zoom_pixel_transform.jsonnet`、`stage2pixel.jsonnet`、`board2pixel.jsonnet`。 |
| `capture/capture_pcb.py` | 单倍率 PCB 图像采集、白框/边距处理和基础拼图。 |
| `capture/capture_pcb_two_zooms.py` | 先拍小倍率全板、再按网格拍大倍率局部图，并生成拼图元数据。 |

### 4.5 元件完整性检测（`detection/component_inspection/`）

处理链为：SAM3 文本分割 → 候选 Mask 清洗/拆分 → ResNet18 正负分类 → RoMaV2 图像配准 → 几何覆盖和一对一真值匹配 → 报告。

| 文件 | 作用 |
|---|---|
| `component_inspection/config.py` | `InspectionConfig`、输入图像发现、模型路径和阈值配置。 |
| `component_inspection/ground_truth.py` | 读取 GUI 导出的 `component_masks.json` 真值元件和多边形。 |
| `component_inspection/segmentation.py` | 调用 SAM3 文本提示分割并进行面积过滤、拆分、去重。 |
| `component_inspection/classifier.py` | 透视矫正候选 Mask Patch，调用训练好的 ResNet18 分类器。 |
| `component_inspection/registration.py` | 用 RoMaV2/DINOv3 特征估计并验证高倍率图到真值图的单应矩阵。 |
| `component_inspection/geometry.py` | 坐标映射、跨视野重叠消除、覆盖比例和一对一元件分配。 |
| `component_inspection/pipeline.py` | 汇总多张高倍率图，输出 JSON、CSV、候选/接受叠加图和缺件结果。 |
| `component_inspection/__init__.py` | 导出配置、真值加载和公共 API。 |
| `component_inspection/README.md` | 命令行用法和四阶段流程说明。 |

### 4.6 内置 RoMaV2 运行时（24 个非缓存文件）

这是随项目内置的稠密匹配模型实现，主要由元件检测的配准阶段调用。

| 文件组 | 作用 |
|---|---|
| `romav2/romav2.py`、`matcher.py`、`refiner.py` | RoMaV2 主模型、粗匹配和局部精炼网络。 |
| `romav2/features.py`、`geometry.py`、`types.py` | 特征提取、坐标/单应几何、数据类型和结果结构。 |
| `romav2/local_correlation.py`、`normalizers.py` | 局部相关计算和 ImageNet 归一化。 |
| `romav2/dpt.py` | DPT 风格预测头/特征解码组件。 |
| `romav2/io.py`、`vis.py` | PIL/NumPy 图像 IO 和匹配可视化。 |
| `romav2/device.py`、`logging.py`、`__init__.py` | 设备选择、日志配置和包初始化。 |
| `romav2/vit/attention.py`、`block.py`、`ffn_layers.py`、`layer_scale.py`、`patch_embed.py`、`rms_norm.py`、`rope.py`、`rope_mixed.py`、`utils.py`、`__init__.py` | RoMaV2 使用的 ViT 基础层、注意力、FFN、位置编码和工具。 |

### 4.7 内置 SAM3 运行时（52 个非缓存文件）

| 文件组 | 作用 |
|---|---|
| `sam3/model_builder.py`、`__init__.py` | 构建并导出 SAM3 图像模型。 |
| `sam3/model/encoder.py`、`decoder.py`、`necks.py`、`vitdet.py` | 图像编码器、掩码解码器、特征颈部和 ViTDet 主干。 |
| `sam3/model/sam3_image.py`、`sam3_image_processor.py`、`sam1_task_predictor.py` | 单图推理、预处理和 SAM1 兼容预测器。 |
| `sam3/model/sam3_tracker_base.py`、`sam3_tracker_utils.py`、`sam3_tracking_predictor.py`、`sam3_video_base.py`、`sam3_video_inference.py`、`sam3_video_predictor.py` | 视频/跟踪接口（当前 PCB 检测主要使用图像接口）。 |
| `sam3/model/geometry_encoders.py`、`position_encoding.py`、`memory.py`、`edt.py`、`masks_ops.py`、`box_ops.py`、`data_misc.py`、`model_misc.py`、`io_utils.py` | 几何编码、位置编码、记忆、距离变换、Mask/框运算、数据和 IO 工具。 |
| `sam3/model/maskformer_segmentation.py`、`vl_combiner.py`、`text_encoder_ve.py`、`tokenizer_ve.py` | MaskFormer 分割、视觉-语言融合、文本编码和分词。 |
| `sam3/model/act_ckpt_utils.py`、`model/utils/misc.py`、`sam1_utils.py`、`sam2_utils.py`、`model/__init__.py` | 激活检查点、兼容层和模型工具。 |
| `sam3/perflib/*.py`、`perflib/triton/*.py` | 高性能 NMS、连通域、Mask 运算、检测跟踪关联及 Triton 实现。 |
| `sam3/sam/common.py`、`mask_decoder.py`、`prompt_encoder.py`、`rope.py`、`transformer.py`、`__init__.py` | SAM 基础提示编码、Transformer、位置编码和掩码解码器。 |
| `sam3/logger.py`、`visualization_utils.py` | 日志和可视化辅助。 |
| `sam3/assets/bpe_simple_vocab_16e6.txt.gz` | 文本编码器使用的 BPE 词表。 |

### 4.8 内置 DINOv3 运行时（144 个非缓存文件）

以下是 DINOv3 的训练/评估代码副本，RoMaV2 配准会复用其视觉特征；大部分文件不是 PCB 业务入口。

| 子目录/文件组 | 作用 |
|---|---|
| `dinov3/models/vision_transformer.py`、`convnext.py`、`models/__init__.py` | ViT/ConvNeXt 主干模型定义。 |
| `dinov3/layers/*.py` | 注意力、Transformer Block、FFN、Patch Embedding、RMSNorm、RoPE、量化/稀疏线性层和 DINO Head。 |
| `dinov3/checkpointer/checkpointer.py`、`checkpointer/__init__.py` | checkpoint 加载、恢复和状态字典处理。 |
| `dinov3/configs/config.py`、`configs/*.yaml`、`configs/train/**/*.yaml` | SSL 默认配置及各 ViT 预训练/蒸馏训练配置。 |
| `dinov3/data/*.py`、`data/datasets/*.py` | 数据增强、变换、Mask、采样、批处理、数据加载器和 ImageNet/COCO/ADE20K 数据集适配。 |
| `dinov3/distributed/*.py`、`env/__init__.py`、`fsdp/ac_compile_parallelize.py` | 分布式 PyTorch 原语、环境初始化、FSDP/编译并行。 |
| `dinov3/train/*.py` | SSL/多蒸馏训练入口、元架构、参数分组和余弦学习率。 |
| `dinov3/loss/*.py` | DINO class token、iBOT patch、Gram 和 KoLeo 损失。 |
| `dinov3/eval/linear.py`、`knn.py`、`log_regression.py`、`metrics/*.py`、`helpers.py`、`results.py`、`data.py`、`accumulators.py`、`utils.py` | 线性/knn/逻辑回归评估、分类指标、结果汇总和评估工具。 |
| `dinov3/eval/detection/**/*.py` | DETR 检测配置、backbone、Transformer/位置编码、解码器、窗口和 box/misc 工具。 |
| `dinov3/eval/segmentation/**/*.py` | Mask2Former 分割推理、主干适配器、像素/Transformer 解码器、位置编码和多尺度可变形注意力。 |
| `dinov3/eval/dense/depth/**/*.py` | DPT 深度估计头、编码器、embedding 和工具。 |
| `dinov3/eval/text/**/*.py` | DINO 文本塔、视觉塔、Tokenizer、CLIP/Gram 损失、训练和配置。 |
| `dinov3/hub/*.py` | backbone、classifier、detector、segmentor、depther、DinoTxt 的构建/导出接口。 |
| `dinov3/run/*.py` | 训练任务初始化和提交辅助。 |
| `dinov3/utils/*.py`、`logging/*.py`、`thirdparty/CLIP/clip/simple_tokenizer.py` | 通用工具、dtype/cluster、日志和 CLIP 简单分词器。 |
| `dinov3/eval/segmentation/models/utils/ops/src/*.{cpp,h,cu,cuh}`、`setup.py`、`ops/test.py` | 多尺度可变形注意力的 C++/CUDA 扩展源码、编译脚本和测试。 |

## 5. `test/` 测试文件（32 个）

| 文件组 | 覆盖内容 |
|---|---|
| `test_camera_control.py`、`test_rfid.py`、`test_motion_controller.py`、`test_motion_control_connection.py`、`test_motion_control_status.py`、`test_lens_control_step.py`、`test_lens_worker_calibrated_positions.py` | 相机、RFID、位移台和镜头硬件控制/工作线程。 |
| `test_capture_pcb_calibration.py`、`test_capture_pcb_margin.py`、`test_two_zoom_capture.py`、`test_capture_optical_center.py` | PCB 采集、双倍率拼图、边距和光学中心。 `capture_optical_center.py` 是测试辅助脚本。 |
| `test_autofocus_calibration_summary.py`、`test_board_calibration_summary.py`、`test_stage_calibration_summary.py`、`test_stage_config_source.py`、`test_zoom_pixel_transform.py` | 标定结果汇总、配置来源和 Zoom 像素变换。 |
| `test_pnp_mask_workflow.py`、`test_component_inspection.py` | PnP Mask 对齐和元件完整性检测流程。 |
| `test_gui_startup.py`、`test_home_background.py`、`test_gui_theme.py`、`test_gui_numeric_wheel.py`、`test_gui_console_separation.py`、`test_gui_calibration_visibility.py`、`test_gui_component_inspection_capture.py`、`test_gui_defect_sim.py`、`test_gui_lens_online_only.py`、`test_gui_mask_editor.py`、`test_gui_two_zoom_capture.py`、`test_height_check_gui.py` | GUI 启动、主题、导航、控制台、任务互斥、各页面和编辑器行为。 |
| `test_jsonnet_store.py` | Jsonnet 单键编辑、注释保留、校验和回滚。 |

## 6. `working_data/` 运行产物

`working_data` 共 **2,891 个文件**（约 2,749 个 PNG、106 个 JSON、10 个 CSV、23 个日志、3 个 Jsonnet 备份）。目录名通常为 `YYYYMMDD_HHMMSS`，表示一次运行。

### 6.1 元件检测：`working_data/component_inspection/`

共有 10 批运行（9 个时间戳批次 + `gpu_smoke_full_capture`），共 2,572 个文件。每批通常包含四个视野 `r00_c00`、`r00_c01`、`r01_c00`、`r01_c01`：

| 文件/目录模式 | 作用 |
|---|---|
| `*/inspection_report.json` | 整批检测状态、缺失元件 ID、统计和输出文件索引。 |
| `*/component_status.csv` | 每个真值元件的 found/missing、覆盖率、匹配图像等表格结果。 |
| `*/component_completeness_overlay.png` | 在 PCB 真值图上标注元件完整性/缺件结果。 |
| `*/detections/<view>/detections.json` | 每个高倍率视野的候选/接受 Mask、分类分数和坐标。 |
| `*/detections/<view>/{sam_mask_overlay,candidate_overlay,accepted_overlay}.png` | SAM 原始 Mask、候选过滤和最终接受结果可视化。 |
| `*/detections/<view>/patches/positive/*.png` | 被分类器判为正样本的元件 Patch。 |
| `*/detections/<view>/patches/negative/*.png` | 可选保存的负样本/被过滤 Patch。 |
| `*/detections/<view>/component_*.png` | 带 SAM 置信度和分类概率的单元件 Patch。 |

### 6.2 双倍率 PCB 拍摄：`working_data/pcb_two_zoom_capture/`

共有 22 个运行目录（262 个文件）。

| 文件/目录模式 | 作用 |
|---|---|
| `<run>/pcb_two_zoom_capture.json` | 采集参数、相机/镜头/位移台位置、视野清单和时间信息。 |
| `<run>/images/rXX_cYY_undistorted.png` | 大倍率视野的去畸变图像，`r/c` 表示网格行列。 |
| `<run>/small_zoom_image/small_undistorted.png` | 小倍率全板去畸变图。 |
| `<run>/small_zoom_image/small_global_cropped.png` | 去除边缘后的全局裁剪图。 |
| `<run>/small_zoom_image/small_alignment_before_*.png` | Mask 对齐前的原图、标注图和中间 Mask。 |
| `<run>/small_zoom_image/small_final_annotated.png`、`small_final_mask.png` | 对齐后的最终标注图和 Mask。 |
| `<run>/pcb_mosaic.json`、`pcb_stitched.json` | 拼图位置、校正位移和拼接结果元数据。 |
| `<run>/pcb_mosaic.png`、`pcb_stitched.png`（部分批次） | 大倍率视野拼接图/内容校正后的拼接图。 |
| `<run>/mask_alignment/alignment_points.json` | 坐标布局图与实拍图的人工对应点。 |
| `<run>/mask_alignment/component_masks.json` | GUI 导出的元件多边形/实例 Mask 真值，供完整性检测使用。 |

### 6.3 PnP Mask 对齐：`working_data/pnp_mask_alignment/`

共有 103 个时间戳目录。多数是交互尝试留下的空目录；已产出文件的运行目录包含：

| 文件/目录模式 | 作用 |
|---|---|
| `<run>/pnp_components.json` | 解析后的 PnP 元件坐标、层信息和像素布局。 |
| `<run>/pnp_component_preview.png` | PnP 元件布局预览图。 |
| `<run>/preview/component_masks.json` | 预览阶段生成的组件 Mask。 |
| `<run>/preview/*.png` | 对齐/Mask 预览图。 |

### 6.4 其他运行数据

| 路径 | 数量/内容 | 作用 |
|---|---|---|
| `working_data/defect_simulation/` | 2 个 JSON 及生成图片 | 缺陷模拟输入框、后端、提示词和输出文件记录。 |
| `working_data/gui_backups/` | 3 个 Jsonnet 备份 | GUI 保存配置前自动备份的旧版本，按配置名分目录。 |
| `working_data/gui_logs/` | 23 个 `.log` | GUI 子进程和拼图任务的完整 stdout/stderr 日志。 |

## 7. 典型数据流

```text
标定脚本
  └─> working_data/<calibration-run>/ + detection/calibration/output/calibrate.json

双倍率拍摄
  └─> working_data/pcb_two_zoom_capture/<run>/images + small_zoom_image + pcb_mosaic.json

PnP 对齐
  └─> pnp_components.json + component_masks.json

元件完整性检测
  └─> SAM3 分割 -> ResNet18 分类 -> RoMaV2/DINOv3 配准
      └─> inspection_report.json + component_status.csv + overlays/patches
```

### 维护建议

- 不要手动编辑模型权重、厂商动态库或 `__pycache__`；模型路径由 `models/README.md` 和检测配置约定。
- `working_data` 是可再生运行产物，清理时优先按时间戳批次删除，并保留 `inspection_report.json`、`component_status.csv` 等需要追溯的结果。
- 新增 GUI 任务时，在 `GUI/core/paths.py` 注册脚本/配置，再在 `GUI/pages/registry.py` 添加 `TaskSpec`。
