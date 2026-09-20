# LensDetect 视觉检测控制台

基于 PySide6 的图形界面, 封装 `detection/calibration` 全部标定功能、
`detection/capture` 主要拍图流程，以及 PnP 坐标到实拍元器件 Mask 的人工对齐流程。
界面编辑各脚本的 jsonnet 配置文件，并以子进程运行脚本、实时转发输出。

## 启动

Linux (`sam` 环境):

```bash
micromamba run -n sam python -m pip install -r requirements-linux.txt
sudo ./linux/install_udev_rules.sh
micromamba run -n sam python detection/hardware_check.py
./linux/run_gui.sh
```

首次安装 udev 规则后，如权限没有立即更新，请重新插拔 LensConnect 和 FT232
位移台串口。相机使用 V4L2，镜头使用项目内的 x86-64 Linux `.so`，位移台使用
Modbus RTU；默认位移台设备为 udev 创建的 `/dev/lensdetect-stage`。

公共依赖: PySide6, opencv-python, Pillow, numpy, jsonnet, minimalmodbus,
pyserial。Windows 按相机名称枚举时还需要 pygrabber，Linux 不需要。

## 功能页

导航是**模式式**的: 启动进入**主界面** (pages/home.py), 六张功能卡片按 3×2 排列,
没有侧边栏; 点击后整屏切换到对应功能界面, 各界面左上有"← 返回主界面"。
**返回主界面时自动断开全部硬件**: 相机、电动镜头 (USB) 与位移台 (串口)
排队断开 (进行中的移动先完成); 若有任务在运行会先确认并强制停止。

- **图像采集** → 采集工作区: 侧边栏 (PCB拍照 / 缺陷模拟 / 坐标 Mask 对齐 / 采集结果) +
  页栈, 任务运行、控制台 Dock、任务互斥逻辑保留在此模式内。
- **相机控制** → 独立的相机预览与参数控制界面: 设备扫描/连接、分辨率与格式、
  拍照/录像、曝光时间与曝光目标、白平衡、色调/饱和度/对比度/伽马调节。
- **标定** → 标定工作区: 脚本位于 detection/calibration/ 下的任务自动归入
  (PCB自动对焦 / 相机内参标定 / Zoom 像素变换标定 / 位移台→像素标定) +
  标定结果页。
- **镜头控制** → 独立的镜头控制界面 (无侧边栏)。
- **位移台控制** → 独立的位移台控制界面 (detection/motion_gui.py 的移植)。
- **RFID检测** → 独立的 E720 RFID 检测界面 (detection/E720_RFID_Tool.py 的
  GUI 移植): 标签识别 (单次寻卡/连续扫描)、标签数据读写 (User 区/EPC 卡号,
  写入自动回读或重扫验证)、功率/地区/信道设置、十六进制调试; 协议实现
  复用原文件, 通讯日志写入 RFID 专属运行控制台。RFID 为独立串口设备, 与检测任务
  不互斥, 返回主界面时自动断开。
- **设置** → 主界面右下角"⚙ 设置"按钮进入独立设置页 (主题/目录/关于)。

## 坐标 Mask 对齐

1. 选择 Altium PnP `.txt`/`.csv` 坐标文件，并选择正面（TopLayer）或反面
   （BottomLayer）生成元器件布局图；反面按实际背面观察方向水平镜像。
2. 拍摄一轮 PCB，或加载最近拍摄结果。程序按实测位移台位置和位移台→像素标定
   拼接大图。
3. 在坐标图和实拍图依次点击至少三组分散且不共线的对应元器件，拟合并预览 Mask。
4. 可在预览中拖动、删除或新增框；导出会保留人工编辑结果。

输出写入本轮拍摄目录的 `mask_alignment/`，包含叠加大图、二值 Mask、16 位实例
Mask、元器件多边形 JSON 和人工对应点 JSON。命令行也可使用：

拼接会用相邻图像内容校正位移台标定位置，并检测 Linux V4L2 旧缓冲帧。升级前
已经拍成重复/滞后帧的历史序列缺少最后一个视野，不能恢复成完整大图，需要重新
执行一次“拍摄一轮”。

```bash
micromamba run -n sam python detection/pnp_mask_workflow.py preview board.txt --layer top --output-dir working_data/pnp_preview
micromamba run -n sam python detection/pnp_mask_workflow.py stitch working_data/pcb_two_zoom_capture/<run>
```

## 相机控制页

相机控制使用独立工作线程持有 OpenCV/UVC 相机，避免预览阻塞界面。驱动不支持的
参数会自动显示为灰色且不可操作，支持的参数显示为白色。拍照和录像结果保存在
`working_data/camera_control`，保存位置写入运行控制台；退出相机控制页或关闭程序时
会停止录像并释放设备。

- **连接与预览**: 扫描相机、选择设备、预览分辨率和像素格式，显示实时画面与 FPS。
- **像素坐标**: 点击预览画面显示原始图像像素坐标，支持逐点撤回和一键清除。
- **采集**: 按当前帧或指定分辨率拍照，使用 MJPG/AVI 录制视频。
- **曝光控制**: 自动曝光开关、可直接编辑的曝光时间 (ms) 和曝光目标值；提交后
  立即写入相机并按驱动回读值更新界面。CamSPC 曝光滑块使用驱动实际支持的
  62.5/125/250/500/1000 ms 五个离散档位，避免回读时滑块跳回。
- **预览缩放**: 支持 25%-400% 逐档缩放、滚轮缩放、适配窗口和放大后拖拽平移，
  像素坐标始终按原始画面计算。
- **颜色调节**: 相机驱动拒绝某个数值时恢复设备实际值并保持控件可用，
  可继续选择其他数值；仅驱动未提供属性时才禁用控件。
- **白平衡**: 自动白平衡、色温/红色分量、一键白平衡（以驱动实际能力为准）。
- **颜色调节**: 色调、饱和度、对比度和伽马值，可恢复连接时的默认值。
- **参数滑块**: 拖动过程中数值框保持不变，释放滑块时一次性更新并提交参数；
  数值框忽略鼠标滚轮，避免滚动控制面板时误改参数。

## 位移台控制页

双轴 Modbus RTU 位移台手动控制, 界面与功能对应 motion_gui.py:

- **连接设置**: 串口枚举/刷新、波特率、X/Y 导程 (mm/转)、细分数 (连接时自动
  读取硬件值)、连接/断开、全局急停。
- **X/Y 轴面板**: 位置 (脉冲 = mm)、状态位 (故障/使能/运行/无效/完成/回零
  完成)、报警显示; 绝对/相对定位 (距离+速度)、急停、正/反点动与停止、
  回零 (有软件零点回相对零点, 无则硬件回零)、设为零点 (软件偏移, 不改硬件)、
  清除报警。
- **通讯日志 → 运行控制台**: 本页不再有独立日志框, 每条 Modbus 读写进入
  位移台专属运行控制台, 按 发送(蓝)/接收(绿)/信息(灰)/错误(红) 着色; 页内保留
  "刷新状态"与 200ms 自动刷新开关。

**运行控制台**: 默认隐藏, 各功能界面的"▤ 运行控制台"按钮点击弹出/收起。
图像采集、相机控制、镜头控制、标定、位移台控制和 RFID 分别保存自己的日志与
弹开状态; 切换功能时 Dock 自动切换到对应内容, 不会混入其他功能日志。任务启动时
只弹出所属功能的控制台, Dock 自身关闭按钮只同步当前功能的按钮状态。

串口通讯全部在 core/motion_worker.py 的 MotionWorker 线程执行 (LoggedMotionController
包装 MotionController 发射日志信号)。适配了新版 motion_controller.py:
位置读取用 read_motor_position_pulses (新 read_motor_position 返回 mm 且每次
读硬件细分数), home(wait=False) 不阻塞队列, 绝对定位先换算硬件坐标 mm。

硬件互斥: 位移台串口与检测任务互斥 (任务运行禁连接, 连接后禁任务启动);
镜头控制为 USB 设备, 可与位移台同时连接。

### 结果浏览的划分

- **标定结果** (标定工作区): 只展示 working_data 下的 `calibrate`、
  `pcb_autofocus`、`zoom_pixel_transform`、`stage_command_to_pixel` 和
  `detection/calibration/output`，附经过过滤的标定汇总表。板→像素数据仍保留给
  采集流程使用，但不在 GUI 任务及结果中显示。
- **采集结果** (图像采集工作区): working_data 下**其余全部**内容
  (pcb_mosaic、pcb_picture、gui_logs、gui_backups 等, 动态计算, 新目录
  自动出现), 不含标定汇总表。

| 页面 | 工作区 | 脚本 | 配置 |
|---|---|---|---|
| PCB 全图/局部图拍照 | 图像采集 | capture/capture_pcb_two_zooms.py | config/pcb_two_zoom_capture/capture.jsonnet |
| 采集结果 | 图像采集 | — | working_data 其余内容浏览 |
| PCB 自动对焦 | 标定 | calibration/autofocus_pcb.py | calibration/configs/autofocus/autofocus.jsonnet |
| 相机内参标定 | 标定 | calibration/calibrate_lens.py | calibration/configs/lens/calibrate_lens.jsonnet |
| Zoom 像素变换标定 | 标定 | calibration/calibrate_zoom_pixel_transform.py | calibration/configs/zoom_pixel_transform/calibrate.jsonnet |
| 位移台→像素标定 | 标定 | calibration/calibrate_stage_to_pixel.py | calibration/configs/stage_to_pixel/stage2pixel.jsonnet |
| 标定结果 | 标定 | — | 标定结果目录浏览 + calibrate.json 汇总 |
| 相机控制 | 相机控制 | (无脚本, 交互式 OpenCV/UVC 控制) | (无配置) |

相机内参标定仅使用实时相机在线采集棋盘格图像，不提供离线图片、LensConnect
镜头控制或自动对焦流程。
| 镜头控制 | 镜头控制 | (无脚本, 交互式硬件控制) | (无配置) |
| 位移台控制 | 位移台控制 | (无脚本, 交互式串口控制) | (无配置) |

设置(深浅主题切换/目录/关于) 从主界面右下角进入。总览页已移除
(pages/dashboard.py 保留未引用)。

## 镜头控制页

厂商程序 `LensConnect_Windows_GUI_x86_2.2.0.exe` 的功能移植, 原样复用
`LensCamera/LensConnect_Controller` SDK, 不新增配置文件:

- **手动控制**: 扫描设备/按序列号连接; zoom / focus / iris 滑条 + 步进 +
  Go to + 初始化; 光学滤镜 (IRCF) 档位切换; 1 s 周期轮询状态位与当前位置。
- **参数设置**: 各电机速度 (PPS) 与间隙补偿开关写入。
- **镜头信息**: 型号/固件/协议/地址/能力位/状态位, 温度刷新, 用户标识读写,
  "详细信息" 对话框 (位置/机械范围、初始化位置、计数等深层寄存器)。
- **预设位**: 4 组位置各存 zoom/focus/iris/滤镜/等待时间, 支持读取当前位、
  按序执行、停止、导出/导入 txt; 内容持久化在 QSettings。

与任务页的"配置 + 子进程"模式不同, 本页是交互式控制: `core/lens_worker.py`
的 QThread 串行执行全部阻塞式 SDK 调用 (USB 读一次 0.1 s, 移动要轮询到位),
空闲轮询失败按连接丢失处理。镜头是独占硬件: 连接镜头后不能启动任务
(app.py 拦截), 任务运行中不能连接镜头 (页面拦截)。

## 设计要点

- **配置编辑是外科手术式的**: 只替换被修改键的值, 注释、jsonnet `import`
  表达式与文件布局保持原样; 写入后立即重载校验全部键值, 失败自动还原;
  每次保存前把旧版本备份到 `working_data/gui_backups/<配置名>/`。
- **子进程运行**: 脚本以 `python -u` 运行于项目根目录 (UTF-8 输出),
  stdout/stderr 彩色实时显示在底部控制台, 完整日志落盘
  `working_data/gui_logs/`。
- **硬件独占**: 相机/位移台/电动镜头同一时间只允许一个任务, 运行中所有
  "保存并运行"按钮禁用。
- **交互窗口**: 标定采集等交互仍由脚本自己的 OpenCV 窗口完成 (快捷键提示
  见各页说明与控制台首行)。
- **停止按钮为强制终止**: 位移台运动中请等当前动作结束再停止。

## 扩展新功能

新脚本接入只需两步:

1. 在 `core/paths.py` 的 `SCRIPTS`/`CONFIGS` 里加一条路径;
2. 在 `pages/registry.py` 的 `TASKS` 里加一个 `TaskSpec` (声明 id、标题、
   脚本、配置和 `F(key, label, kind=...)` 字段列表)。

导航项、设置表单、配置写回、子进程运行、日志全部自动生成。
表单字段类型: `bool / int / float / str / enum / int_list / float_list /
vec2 / int_null / float_null / str_null / vec2_null`。

## 目录结构

```
GUI/
├── main.py            入口
├── app.py             主窗口 (模式栈: 主界面/图像采集/相机控制/标定/镜头控制/位移台/RFID + 控制台 Dock)
├── core/
│   ├── paths.py       路径常量
│   ├── theme.py       主题 (深/浅) 与 QSS
│   ├── jsonnet_store.py  jsonnet 外科手术式读写 + 校验回滚
│   ├── runner.py      QProcess 任务运行器 (单任务互斥 + 日志落盘)
│   ├── camera_worker.py OpenCV/UVC 相机预览与采集线程
│   ├── lens_worker.py LensConnect 镜头控制线程 (镜头控制页专用)
│   └── motion_worker.py 位移台控制线程 (位移台控制页专用)
├── widgets/
│   ├── forms.py       数据驱动设置表单
│   ├── console.py     运行控制台
│   └── resultview.py  汇总表 / JSON 树 / 图像查看
└── pages/
    ├── home.py        主界面 (六个功能入口)
    ├── registry.py    任务注册表 (新功能在这加)
    ├── task_page.py   通用任务页
    ├── camera_control.py 相机控制页 (预览/拍照/录像/参数调节)
    ├── lens_control.py 镜头控制页 (厂商 GUI 移植, 非任务页)
    ├── motion_control.py 位移台控制页 (motion_gui.py 移植, 非任务页)
    ├── dashboard.py   总览
    ├── results.py     结果浏览
    └── settings_page.py  设置与关于
```
