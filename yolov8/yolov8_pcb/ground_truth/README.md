# PCB 人工标注 Ground Truth

本目录保存独立于 SAM3 和 YOLOv8-Seg 预测结果的人工标注，用于训练集构建和两种方法的公平评估。

## 目录结构

```text
ground_truth/
├── images/       # 待标注的原始 PCB 图像
├── annotations/  # Labelme/CVAT 导出的原始人工标注
├── yolo/         # 转换后的 YOLOv8-Seg 数据集
└── README.md
```

## 标注要求

- 每个 PCB 元件必须作为一个独立实例标注。
- 所有元件统一使用类别名 `component`。
- 使用多边形沿元件主体边缘标注，不使用简单矩形框代替分割轮廓。
- 不标注 PCB 走线、丝印、焊盘和背景区域。
- 相邻元件不得合并；同一元件不得重复标注。
- 对图像边缘处可见的元件，按实际可见区域标注。
- 标注人员不得参考 SAM3 或 YOLOv8-Seg 的预测叠加图。

## 文件命名

图像与原始标注必须保持相同文件名主干，例如：

```text
images/r00_c00_undistorted.png
annotations/r00_c00_undistorted.json
```

原始人工标注属于不可修改的评价依据。格式转换、训练切片和数据增强结果只能写入 `yolo/`，不得覆盖 `annotations/`。

## YOLOv8-Seg 目标结构

完成格式转换后，`yolo/` 应组织为：

```text
yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

每个标签文件一行表示一个实例：

```text
0 x1 y1 x2 y2 x3 y3 ...
```

其中 `0` 表示 `component`，多边形坐标必须按图像宽高归一化至 `[0, 1]`。

## 数据划分

当前四张原始图像应按原图进行四折交叉验证。禁止把同一原图的重叠切片同时放入训练集和验证集，以免数据泄漏。

每一折使用三张原图训练，并将剩余一张原图作为测试图。YOLOv8-Seg 与 SAM3 必须使用同一测试图和同一份人工 Ground Truth 计算指标。
