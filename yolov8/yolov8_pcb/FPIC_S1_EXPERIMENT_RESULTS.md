# FPIC s1 YOLOv8-Seg 实验结果

## 1. 实验概况

本实验使用本地 FPIC `s1` 数据集训练 YOLOv8l-Seg，并在一个未参与训练的采集组上进行测试。

数据集路径：

```text
/home/vision/Users/DSC/data/s1
```

类别：

```text
resistors, capacitors, ICs, inductors, diodes
```

数据划分：

| 划分 | 采集组 | 切片数 | 实例数 |
|---|---|---:|---:|
| Train | DSLR_s1_front，以及 5 组 Microscope 数据 | 309 | 856 |
| Test | Microscope_s1_p1_1x_40_ring | 32 | 93 |

测试集使用 YOLO 数据集中的 `val` 目录，但在本文档中将其作为 held-out test split 使用。

## 2. 训练配置

| 参数 | 设置 |
|---|---|
| 模型 | YOLOv8l-Seg |
| 初始权重 | `yolov8l-seg.pt` |
| 输入尺寸 | 1024 × 1024 |
| Batch size | 4 |
| 最大 epoch | 100 |
| 早停 patience | 20 |
| AMP | 开启 |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Tile size | 1024 |
| Tile overlap | 256 |
| 最终停止 epoch | 67 |
| 最佳 epoch | 47 |

最佳权重：

```text
runs/fpic_train/yolov8l_seg_s1/weights/best.pt
```

最佳 epoch 的训练记录：

| 指标 | 数值 |
|---|---:|
| Box Precision | 0.5814 |
| Box Recall | 1.0000 |
| Box mAP50 | 0.9373 |
| Box mAP50-95 | 0.7764 |
| Mask Precision | 0.5814 |
| Mask Recall | 1.0000 |
| Mask mAP50 | 0.9373 |
| Mask mAP50-95 | 0.7174 |

完整训练曲线和日志：

```text
runs/fpic_train/yolov8l_seg_s1/results.csv
runs/fpic_train/yolov8l_seg_s1/results.png
```

## 3. 测试集结果

最佳权重在 held-out test split 上的结果如下：

| 指标 | 数值 |
|---|---:|
| Mask AP50 | 0.9373 |
| Mask AP75 | 0.9143 |
| Mask AP50:95 | 0.7162 |
| Boundary F1 | 0.7253 |
| Boundary IoU | 0.5029 |
| Macro-F1 | 0.7796 |
| 联合 F1 | 0.7830 |
| Small Component Recall | 1.0000 |
| Center Error Mean | 3.194 px |
| Center Error Median | 1.586 px |

实例统计：

| 项目 | 数值 |
|---|---:|
| Ground-truth instances | 93 |
| Matched instances | 92 |
| Predicted instances | 142 |
| Ultralytics Mask Precision | 0.5801 |
| Ultralytics Mask Recall | 1.0000 |

## 4. 类别级 F1

| 类别 | F1 |
|---|---:|
| resistors | 0.8548 |
| capacitors | 0.6761 |
| ICs | 0.6400 |
| inductors | 0.7273 |
| diodes | 1.0000 |

测试集类别实例数量较少，尤其是 `diodes` 仅 2 个、`inductors` 仅 4 个，因此少数类指标的统计稳定性有限。

## 5. 指标定义

- **Mask AP50**：掩码 IoU 阈值为 0.50 时的平均精度。
- **Mask AP75**：掩码 IoU 阈值为 0.75 时的平均精度。
- **Mask AP50:95**：IoU 从 0.50 到 0.95、步长 0.05 的平均 AP。
- **Boundary F1**：预测边界与真值边界在 2 px 容差下的 F1 值。
- **Boundary IoU**：2 px 边界容差下的边界区域 IoU。
- **Macro-F1**：五个类别的类别感知实例 F1 的算术平均值。
- **联合 F1**：掩码 IoU 至少为 0.50 且类别预测正确时的全局 micro-F1。
- **Small Component Recall**：真值掩码面积小于 `4096 px²` 的元件 Recall。
- **Center Error**：预测掩码质心与真值掩码质心之间的欧氏距离，单位为像素。

AP 指标使用置信度扫描计算；F1、边界指标和中心误差使用 `conf=0.25`，实例匹配阈值为 IoU `0.50`。

## 6. 复现实验

生成 FPIC YOLO 数据集：

```bash
python main.py prepare-fpic \
  --validation-group Microscope_s1_p1_1x_40_ring \
  --force
```

训练：

```bash
python main.py train \
  --data ground_truth/fpic_yolo/data.yaml \
  --project runs/fpic_train \
  --name yolov8l_seg_s1 \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
  --amp \
  --patience 20
```

测试评估：

```bash
python main.py evaluate \
  --weights runs/fpic_train/yolov8l_seg_s1/weights/best.pt \
  --data ground_truth/fpic_yolo/data.yaml \
  --output runs/fpic_evaluation \
  --imgsz 1024 \
  --batch 4 \
  --device 0 \
  --conf 0.25 \
  --boundary-tolerance 2 \
  --small-area 4096
```

指标 JSON：

```text
runs/fpic_evaluation/metrics.json
```

## 7. 结果限制

当前 `s1` 数据集中共有 435 个矩形标注和 2 个多边形标注。矩形标注被转换成四点多边形后用于 YOLOv8-Seg 训练，因此大多数真值并不是像素级精确元件轮廓。

因此，当前 Boundary F1、Boundary IoU 和 Mask AP 反映的是模型对 FPIC 矩形弱分割标签的拟合效果，不能直接等同于高质量人工轮廓上的真实分割精度。若用于论文最终结论，建议使用 Labelme 或 CVAT 对测试图像进行独立像素级多边形标注。

此外，Train 和 Test 均来自同一块 `s1` PCB 的不同采集组，测试结果不能代表跨 PCB、跨板型或跨成像设备的泛化性能。

## 8. 方案 B：冻结 SAM3 + 五类分类头

为与 YOLOv8-Seg 对比，采用冻结 `sam3.pt`、仅训练独立五类 ResNet18 分类头的方案。本实验不是 SAM3 端到端微调。

训练脚本：

```text
sam3_fpic_scheme_b.py
```

分类头权重：

```text
runs/sam3_fpic_scheme_b/classifier_best.pt
```

| 参数 | 设置 |
|---|---|
| Backbone | 冻结 SAM3 `sam3.pt` |
| 分类头 | ResNet18 |
| 类别数 | 5 |
| 分类输入 | 96 × 224 |
| 最大 epoch | 40 |
| Batch size | 32 |
| 优化器 | AdamW |
| 分类头内部验证 Macro-F1 | 1.0000 |

方案 B 测试集结果：

| 指标 | 冻结 SAM3 + 分类头 | YOLOv8-Seg |
|---|---:|---:|
| Mask AP50 | 0.2437 | 0.9373 |
| Mask AP75 | 0.1776 | 0.9143 |
| Mask AP50:95 | 0.1822 | 0.7162 |
| Boundary F1 | 0.4287 | 0.7253 |
| Boundary IoU | 0.2524 | 0.5029 |
| Macro-F1 | 0.1079 | 0.7796 |
| 联合 F1 | 0.1664 | 0.7830 |
| 小元件 Recall | 0.9048 | 1.0000 |
| Center Error Mean | 4.298 px | 3.194 px |
| Center Error Median | 2.818 px | 1.586 px |

方案 B 实例统计：

| 项目 | 数值 |
|---|---:|
| Ground-truth instances | 93 |
| SAM3 predicted instances | 484 |
| IoU ≥ 0.50 matched instances | 85 |

结果 JSON：

```text
runs/sam3_fpic_scheme_b/metrics.json
```

在当前相同测试切片上，YOLOv8-Seg 的 Mask AP、边界指标和类别 F1 均明显高于冻结 SAM3 + 分类头。方案 B 的主要问题是 SAM3 文本提示生成了大量候选掩码，分类头只负责候选类别判断，没有学习与 FPIC 五类标注对齐的端到端候选筛选器，导致 484 个预测实例中存在大量误检。

因此，方案 B 可以作为“冻结通用分割模型 + 专用分类头”的基线，但不能代表经过 FPIC 标注微调的 SAM3 性能。

## 9. 100 epoch 与 1000 epoch 对比

在完全相同的 FPIC `s1` 数据划分、冻结 SAM3 checkpoint、ResNet18 分类头结构、输入尺寸和 batch size 下，分别训练 100 epoch 和 1000 epoch。两次训练使用独立输出目录：

```text
runs/sam3_fpic_scheme_b_100/
runs/sam3_fpic_scheme_b_1000/
```

分类头内部验证的最佳 Macro-F1 均为 `1.0000`。100 epoch 最佳内部验证 epoch 为 8，1000 epoch 最佳内部验证 epoch 为 6，说明分类头在极少训练轮次后已经饱和。

| 指标 | 100 epoch | 1000 epoch | 变化（1000 - 100） |
|---|---:|---:|---:|
| Mask AP50 | 0.3170 | 0.2605 | -0.0565 |
| Mask AP75 | 0.1883 | 0.1363 | -0.0520 |
| Mask AP50:95 | 0.2025 | 0.1508 | -0.0516 |
| Boundary F1 | 0.4287 | 0.4287 | 0.0000 |
| Boundary IoU | 0.2524 | 0.2524 | 0.0000 |
| Macro-F1 | 0.1565 | 0.1246 | -0.0319 |
| 联合 F1 | 0.2357 | 0.1976 | -0.0381 |
| 小元件 Recall | 0.9048 | 0.9048 | 0.0000 |
| Center Error Mean | 4.298 px | 4.298 px | 0.000 px |
| Center Error Median | 2.818 px | 2.818 px | 0.000 px |

两次评估的结构性统计保持不变：测试集真值 93 个、SAM3 预测候选 484 个、IoU ≥ 0.50 匹配 85 个。100 epoch 和 1000 epoch 的指标 JSON 分别为：

```text
runs/sam3_fpic_scheme_b_100/metrics.json
runs/sam3_fpic_scheme_b_1000/metrics.json
```

### 原因分析

1. **SAM3 掩码被冻结**。两次实验调用的是同一个 `sam3.pt`，候选掩码、候选数量和几何位置完全相同。因此 Boundary F1、Boundary IoU、Center Error 和小元件 Recall 不会随分类头 epoch 改变。
2. **分类头很早就过拟合/饱和**。训练样本只有 856 个元件裁剪图，分类头内部验证在第 6–8 epoch 就达到 Macro-F1=1.0，继续训练只会降低训练 loss，不能增加新的外部泛化信息。
3. **AP/F1 下降来自置信度排序变化**。分类头输出的类别概率参与了预测置信度，1000 epoch 后模型在训练裁剪图上更加自信，但在 held-out 测试采集组上排序变差，所以 AP50、AP75、AP50:95、Macro-F1 和联合 F1 下降。
4. **测试集规模和类别分布有限**。测试集只有 93 个实例，其中电感 4 个、二极管 2 个，少数类别的 F1 对少量错误非常敏感。
5. **训练仍有随机性**。DataLoader 多进程和随机水平翻转使两次独立训练的概率输出不完全相同；因此严格论文比较应固定所有随机源并重复多次报告均值和标准差。

结论：对当前“冻结 SAM3 + 独立分类头”的方案，100 epoch 已经足够，1000 epoch 没有改善分割边界，反而造成测试集分类排序退化。若要提升 Mask AP、Boundary F1 或 Center Error，必须改进 SAM3 的候选生成/掩码质量或进行端到端 SAM3 微调，而不是继续增加分类头训练轮次。

## 10. 仅测试程序结果

本次使用独立测试程序，不读取训练日志、不重新训练：

```text
test_sam3_classifier.py
```

测试命令：

```bash
python test_sam3_classifier.py \
  --dataset ground_truth/fpic_yolo \
  --sam-checkpoint /home/vision/Users/DSC/sam3/models/sam3/sam3.pt \
  --classifier runs/sam3_fpic_scheme_b_1000/classifier_best.pt \
  --output runs/sam3_fpic_test_1000 \
  --device 0
```

测试结果 JSON：

```text
runs/sam3_fpic_test_1000/metrics.json
```

| 指标 | 测试程序结果 |
|---|---:|
| Mask AP50 | 0.2605 |
| Mask AP75 | 0.1363 |
| Mask AP50:95 | 0.1508 |
| Boundary F1 | 0.4287 |
| Boundary IoU | 0.2524 |
| Macro-F1 | 0.1246 |
| 联合 F1 | 0.1976 |
| 小元件 Recall | 0.9048 |
| Center Error Mean | 4.298 px |
| Center Error Median | 2.818 px |

测试集共 32 张切片、93 个真值实例；冻结 SAM3 生成 484 个候选实例，IoU ≥ 0.50 匹配 85 个实例。

### 9.1 指标差异核对

从两份 `metrics.json` 逐项复核得到的相对变化如下：

| 指标 | 绝对变化 | 相对变化 |
|---|---:|---:|
| Mask AP50 | -0.0565 | -17.82% |
| Mask AP75 | -0.0520 | -27.63% |
| Mask AP50:95 | -0.0516 | -25.51% |
| Macro-F1 | -0.0319 | -20.41% |
| 联合 F1 | -0.0381 | -16.18% |
| Boundary F1 | 0.0000 | 0.00% |
| Boundary IoU | 0.0000 | 0.00% |
| 小元件 Recall | 0.0000 | 0.00% |
| Center Error Mean | 0.000 px | 0.00% |

类别 F1 的变化为：

| 类别 | 100 epoch | 1000 epoch | 变化 |
|---|---:|---:|---:|
| resistors | 0.3502 | 0.2345 | -0.1157 |
| capacitors | 0.1657 | 0.1630 | -0.0027 |
| ICs | 0.2667 | 0.2254 | -0.0413 |
| inductors | 0.0000 | 0.0000 | 0.0000 |
| diodes | 0.0000 | 0.0000 | 0.0000 |

### 9.2 差异定位

两次评估的 `ground_truth_instances=93`、`predicted_instances=484`、`matched_instances=85` 完全一致，说明 100 和 1000 epoch 没有改变 SAM3 的候选掩码集合。由于边界指标和质心误差也完全一致，可以排除“掩码形状变差”这一原因。

真正变化的是分类头的类别概率和置信度排序。AP 是依赖置信度排序的指标；1000 epoch 后分类头在训练裁剪图上更加自信，但这种置信度在 held-out 采集组上校准变差，导致正确预测排位下降、错误候选排位上升，所以 AP50、AP75 和 AP50:95 下降。Macro-F1 和联合 F1 的下降则说明部分候选类别预测也发生了变化，其中 `resistors` F1 下降最大，为 `-0.1157`。

分类头内部验证 Macro-F1 在第 6–8 epoch 已经达到 1.0，但这个内部验证集是从训练采集组的裁剪实例中随机划分的，不是最终的 held-out 采集组。因此内部验证饱和不能证明跨采集组泛化已经饱和，1000 epoch 反而加重了对训练采集组外观、裁剪和增强模式的过拟合。

此外，DataLoader 多进程和随机水平翻转使两次独立训练的概率输出存在随机差异。因此 100/1000 对比应解释为当前单次重复实验的趋势；若论文需要严格结论，应固定所有随机源并进行多次重复，报告均值和标准差。
