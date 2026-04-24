# K230 YOLO 模型部署完整指南

## 目录
1. [任务总结](#任务总结)
2. [整体流程概览](#整体流程概览)
3. [阶段一：模型训练](#阶段一模型训练)
4. [阶段二：模型转换与量化](#阶段二模型转换与量化)
5. [阶段三：K230 部署](#阶段三k230-部署)
6. [关键对齐点详解](#关键对齐点详解)
7. [常见问题与解决方案](#常见问题与解决方案)

---

## 任务总结

### 问题
K230 部署 YOLO 模型后检测效果极差，模型几乎"失明"，无法正确识别任何物体。

### 根本原因
**预处理不对齐**。这是嵌入式部署中最常见也最致命的问题：

| 阶段 | 预处理 | 数值范围 |
|------|--------|---------|
| 训练 | `/255.0` 归一化 | 0~1 |
| 转换校准 | `/255.0` 归一化 | 0~1 |
| K230 部署（修改前） | **无归一化** | 0~255 |

量化是基于 0~1 范围计算的 Scale 和 Zero-point，但实际推理输入是 0~255，导致量化参数完全错位，模型输出毫无意义。

### 解决方案
修改了三个文件：

1. **`K230_demo/convert_k230.py`**
   - 添加预处理参数注释，明确记录：RGB格式、NCHW布局、归一化范围[0,1]

2. **`K230_demo/github_converter.py`**
   - 添加预处理说明注释
   - 确认量化参数为 uint8

3. **`fridge.v1i.yolov12/ob_demo_debug.py`**（核心修改）
   - 新增 `run_with_normalize()` 方法
   - ai2d resize 后手动执行归一化：`img * (1.0/255.0)`
   - 添加预处理缓冲区管理

### 关键教训
```
╔═══════════════════════════════════════════════════════════════════════════╗
║  嵌入式部署第一定律：预处理必须形成闭环！                                  ║
║                                                                           ║
║  训练时的预处理 = 转换校准时的预处理 = 部署时的预处理                      ║
║                                                                           ║
║  任何一环不一致，模型就会"失明"或精度大幅下降。                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 整体流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOLO 模型 K230 部署全流程                            │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │   阶段一     │      │   阶段二     │      │   阶段三     │
  │  模型训练    │ ───▶ │  模型转换    │ ───▶ │  K230 部署   │
  │   (PC端)     │      │   + 量化     │      │   (嵌入式)   │
  └──────────────┘      └──────────────┘      └──────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   best.pt               best.onnx              best.kmodel
   (权重文件)            (中间格式)             (K230专用)
        │                      │                      │
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               │
                               ▼
                    ╔═════════════════════════════╗
                    ║   关键：每一步都要对齐！    ║
                    ║   - 图像尺寸               ║
                    ║   - 颜色格式 (RGB/BGR)     ║
                    ║   - 归一化参数             ║
                    ║   - Layout (NCHW/NHWC)     ║
                    ╚═════════════════════════════╝
```

---

## 阶段一：模型训练

### 1.1 训练脚本核心要素

```python
# mytrain_demo.py
from ultralytics import YOLO

model = YOLO("yolo12n.pt")  # 选择模型规模

model.train(
    data="data.yaml",        # 数据集配置
    epochs=200,              # 训练轮数
    imgsz=320,               # ★ 输入尺寸（部署时必须一致）
    batch=16,
    device=0,                # GPU
)
```

### 1.2 训练时的隐式预处理

**这是大多数人忽略的地方！** Ultralytics YOLO 在训练时会自动执行以下预处理：

```python
# Ultralytics 内部预处理流程（隐式）
def preprocess(image):
    # 1. Resize 到 imgsz
    image = resize(image, (imgsz, imgsz))
    
    # 2. 颜色格式：RGB（OpenCV 读取是 BGR，会转换）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. 归一化到 0~1
    image = image.astype(np.float32) / 255.0
    
    # 4. 转换为 NCHW 格式
    image = image.transpose(2, 0, 1)  # HWC → CHW
    
    return image
```

### 1.3 训练阶段的关键记录

训练完成后，你需要记录以下参数，**部署时必须完全一致**：

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `imgsz` | 320 | 输入图像尺寸 |
| 颜色格式 | RGB | 训练时使用的颜色顺序 |
| 归一化 | `/255.0` | 数值范围 0~1 |
| Layout | NCHW | 通道在前 |
| 均值/方差 | 无 | YOLO 默认不使用 |

### 1.4 输出文件

```
runs/detect/fridge_yolo12n_v2/weights/
├── best.pt      # 最佳权重（用于转换）
├── last.pt      # 最后一轮权重
└── ...
```

---

## 阶段二：模型转换与量化

### 2.1 为什么需要转换？

```
best.pt (PyTorch) ──▶ best.onnx (通用格式) ──▶ best.kmodel (K230专用)
     │                      │                        │
     │                      │                        │
   PC训练                中间过渡                 嵌入式运行
   浮点运算              标准算子                定点运算(量化)
```

### 2.2 ONNX 导出

```python
# convert_k230.py 中的导出逻辑
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="onnx",
    imgsz=320,           # ★ 必须与训练时一致
    opset=11,            # K230 最稳定的是 opset 11
    simplify=True,       # 简化 ONNX 图
    nms=False,           # 板端后处理，建议 False
)
```

### 2.3 量化（PTQ）详解

**什么是量化？**

量化是将浮点模型转换为定点模型的过程，可以大幅减小模型体积并加速推理：

```
float32 (32位) ──▶ uint8 (8位)
模型体积: 1/4
推理速度: 2~4倍提升
精度损失: 通常 1~3%
```

**量化的数学原理：**

```
原始值 (float32):  x
量化值 (uint8):   q
反量化值:         x' ≈ x

量化公式:
q = round(x / scale) + zero_point

反量化公式:
x' = (q - zero_point) * scale

其中 scale 和 zero_point 通过校准数据统计得到
```

### 2.4 校准数据集

**这是量化的关键！** 校准数据用于统计每层激活值的分布，从而确定最优的 scale 和 zero_point。

```python
# convert_k230.py 中的校准数据准备
def read_calibration_images(img_dir, shape, max_num):
    for img_path in images:
        img = cv2.imread(img_path)
        
        # ★★★ 预处理必须与训练时完全一致 ★★★
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # 归一化！
        img = np.transpose(img, (2, 0, 1))
        
        data_list.append(img)
```

**校准数据集要求：**
- 数量：100~500 张（太少不够代表性，太多浪费时间）
- 来源：验证集或真实场景图片
- 分布：覆盖各种光照、角度、目标大小
- **预处理：必须与训练时完全一致！**

### 2.5 量化参数选择

| 参数 | 可选值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `quant_type` | uint8, int8, float16 | uint8 | 激活值量化类型 |
| `w_quant_type` | uint8, int8 | uint8 | 权重量化类型 |
| `calib_method` | Kld, NoClip | Kld | 校准方法，Kld 精度更高 |

**量化类型对比：**

| 类型 | 精度 | 速度 | 模型大小 | 适用场景 |
|------|------|------|---------|---------|
| uint8 | 中等 | 最快 | 最小 | 一般检测（推荐） |
| float16 | 高 | 较慢 | 2倍 | 精度敏感任务 |
| 混合精度 | 平衡 | 中等 | 中等 | 特定层需要高精度 |

### 2.6 转换命令

```python
# 使用 github_converter.py（云端转换）
python github_converter.py

# 或使用 convert_k230.py（本地转换）
python convert_k230.py
```

### 2.7 转换阶段的关键配置

```python
# github_converter.py 顶部配置
INPUT_SHAPE = [1, 3, 320, 320]  # [N, C, H, W]
ONNX_IMGSZ = 320                 # 必须与 INPUT_SHAPE 一致
QUANT_TYPE = "uint8"             # 量化类型
W_QUANT_TYPE = "uint8"           # 权重量化
CALIB_METHOD = "Kld"             # 校准方法
TARGET = "k230"                  # 目标芯片
```

---

## 阶段三：K230 部署

### 3.1 K230 推理架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        K230 推理流程                            │
└─────────────────────────────────────────────────────────────────┘

摄像头采集 ──▶ PipeLine ──▶ AI2D预处理 ──▶ KPU推理 ──▶ 后处理 ──▶ 显示
    │              │             │            │           │
    │              │             │            │           │
  RGB/BGR      格式转换      Resize/Pad     模型推理    NMS/画框
  原始尺寸      颜色空间      归一化?        .kmodel     坐标映射
```

### 3.2 AI2D 预处理

AI2D 是 K230 的硬件预处理加速器，支持：
- Resize（缩放）
- Pad（填充）
- Crop（裁剪）
- 颜色空间转换

**但 AI2D 不支持归一化！** 这是导致问题的根源。

```python
# ob_demo_debug.py 中的 AI2D 配置
self.ai2d = Ai2d(debug_mode)
self.ai2d.set_ai2d_dtype(
    nn.ai2d_format.NCHW_FMT,   # 输入格式
    nn.ai2d_format.NCHW_FMT,   # 输出格式
    np.uint8,                  # 输入数据类型
    np.uint8                   # 输出数据类型（无法直接输出 float32）
)

self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
self.ai2d.build([1, 3, H_in, W_in], [1, 3, H_out, W_out])
```

### 3.3 手动归一化（解决方案）

```python
def run_with_normalize(self, input_np):
    """
    带归一化的推理流程
    """
    # Step 1: AI2D resize (uint8 → uint8)
    ai2d_input = nn.from_numpy(input_np)
    self.ai2d.run(ai2d_input, self.ai2d_output_tensor)
    
    # Step 2: 手动归一化 (uint8 [0,255] → float32 [0,1])
    ai2d_out = self.ai2d_output_tensor.to_numpy()
    self.normalized_input_np = ai2d_out * (1.0 / 255.0)  # 乘法比除法快
    self.normalized_input_tensor = nn.from_numpy(self.normalized_input_np)
    
    # Step 3: KPU 推理
    self.kpu.set_input_tensor(0, self.normalized_input_tensor)
    self.kpu.run()
    
    # Step 4: 获取输出
    results = [self.kpu.get_output_tensor(i).to_numpy() 
               for i in range(self.kpu.outputs_size)]
    
    return self.postprocess(results)
```

### 3.4 后处理

后处理包括：
1. **输出解码**：将模型输出转换为边界框坐标
2. **置信度过滤**：过滤低于阈值的检测框
3. **NMS**：非极大值抑制，去除重复框
4. **坐标映射**：将模型坐标映射回原图

```python
def postprocess(self, results):
    # 1. 解码输出
    output_data = results[0].transpose()
    boxes = output_data[:, 0:4]    # x, y, w, h
    scores = output_data[:, 4:]    # 类别分数
    
    # 2. 置信度过滤
    confs = np.max(scores, axis=-1)
    mask = confs > self.confidence_threshold
    
    # 3. NMS
    keep = self.nms(boxes[mask], confs[mask], self.nms_threshold)
    
    # 4. 坐标映射（模型坐标 → 原图坐标）
    for i in keep:
        x, y, w, h = boxes[i]
        left = int((x - 0.5 * w) * self.x_factor)
        top = int((y - 0.5 * h) * self.y_factor)
        ...
```

### 3.5 部署文件清单

```
K230 SD卡/
├── food.kmodel           # 量化后的模型
├── ob_demo_debug.py      # 主程序（含归一化）
└── libs/                 # K230 SDK 库
    ├── PipeLine.py
    ├── AIBase.py
    ├── AI2D.py
    └── ...
```

---

## 关键对齐点详解

### 对齐点 1：图像尺寸

```
训练 imgsz = 320
    ↓
转换 INPUT_SHAPE = [1, 3, 320, 320]
    ↓
部署 model_input_size = [320, 320]
```

**不一致的后果**：模型输入尺寸错误，推理失败或输出乱码。

### 对齐点 2：颜色格式

```
训练：RGB（Ultralytics 默认）
    ↓
转换校准：cv2.COLOR_BGR2RGB
    ↓
部署：PipeLine 输出 RGB888p
```

**不一致的后果**：颜色通道错位，检测精度大幅下降。

**验证方法**：用一张纯红色图片测试，检查模型输入的通道值。

### 对齐点 3：归一化（最关键！）

```
训练：img / 255.0 → [0, 1]
    ↓
转换校准：img.astype(np.float32) / 255.0 → [0, 1]
    ↓
部署：ai2d_out * (1.0/255.0) → [0, 1]
```

**不一致的后果**：量化参数错位，模型"失明"。

**数学解释**：
```
量化校准时：
  输入范围 [0, 1]
  统计得到 scale = 0.01, zero_point = 0
  
部署时（错误）：
  输入范围 [0, 255]
  实际量化值 q = 255 / 0.01 = 25500（溢出！）
  
部署时（正确）：
  输入范围 [0, 1]（归一化后）
  实际量化值 q = 1 / 0.01 = 100（正常）
```

### 对齐点 4：Layout

```
训练：NCHW（PyTorch 默认）
    ↓
转换：NCHW（ONNX 导出）
    ↓
部署：NCHW（ai2d 配置）
```

**K230 内部偏好 NHWC，但 nncase 会自动处理转换。** 建议全程使用 NCHW。

### 对齐点 5：均值/方差

```
YOLO 默认：无均值/方差归一化
    ↓
转换校准：无
    ↓
部署：无
```

**注意**：某些模型（如 ImageNet 预训练）使用均值/方差归一化：
```python
# 如果训练时使用了均值/方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = (img / 255.0 - mean) / std
```

部署时必须执行相同的操作。

---

## 常见问题与解决方案

### Q1: 模型部署后检测不到任何物体

**原因**：预处理不对齐（最可能是归一化）

**排查步骤**：
1. 检查训练时的预处理代码
2. 检查转换校准时的预处理代码
3. 检查部署时的预处理代码
4. 确保三者完全一致

### Q2: 检测框位置偏移

**原因**：坐标映射错误

**解决方案**：
```python
# 检查 x_factor 和 y_factor
self.x_factor = rgb888p_size[0] / model_input_size[0]
self.y_factor = rgb888p_size[1] / model_input_size[1]

# 如果使用了 letterbox（填充），需要扣除填充
left = int((x - 0.5 * w) * x_factor - pad_x)
top = int((y - 0.5 * h) * y_factor - pad_y)
```

### Q3: 检测精度比 PC 端低很多

**原因**：量化损失

**解决方案**：
1. 增加校准图片数量（100 → 500）
2. 使用更具代表性的校准数据
3. 尝试不同的校准方法（Kld vs NoClip）
4. 如果仍不够，使用 float16 量化

### Q4: 推理速度太慢

**原因**：预处理或后处理耗时

**排查**：
```python
# 开启 debug_mode 查看各阶段耗时
fridge_det = FridgeDetectionApp(..., debug_mode=1)
```

**优化方向**：
- AI2D 硬件加速预处理
- 优化 NMS 实现
- 减少后处理中的循环

### Q5: 内存不足

**原因**：K230 内存有限

**解决方案**：
1. 减小输入尺寸（320 → 224）
2. 使用更小的模型（yolo12n → yolo12n-nano）
3. 减少后处理缓冲区大小

---

## 附录：完整对齐检查清单

在每次部署前，请逐项检查：

```
□ 图像尺寸
  □ 训练 imgsz = 320
  □ 转换 INPUT_SHAPE = [1, 3, 320, 320]
  □ 部署 model_input_size = [320, 320]

□ 颜色格式
  □ 训练：RGB
  □ 转换校准：BGR → RGB
  □ 部署：PipeLine 输出 RGB

□ 归一化
  □ 训练：/ 255.0
  □ 转换校准：/ 255.0
  □ 部署：* (1.0/255.0)

□ Layout
  □ 训练：NCHW
  □ 转换：NCHW
  □ 部署：NCHW

□ 均值/方差
  □ 训练：无
  □ 转换校准：无
  □ 部署：无

□ 量化参数
  □ 量化类型：uint8
  □ 校准方法：Kld
  □ 校准图片：100~500张

□ 后处理
  □ 置信度阈值：合理（0.1~0.5）
  □ NMS 阈值：合理（0.4~0.5）
  □ 坐标映射：正确
```

---

## 结语

嵌入式模型部署不是简单的"复制粘贴"，而是一个需要精确对齐的系统工程。

**记住三个关键点**：

1. **预处理对齐**：训练、转换、部署三者的预处理必须完全一致
2. **量化校准**：校准数据的预处理和代表性决定了量化精度
3. **后处理正确**：坐标映射和 NMS 参数需要根据实际场景调整

祝你在 K230 上成功部署你的模型！

---

*本文档由 AI 助手生成，如有疑问请参考 K230 官方文档或 nncase 文档。*
