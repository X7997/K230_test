<p align="center">
  <h1 align="center">🔧 Yolo_to_K230</h1>
  <p align="center">
    <b>YOLO 模型一键转 K230 Kmodel</b><br>
    本地零环境配置 · GitHub Actions 云端转换 · 自动下载结果
  </p>
</p>

---

## 🆚 为什么用 Yolo_to_K230？

| | 手动配置 | 本项目 |
|---|---|---|
| 本地环境 | 装 .NET 7.0 + nncase + Python 3.10 + 各种依赖，Windows/Linux 各种踩坑 | ❌ **不需要任何本地环境**（除 Git 和 gh CLI） |
| 转换过程 | 手动写脚本、调参数、跑命令、看报错、反复试 | ✅ 一条命令全自动：推送 → 云端转换 → 下载回本地 |
| 校准图 | 自己写预处理代码、调尺寸、凑格式 | ✅ 自动全量上传，无需手动筛选 |
| 出错排查 | 本地报错信息不完整 | ✅ 云端有完整日志，脚本自动获取失败原因 |
| 跨平台 | Windows 排坑半天 | ✅ GitHub Actions Ubuntu runner，稳定可靠 |

---

## 🚀 快速开始

### 前置要求

| 工具 | 版本 | 安装方式 |
|------|------|---------|
| **Git** | 最新 | [git-scm.com](https://git-scm.com/) |
| **GitHub CLI (gh)** | 最新 | [cli.github.com](https://cli.github.com/) → 安装后运行 `gh auth login` |
| **Python** | 3.8+ | 系统自带即可，脚本只用到标准库 |

### 三步转换

#### 1️⃣ 克隆本项目

```bash
git clone https://github.com/X7997/Yolo_to_K230.git
cd Yolo_to_K230
```

#### 2️⃣ 运行脚本（首次会进入交互式配置）

```bash
python easy_k230.py
```

首次运行会引导你填写：
- **GitHub 仓库地址** — 新建一个空仓库（不要勾选 README / .gitignore）
- **模型路径** — 直接粘贴你训练好的 `best.pt` 绝对地址
- **校准图目录** — 粘贴验证集图片目录路径
- **输入尺寸** — 默认 320x320，K210 改 224
- **代理端口** — 有翻墙就填（如 10808），没有填 0

配置自动保存到 `config.json`，之后只需改模型路径即可。

#### 3️⃣ 等待完成

脚本会自动：推送模型到 GitHub → 云端转换 → 下载 `.kmodel` 回本地

```
✅ 已复制模型 -> models\input.pt  (6.2 MB)
✅ 已复制全部 770 张校准图 -> calib
✅ 已推送到 GitHub（convert-request 分支）
🔍 运行 ID: 24756880699
⏳ 等待转换完成（通常 3~8 分钟）...
✅ Actions 运行成功！开始下载结果...
🎉 转换成功！Kmodel 已下载到本地:
   📦 output\food.kmodel  (3275.1 KB)
   📋 Q:\Yolo_to_K230\food.kmodel
```

**转换后的 `.kmodel` 文件出现在项目根目录，直接拷贝到 K230 板子上即可！**

---

## ⚙️ 配置参数

编辑 `config.json` 或重新运行 `python easy_k230.py` 会进入配置向导：

```json
{
  "source_pt": "C:\\Users\\xxx\\best.pt",
  "source_calib": "C:\\Users\\xxx\\valid\\images",
  "input_shape": [1, 3, 320, 320],
  "onnx_imgsz": 320,
  "quant_type": "uint8",
  "w_quant_type": "uint8",
  "calib_method": "Kld",
  "target": "k230",
  "kmodel_filename": "best.kmodel"
}
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input_shape` | 模型输入尺寸 [N,C,H,W]，K230 用 320，K210 用 224 | `[1,3,320,320]` |
| `onnx_imgsz` | ONNX 导出尺寸，必须和 input_shape 的 H/W 一致 | `320` |
| `quant_type` | 激活量化类型 | `uint8` |
| `w_quant_type` | 权重量化类型 | `uint8` |
| `calib_method` | 校准方法 | `Kld` |
| `target` | 目标芯片 | `k230` |
| `kmodel_filename` | 输出文件名 | 自动与原模型同名 |

---

## ⚠️ 重要：预处理对齐

**这是 K230 部署成功的关键！** 如果预处理不对齐，模型会"失明"。

### 转换时的预处理参数

本工具在量化校准时使用的预处理：

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入范围 | 0~1 | `img / 255.0` 归一化 |
| 颜色格式 | RGB | BGR → RGB 转换 |
| Layout | NCHW | 通道在前 |
| 均值/方差 | 无 | YOLO 默认不使用 |

### K230 部署代码必须对齐

```python
# K230 部署时的预处理示例
def run_with_normalize(self, input_np):
    # Step 1: ai2d resize (uint8 → uint8)
    ai2d_input = nn.from_numpy(input_np)
    self.ai2d.run(ai2d_input, self.ai2d_output_tensor)
    
    # Step 2: 手动归一化 (uint8 [0,255] → float32 [0,1])
    ai2d_out = self.ai2d_output_tensor.to_numpy()
    normalized_input = ai2d_out * (1.0 / 255.0)  # 关键！
    normalized_tensor = nn.from_numpy(normalized_input)
    
    # Step 3: KPU 推理
    self.kpu.set_input_tensor(0, normalized_tensor)
    self.kpu.run()
    ...
```

### 对齐检查清单

```
□ 图像尺寸：训练 imgsz = 转换 input_shape = 部署 model_input_size
□ 颜色格式：训练 RGB = 转换校准 RGB = 部署 RGB
□ 归一化：  训练 /255 = 转换校准 /255 = 部署 *1/255
□ Layout：  训练 NCHW = 转换 NCHW = 部署 NCHW
□ 均值方差：训练无 = 转换无 = 部署无
```

---

## 📁 项目结构

```
Yolo_to_K230/
├── .github/workflows/
│   └── convert_k230.yml    ← GitHub Actions 工作流（自动触发）
├── convert_k230.py          ← 云端转换核心脚本（无需手动修改）
├── easy_k230.py             ← 本地一键脚本（唯一需要运行的入口）
├── .gitignore
├── config.json              ← 自动生成，保存你的配置
├── README.md
├── models/                  ← 运行时自动创建，存放上传的模型
├── calib/                   ← 运行时自动创建，存放校准图
└── output/                  ← 运行时自动创建，存放下载结果
```

---

## ❓ 常见问题

### `gh auth login` 怎么做？

在终端运行 `gh auth login`，选择 **HTTPS** → **通过浏览器登录**，在弹出的 GitHub 页面授权即可。

### 网络连接失败怎么办？

如果你使用代理（v2rayN / Clash），在配置向导中填入代理端口号：
- v2rayN: `10808`
- Clash: `7890`
- 无代理: `0`

脚本会自动为 Git 和环境变量设置代理。

### 校准图用什么？

直接用训练数据集的**验证集图片**即可，越多越好（本项目全量上传，不限制数量）。

### 支持哪些模型格式？

- `.pt` — YOLO 训练权重（推荐，自动导出 ONNX）
- `.onnx` — 已导出的 ONNX 模型

### K210 怎么用？

修改 `config.json`：
```json
"input_shape": [1, 3, 224, 224],
"onnx_imgsz": 224,
"target": "k210"
```

---

## 📜 许可证

MIT License

---

<p align="center">
  觉得有用？给个 ⭐ Star 吧！
</p>