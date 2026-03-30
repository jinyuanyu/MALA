# MALA项目

## 项目简介

MALA（MAE-LaMa）是一个基于深度学习的遥感卫星图像时序修复项目。该项目使用掩码自编码器（MAE）与LaMa图像修复技术相结合，实现对Sentinel-2卫星遥感图像序列中缺失区域（如云遮挡、条带缺失等）的高质量修复。

### 核心特性

- **时序注意力机制**：捕捉遥感图像时间序列中的依赖关系
- **多掩码支持**：支持云掩码、条带掩码、薄云掩码和混合掩码
- **MAE-LaMa协作**：结合MAE重建和LaMa精细修复
- **海洋先验**：可选的海洋区域先验知识利用

## 项目结构

```
MALA/
├── data/                    # 数据加载模块
│   └── dataset.py          # 数据集类定义
├── models/                  # 模型定义模块
│   ├── __init__.py
│   ├── modules.py          # 核心组件（注意力机制、解码器等）
│   └── video_completion.py # 完整视频修复模型
├── utils/                   # 工具模块
│   ├── __init__.py
│   ├── metrics.py          # 评估指标计算
│   └── visualization.py   # 可视化工具
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── README.md               # 项目文档
└── requirements.txt        # 依赖列表
```

## 环境配置

### 1. 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n mala python=3.8
conda activate mala

# 或使用pip创建虚拟环境
python -m venv mala_env
source mala_env/bin/activate  # Linux/Mac
# 或
mala_env\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
# 安装PyTorch（根据CUDA版本选择合适的命令）
# CUDA 11.3
pip install torch==1.10.0 torchvision==0.11.0

# 或 CUDA 11.8+
pip install torch torchvision

# 安装项目依赖
pip install -r requirements.txt
```

## 数据准备

### 目录结构

项目期望的数据目录结构如下：

```
data/
├── images/
│   ├── frame_001.png
│   ├── frame_002.png
│   └── ...
├── mask_img/               # 预定义掩码（可选）
│   ├── mask_001.png
│   └── ...
└── inpainted_img/
    └── lama_init/         # LaMa初始修复结果（可选）
        ├── init_001.png
        └── ...
```

### 图像要求

- 格式：PNG或JPG
- 通道：RGB三通道
- 命名：格式为 `frame_XXXX.png`（XXXX为时间标识）

## 使用方法

### 训练模型

```bash
python train.py \
    --data_dir /path/to/training/data \
    --ocean_mask_path /path/to/ocean/mask.png \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --pretrained_path /path/to/pretrained/model.pth \
    --checkpoint_dir ./checkpoints
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | 必填 | 训练数据目录路径 |
| `--ocean_mask_path` | None | 海洋掩码图像路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--use_lama_init` | False | 是否使用LaMa初始修复 |
| `--use_ocean_prior` | False | 是否使用海洋先验 |

### 模型推理

```bash
python inference.py \
    --model_path ./checkpoints/model_best.pth \
    --data_dir /path/to/test/data \
    --ocean_mask_path /path/to/ocean/mask.png \
    --output_dir ./results \
    --save_images \
    --save_visualization
```

主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 模型权重路径 |
| `--data_dir` | 必填 | 测试数据目录 |
| `--mask_type` | random | 掩码类型 |
| `--output_dir` | results | 输出目录 |

### 掩码类型

支持以下掩码类型：

- `random`：随机薄云掩码
- `cloud`：厚云掩码
- `strip`：条带缺失掩码
- `mixed`：混合掩码
- `predefined`：使用预定义的掩码

## 评估指标

项目使用以下指标评估修复质量：

- **SSIM**：结构相似性指数（范围0-1，越高越好）
- **PSNR**：峰值信噪比（单位dB，越高越好）
- **MAE**：平均绝对误差（范围0-255，越低越好）

## 模型架构

### 核心组件

1. **PatchEmbedding**：将图像分割成patch并嵌入到高维空间
2. **TemporalAttention**：时间维度注意力机制
3. **PatchDecoder**：将特征解码回图像空间
4. **LamaInpaintingModule**（可选）：LaMa精细修复模块

### 损失函数

训练过程中使用多种损失函数组合：

- **重构损失**：MSELoss，在掩码区域计算
- **颜色一致性损失**：保持时序颜色一致性
- **梯度损失**：提高空间连续性

## 依赖列表

主要依赖包括：

- torch >= 1.10.0
- torchvision
- numpy
- opencv-python
- Pillow
- scikit-image
- matplotlib
- einops
- seaborn

详细依赖见 `requirements.txt`。

## 注意事项

1. **GPU要求**：建议使用NVIDIA GPU以获得最佳性能
2. **内存要求**：根据图像尺寸和批次大小，可能需要8GB+显存
3. **数据预处理**：确保图像尺寸一致或使用脚本自动处理

## 许可证

本项目仅供研究使用。

## 引用

如果使用本项目进行科学研究，请引用相关论文。

## 联系方式

如有问题，请提交Issue或联系项目维护者。
