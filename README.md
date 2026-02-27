# MALA: Masked Autoencoder for Remote Sensing Image Completion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 项目简介 | Introduction

MALA (Masked Autoencoder for Remote Sensing Image Completion) 是一个基于深度学习的遥感图像缺失区域填补与插补框架。该项目实现了多种先进的时间序列插补算法，包括基于Transformer的掩码自编码器 (Masked Autoencoder, MAE) 架构，以及多种传统插值方法。

MALA is a deep learning framework for remote sensing image completion and interpolation. This project implements various advanced time-series interpolation algorithms, including Masked Autoencoder (MAE) architecture based on Transformer and multiple traditional interpolation methods.

## 主要特性 | Key Features

- **深度学习模型**: 基于Transformer的MAE架构，支持时空特征学习
- **多算法支持**: 集成DINEOF、样条插值、最近邻、IDW、克里金等经典方法
- **完整实验流程**: 包含数据处理、模型训练、评估、可视化全流程
- **丰富的可视化**: 热力图、时序图、散点图、箱线图等多种分析图表
- **模块化设计**: 清晰的代码结构，便于扩展和定制

## 目录结构 | Directory Structure

```
MALA/
├── models/                 # 深度学习模型定义
│   ├── __init__.py
│   ├── mae_lama.py        # 主模型：MAE-LaMa
│   └── components/         # 模型组件
├── data/                   # 数据加载与处理
│   ├── __init__.py
│   └── datasets.py         # 数据集定义
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── metrics.py          # 评估指标
│   └── visualization.py    # 可视化工具
├── experiments/            # 实验脚本
│   ├── train.py           # 训练脚本
│   ├── evaluate.py        # 评估脚本
│   └── run_experiments.py # 实验运行
├── scripts/               # 辅助脚本
│   ├── data_preprocessing.py
│   └── analysis.py
├── docs/                  # 详细文档
│   ├── installation.md
│   ├── usage.md
│   └── api.md
├── configs/               # 配置文件
│   └── default.yaml
├── requirements.txt        # 依赖
└── README.md             # 本文件
```

## 安装 | Installation

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.0 (GPU支持)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/jinyuanyu/MALA.git
cd MALA

# 2. 创建虚拟环境
conda create -n mala python=3.9
conda activate mala

# 3. 安装依赖
pip install -r requirements.txt

# 4. (可选) 安装PyTorch CUDA版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始 | Quick Start

### 1. 数据准备

将您的遥感数据整理为以下格式：

```
data/
├── S2_Daily_Mosaic/           # 原始图像序列
│   ├── 20230101.png
│   ├── 20230102.png
│   └── ...
└── masks/                     # 掩码文件
    ├── mask_01.png
    └── ...
```

### 2. 运行训练

```python
from models.mae_lama import VideoCompletionModel
from data.datasets import RemoteSensingDataset
from experiments.train import train_model

# 1. 加载数据
dataset = RemoteSensingDataset(
    data_dir='data/your_data/',
    max_seq_len=8
)

# 2. 初始化模型
model = VideoCompletionModel(
    in_channels=3,
    out_channels=3,
    hidden_dim=256,
    num_heads=8,
    num_layers=6
)

# 3. 训练模型
train_model(model, dataset, epochs=100)
```

### 3. 运行评估

```python
from experiments.evaluate import evaluate_model

results = evaluate_model(
    model=model,
    test_dataset=test_dataset,
    methods=['Proposed', 'DINEOF', 'Spline', 'Nearest_Neighbor']
)

print(f"SSIM: {results['SSIM']:.4f}")
print(f"PSNR: {results['PSNR']:.2f}")
print(f"MAE: {results['MAE']:.2f}")
```

## 支持的算法 | Supported Algorithms

### 深度学习方法

| 算法名称 | 描述 | 论文 |
|---------|------|------|
| **MALA** | 掩码自编码器方法 | 原创 |
| **Proposed** | 提出的深度学习方法 | - |
| **MaskAE** | 掩码自编码器 | - |

### 传统插值方法

| 算法名称 | 描述 | 特点 |
|---------|------|------|
| **DINEOF** | 数据插值正交函数法 | 适合大范围时空缺失 |
| **Spline** | 三次样条插值 | 时间序列平滑 |
| **Nearest Neighbor** | 最近邻插值 | 计算简单快速 |
| **IDW** | 反距离加权插值 | 空间局部性 |
| **Kriging** | 克里金插值 | 地统计学最优 |

## 评估指标 | Metrics

- **SSIM**: 结构相似性指数
- **PSNR**: 峰值信噪比
- **MAE**: 平均绝对误差
- **TCC**: 时间相关系数

## 可视化示例 | Visualization Examples

项目提供丰富的可视化功能：

```python
# 1. 误差热力图
from utils.visualization import plot_error_heatmap

plot_error_heatmap(results, save_path='error_heatmap.png')

# 2. 时序对比图
from utils.visualization import plot_timeseries

plot_timeseries(pixel_data, missing_regions, save_path='timeseries.png')

# 3. 箱线图比较
from utils.visualization import plot_boxplot

plot_boxplot(all_metrics, save_path='boxplot.png')

# 4. 散点图
from utils.visualization import plot_scatter

plot_scatter(ground_truth, predictions, save_path='scatter.png')
```

## 实验配置 | Experiment Configuration

支持多种实验设置：

```yaml
# configs/experiment.yaml
experiment:
  name: "remote_sensing_interpolation"
  
  # 数据配置
  data:
    data_dir: "data/S2_Daily_Mosaic/"
    mask_dir: "data/masks/"
    max_seq_len: 8
    image_size: [256, 256]
  
  # 模型配置
  model:
    name: "MAE-LaMa"
    hidden_dim: 256
    num_heads: 8
    num_layers: 6
    dropout: 0.1
  
  # 训练配置
  training:
    epochs: 100
    batch_size: 4
    learning_rate: 0.0001
    optimizer: "Adam"
  
  # 评估配置
  evaluation:
    metrics: ["SSIM", "PSNR", "MAE", "TCC"]
    methods: ["Proposed", "DINEOF", "Spline"]
```

## 文档 | Documentation

详细的文档请参阅 [docs/](docs/) 目录：

- [安装指南](docs/installation.md) - 环境配置与依赖安装
- [使用教程](docs/usage.md) - 完整的使用流程
- [API文档](docs/api.md) - 详细的API说明
- [算法说明](docs/algorithms.md) - 各算法的原理介绍

## 常见问题 | FAQ

### Q: 如何使用自己的数据集？

A: 请参考 [数据准备指南](docs/usage.md#数据准备) 创建与项目兼容的数据格式。

### Q: 支持哪些图像格式？

A: 支持 PNG、JPG、TIFF 等常见图像格式。

### Q: 如何在CPU上运行？

A: 模型支持CPU和GPU运行，只需将模型和数据放到CPU设备上即可。

### Q: 如何添加新的插值算法？

A: 请参考 [扩展指南](docs/usage.md#添加新算法) 添加自定义算法。

## 贡献 | Contributing

欢迎提交Pull Request或Issue！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 许可证 | License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用 | Citation

如果您使用了本项目，请引用：

```bibtex
@software{MALA,
  title = {MALA: Masked Autoencoder for Remote Sensing Image Completion},
  author = {Jin Yuanyu},
  year = {2024},
  url = {https://github.com/jinyuanyu/MALA}
}
```

## 联系方式 | Contact

- GitHub Issues: https://github.com/jinyuanyu/MALA/issues
- Email: jinyuanyu@example.com

---

<p align="center">
  如果您觉得这个项目有帮助，请给我们一个 ⭐️！
</p>
