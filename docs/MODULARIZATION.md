# MALA 模块化说明

## 当前结构

```text
MALA/
├── data/                 # 数据集与掩码组织
├── models/               # 模型结构与组件
├── utils/                # 指标与可视化
├── engine/               # 新增：训练/推理/配置/构建器
├── analysis/             # 新增：分析脚本共享辅助模块
├── MAE_LaMa.py           # 兼容层，转发到模块化实现
├── train.py              # 轻量训练入口
└── inference.py          # 轻量推理入口
```

## engine/ 目录职责

- `config.py`
  - 定义 `DataConfig`、`ModelConfig`、`TrainConfig`、`InferenceConfig`
  - 统一训练和推理参数对象
- `builders.py`
  - 负责数据集、DataLoader、模型实例化
- `losses.py`
  - 负责梯度损失、重建损失和 LaMA 协同损失
- `trainer.py`
  - 负责训练循环、预训练权重加载、检查点保存
- `inference.py`
  - 负责推理循环、指标聚合与结果输出

## analysis/ 目录职责

- `common.py`
  - 统一算法名称映射与中文绘图配置
- `experiment.py`
  - 统一实验目录遍历、结果图像发现与帧号提取

## 路径配置策略

当前主干已经尽量减少对固定 `E:/lama/...` 路径的依赖。

- `MALA_DATA_ROOT`
  - 将 `E:/...` 样式路径映射到新的数据根目录
- `MALA_MASK_DIR`
  - 覆盖默认掩码目录
- `MALA_LAMA_INIT_DIR`
  - 覆盖默认 LaMa 初始结果目录

这样在不同机器或不同挂载目录下运行时，不需要再逐个修改源码文件。

## 模块化后的好处

- 入口脚本更薄，更容易维护
- 训练与推理逻辑可复用，减少重复代码
- 新模型或新实验可以直接复用 `engine/` 能力
- 后续接入 `finetune`、`evaluate`、`ablation` 更容易扩展

## 兼容性策略

- `MAE_LaMa.py` 没有被直接删除
- 历史常用类名与函数名仍然保留
- 内部实现已经改为复用模块化后的实现
- 这样旧 notebook 和旧脚本可以逐步迁移，而不是一次性全部重写
