# MALA 模块化说明

## 当前结构

```text
MALA/
├── data/                 # 数据集与掩码组织
├── models/               # 模型结构与组件
├── utils/                # 指标与可视化
├── engine/               # 新增：训练/推理/配置/构建器
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
