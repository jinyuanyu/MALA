"""
MALA工程化执行模块
==================

统一收拢训练、推理、配置与构建逻辑。
"""

from .builders import build_dataloader, build_inference_dataset, build_model, build_train_dataset
from .config import DataConfig, InferenceConfig, ModelConfig, TrainConfig, resolve_device
from .inference import run_inference, save_metrics_report
from .trainer import create_scheduler, load_pretrained_weights, run_training, train_epoch, validate_epoch

__all__ = [
    "DataConfig",
    "InferenceConfig",
    "ModelConfig",
    "TrainConfig",
    "resolve_device",
    "build_train_dataset",
    "build_inference_dataset",
    "build_dataloader",
    "build_model",
    "load_pretrained_weights",
    "create_scheduler",
    "train_epoch",
    "validate_epoch",
    "run_training",
    "run_inference",
    "save_metrics_report",
]
