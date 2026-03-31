"""
MALA对象构建器
==============

将数据集、DataLoader 与模型实例化逻辑集中管理。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from data.dataset import Datasets, Datasets_inference
from models.video_completion import VideoCompletionModel

from .config import DataConfig, ModelConfig


def build_train_dataset(config: DataConfig) -> Datasets:
    return Datasets(
        data_dir=config.data_dir,
        max_seq_len=config.max_seq_len,
        ocean_mask_path=config.ocean_mask_path,
    )


def build_inference_dataset(config: DataConfig) -> Datasets_inference:
    return Datasets_inference(
        data_dir=config.data_dir,
        max_seq_len=config.max_seq_len,
        ocean_mask_path=config.ocean_mask_path,
        mask_type=config.mask_type,
        mask_ratio=config.mask_ratio,
    )


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_model(config: ModelConfig, device: str) -> VideoCompletionModel:
    model = VideoCompletionModel(**config.to_model_kwargs())
    return model.to(device)
