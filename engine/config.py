"""
MALA配置对象
===========

使用 dataclass 统一管理训练与推理的核心参数，
避免命令行脚本中散落大量同义字段。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class DataConfig:
    data_dir: str
    ocean_mask_path: str | None = None
    max_seq_len: int = 8
    batch_size: int = 4
    num_workers: int = 2
    mask_type: str = "random"
    mask_ratio: float = 0.5


@dataclass(slots=True)
class ModelConfig:
    img_size_h: int = 224
    img_size_w: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    max_seq_len: int = 8
    use_lama_init: bool = False
    use_ocean_prior: bool = False
    freeze_backbone: bool = False
    use_mask_channel: bool = False
    out_channels: int = 3
    dropout: float = 0.1

    def to_model_kwargs(self) -> dict:
        return {
            "img_size_h": self.img_size_h,
            "img_size_w": self.img_size_w,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "use_lama_init": self.use_lama_init,
            "use_ocean_prior": self.use_ocean_prior,
            "freeze_backbone": self.freeze_backbone,
            "use_mask_channel": self.use_mask_channel,
            "out_channels": self.out_channels,
            "dropout": self.dropout,
        }


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    pretrained_path: str | None = None
    log_interval: int = 10
    checkpoint_every: int = 10
    device: str = "cuda"


@dataclass(slots=True)
class InferenceConfig:
    model_path: str
    output_dir: str = "results"
    save_images: bool = False
    save_visualization: bool = False
    device: str = "cuda"


def resolve_device(requested_device: str = "cuda") -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested_device
