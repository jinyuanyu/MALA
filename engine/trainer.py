"""
MALA训练引擎
============

封装训练循环、验证流程、权重加载与检查点保存。
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainConfig
from .losses import compute_train_batch_loss, ensure_target_channels


def _move_batch_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def load_pretrained_weights(model: nn.Module, pretrained_path: str | None, device: str) -> nn.Module:
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"从 {pretrained_path} 加载预训练权重")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {
            key: value
            for key, value in pretrained_dict.items()
            if key in model_dict and value.shape == model_dict[key].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"已加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    return model


def create_scheduler(optimizer: optim.Optimizer):
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )


def train_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    use_lama: bool = False,
    log_interval: int = 10,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        batch = _move_batch_to_device(batch, device)
        video = batch["video"]
        masked_video = batch["masked"]
        mask = batch["mask"]
        ocean_mask = batch["ocean_mask"]

        optimizer.zero_grad()
        batch_loss, _, stats = compute_train_batch_loss(
            model=model,
            video=video,
            masked_video=masked_video,
            mask=mask,
            ocean_mask=ocean_mask,
            criterion=criterion,
            use_lama=use_lama,
        )
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        if batch_idx % log_interval == 0:
            stats_text = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {batch_loss.item():.4f} ({stats_text})"
            )

    return total_loss / max(len(dataloader), 1)


def validate_epoch(
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            video = batch["video"]
            masked_video = batch["masked"]
            mask = batch["mask"]
            ocean_mask = batch["ocean_mask"]

            reconstructed = model(masked_video, mask, ocean_mask)
            target = ensure_target_channels(reconstructed, video, mask)
            loss = criterion(255 * reconstructed * mask, 255 * target * mask)
            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def run_training(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    config: TrainConfig,
    use_lama: bool = False,
) -> str:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    best_checkpoint_path = os.path.join(config.checkpoint_dir, "model_best.pth")

    print("开始训练...")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
            epoch=epoch,
            use_lama=use_lama,
            log_interval=config.log_interval,
        )
        print(f"Epoch [{epoch}/{config.epochs}] Train Loss: {train_loss:.4f}")
        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"保存最佳模型到 {best_checkpoint_path}")

        if epoch % config.checkpoint_every == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    print("训练完成!")
    return best_checkpoint_path
