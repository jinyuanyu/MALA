"""
MALA损失函数模块
================

统一管理训练时的重建损失与 LaMA 协同损失。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dx_y = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    dy_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dy_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


def ensure_target_channels(
    reconstructed: torch.Tensor,
    video: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if reconstructed.shape[2] != video.shape[2]:
        return torch.cat([video, mask], dim=2)
    return video


def compute_train_batch_loss(
    model,
    video: torch.Tensor,
    masked_video: torch.Tensor,
    mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    criterion,
    use_lama: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if use_lama:
        mae_output = model.forward_mae_only(masked_video, mask, ocean_mask)
        reconstructed = model(masked_video, mask, ocean_mask)

        mae_rgb = mae_output[:, :, :3]
        mae_loss = criterion(mae_rgb * mask, video * mask)
        color_loss = F.l1_loss(mae_rgb.mean(dim=[3, 4]), video.mean(dim=[3, 4]))
        grad = gradient_loss(mae_rgb * mask, video * mask)
        final_loss = criterion(reconstructed * mask, video * mask)

        total_loss = 125 * final_loss + 125 * mae_loss + 10 * color_loss + 5 * grad
        stats = {
            "final_loss": float(final_loss.detach().item()),
            "mae_loss": float(mae_loss.detach().item()),
            "color_loss": float(color_loss.detach().item()),
            "grad_loss": float(grad.detach().item()),
        }
        return total_loss, reconstructed, stats

    reconstructed = model(masked_video, mask, ocean_mask)
    target = ensure_target_channels(reconstructed, video, mask)
    total_loss = criterion(255 * reconstructed * mask, 255 * target * mask)
    stats = {"reconstruction_loss": float(total_loss.detach().item())}
    return total_loss, reconstructed, stats
