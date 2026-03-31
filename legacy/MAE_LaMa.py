"""
MALA 兼容层
===========

该文件保留历史研究脚本常用的类名与函数名，
内部实现已经转发到模块化后的 data / models / engine / utils 目录。
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from data.dataset import Datasets, Datasets_inference
from engine.inference import select_rgb_channels
from engine.losses import compute_train_batch_loss
from engine.trainer import create_scheduler, load_pretrained_weights
from models.modules import PatchDecoder, PatchEmbedding, TemporalAttention
from models.video_completion import LamaInpaintingModule, VideoCompletionModel
from utils.metrics import calculate_metrics_torch
from utils.visualization import denormalize_image

__all__ = [
    "Datasets",
    "Datasets_inference",
    "PatchEmbedding",
    "TemporalAttention",
    "PatchDecoder",
    "LamaInpaintingModule",
    "VideoCompletionModel",
    "train",
    "inference_with_pretrained",
]


def train(
    model,
    dataloader,
    optimizer,
    device,
    criterion,
    epochs: int = 10,
    pretrained_path: str | None = None,
):
    """
    历史训练接口兼容包装。

    保留原有签名，内部复用模块化后的损失计算与调度逻辑。
    """
    if pretrained_path:
        model = load_pretrained_weights(model, pretrained_path, device)
        trainable_params = filter(lambda parameter: parameter.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=1e-2)

    model.train()
    scheduler = create_scheduler(optimizer)

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            video = batch["video"].to(device)
            masked_video = batch["masked"].to(device)
            mask = batch["mask"].to(device)
            ocean_mask = batch["ocean_mask"].to(device)

            optimizer.zero_grad()
            batch_loss, _, _ = compute_train_batch_loss(
                model=model,
                video=video,
                masked_video=masked_video,
                mask=mask,
                ocean_mask=ocean_mask,
                criterion=criterion,
                use_lama=getattr(model, "use_lama_init", False),
            )
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / max(len(dataloader), 1)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def _save_legacy_visualization(video, masked_video, combined, times, frame_index: int) -> None:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    original_img = denormalize_image(video[0, frame_index]).cpu()
    plt.imshow(original_img.permute(1, 2, 0))
    plt.title(f"Original Frame\nTime: {times[0, frame_index].item()}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    masked_img = denormalize_image(masked_video[0, frame_index]).cpu()
    plt.imshow(masked_img.permute(1, 2, 0))
    plt.title("Masked Frame")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    recon_img = denormalize_image(combined[0, frame_index]).cpu()
    plt.imshow(recon_img.permute(1, 2, 0))
    plt.title("Reconstructed Frame")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("inference_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def _save_legacy_frames(combined, times, input_seq_len: int) -> None:
    os.makedirs("./inpainted_VMAE", exist_ok=True)
    for frame_index in range(min(input_seq_len, combined.shape[1])):
        reconstructed = denormalize_image(combined[0, frame_index]).cpu()
        if reconstructed.shape[0] == 1:
            recon_img = Image.fromarray((reconstructed.squeeze(0).numpy() * 255).astype("uint8"), "L")
        else:
            recon_img = Image.fromarray(
                (reconstructed.permute(1, 2, 0).numpy() * 255).astype("uint8")
            )
        recon_img.save(f"./inpainted_VMAE/reconstructed_frame_{times[0, frame_index].item():04d}.png")


def inference_with_pretrained(
    out_channels,
    model_path,
    data_dir,
    model,
    dataset,
    dataloader,
    input_seq_len: int = 8,
):
    """
    历史推理接口兼容包装。

    保留原有参数列表，内部使用模块化后的输出后处理与指标计算。
    """
    del data_dir, dataset  # 兼容旧接口，当前逻辑不再直接依赖它们

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sample = next(iter(dataloader))
    video = sample["video"].to(device)
    masked_video = sample["masked"].to(device)
    mask = sample["mask"].to(device)
    times = sample["times"]
    ocean_mask = sample["ocean_mask"].to(device)

    with torch.no_grad():
        output = model(masked_video, mask, ocean_mask)
        output_rgb = select_rgb_channels(output)
        if output_rgb.shape[2] > 3:
            output_rgb = output_rgb[:, :, : max(1, min(3, out_channels - 1)), :, :]
        combined = output_rgb * mask + video * (1 - mask)

    frame_index = min(2, combined.shape[1] - 1)
    _save_legacy_visualization(video, masked_video, combined, times, frame_index)

    print("\n逐帧评估指标（仅mask区域）:")
    print(f"{'Time':<10}{'SSIM':<10}{'PSNR (dB)':<12}{'MAE':<8}")
    print("-" * 40)

    for current_index in range(min(input_seq_len, combined.shape[1])):
        original = denormalize_image(video[0, current_index])
        reconstructed = denormalize_image(combined[0, current_index])
        ssim_val, psnr_val, mae_val = calculate_metrics_torch(
            original,
            reconstructed,
            mask[0, current_index].cpu(),
        )
        print(f"{times[0, current_index].item():<10}{ssim_val:.4f}{'':<2}{psnr_val:.2f}{'':<5}{mae_val:.4f}")

    _save_legacy_frames(combined, times, input_seq_len)
    return combined
