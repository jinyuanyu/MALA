"""
MALA推理引擎
============

封装推理循环、指标汇总与结果文件写出。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from utils.metrics import calculate_metrics_torch
from utils.visualization import denormalize_image, save_reconstructed_frames, visualize_comparison


def select_rgb_channels(output: torch.Tensor) -> torch.Tensor:
    if output.shape[2] <= 3:
        return output
    return output[:, :, :3, :, :]


def run_inference(
    model: torch.nn.Module,
    dataloader,
    device: str,
    save_images: bool = False,
    save_visualization: bool = False,
    output_dir: str | None = None,
) -> dict[str, float]:
    model.eval()

    all_ssim: list[float] = []
    all_psnr: list[float] = []
    all_mae: list[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}...")

            video = batch["video"].to(device)
            masked_video = batch["masked"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"]
            ocean_mask = batch["ocean_mask"].to(device)

            output = model(masked_video, mask, ocean_mask)
            output_rgb = select_rgb_channels(output)
            combined = output_rgb * mask + video * (1 - mask)

            for t in range(output_rgb.shape[1]):
                original = denormalize_image(video[0, t])
                reconstructed = denormalize_image(combined[0, t])
                ssim_val, psnr_val, mae_val = calculate_metrics_torch(
                    original,
                    reconstructed,
                    mask[0, t].cpu(),
                )
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                all_mae.append(mae_val)
                print(f"  帧 {t}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, MAE={mae_val:.4f}")

            if save_images and output_dir:
                save_reconstructed_frames(
                    combined[0],
                    times[0],
                    output_dir,
                    prefix=f"batch_{batch_idx}_frame",
                )

            if save_visualization and output_dir:
                for t in range(min(3, output_rgb.shape[1])):
                    original_img = denormalize_image(video[0, t])
                    masked_img = denormalize_image(masked_video[0, t])
                    reconstructed_img = denormalize_image(combined[0, t])
                    visualize_comparison(
                        original_img.permute(1, 2, 0).cpu().numpy(),
                        masked_img.permute(1, 2, 0).cpu().numpy(),
                        reconstructed_img.permute(1, 2, 0).cpu().numpy(),
                        times=times[0, t].item(),
                        save_path=os.path.join(output_dir, f"comparison_batch{batch_idx}_frame{t}.png"),
                    )

    metrics = {
        "SSIM": float(np.mean(all_ssim)) if all_ssim else float("nan"),
        "PSNR": float(np.mean(all_psnr)) if all_psnr else float("nan"),
        "MAE": float(np.mean(all_mae)) if all_mae else float("nan"),
        "SSIM_std": float(np.std(all_ssim)) if all_ssim else float("nan"),
        "PSNR_std": float(np.std(all_psnr)) if all_psnr else float("nan"),
        "MAE_std": float(np.std(all_mae)) if all_mae else float("nan"),
    }
    return metrics


def save_metrics_report(metrics: dict[str, float], output_dir: str, filename: str = "metrics.txt") -> str:
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, filename)
    with open(metrics_path, "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    return metrics_path
