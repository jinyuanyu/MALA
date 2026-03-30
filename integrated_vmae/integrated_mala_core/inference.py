"""
MALA项目推理脚本
================

本脚本提供模型推理功能，支持：
- 批量推理
- 结果保存
- 指标计算

使用方法:
    python inference.py --model_path <模型路径> --data_dir <数据路径>

作者: MALA Team
日期: 2024
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入项目模块
from data.dataset import Datasets_inference
from models.video_completion import VideoCompletionModel
from utils.metrics import calculate_metrics_torch
from utils.visualization import (
    denormalize_image,
    visualize_comparison,
    save_reconstructed_frames
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MALA模型推理')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='推理数据目录路径')
    parser.add_argument('--ocean_mask_path', type=str, default=None,
                       help='海洋掩码路径')
    parser.add_argument('--max_seq_len', type=int, default=8,
                       help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批次大小')
    parser.add_argument('--mask_type', type=str, default='random',
                       choices=['random', 'cloud', 'strip', 'mixed', 'predefined'],
                       help='掩码类型')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                       help='掩码缺失比例')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--img_size_h', type=int, default=224,
                       help='图像高度')
    parser.add_argument('--img_size_w', type=int, default=224,
                       help='图像宽度')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch大小')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='注意力头数')
    parser.add_argument('--use_lama_init', action='store_true',
                       help='是否使用LaMa初始修复')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--save_images', action='store_true',
                       help='是否保存重建图像')
    parser.add_argument('--save_visualization', action='store_true',
                       help='是否保存可视化结果')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    return parser.parse_args()


def inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    save_images: bool = False,
    output_dir: str = None
):
    """
    执行推理
    
    参数:
        model: 视频修复模型
        dataloader: 数据加载器
        device: 计算设备
        save_images: 是否保存图像
        output_dir: 输出目录
        
    返回:
        指标字典
    """
    model.eval()
    
    all_ssim = []
    all_psnr = []
    all_mae = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"处理批次 {batch_idx + 1}...")
            
            # 获取数据
            video = batch['video'].to(device)
            masked_video = batch['masked'].to(device)
            mask = batch['mask'].to(device)
            times = batch['times']
            ocean_mask = batch['ocean_mask'].to(device)
            
            # 前向传播
            output = model(masked_video, mask, ocean_mask)
            
            # 处理输出通道
            if output.shape[2] in (1, 3):
                output_rgb = output[:, :, :, :]
            else:
                output_rgb = output[:, :, :2, :]
            
            # 组合未缺失和重构区域
            combined = output_rgb * mask + video * (1 - mask)
            
            # 计算指标（仅在掩码区域）
            for t in range(output.shape[1]):
                original = denormalize_image(video[0, t])
                reconstructed = denormalize_image(combined[0, t])
                
                ssim_val, psnr_val, mae_val = calculate_metrics_torch(
                    original,
                    reconstructed,
                    mask[0, t].cpu()
                )
                
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)
                all_mae.append(mae_val)
                
                print(f"  帧 {t}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}, MAE={mae_val:.4f}")
            
            # 保存图像
            if save_images and output_dir:
                save_reconstructed_frames(
                    combined[0],
                    times[0],
                    output_dir,
                    prefix=f"batch_{batch_idx}_frame"
                )
            
            # 保存可视化
            if save_images and output_dir:
                for t in range(min(3, output.shape[1])):
                    original_img = denormalize_image(video[0, t])
                    masked_img = denormalize_image(masked_video[0, t])
                    reconstructed_img = denormalize_image(combined[0, t])
                    
                    visualize_comparison(
                        original_img.permute(1, 2, 0).cpu().numpy(),
                        masked_img.permute(1, 2, 0).cpu().numpy(),
                        reconstructed_img.permute(1, 2, 0).cpu().numpy(),
                        times=times[0, t].item(),
                        save_path=os.path.join(
                            output_dir, 
                            f'comparison_batch{batch_idx}_frame{t}.png'
                        )
                    )
    
    # 计算平均指标
    metrics = {
        'SSIM': np.mean(all_ssim),
        'PSNR': np.mean(all_psnr),
        'MAE': np.mean(all_mae),
        'SSIM_std': np.std(all_ssim),
        'PSNR_std': np.std(all_psnr),
        'MAE_std': np.std(all_mae)
    }
    
    return metrics


def main():
    """主推理函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    print("加载数据集...")
    dataset = Datasets_inference(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        ocean_mask_path=args.ocean_mask_path,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"推理样本数: {len(dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = VideoCompletionModel(
        img_size_h=args.img_size_h,
        img_size_w=args.img_size_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_lama_init=args.use_lama_init
    )
    model = model.to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # 执行推理
    print("开始推理...")
    metrics = inference(
        model,
        dataloader,
        device,
        save_images=args.save_images,
        output_dir=args.output_dir
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("推理结果")
    print("=" * 50)
    print(f"SSIM: {metrics['SSIM']:.4f} ± {metrics['SSIM_std']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.2f} ± {metrics['PSNR_std']:.2f}")
    print(f"MAE:  {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")
    
    # 保存指标到文件
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"\n指标已保存到: {metrics_path}")


if __name__ == '__main__':
    main()
