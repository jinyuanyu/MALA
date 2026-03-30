"""
MALA项目训练脚本
================

本脚本提供模型训练功能，支持：
- 加载预训练权重
- 多种损失函数
- 学习率调度
- 断点续训

使用方法:
    python train.py --data_dir <数据路径> --epochs <轮数>

作者: MALA Team
日期: 2024
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 导入项目模块
from data.dataset import Datasets, Datasets_inference
from models.video_completion import VideoCompletionModel, create_model
from utils.metrics import calculate_metrics_torch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MALA模型训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练数据目录路径')
    parser.add_argument('--ocean_mask_path', type=str, default=None,
                       help='海洋掩码路径')
    parser.add_argument('--max_seq_len', type=int, default=8,
                       help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    
    # 模型参数
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
    parser.add_argument('--use_ocean_prior', action='store_true',
                       help='是否使用海洋先验')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='预训练模型路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志打印间隔')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    return parser.parse_args()


def gradient_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算梯度损失，提高空间连续性
    
    参数:
        x: 预测图像
        y: 目标图像
        
    返回:
        梯度损失值
    """
    dx_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dx_y = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    dy_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dy_y = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return nn.functional.l1_loss(dx_x, dx_y) + nn.functional.l1_loss(dy_x, dy_y)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    use_lama: bool = False
) -> float:
    """
    训练一个epoch
    
    参数:
        model: 视频修复模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前轮数
        use_lama: 是否使用LaMa
        
    返回:
        平均损失值
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据
        video = batch['video'].to(device)
        masked_video = batch['masked'].to(device)
        mask = batch['mask'].to(device)
        ocean_mask = batch['ocean_mask'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        if use_lama:
            # 获取MAE重构结果
            mae_output = model.forward_mae_only(masked_video, mask, ocean_mask)
            reconstructed = model(masked_video, mask, ocean_mask)
            
            # 分离RGB和掩码通道
            mae_rgb = mae_output[:, :, :3]
            mae_mask_pred = mae_output[:, :, 3:]
            
            # 1. MAE重构损失（在掩码区域）
            mae_loss = criterion(mae_rgb * mask, video * mask)
            
            # 2. 颜色一致性损失
            color_loss = nn.functional.l1_loss(
                mae_rgb.mean(dim=[3, 4]), 
                video.mean(dim=[3, 4])
            )
            
            # 3. 梯度损失
            grad_loss = gradient_loss(mae_rgb * mask, video * mask)
            
            # 4. 最终输出损失（LaMa处理后的结果）
            final_loss = criterion(reconstructed * mask, video * mask)
            
            # 组合损失
            total_batch_loss = 125 * final_loss + 125 * mae_loss + 10 * color_loss + 5 * grad_loss
        else:
            reconstructed = model(masked_video, mask, ocean_mask)
            
            if reconstructed.shape[2] != video.shape[2]:
                video = torch.cat([video, mask], dim=2)
            
            total_batch_loss = criterion(255 * reconstructed * mask, 255 * video * mask)
        
        # 反向传播
        total_batch_loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += total_batch_loss.item()
        
        # 打印日志
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {total_batch_loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """
    验证模型
    
    参数:
        model: 视频修复模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    返回:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            masked_video = batch['masked'].to(device)
            mask = batch['mask'].to(device)
            ocean_mask = batch['ocean_mask'].to(device)
            
            reconstructed = model(masked_video, mask, ocean_mask)
            
            if reconstructed.shape[2] != video.shape[2]:
                video = torch.cat([video, mask], dim=2)
            
            loss = criterion(255 * reconstructed * mask, 255 * video * mask)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def load_pretrained_weights(model: nn.Module, pretrained_path: str, device: str):
    """
    加载预训练权重
    
    参数:
        model: 模型
        pretrained_path: 预训练权重路径
        device: 计算设备
        
    返回:
        更新后的模型
    """
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"从 {pretrained_path} 加载预训练权重")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        
        # 过滤可加载的权重
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        # 更新模型字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"已加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    
    return model


def main():
    """主训练函数"""
    # 解析参数
    args = parse_args()
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("加载数据集...")
    train_dataset = Datasets(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        ocean_mask_path=args.ocean_mask_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = VideoCompletionModel(
        img_size_h=args.img_size_h,
        img_size_w=args.img_size_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_lama_init=args.use_lama_init,
        use_ocean_prior=args.use_ocean_prior
    )
    model = model.to(device)
    
    # 加载预训练权重
    if args.pretrained_path:
        model = load_pretrained_weights(model, args.pretrained_path, device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, epoch, args.use_lama_init
        )
        
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f}")
        
        # 更新学习率
        scheduler.step(train_loss)
        
        # 保存检查点
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'model_best.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存最佳模型到 {checkpoint_path}")
        
        # 定期保存
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'model_epoch_{epoch}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
    
    print("训练完成!")


if __name__ == '__main__':
    main()
