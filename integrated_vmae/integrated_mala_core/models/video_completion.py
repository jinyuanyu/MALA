"""
MALA项目主模型定义
==================

本模块整合了MAE编码器、LaMa修复模块和完整视频修复模型。
支持仅MAE前向传播和MAE-LaMa协作式修复两种模式。

作者: MALA Team
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List
from .modules import PatchEmbedding, TemporalAttention, PatchDecoder, MAEEncoderBlock


class LamaInpaintingModule(nn.Module):
    """
    LaMa图像修复模块
    
    基于LaMa算法的图像修复模块，用于对掩码区域进行修复。
    该模块需要配合simple_lama_inpainting库使用。
    
    输入：RGB图像 + 二值掩码
    输出：修复后的RGB图像
    
    注意：此模块需要安装 simple_lama_inpainting 库
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化LaMa修复模块
        
        参数:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        super().__init__()
        self.device = device
        
        # 图像转换工具
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 反归一化变换：将[-1, 1]映射到[0, 1]
        self.denormalize = transforms.Normalize(
            mean=[-1.0, -1.0, -1.0],
            std=[2.0, 2.0, 2.0]
        )
        
        # 归一化变换：将[0, 1]映射到[-1, 1]
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
        
        # 初始化LaMa模型
        try:
            from simple_lama_inpainting import SimpleLama
            self.lama = SimpleLama()
        except ImportError:
            print("警告: simple_lama_inpainting库未安装，LaMa模块将使用占位符")
            self.lama = None

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：逐帧修复
        
        参数:
            x: 输入图像，形状 (B, T, C, H, W)
            mask: 掩码张量，形状 (B, T, 1, H, W)
            
        返回:
            修复后的图像，形状 (B, T, C, H, W)
        """
        if self.lama is None:
            # 如果LaMa未初始化，返回输入
            return x
            
        batch_size = x.shape[0]
        repaired_images = []
        
        for i in range(batch_size):
            for j in range(x.shape[1]):
                # 获取当前帧
                img_tensor = self.denormalize(x[i, j]).cpu()
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                
                # 获取当前帧掩码
                mask_tensor = mask[i, j, 0].cpu()
                
                # 转换为PIL图像
                img_pil = self.to_pil(img_tensor)
                mask_array = (mask_tensor.numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_array, mode='L')
                
                # LaMa修复
                repaired_pil = self.lama(img_pil, mask_pil)
                
                # 确保尺寸一致
                if repaired_pil.size != img_pil.size:
                    repaired_pil = repaired_pil.resize(img_pil.size, Image.BILINEAR)
                
                # 转换回张量并归一化
                repaired_tensor = self.to_tensor(repaired_pil)
                repaired_tensor = self.normalize(repaired_tensor)
                repaired_images.append(repaired_tensor)
        
        # 重组为批次
        repaired_batch = torch.stack(repaired_images).to(self.device)
        repaired_batch = repaired_batch.view(
            batch_size, 
            x.shape[1], 
            -1, 
            repaired_batch.shape[-2], 
            repaired_batch.shape[-1]
        )
        
        return repaired_batch


class VideoCompletionModel(nn.Module):
    """
    视频修复完整模型
    
    整合MAE编码器、时序注意力、解码器和可选的LaMa修复模块。
    支持两种工作模式：
    1. 仅MAE：快速但质量有限
    2. MAE-LaMa协作：质量更高但速度较慢
    
    模型架构：
    输入 -> PatchEmbedding -> TemporalAttention -> 多层EncoderBlock 
    -> PatchDecoder -> 可选的LaMa refinement -> 输出
    """
    
    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        max_seq_len: int = 8,
        use_lama_init: bool = False,
        use_ocean_prior: bool = False,
        freeze_backbone: bool = False,
        fine_tune_layers: Optional[List[str]] = None,
        use_mask_channel: bool = False,
        out_channels: int = 3,
        dropout: float = 0.1
    ):
        """
        初始化视频修复模型
        
        参数:
            img_size_h: 输入图像高度
            img_size_w: 输入图像宽度
            patch_size: patch大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            use_lama_init: 是否使用LaMa初始修复
            use_ocean_prior: 是否使用海洋先验
            freeze_backbone: 是否冻结骨干网络
            fine_tune_layers: 需要微调的层列表
            use_mask_channel: 是否使用掩码通道
            out_channels: 输出通道数
            dropout: Dropout比例
        """
        super().__init__()
        
        # Patch嵌入层
        self.patch_embedding = PatchEmbedding(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=out_channels,
            embed_dim=embed_dim,
            use_mask_channel=use_mask_channel,
            use_lama_init=False
        )
        
        # 时间注意力层
        self.temporal_attention = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = PatchDecoder(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            embed_dim=embed_dim,
            out_channels=out_channels
        )
        
        # 多层时间注意力编码器块
        self.atten_layers = nn.ModuleList([
            TemporalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout
            ) for _ in range(3)
        ])
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 配置标志
        self.use_lama_init = use_lama_init
        self.use_ocean_prior = use_ocean_prior
        
        # LaMa修复模块
        if self.use_lama_init:
            self.lama_module = LamaInpaintingModule(
                device=next(self.parameters()).device
            )
        
        # 掩码更新模块：基于MAE结果学习更好的掩码
        if use_ocean_prior:
            self.mask_update_layer = nn.Sequential(
                nn.Conv2d(out_channels + 1, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.mask_update_layer = nn.Sequential(
                nn.Conv2d(out_channels, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 冻结控制
        self.freeze_backbone = freeze_backbone
        self.fine_tune_layers = fine_tune_layers or []
        
        if self.freeze_backbone:
            self.set_freeze_status()

    def set_freeze_status(self):
        """
        设置参数冻结状态
        
        冻结骨干网络参数，只保留指定层的可训练性
        """
        # 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻指定层
        for name, param in self.named_parameters():
            # 解冻骨干网络
            if not self.freeze_backbone and ('patch_embedding' in name or 'temporal_attention' in name):
                param.requires_grad = True
            
            # 解冻微调层
            for layer_name in self.fine_tune_layers:
                if layer_name in name:
                    param.requires_grad = True
            
            # 确保这些层总是可训练
            if 'mask_update_layer' in name or 'decoder' in name or 'lama_module' in name:
                param.requires_grad = True
        
        # 打印可训练参数信息
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"可训练参数: {trainable}/{total} ({100. * trainable / total:.2f}%)")

    def enhance_lama_input_with_mae(
        self,
        mae_reconstructed: torch.Tensor,
        original_input: torch.Tensor,
        mask: torch.Tensor,
        ocean_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用MAE重构结果增强LaMa的输入和掩码
        
        参数:
            mae_reconstructed: MAE重构结果 (B, T, C+1, H, W)
            original_input: 原始输入 (B, T, C, H, W)
            mask: 原始掩码 (B, T, 1, H, W)
            ocean_mask: 海洋掩码 (B, 1, H, W)
            
        返回:
            updated_mask: 更新后的掩码 (B, T, 1, H, W)
        """
        B, T, C_plus, H, W = mae_reconstructed.shape
        C = C_plus - 1  # RGB通道数
        
        updated_masks = []
        
        for t in range(T):
            mae_frame = mae_reconstructed[:, t, :C]
            mae_mask = mae_reconstructed[:, t, C:]
            original_frame = original_input[:, t]
            mask_frame = mask[:, t]
            
            # 计算MAE重构质量
            mae_quality = 1.0 - torch.abs(mae_frame - original_frame).mean(dim=1, keepdim=True)
            mae_quality = torch.sigmoid(mae_quality * 10 - 5)
            
            if self.use_ocean_prior and ocean_mask is not None:
                # 使用海洋先验
                mask_update_weight = self.mask_update_layer(
                    torch.cat((mae_reconstructed[:, t], ocean_mask), dim=1)
                )
                
                ocean_weight = (ocean_mask > 0.5).float()
                updated_mask_frame = (
                    mask_frame * (1 - mask_update_weight * mae_quality * 0.7) *
                    (1 - ocean_weight * 0.3) +
                    mae_mask * mask_update_weight * (1 - mae_quality) * 0.5 *
                    (1 + ocean_weight * 0.5) +
                    ocean_weight * 0.1
                ).clamp(0, 1)
            else:
                # 不使用海洋先验
                mask_update_weight = self.mask_update_layer(mae_reconstructed[:, t])
                updated_mask_frame = (
                    mask_frame * (1 - mask_update_weight * mae_quality * 0.5) +
                    mae_mask * mask_update_weight * (1 - mae_quality) * 0.3
                ).clamp(0, 1)
            
            updated_masks.append(updated_mask_frame)
        
        updated_mask = torch.stack(updated_masks, dim=1)
        
        return updated_mask

    def iterative_mae_lama_refinement(
        self,
        mae_reconstructed: torch.Tensor,
        original_input: torch.Tensor,
        mask: torch.Tensor,
        ocean_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        迭代式MAE-LaMa协作优化
        
        参数:
            mae_reconstructed: MAE重构结果 (B, T, C+1, H, W)
            original_input: 原始输入 (B, T, C, H, W)
            mask: 原始掩码 (B, T, 1, H, W)
            ocean_mask: 海洋掩码
            
        返回:
            final_result: 最终重构结果 (B, T, C, H, W)
        """
        # 增强LaMa输入
        updated_mask = self.enhance_lama_input_with_mae(
            mae_reconstructed, original_input, mask, ocean_mask
        )
        
        # 分离梯度
        with torch.no_grad():
            updated_mask_detached = updated_mask.detach()
            lama_results = self.lama_module(mae_reconstructed, updated_mask_detached)
        
        # 重新连接到计算图
        lama_results = lama_results.detach().requires_grad_(True)
        
        return lama_results

    def forward_mae_only(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ocean_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        仅执行MAE部分，返回重构结果（不经过LaMa处理）
        
        参数:
            x: 输入图像 (B, T, C, H, W)
            mask: 掩码 (B, T, 1, H, W)
            ocean_mask: 海洋掩码
            
        返回:
            mae_reconstructed: MAE重构结果 (B, T, C+1, H, W)
        """
        B, T, C, H, W = x.shape
        
        # 1. Patch嵌入 (B, T, N, D)
        x_embedded = self.patch_embedding(x, mask)
        
        # 2. 重新排列为 (B, N, T, D)
        x_pos = rearrange(x_embedded, 'b t n d -> b n t d')
        
        # 3. 时间注意力机制
        attn_output = self.temporal_attention(x_pos)
        x_pos = x_pos + attn_output
        x_pos = self.norm1(x_pos)
        
        # 4. 多层时间注意力
        for layer in self.atten_layers:
            attn_output = layer(x_pos)
            x_pos = x_pos + attn_output
            x_pos = self.norm1(x_pos)
            
            ff_output = self.ffn(x_pos)
            x_pos = x_pos + ff_output
            x_pos = self.norm2(x_pos)
        
        # 5. 恢复原始形状 (B, T, N, D)
        x_out = rearrange(x_pos, 'b n t d -> b t (n d)')
        
        # 6. 解码器
        mae_reconstructed = self.decoder(x_out)
        
        return mae_reconstructed

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ocean_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整前向传播
        
        参数:
            x: 输入图像 (B, T, C, H, W)
            mask: 掩码 (B, T, 1, H, W)
            ocean_mask: 海洋掩码
            
        返回:
            重构结果 (B, T, C, H, W)
        """
        # 1. Patch嵌入
        x_embedded = self.patch_embedding(x, mask)
        
        # 2. 重排列
        x_pos = rearrange(x_embedded, 'b t n d -> b n t d')
        
        # 3. 时间注意力
        attn_output = self.temporal_attention(x_pos)
        x_pos = x_pos + attn_output
        x_pos = self.norm1(x_pos)
        
        # 4. 多层时间注意力
        for layer in self.atten_layers:
            attn_output = layer(x_pos)
            x_pos = x_pos + attn_output
            x_pos = self.norm1(x_pos)
            
            ff_output = self.ffn(x_pos)
            x_pos = x_pos + ff_output
            x_pos = self.norm2(x_pos)
        
        # 5. 恢复形状
        x_out = rearrange(x_pos, 'b n t d -> b t (n d)')
        
        # 6. 解码器
        mae_reconstructed = self.decoder(x_out)
        
        # 7. 如果启用LaMa，则进行协作式重构
        if self.use_lama_init:
            final_result = self.iterative_mae_lama_refinement(
                mae_reconstructed, x, mask, ocean_mask
            ).to(x.device)
            return final_result
        else:
            return mae_reconstructed


# 导入必要的类型和模块
from typing import Optional
import numpy as np
from PIL import Image
from torchvision import transforms


def create_model(config: dict) -> VideoCompletionModel:
    """
    根据配置创建模型的工厂函数
    
    参数:
        config: 模型配置字典
        
    返回:
        初始化好的模型
    """
    return VideoCompletionModel(
        img_size_h=config.get('img_size_h', 224),
        img_size_w=config.get('img_size_w', 224),
        patch_size=config.get('patch_size', 16),
        embed_dim=config.get('embed_dim', 768),
        num_heads=config.get('num_heads', 12),
        max_seq_len=config.get('max_seq_len', 8),
        use_lama_init=config.get('use_lama_init', False),
        use_ocean_prior=config.get('use_ocean_prior', False),
        freeze_backbone=config.get('freeze_backbone', False),
        fine_tune_layers=config.get('fine_tune_layers', None),
        use_mask_channel=config.get('use_mask_channel', False),
        out_channels=config.get('out_channels', 3),
        dropout=config.get('dropout', 0.1)
    )
