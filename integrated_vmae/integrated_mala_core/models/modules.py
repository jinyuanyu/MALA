"""
MALA项目模型模块
================

本模块包含视频修复模型的核心组件：
- Patch嵌入层：将图像分割成patch并转换为特征向量
- 时间注意力机制：捕捉时序依赖关系
- Patch解码器：将特征向量重建为图像

作者: MALA Team
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Patch嵌入层
    
    将输入图像序列分割成固定大小的patch，并将每个patch映射到嵌入向量。
    支持可选的掩码通道输入，用于告知模型哪些区域被遮挡。
    
    输入形状: (B, T, C, H, W) - 批次、时间步、通道、高度、宽度
    输出形状: (B, T, N, D) - 批次、时间步、patch数量、嵌入维度
    """
    
    def __init__(
        self, 
        img_size_h: int = 224, 
        img_size_w: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3,
        embed_dim: int = 768,
        use_mask_channel: bool = False
    ):
        """
        初始化Patch嵌入层
        
        参数:
            img_size_h: 输入图像高度
            img_size_w: 输入图像宽度
            patch_size: patch大小（正方形）
            in_channels: 输入通道数（RGB为3）
            embed_dim: 嵌入向量维度
            use_mask_channel: 是否将掩码作为额外通道输入
        """
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.embed_dim = embed_dim
        self.use_mask_channel = use_mask_channel
        
        # 计算输入通道数：如果使用掩码通道，则+1
        conv_in_channels = in_channels  # + 1 if use_mask_channel else in_channels
        
        # 卷积层：将patch映射到嵌入向量
        # 使用卷积实现patch embedding，相当于对每个patch进行线性投影
        self.projection = nn.Conv2d(
            conv_in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B, T, C, H, W)
            mask: 可选的掩码张量，形状 (B, T, 1, H, W)
            
        返回:
            嵌入后的张量，形状 (B, T, N, D)
        """
        B, T, C, or_H, or_W = x.shape
        
        # 如果输入尺寸与模型期望尺寸不同，进行插值调整
        if or_H != self.img_size_h or or_W != self.img_size_w:
            x = F.interpolate(
                x.view(-1, C, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='bilinear', 
                align_corners=False
            )
            x = x.view(B, T, C, self.img_size_h, self.img_size_w)
        
        # 如果使用掩码通道，将掩码与图像拼接
        if self.use_mask_channel and mask is not None:
            mask = F.interpolate(
                mask.view(-1, 1, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='nearest'
            )
            mask = mask.view(B, T, 1, self.img_size_h, self.img_size_w)
            
            # 拼接图像和掩码： (B, T, C+1, H, W)
            x_with_mask = torch.cat([x, mask], dim=2)
            
            # 重排列并投影： (B*T, C+1, H, W) -> (B*T, D, H', W')
            x_with_mask = rearrange(x_with_mask, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_with_mask)
            
            # 再次重排列： (B*T, D, H', W') -> (B, T, N, D)
            x_embedded = rearrange(x_embedded, '(b t) d n_h n_w -> b t (n_h n_w) d', b=B, t=T)
        else:
            # 不使用掩码通道的简化版本
            x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_reshaped)
            x_embedded = rearrange(x_embedded, '(b t) d n_h n_w -> b t (n_h n_w) d', b=B, t=T)
            
        return x_embedded


class TemporalAttention(nn.Module):
    """
    时间注意力机制
    
    专注于捕捉时间序列中的依赖关系。使用可学习的时间偏置来编码时间位置信息。
    
    注意力计算：在每个空间位置（patch）上，计算不同时间步之间的注意力分数。
    这类似于Transformer中的自注意力，但只在时间维度上计算。
    
    输入形状: (B, N, T, D) - 批次、空间位置数量、时间步、特征维度
    输出形状: (B, N, T, D) - 相同的形状
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1
    ):
        """
        初始化时间注意力层
        
        参数:
            embed_dim: 嵌入向量维度
            num_heads: 注意力头数量
            max_seq_len: 最大序列长度（时间步数）
            dropout: Dropout比例
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 确保embed_dim能被num_heads整除
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 缩放因子，用于归一化注意力分数
        self.scale = self.head_dim ** -0.5
        
        # 可学习的时间位置偏置：编码不同时间步之间的相对位置关系
        self.temporal_bias = nn.Parameter(torch.randn(1, num_heads, max_seq_len, max_seq_len))
        
        # QKV投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B, N, T, D)
            
        返回:
            注意力增强后的张量，形状 (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        # 合并批次和空间维度： (B, N, T, D) -> (B*N, T, D)
        x_flat = x.reshape(B * N, T, D)
        
        # 线性变换得到Q, K, V
        # 然后重排列为多头格式： (B*N, T, D) -> (B*N, H, T, d)
        q = self.q_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数： (B*N, H, T, T)
        # QK^T / sqrt(d)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加时间偏置（广播到所有批次）
        attn_scores = attn_scores + self.temporal_bias[:, :, :T, :T]
        
        # 计算注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力到V： (B*N, H, T, d) -> (B*N, T, D)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).reshape(B * N, T, D)
        
        # 输出投影
        output = self.out_proj(context)
        
        # 恢复原始形状： (B*N, T, D) -> (B, N, T, D)
        output = output.view(B, N, T, D)
        
        return output


class PatchDecoder(nn.Module):
    """
    Patch解码器
    
    将编码后的patch特征向量重建为图像。
    使用转置卷积进行上采样，将patch特征图转换为完整图像。
    
    输入形状: (B, T, N, D) - 批次、时间步、patch数量、特征维度
    输出形状: (B, T, C, H, W) - 批次、时间步、通道、高度、宽度
    """
    
    def __init__(
        self, 
        img_size_h: int, 
        img_size_w: int, 
        patch_size: int, 
        embed_dim: int, 
        out_channels: int = 3
    ):
        """
        初始化Patch解码器
        
        参数:
            img_size_h: 输出图像高度
            img_size_w: 输出图像宽度
            patch_size: patch大小
            embed_dim: 嵌入维度
            out_channels: 输出通道数（RGB为3）
        """
        super().__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # 计算patch数量
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        
        # 转置卷积解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim, 
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            ),
        )
        
        # 如果输出尺寸不能被patch_size整除，需要额外的上采样层
        self.resize = None
        if (self.num_patches_h * patch_size != img_size_h) or \
           (self.num_patches_w * patch_size != img_size_w):
            self.resize = nn.Upsample(
                size=(img_size_h, img_size_w),
                mode='bilinear',
                align_corners=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B, T, N, D)
            
        返回:
            重建的图像，形状 (B, T, C, H, W)
        """
        B, T, N_D = x.shape
        D = N_D // (self.num_patches_h * self.num_patches_w)
        
        # 重排列维度： (B, T, N, D) -> (B, T, H', W', D)
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, D)
        
        # 重排列： (B, T, H', W', D) -> (B*T, D, H', W')
        x = rearrange(x, 'b t n_h n_w d -> (b t) d n_h n_w')
        
        # 解码： (B*T, D, H', W') -> (B*T, C, H, W)
        output = self.decoder(x)
        
        # 如有需要，调整输出尺寸
        if self.resize is not None:
            output = self.resize(output)
        
        # 恢复时间维度： (B*T, C, H, W) -> (B, T, C, H, W)
        output = rearrange(output, '(b t) c h w -> b t c h w', b=B)
        
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    Transformer中的FFN模块，包含两个线性变换和激活函数。
    
    输入形状: (B, N, T, D)
    输出形状: (B, N, T, D)
    """
    
    def __init__(self, embed_dim: int, ff_dim: int = None, dropout: float = 0.1):
        """
        初始化前馈网络
        
        参数:
            embed_dim: 输入/输出维度
            ff_dim: 中间层维度，默认为embed_dim * 4
            dropout: Dropout比例
        """
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
            
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.net(x)


class MAEEncoderBlock(nn.Module):
    """
    MAE编码器块
    
    包含时间注意力和前馈网络的组合块，使用残差连接和层归一化。
    
    结构：
    x -> LayerNorm -> TemporalAttention -> Add & Norm -> LayerNorm -> FFN -> Add & Norm -> output
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1
    ):
        """
        初始化编码器块
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout比例
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TemporalAttention(embed_dim, num_heads, max_seq_len, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量，形状 (B, N, T, D)
            
        返回:
            输出张量，形状 (B, N, T, D)
        """
        # 注意力块 with 残差连接
        x = x + self.attn(self.norm1(x))
        
        # 前馈块 with 残差连接
        x = x + self.ffn(self.norm2(x))
        
        return x
