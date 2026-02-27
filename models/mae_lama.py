# -*- coding: utf-8 -*-
"""
================================================================================
MALA 主模型模块 - MAE-LaMa 视频补全模型
================================================================================

该模块实现了基于掩码自编码器(MAE)的遥感图像时序填补模型。

Model Architecture:
-------------------
1. PatchEmbedding: 将输入图像转换为patch序列，并添加位置编码
2. TemporalAttention: 时间维度的自注意力机制，捕捉时序依赖关系
3. PatchDecoder: 将patch序列解码回图像空间

主要功能:
---------
- 支持随机掩码、厚云掩码、薄云掩码、条带掩码等多种缺失类型
- 集成LaMa图像修复模块作为辅助
- 支持Ocean_mask进行海洋区域识别
- 完整的训练和推理流程

Author: Jin Yuanyu
Email: jinyuanyu@example.com
Institution: Remote Sensing Laboratory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Dict, List
import math
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


class PatchEmbedding(nn.Module):
    """
    =====================================================================
    Patch Embedding 层
    =====================================================================
    
    将输入的图像序列转换为patch序列，并投影到隐藏维度空间。
    这是Transformer架构的标准做法，将图像划分为固定大小的patches。
    
    参数:
    ------
    img_size_h : int
        输入图像的高度
    img_size_w : int  
        输入图像的宽度
    patch_size : int
        每个patch的大小（通常为16x16）
    in_channels : int
        输入通道数（RGB为3）
    embed_dim : int
        嵌入维度，即每个patch的向量长度
    use_mask_channel : bool
        是否在输入中添加mask通道
        
    示例:
    -----
    >>> patch_embed = PatchEmbedding(224, 224, 16, 3, 768)
    >>> x = torch.randn(2, 8, 3, 224, 224)  # (B, T, C, H, W)
    >>> mask = torch.randn(2, 8, 1, 224, 224)
    >>> output = patch_embed(x, mask)  # (B, T, N, D)
    """
    
    def __init__(
        self, 
        img_size_h: int = 224, 
        img_size_w: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768,
        use_mask_channel: bool = True
    ):
        super().__init__()
        
        # 图像和patch参数
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_mask_channel = use_mask_channel
        
        # 计算patch数量
        self.num_patches_h = img_size_h // patch_size
        self.num_patches_w = img_size_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # 卷积投影层：将每个patch映射到embed_dim维度
        # 输入: (B*T, C+1, H, W) 如果使用mask通道
        # 输出: (B*T, embed_dim, H/patch_size, W/patch_size)
        conv_in_channels = in_channels + 1 if use_mask_channel else in_channels
        
        self.projection = nn.Conv2d(
            conv_in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )
        
        # 初始化权重（可选）
        self._init_weights()
    
    def _init_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
        ------
        x : torch.Tensor
            输入张量，形状为 (B, T, C, H, W)
        mask : torch.Tensor, optional
            掩码张量，形状为 (B, T, 1, H, W)
            
        返回:
        ------
        torch.Tensor
            嵌入后的patch序列，形状为 (B, T, N, D)
            其中 N = (H/patch_size) * (W/patch_size)
        """
        B, T, C, or_H, or_W = x.shape
        
        # 如果输入尺寸与设定不同，进行插值
        if or_H != self.img_size_h or or_W != self.img_size_w:
            x = F.interpolate(
                x.view(-1, C, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='bilinear', 
                align_corners=False
            )
            x = x.view(B, T, C, self.img_size_h, self.img_size_w)
        
        # 如果使用mask通道，将mask与图像拼接
        if self.use_mask_channel and mask is not None:
            # 对mask进行插值到目标尺寸
            mask = F.interpolate(
                mask.view(-1, 1, or_H, or_W),
                size=(self.img_size_h, self.img_size_w),
                mode='nearest'
            )
            mask = mask.view(B, T, 1, self.img_size_h, self.img_size_w)
            
            # 拼接: (B, T, C+1, H, W)
            x_with_mask = torch.cat([x, mask], dim=2)
            
            # 重排并通过卷积投影
            # (B, T, C+1, H, W) -> (B*T, C+1, H, W) -> (B*T, D, N_h, N_w)
            x_with_mask = rearrange(x_with_mask, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_with_mask)
            
            # 重排回序列形式: (B*T, D, N_h, N_w) -> (B, T, N, D)
            x_embedded = rearrange(
                x_embedded, 
                '(b t) d n_h n_w -> b t (n_h n_w) d', 
                b=B, 
                t=T
            )
        else:
            # 不使用mask通道的情况
            x_reshaped = rearrange(x, 'b t c h w -> (b t) c h w')
            x_embedded = self.projection(x_reshaped)
            x_embedded = rearrange(
                x_embedded, 
                '(b t) d n_h n_w -> b t (n_h n_w) d', 
                b=B, 
                t=T
            )
        
        return x_embedded


class TemporalAttention(nn.Module):
    """
    =====================================================================
    时间注意力机制模块
    =====================================================================
    
    实现仅在时间维度的自注意力机制，用于捕捉时间序列中的依赖关系。
    相比空间注意力，计算效率更高，特别适合时序数据的处理。
    
    参数:
    ------
    embed_dim : int
        嵌入维度
    num_heads : int
        注意力头数
    max_seq_len : int
        最大序列长度（时间步数）
    dropout : float
        Dropout比率
        
    注意力计算:
    -----------
    使用可学习的时间偏置(temporal_bias)来编码时间位置信息，
    这比传统的正弦位置编码更适合于时序建模。
    
    Q = x * W_Q, K = x * W_K, V = x * W_V
    Attention(Q, K, V) = softmax(QK^T / sqrt(d) + Bias) * V
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim ** -0.5
        
        # 验证维度兼容性
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim必须能被num_heads整除"
        
        # 可学习的时间偏置 - 这是本模型的关键创新点
        # 形状: (1, num_heads, max_seq_len, max_seq_len)
        # 用于编码不同时间步之间的相对位置关系
        self.temporal_bias = nn.Parameter(
            torch.randn(1, num_heads, max_seq_len, max_seq_len)
        )
        
        # Q, K, V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
        ------
        x : torch.Tensor
            输入张量，形状为 (B, N, T, D)
            B: 批次大小, N: patch数量, T: 时间步数, D: 嵌入维度
            
        返回:
        ------
        torch.Tensor
            注意力输出，形状为 (B, N, T, D)
        """
        B, N, T, D = x.shape
        
        # 合并批次和空间维度: (B*N, T, D)
        x_flat = x.reshape(B * N, T, D)
        
        # 线性变换得到 Q, K, V
        # (B*N, T, D) -> (B*N, T, D)
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        # 分离多头: (B*N, T, D) -> (B*N, H, T, d)
        q = q.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * N, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        # (B*N, H, T, d) @ (B*N, H, d, T) -> (B*N, H, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加时间偏置（广播到所有批次）
        attn_scores = attn_scores + self.temporal_bias[:, :, :T, :T]
        
        # Softmax归一化
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力到V
        # (B*N, H, T, T) @ (B*N, H, T, d) -> (B*N, H, T, d)
        context = torch.matmul(attn_probs, v)
        
        # 合并多头: (B*N, H, T, d) -> (B*N, T, D)
        context = context.transpose(1, 2).reshape(B * N, T, D)
        
        # 输出投影
        output = self.out_proj(context)
        
        # 恢复原始形状: (B*N, T, D) -> (B, N, T, D)
        output = output.view(B, N, T, D)
        
        return output


class PatchDecoder(nn.Module):
    """
    =====================================================================
    Patch解码器模块
    =====================================================================
    
    将编码后的patch序列解码回图像空间。
    使用转置卷积进行上采样，恢复原始图像尺寸。
    
    参数:
    ------
    img_size_h : int
        输出图像高度
    img_size_w : int
        输出图像宽度  
    patch_size : int
        Patch大小
    embed_dim : int
        嵌入维度
    out_channels : int
        输出通道数（通常为3 for RGB）
    """
    
    def __init__(
        self, 
        img_size_h: int, 
        img_size_w: int, 
        patch_size: int, 
        embed_dim: int, 
        out_channels: int = 3
    ):
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
        # 输入: (B*T, embed_dim, N_h, N_w)
        # 输出: (B*T, out_channels, H, W)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim, 
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0
            ),
        )
        
        # 尺寸调整层（如果需要）
        # 当图像尺寸不能被patch_size整除时使用
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
        ------
        x : torch.Tensor
            输入张量，形状为 (B, T, N*D)
            
        返回:
        ------
        torch.Tensor
            解码后的图像，形状为 (B, T, C, H, W)
        """
        B, T, N_D = x.shape
        
        # 计算每个patch的维度
        D = N_D // (self.num_patches_h * self.num_patches_w)
        
        # 重排维度: (B, T, N_h, N_w, D) -> (B*T, D, N_h, N_w)
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, D)
        x = rearrange(x, 'b t n_h n_w d -> (b t) d n_h n_w')
        
        # 解码
        output = self.decoder(x)
        
        # 调整尺寸（如需要）
        if self.resize is not None:
            output = self.resize(output)
        
        # 重排回时序格式: (B*T, C, H, W) -> (B, T, C, H, W)
        output = rearrange(output, '(b t) c h w -> b t c h w', b=B)
        
        return output


class VideoCompletionModel(nn.Module):
    """
    =====================================================================
    视频/时序图像补全模型
    =====================================================================
    
    完整的MAE-LaMa视频补全模型，整合编码器、时间注意力和解码器。
    
    架构流程:
    ---------
    1. PatchEmbedding: 将输入图像序列转换为patch嵌入
    2. Encoder: 若干层Transformer编码器（可选）
    3. TemporalAttention: 时间注意力层
    4. Decoder: 将编码特征解码回图像
    
    参数:
    ------
    in_channels : int
        输入通道数（默认3 for RGB）
    out_channels : int
        输出通道数
    hidden_dim : int
        隐藏层维度
    num_heads : int
        注意力头数
    num_layers : int
        编码器层数
    max_seq_len : int
        最大时间序列长度
    img_size : tuple
        输入图像尺寸 (H, W)
    patch_size : int
        Patch大小
    dropout : float
        Dropout比率
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 8,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        dropout: float = 0.1,
        use_lama_init: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size_h=img_size[0],
            img_size_w=img_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
            use_mask_channel=True
        )
        
        # 时间注意力层
        self.temporal_attn = TemporalAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # 编码器层（可选，用于更深的特征提取）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 解码器
        self.decoder = PatchDecoder(
            img_size_h=img_size[0],
            img_size_w=img_size[1],
            patch_size=patch_size,
            embed_dim=hidden_dim,
            out_channels=out_channels
        )
        
        # 投影层（用于特征对齐）
        self.encoder_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化所有权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        masked_video: torch.Tensor, 
        mask: torch.Tensor,
        ocean_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
        ------
        masked_video : torch.Tensor
            带掩码的视频序列，形状 (B, T, C, H, W)
        mask : torch.Tensor
            掩码，形状 (B, T, 1, H, W)，值为0或1
        ocean_mask : torch.Tensor, optional
            海洋掩码，形状 (B, 1, H, W)
            
        返回:
        ------
        torch.Tensor
            重建的视频，形状 (B, T, C, H, W)
        """
        B, T, C, H, W = masked_video.shape
        
        # 1. Patch Embedding
        # (B, T, C, H, W) -> (B, T, N, D)
        x = self.patch_embed(masked_video, mask)
        
        # 2. 编码器处理
        # (B, T, N, D) -> (B, T, N, D)
        x = self.encoder_proj(x)
        
        # 通过Transformer编码器
        # 需要调整形状以适应TransformerEncoder
        x = x.permute(0, 2, 1, 3)  # (B, N, T, D)
        x = x.reshape(B * x.shape[1], T, -1)  # (B*N, T, D)
        x = self.encoder(x)
        x = x.reshape(B, -1, T, self.hidden_dim)  # (B, N, T, D)
        x = x.permute(0, 2, 1, 3)  # (B, T, N, D)
        
        # 3. 时间注意力
        x = self.temporal_attn(x)
        
        # 4. 解码器
        # (B, T, N, D) -> (B, T, C, H, W)
        x = self.decoder_proj(x)
        x = x.reshape(B, T, -1)  # (B, T, N*D)
        output = self.decoder(x)
        
        return output
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MaskGenerator:
    """
    =====================================================================
    掩码生成器
    =====================================================================
    
    生成各种类型的掩码用于训练和测试：
    - 随机掩码: 随机选择像素进行掩码
    - 厚云掩码: 模拟厚云覆盖
    - 薄云掩码: 模拟薄云/半透明覆盖
    - 条带掩码: 模拟卫星条带缺失
    - 混合掩码: 多种类型组合
    
    使用示例:
    ---------
    >>> generator = MaskGenerator(img_size=(224, 224), max_seq_len=8)
    >>> mask = generator.generate_mask(mask_type='cloud', mask_ratio=0.3)
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        max_seq_len: int = 8
    ):
        self.img_size = img_size
        self.max_seq_len = max_seq_len
    
    def generate_mask(
        self,
        mask_type: str = 'random',
        mask_ratio: float = 0.3
    ) -> torch.Tensor:
        """
        生成指定类型的掩码
        
        参数:
        ------
        mask_type : str
            掩码类型: 'random', 'cloud', 'thin_cloud', 'strip', 'mixed'
        mask_ratio : float
            掩码比例（0-1之间）
            
        返回:
        ------
        torch.Tensor
            掩码张量，形状为 (T, 1, H, W)
        """
        mask_generators = {
            'random': self._generate_random_mask,
            'cloud': self._generate_cloud_mask,
            'thin_cloud': self._generate_thin_cloud_mask,
            'strip': self._generate_strip_mask,
            'mixed': self._generate_mixed_mask
        }
        
        generator = mask_generators.get(mask_type, self._generate_random_mask)
        return generator(mask_ratio)
    
    def _generate_random_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成随机掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        for t in range(self.max_seq_len):
            # 计算当前时间步需要掩码的像素数
            num_pixels = int(mask_ratio * self.img_size[0] * self.img_size[1])
            
            # 随机选择像素位置
            pixel_indices = torch.randperm(self.img_size[0] * self.img_size[1])[:num_pixels]
            
            # 设置掩码
            for idx in pixel_indices:
                h = idx // self.img_size[1]
                w = idx % self.img_size[1]
                mask[t, 0, h, w] = 1
        
        return mask
    
    def _generate_cloud_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成厚云掩码（块状）"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        for t in range(self.max_seq_len):
            # 计算云块数量
            num_clouds = max(1, int(mask_ratio * 5))
            
            for _ in range(num_clouds):
                # 随机选择云块中心
                center_h = np.random.randint(0, self.img_size[0])
                center_w = np.random.randint(0, self.img_size[1])
                
                # 随机选择云块大小
                cloud_h = np.random.randint(self.img_size[0] // 10, self.img_size[0] // 4)
                cloud_w = np.random.randint(self.img_size[1] // 10, self.img_size[1] // 4)
                
                # 创建圆形/椭圆形云块
                y, x = np.ogrid[:self.img_size[0], :self.img_size[1]]
                dist = ((y - center_h) / cloud_h) ** 2 + ((x - center_w) / cloud_w) ** 2
                cloud = (dist <= 1).astype(np.float32)
                
                # 添加到掩码
                mask[t, 0] = torch.from_numpy(
                    np.maximum(mask[t, 0].numpy(), cloud)
                )
        
        return mask
    
    def _generate_thin_cloud_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成薄云掩码（更柔和的覆盖）"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        # 使用噪声生成更自然的薄云效果
        for t in range(self.max_seq_len):
            # 生成多尺度噪声
            noise = np.random.randn(*self.img_size).astype(np.float32)
            
            # 高斯平滑
            noise = cv2.GaussianBlur(noise, (15, 15), 0)
            
# 归一化
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            # 阈值化得到指定比例的掩码
            threshold = np.percentile(noise, (1 - mask_ratio) * 100)
            cloud_mask = (noise > threshold).astype(np.float32)
            
            mask[t, 0] = torch.from_numpy(cloud_mask)
        
        return mask
    
    def _generate_strip_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成条带掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        strip_width = max(1, int(self.img_size[1] * 0.05))
        num_strips = int(self.img_size[1] * mask_ratio / strip_width)
        
        for t in range(self.max_seq_len):
            for _ in range(num_strips):
                strip_pos = np.random.randint(0, self.img_size[1] - strip_width)
                orientation = np.random.choice(['horizontal', 'vertical'])
                
                if orientation == 'horizontal':
                    mask[t, 0, :, strip_pos:strip_pos + strip_width] = 1
                else:
                    mask[t, 0, strip_pos:strip_pos + strip_width, :] = 1
        
        return mask
    
    def _generate_mixed_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成混合掩码"""
        # 分配比例
        cloud_ratio = mask_ratio * 0.5
        thin_cloud_ratio = mask_ratio * 0.3
        strip_ratio = mask_ratio * 0.2
        
        cloud_mask = self._generate_cloud_mask(cloud_ratio)
        thin_cloud_mask = self._generate_thin_cloud_mask(thin_cloud_ratio)
        strip_mask = self._generate_strip_mask(strip_ratio)
        
        # 组合掩码
        combined_mask = torch.clamp(
            cloud_mask + thin_cloud_mask + strip_mask, 
            0, 
            1
        )
        
        return combined_mask


# =====================================================================
# 工具函数
# =====================================================================

def create_model(
    in_channels: int = 3,
    out_channels: int = 3,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 6,
    max_seq_len: int = 8,
    img_size: Tuple[int, int] = (224, 224),
    patch_size: int = 16,
    device: str = 'cuda'
) -> VideoCompletionModel:
    """
    工厂函数：创建并初始化模型
    
    参数:
    ------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    hidden_dim : int
        隐藏层维度
    num_heads : int
        注意力头数
    num_layers : int
        编码器层数
    max_seq_len : int
        最大时间序列长度
    img_size : tuple
        输入图像尺寸
    patch_size : int
        Patch大小
    device : str
        设备 ('cuda' or 'cpu')
        
    返回:
    ------
    VideoCompletionModel
        初始化的模型
        
    示例:
    -----
    >>> model = create_model(
    ...     in_channels=3,
    ...     out_channels=3,
    ...     hidden_dim=256,
    ...     num_heads=8,
    ...     num_layers=6,
    ...     max_seq_len=8,
    ...     img_size=(224, 224),
    ...     device='cuda'
    ... )
    >>> print(f"模型参数数量: {model.get_num_params():,}")
    """
    model = VideoCompletionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        img_size=img_size,
        patch_size=patch_size
    )
    
    model = model.to(device)
    
    return model


# =====================================================================
# 主程序入口（用于测试）
# =====================================================================

if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("MALA 模型测试")
    print("=" * 60)
    
    # 创建模型
    model = create_model(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_len=8,
        img_size=(224, 224)
    )
    
    print(f"模型参数数量: {model.get_num_params():,}")
    
    # 测试前向传播
    B, T, C, H, W = 2, 8, 3, 224, 224
    
    # 随机输入
    masked_video = torch.randn(B, T, C, H, W)
    mask = torch.rand(B, T, 1, H, W) > 0.7
    mask = mask.float()
    
    # 前向传播
    output = model(masked_video, mask)
    
    print(f"输入形状: {masked_video.shape}")
    print(f"掩码形状: {mask.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试掩码生成器
    print("\n测试掩码生成器:")
    mask_gen = MaskGenerator(img_size=(224, 224), max_seq_len=8)
    
    for mask_type in ['random', 'cloud', 'thin_cloud', 'strip', 'mixed']:
        mask = mask_gen.generate_mask(mask_type=mask_type, mask_ratio=0.3)
        actual_ratio = mask.mean().item()
        print(f"  {mask_type}: 实际掩码比例 = {actual_ratio:.4f}")
    
    print("\n测试完成!")
