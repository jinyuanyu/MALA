# -*- coding: utf-8 -*-
"""
MALA 模型模块

该模块包含遥感图像填补的核心深度学习模型实现。
Based on Masked Autoencoder (MAE) architecture for remote sensing image completion.

Author: Jin Yuanyu
Email: jinyuanyu@example.com
"""

# 模型组件
from .mae_lama import VideoCompletionModel, PatchEmbedding, TemporalAttention, PatchDecoder

__all__ = [
    'VideoCompletionModel',
    'PatchEmbedding', 
    'TemporalAttention',
    'PatchDecoder'
]

__version__ = '1.0.0'
