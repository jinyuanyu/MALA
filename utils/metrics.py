# -*- coding: utf-8 -*-
"""
================================================================================
评估指标模块
================================================================================

该模块提供遥感图像填补任务的评估指标计算功能，包括：
- SSIM (结构相似性指数)
- PSNR (峰值信噪比)
- MAE (平均绝对误差)
- TCC (时间相关系数)

Author: Jin Yuanyu
Email: jinyuanyu@example.com
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_ssim(
    original: np.ndarray, 
    reconstructed: np.ndarray,
    data_range: float = 1.0
) -> float:
    """
    =====================================================================
    计算结构相似性指数 (SSIM)
    =====================================================================
    
    SSIM衡量两幅图像之间的结构相似性，考虑了亮度、对比度和结构三个因素。
    值越高表示两幅图像越相似，最好的值为1.0。
    
    参数:
    ------
    original : np.ndarray
        原始图像，形状为 (H, W) 或 (C, H, W)
    reconstructed : np.ndarray
        重建图像，与original形状相同
    data_range : float
        数据范围，通常为1.0（归一化后）或255（原始像素值）
        
    返回:
    ------
    float
        SSIM值，范围[0, 1]
        
    示例:
    -----
    >>> original = np.random.rand(256, 256, 3)
    >>> reconstructed = np.random.rand(256, 256, 3)
    >>> ssim_value = calculate_ssim(original, reconstructed)
    >>> print(f"SSIM: {ssim_value:.4f}")
    """
    # 处理单通道情况
    if original.ndim == 2:
        return ssim(
            original, 
            reconstructed, 
            data_range=data_range
        )
    # 处理多通道情况
    elif original.ndim == 3:
        # 转换为 (H, W, C) 格式
        if original.shape[0] == 3 or original.shape[0] == 1:
            original = np.transpose(original, (1, 2, 0))
            reconstructed = np.transpose(reconstructed, (1, 2, 0))
        
        return ssim(
            original,
            reconstructed,
            data_range=data_range,
            channel_axis=2,
            multichannel=True
        )
    else:
        raise ValueError(f"不支持的图像维度: {original.ndim}")


def calculate_psnr(
    original: np.ndarray,
    reconstructed: np.ndarray,
    data_range: float = 1.0
) -> float:
    """
    =====================================================================
    计算峰值信噪比 (PSNR)
    =====================================================================
    
    PSNR是最常用的图像质量评估指标之一，基于像素级别的误差计算。
    值越高表示图像质量越好，通常PSNR > 30dB表示较好的重建质量。
    
    参数:
    ------
    original : np.ndarray
        原始图像
    reconstructed : np.ndarray
        重建图像
    data_range : float
        数据范围，通常为1.0或255
        
    返回:
    ------
    float
        PSNR值，单位为dB
        
    示例:
    -----
    >>> original = np.random.rand(256, 256, 3)
    >>> reconstructed = np.random.rand(256, 256, 3)
    >>> psnr_value = calculate_psnr(original, reconstructed)
    >>> print(f"PSNR: {psnr_value:.2f} dB")
    """
    return psnr(
        original, 
        reconstructed, 
        data_range=data_range
    )


def calculate_mae(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    =====================================================================
    计算平均绝对误差 (MAE)
    =====================================================================
    
    MAE是最简单的误差度量，计算两幅图像对应像素差值的绝对值的平均。
    值越小表示重建效果越好。
    
    参数:
    ------
    original : np.ndarray
        原始图像
    reconstructed : np.ndarray
        重建图像
    mask : np.ndarray, optional
        掩码，仅计算掩码区域的MAE
        
    返回:
    ------
    float
        MAE值
        
    示例:
    -----
    >>> original = np.random.rand(256, 256, 3)
    >>> reconstructed = np.random.rand(256, 256, 3)
    >>> mae_value = calculate_mae(original, reconstructed)
    >>> print(f"MAE: {mae_value:.4f}")
    """
    if mask is not None:
        # 仅计算掩码区域的误差
        error = np.abs(original - reconstructed)
        # 确保掩码形状与图像匹配
        if mask.ndim == 2 and error.ndim == 3:
            mask = np.expand_dims(mask, axis=-1)
        return float(np.mean(error[mask > 0]))
    else:
        # 计算整体误差
        return float(np.mean(np.abs(original - reconstructed)))


def calculate_tcc(
    original_seq: np.ndarray,
    reconstructed_seq: np.ndarray,
    mask_seq: Optional[np.ndarray] = None,
    sample_size: int = 1000
) -> float:
    """
    =====================================================================
    计算时间相关系数 (TCC)
    =====================================================================
    
    TCC衡量重建图像序列与原始序列之间的时间相关性。
    特别适合评估时序填补任务，考虑了时间维度的重建质量。
    
    参数:
    ------
    original_seq : np.ndarray
        原始图像序列，形状为 (T, C, H, W) 或 (T, H, W)
    reconstructed_seq : np.ndarray
        重建图像序列，与original_seq形状相同
    mask_seq : np.ndarray, optional
        掩码序列，形状为 (T, H, W)
    sample_size : int
        采样像素数量，用于加速计算
        
    返回:
    ------
    float
        TCC值，范围[-1, 1]，1表示完全相关
        
    算法说明:
    ---------
    1. 随机采样指定数量的像素位置
    2. 对每个采样位置，提取其在时间序列上的值
    3. 计算原始序列和重建序列的皮尔逊相关系数
    4. 返回所有采样位置相关系数的平均值
    """
    T, H, W = original_seq.shape[:3]
    
    # 随机采样像素
    total_pixels = H * W
    sample_size = min(sample_size, total_pixels)
    pixel_indices = np.random.choice(total_pixels, sample_size, replace=False)
    
    total_corr = 0.0
    total_count = 0
    
    # 对每个通道计算TCC
    if original_seq.ndim == 4:
        C = original_seq.shape[1]
    else:
        C = 1
        original_seq = np.expand_dims(original_seq, axis=1)
        reconstructed_seq = np.expand_dims(reconstructed_seq, axis=1)
    
    for c in range(C):
        # 重塑为 (T, H*W)
        orig_2d = original_seq[:, c, :, :].reshape(T, -1)
        recon_2d = reconstructed_seq[:, c, :, :].reshape(T, -1)
        
        if mask_seq is not None:
            mask_2d = mask_seq.reshape(T, -1)
        
        for idx in pixel_indices:
            # 提取时间序列
            orig_ts = orig_2d[:, idx]
            recon_ts = recon_2d[:, idx]
            
            # 如果有掩码，只考虑掩码区域
            if mask_seq is not None:
                if not np.any(mask_2d[:, idx] > 0):
                    continue
            
            # 检查方差
            if np.std(orig_ts) < 1e-10 or np.std(recon_ts) < 1e-10:
                continue
            
            # 计算皮尔逊相关系数
            try:
                corr = np.corrcoef(orig_ts, recon_ts)[0, 1]
                if not np.isnan(corr):
                    total_corr += corr
                    total_count += 1
            except:
                continue
    
    return total_corr / total_count if total_count > 0 else 0.0


def calculate_all_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    =====================================================================
    计算所有评估指标
    =====================================================================
    
    一次性计算SSIM、PSNR、MAE三个常用指标。
    
    参数:
    ------
    original : np.ndarray
        原始图像
    reconstructed : np.ndarray
        重建图像
    mask : np.ndarray, optional
        掩码
    data_range : float
        数据范围
        
    返回:
    ------
    dict
        包含SSIM、PSNR、MAE的字典
        
    示例:
    -----
    >>> original = np.random.rand(256, 256, 3)
    >>> reconstructed = np.random.rand(256, 256, 3)
    >>> metrics = calculate_all_metrics(original, reconstructed)
    >>> for name, value in metrics.items():
    ...     print(f"{name}: {value:.4f}")
    """
    return {
        'SSIM': calculate_ssim(original, reconstructed, data_range),
        'PSNR': calculate_psnr(original, reconstructed, data_range),
        'MAE': calculate_mae(original, reconstructed, mask)
    }


def calculate_metrics_from_tensors(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    =====================================================================
    从PyTorch张量计算评估指标
    =====================================================================
    
    方便的包装函数，处理PyTorch张量到NumPy的转换。
    
    参数:
    ------
    original : torch.Tensor
        原始图像张量，形状为 (C, H, W) 或 (B, C, H, W)
    reconstructed : torch.Tensor
        重建图像张量
    mask : torch.Tensor, optional
        掩码张量
    data_range : float
        数据范围
        
    返回:
    ------
    dict
        评估指标字典
    """
    # 确保在CPU上
    original = original.cpu()
    reconstructed = reconstructed.cpu()
    
    # 去除批次维度（如果有）
    if original.ndim == 4:
        original = original[0]
    if reconstructed.ndim == 4:
        reconstructed = reconstructed[0]
    
    # 转换为numpy
    original = original.numpy()
    reconstructed = reconstructed.numpy()
    
    # 转换为 (H, W, C) 格式
    if original.ndim == 3:
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
    
    # 处理掩码
    mask_np = None
    if mask is not None:
        mask = mask.cpu()
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]  # (H, W)
        mask_np = mask.numpy()
    
    return calculate_all_metrics(original, reconstructed, mask_np, data_range)


# =====================================================================
# 时间序列指标计算
# =====================================================================

def calculate_temporal_metrics(
    original_seq: torch.Tensor,
    reconstructed_seq: torch.Tensor,
    mask_seq: Optional[torch.Tensor] = None,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    =====================================================================
    计算时序图像序列的评估指标
    =====================================================================
    
    对整个时间序列计算平均指标。
    
    参数:
    ------
    original_seq : torch.Tensor
        原始序列，形状为 (T, C, H, W)
    reconstructed_seq : torch.Tensor
        重建序列
    mask_seq : torch.Tensor, optional
        掩码序列
    data_range : float
        数据范围
        
    返回:
    ------
    dict
        评估指标字典
    """
    T = original_seq.shape[0]
    
    ssim_values = []
    psnr_values = []
    mae_values = []
    
    for t in range(T):
        metrics = calculate_metrics_from_tensors(
            original_seq[t],
            reconstructed_seq[t],
            mask_seq[t] if mask_seq is not None else None,
            data_range
        )
        
        ssim_values.append(metrics['SSIM'])
        psnr_values.append(metrics['PSNR'])
        mae_values.append(metrics['MAE'])
    
    # 计算TCC（整个序列）
    original_np = original_seq.cpu().numpy()
    recon_np = reconstructed_seq.cpu().numpy()
    
    # 转换为 (T, H, W, C) 格式
    if original_np.ndim == 4:
        original_np = np.transpose(original_np, (0, 2, 3, 1))
        recon_np = np.transpose(recon_np, (0, 2, 3, 1))
    
    mask_np = None
    if mask_seq is not None:
        mask_np = mask_seq.cpu().numpy()
        if mask_np.ndim == 4:
            mask_np = mask_np[:, 0, :, :]  # (T, H, W)
    
    tcc = calculate_tcc(original_np, recon_np, mask_np)
    
    return {
        'SSIM': float(np.mean(ssim_values)),
        'PSNR': float(np.mean(psnr_values)),
        'MAE': float(np.mean(mae_values)),
        'TCC': float(tcc)
    }


# =====================================================================
# 主程序入口
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MALA 评估指标测试")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    
    # 创建测试图像
    original = np.random.rand(256, 256, 3).astype(np.float32)
    noise = np.random.randn(256, 256, 3).astype(np.float32) * 0.1
    reconstructed = np.clip(original + noise, 0, 1).astype(np.float32)
    
    # 创建测试掩码
    mask = np.zeros((256, 256), dtype=bool)
    mask[100:150, 100:150] = True
    
    # 计算各项指标
    print("\n无掩码评估:")
    metrics = calculate_all_metrics(original, reconstructed)
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 创建时间序列数据
    print("\n时序评估:")
    original_seq = np.random.rand(8, 256, 256, 3).astype(np.float32)
    noise_seq = np.random.randn(8, 256, 256, 3).astype(np.float32) * 0.1
    reconstructed_seq = np.clip(original_seq + noise_seq, 0, 1).astype(np.float32)
    
    # 转换为torch tensor
    original_t = torch.from_numpy(original_seq).permute(0, 3, 1, 2)
    reconstructed_t = torch.from_numpy(reconstructed_seq).permute(0, 3, 1, 2)
    
    temporal_metrics = calculate_temporal_metrics(original_t, reconstructed_t)
    for name, value in temporal_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print("\n测试完成!")
