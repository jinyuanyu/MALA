"""
MALA项目评估指标模块
====================

本模块提供图像修复质量的评估指标计算功能：
- MSE (均方误差)
- PSNR (峰值信噪比)
- SSIM (结构相似性指数)
- MAE (平均绝对误差)

支持在完整图像和掩码区域上计算指标。

作者: MALA Team
日期: 2024
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from skimage.metrics import (
    structural_similarity as ssim,
    mean_squared_error,
    peak_signal_noise_ratio
)


def calculate_mse(
    original: np.ndarray, 
    processed: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算均方误差（MSE）
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 可选的掩码，仅在掩码区域计算
        
    返回:
        MSE值
    """
    if original is None or processed is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, 
            (original.shape[1], original.shape[0])
        )
    
    # 处理掩码
    if mask is not None:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        mask_binary = (mask > 127)
        if not np.any(mask_binary):
            return np.nan
        
        original_masked = original[mask_binary]
        processed_masked = processed[mask_binary]
        mse = mean_squared_error(original_masked, processed_masked)
    else:
        mse = mean_squared_error(original, processed)
    
    return mse


def calculate_psnr(
    original: np.ndarray, 
    processed: np.ndarray, 
    mask: Optional[np.ndarray] = None,
    max_val: float = 255.0
) -> float:
    """
    计算峰值信噪比（PSNR）
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 可选的掩码
        max_val: 像素最大值
        
    返回:
        PSNR值（dB）
    """
    mse = calculate_mse(original, processed, mask)
    
    if np.isnan(mse) or mse == 0:
        return np.inf if mse == 0 else np.nan
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim(
    original: np.ndarray, 
    processed: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算结构相似性指数（SSIM）
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 可选的掩码
        
    返回:
        SSIM值（范围0-1）
    """
    if original is None or processed is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, 
            (original.shape[1], original.shape[0])
        )
    
    # 处理掩码
    if mask is not None:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        mask_binary = (mask > 127)
        if not np.any(mask_binary):
            return np.nan
    
    try:
        # 对于彩色图像，计算每个通道的SSIM然后取平均
        if len(original.shape) == 3:
            ssim_values = []
            for channel in range(original.shape[2]):
                original_channel = original[:, :, channel]
                processed_channel = processed[:, :, channel]
                
                ssim_val = ssim(
                    original_channel, 
                    processed_channel,
                    data_range=255, 
                    win_size=7
                )
                ssim_values.append(ssim_val)
            
            return np.mean(ssim_values)
        else:
            return ssim(original, processed, data_range=255, win_size=7)
            
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return np.nan


def calculate_mae(
    original: np.ndarray, 
    processed: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> float:
    """
    计算平均绝对误差（MAE）
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 可选的掩码
        
    返回:
        MAE值（0-255范围）
    """
    if original is None or processed is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, 
            (original.shape[1], original.shape[0])
        )
    
    # 转换为浮点数
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    
    # 处理掩码
    if mask is not None:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        
        mask_binary = (mask > 127)
        if not np.any(mask_binary):
            return np.nan
        
        original_masked = original[mask_binary]
        processed_masked = processed[mask_binary]
        mae = np.mean(np.abs(original_masked - processed_masked))
    else:
        mae = np.mean(np.abs(original - processed))
    
    return mae


def calculate_all_metrics(
    original: np.ndarray, 
    processed: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 可选的掩码
        
    返回:
        包含所有指标的字典
    """
    return {
        'MSE': calculate_mse(original, processed, mask),
        'PSNR': calculate_psnr(original, processed, mask),
        'SSIM': calculate_ssim(original, processed, mask),
        'MAE': calculate_mae(original, processed, mask)
    }


def calculate_metrics_for_frame(
    original_path: str,
    processed_path: str,
    mask_path: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """
    计算单帧的所有评估指标
    
    参数:
        original_path: 原始图像路径
        processed_path: 处理后图像路径
        mask_path: 可选的掩码路径
        
    返回:
        包含指标的字典，或None（如果加载失败）
    """
    # 加载图像
    original = cv2.imread(original_path)
    if original is not None:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    processed = cv2.imread(processed_path)
    if processed is not None:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    mask = None
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None:
        return None
    
    return calculate_all_metrics(original, processed, mask)


# ============ PyTorch版本指标计算（用于训练时评估）============

import torch


def calculate_metrics_torch(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    use_mask区域: bool = True
) -> Tuple[float, float, float]:
    """
    使用PyTorch张量计算评估指标
    
    参数:
        original: 原始图像张量 (C, H, W) 或 (B, C, H, W)
        reconstructed: 重建图像张量
        mask: 掩码张量
        use_mask区域: 是否仅在掩码区域计算
        
    返回:
        (SSIM, PSNR, MAE) 元组
    """
    # 确保在CPU上计算
    original_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    # 处理维度
    if original_np.ndim == 4:  # (B, C, H, W)
        # 取第一帧
        original_np = original_np[0]
        recon_np = recon_np[0]
        mask_np = mask_np[0]
    
    # 统一通道顺序 (C, H, W) -> (H, W, C)
    if original_np.ndim == 3 and original_np.shape[0] in [1, 3]:
        original_np = np.moveaxis(original_np, 0, -1)
        recon_np = np.moveaxis(recon_np, 0, -1)
        mask_np = np.squeeze(mask_np, axis=0)
    
    # 确保掩码是二维
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    
    # 转换到0-255范围
    original_np = (original_np * 255).astype(np.uint8)
    recon_np = (recon_np * 255).astype(np.uint8)
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    
    # 计算指标
    if use_mask区域:
        ssim_val = calculate_ssim(original_np, recon_np, mask_np)
        psnr_val = calculate_psnr(original_np, recon_np, mask_np)
        mae_val = calculate_mae(original_np, recon_np, mask_np)
    else:
        ssim_val = calculate_ssim(original_np, recon_np)
        psnr_val = calculate_psnr(original_np, recon_np)
        mae_val = calculate_mae(original_np, recon_np)
    
    return ssim_val, psnr_val, mae_val
