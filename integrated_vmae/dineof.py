import torch
import numpy as np
from scipy.spatial import KDTree
from PIL import Image
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# 假设这些模块存在，根据实际情况调整导入
from MAE_LaMa import Datasets_inference

def apply_dineof_improved(masked_data, mask, device, max_modes=10, n_iter=20, tolerance=1e-6):
    """
    改进的DINEOF实现
    基于经验正交函数(EOF)的时空数据插值方法
    """
    result = masked_data.clone()
    B, T, C, H, W = masked_data.shape
    
    for b in range(B):
        for c in range(C):
            # 获取该通道的时空数据 [T, H, W]
            channel_data = masked_data[b, :, c, :, :].cpu().numpy()
            channel_mask = mask[b, :, 0, :, :].cpu().numpy() > 0.5
            
            # 重塑为二维矩阵 [时间, 空间]
            spatial_shape = (H, W)
            data_2d = channel_data.reshape(T, -1)  # [T, H*W]
            mask_2d = channel_mask.reshape(T, -1)  # [T, H*W]
            
            # 初始填充：使用时间均值填充缺失值
            initial_guess = initialize_dineof(data_2d, mask_2d)
            
            # DINEOF迭代过程
            reconstructed = dineof_iteration(initial_guess, mask_2d, max_modes, n_iter, tolerance)
            
            # 重塑回原始形状
            reconstructed_3d = reconstructed.reshape(T, H, W)
            result[b, :, c, :, :] = torch.from_numpy(reconstructed_3d).to(device)
    
    return result

def initialize_dineof(data_2d, mask_2d):
    """DINEOF初始化：使用时间均值填充缺失值"""
    initialized = data_2d.copy()
    
    # 对每个空间点，用时间均值填充缺失值
    for j in range(data_2d.shape[1]):
        if np.any(mask_2d[:, j]):
            valid_values = data_2d[~mask_2d[:, j], j]
            if len(valid_values) > 0:
                time_mean = np.mean(valid_values)
                initialized[mask_2d[:, j], j] = time_mean
            else:
                # 如果所有时间点都缺失，使用全局均值
                initialized[mask_2d[:, j], j] = np.nanmean(data_2d)
    
    # 处理剩余的NaN值
    initialized = np.nan_to_num(initialized, nan=np.nanmean(data_2d))
    return initialized

def dineof_iteration(initial_data, mask, max_modes=10, n_iter=20, tolerance=1e-6):
    """
    DINEOF核心迭代过程
    """
    current_guess = initial_data.copy()
    prev_rmse = float('inf')
    
    for iteration in range(n_iter):
        # 1. 对当前猜测进行SVD分解
        U, s, Vt = np.linalg.svd(current_guess, full_matrices=False)
        
        # 2. 交叉验证确定最优模态数量（简化版）
        optimal_modes = find_optimal_modes(current_guess, mask, U, s, Vt, max_modes)
        
        # 3. 使用选定模态重建数据
        reconstructed = U[:, :optimal_modes] @ np.diag(s[:optimal_modes]) @ Vt[:optimal_modes, :]
        
        # 4. 保持已知值不变，只更新缺失值
        current_guess[mask] = reconstructed[mask]
        
        # 5. 检查收敛性
        rmse = calculate_rmse(reconstructed, current_guess, mask)
        if abs(prev_rmse - rmse) < tolerance:
            break
        
        prev_rmse = rmse
    
    return current_guess

def find_optimal_modes(data, mask, U, s, Vt, max_modes):
    """
    通过交叉验证确定最优EOF模态数量（简化版）
    """
    # 在实际DINEOF中，这会涉及复杂的交叉验证过程
    # 这里使用简化版本：选择解释方差超过95%的模态
    
    total_variance = np.sum(s ** 2)
    explained_variance = np.cumsum(s ** 2) / total_variance
    
    # 找到解释方差超过95%的最小模态数
    optimal_modes = np.argmax(explained_variance >= 0.95) + 1
    return min(optimal_modes, max_modes)

def calculate_rmse(reconstructed, original, mask):
    """计算均方根误差"""
    diff = reconstructed - original
    return np.sqrt(np.mean(diff[mask] ** 2))

def save_sample_images(results, save_path):
    """保存样本图像"""
    def unnorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)
    
    os.makedirs(f'experiment_results/{save_path}/images', exist_ok=True)
    
    # 保存第一帧的对比图像
    for t in range(8):
        if 'methods_results' in results:
            for name, result in results['methods_results'].items():
                if result.shape[1] > t:
                    img_data = unnorm(result[0, t])
                    if img_data.shape[0] == 1:
                        img = Image.fromarray((img_data.squeeze(0).numpy() * 255).astype(np.uint8), 'L')
                    else:
                        img_array = (img_data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        if img_array.shape[2] == 3:
                            img = Image.fromarray(img_array, 'RGB')
                        else:
                            img = Image.fromarray(img_array[:,:,0], 'L')
                    img.save(f'experiment_results/{save_path}/images/{name}_frame{t}.png')
