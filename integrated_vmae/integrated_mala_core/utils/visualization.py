"""
MALA项目可视化工具模块
======================

本模块提供图像和结果的可视化功能：
- 图像展示和保存
- 误差热力图生成
- 散点图绘制
- 时序分析图表

作者: MALA Team
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import seaborn as sns


def denormalize_image(img):
    """
    反归一化图像，将[-1, 1]范围转换到[0, 1]
    
    参数:
        img: 归一化后的图像张量
        
    返回:
        反归一化后的图像张量
    """
    return (img * 0.5 + 0.5).clamp(0, 1)


def visualize_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    reconstructed: np.ndarray,
    times: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    可视化原始、掩码和重建图像的对比
    
    参数:
        original: 原始图像
        masked: 掩码后的图像
        reconstructed: 重建图像
        times: 时间标识数组
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始帧
    axes[0].imshow(original)
    if times is not None:
        axes[0].set_title(f'原始帧\n时间: {times}')
    else:
        axes[0].set_title('原始帧')
    axes[0].axis('off')
    
    # 掩码帧
    axes[1].imshow(masked)
    axes[1].set_title('掩码帧')
    axes[1].axis('off')
    
    # 重建帧
    axes[2].imshow(reconstructed)
    axes[2].set_title('重建帧')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_error_heatmap(
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    algorithm_name: str = "Algorithm",
    rect_points: Optional[list] = None,
    rect_label: Optional[str] = None
):
    """
    创建并保存误差热力图
    
    参数:
        original: 原始图像
        processed: 处理后的图像
        mask: 掩码
        output_path: 输出路径
        algorithm_name: 算法名称
        rect_points: 关注区域矩形点
        rect_label: 矩形标签
    """
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(
            processed, 
            (original.shape[1], original.shape[0])
        )
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # 计算误差
    original_float = original.astype(np.float32)
    processed_float = processed.astype(np.float32)
    error = np.mean(np.abs(original_float - processed_float), axis=2) / 255.0 * 10000
    
    # 创建掩码
    mask_binary = (mask > 127).astype(np.uint8)
    error_masked = error * mask_binary
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 掩码区域误差，非掩码区域透明
    mask_indices = mask_binary > 0
    if np.any(mask_indices):
        error_display = error_masked.copy().astype(float) * 2
        error_display[~mask_indices] = np.nan
        
        # 使用YlOrRd颜色映射
        im = ax.imshow(error_display, cmap='YlOrRd', interpolation='nearest')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('MAE ×10k', rotation=270, labelpad=15, fontsize=20)
        cbar.ax.tick_params(labelsize=14)
        
        # 设置颜色范围
        im.set_clim(vmin=0, vmax=10000)
        
        # 绘制关注区域
        if rect_points is not None and len(rect_points) == 4:
            x_coords = [p[0] for p in rect_points + [rect_points[0]]]
            y_coords = [p[1] for p in rect_points + [rect_points[0]]]
            ax.plot(x_coords, y_coords, 'g-', linewidth=3, label='关注区域')
            
            if rect_label:
                center_x = np.mean(x_coords[:-1])
                center_y = np.mean(y_coords[:-1])
                ax.text(center_x, center_y, rect_label,
                    fontsize=20, color='green', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            alpha=0.8, edgecolor="green"))
            
            ax.legend(loc='upper right', fontsize=20)
    else:
        ax.imshow(np.zeros_like(error), cmap='gray')
        ax.text(0.5, 0.5, '未找到掩码区域', transform=ax.transAxes,
                ha='center', va='center', fontsize=18, color='red')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_1to1(
    ref_pixels: np.ndarray,
    rec_pixels: np.ndarray,
    band_names: list = ['B2', 'B3', 'B4'],
    output_path: Optional[str] = None
):
    """
    绘制学术论文级1:1散点图
    
    参数:
        ref_pixels: 参考像素数据 (N, 3)
        rec_pixels: 重建像素数据 (N, 3)
        band_names: 波段名称列表
        output_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    
    for i, (ax, band_name) in enumerate(zip(axes, band_names)):
        ref_band = ref_pixels[:, i]
        rec_band = rec_pixels[:, i]
        
        # 绘制散点
        ax.scatter(ref_band, rec_band, alpha=0.3, s=1)
        
        # 绘制1:1线
        min_val = min(ref_band.min(), rec_band.min())
        max_val = max(ref_band.max(), rec_band.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        
        ax.set_xlabel(f'参考值 ({band_name})')
        ax.set_ylabel(f'重建值 ({band_name})')
        ax.set_title(band_name)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_timeseries(
    timestamps: np.ndarray,
    values: np.ndarray,
    labels: list,
    title: str = "时序分析",
    xlabel: str = "时间",
    ylabel: str = "值",
    output_path: Optional[str] = None
):
    """
    绘制时序折线图
    
    参数:
        timestamps: 时间戳数组
        values: 数值数组 (T, N)
        labels: 曲线标签列表
        title: 图表标题
        xlabel: X轴标签
        ylabel: Y轴标签
        output_path: 保存路径
    """
    plt.figure(figsize=(12, 6))
    
    for i, (value, label) in enumerate(zip(values, labels)):
        plt.plot(timestamps, value, label=label, marker='o')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_reconstructed_frames(
    reconstructed,
    times,
    output_dir: str,
    prefix: str = "reconstructed_frame"
):
    """
    保存重建的图像帧
    
    参数:
        reconstructed: 重建图像张量 (T, C, H, W)
        times: 时间标识 (T,)
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    from PIL import Image
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    reconstructed = denormalize_image(reconstructed)
    
    for t in range(reconstructed.shape[0]):
        img_tensor = reconstructed[t]
        
        # 转换到CPU和numpy
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # 保存图像
        img_pil = Image.fromarray(img_np)
        img_pil.save(output_path / f"{prefix}_{times[t].item():04d}.png")


def create_mask_visualization(
    mask: np.ndarray,
    output_path: str,
    colormap: str = 'gray'
):
    """
    创建掩码可视化
    
    参数:
        mask: 掩码数组
        output_path: 保存路径
        colormap: 颜色映射
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap=colormap)
    plt.colorbar(label='掩码值')
    plt.title('掩码可视化')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
