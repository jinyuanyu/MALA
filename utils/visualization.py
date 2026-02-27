# -*- coding: utf-8 -*-
"""
================================================================================
可视化工具模块
================================================================================

该模块提供遥感图像填补结果的可视化功能，包括：
- 误差热力图 (Error Heatmap)
- 时序对比图 (Timeseries Comparison)
- 箱线图 (Boxplot)
- 散点图 (Scatter Plot)
- 散点一对一图 (Scatter One-to-One)

Author: Jin Yuanyu
Email: jinyuanyu@example.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_error_heatmap(
    error_map: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Error Heatmap",
    cmap: str = 'YlOrRd',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    =====================================================================
    绘制误差热力图
    =====================================================================
    
    可视化重建图像与原始图像之间的误差分布。
    
    参数:
    ------
    error_map : np.ndarray
        误差图，形状为 (H, W)
    save_path : str, optional
        保存路径
    title : str
        图表标题
    cmap : str
        颜色映射
    vmin, vmax : float, optional
        颜色范围限制
        
    返回:
    ------
    plt.Figure
        matplotlib图表对象
        
    示例:
    -----
    >>> error = np.random.rand(256, 256)
    >>> fig = plot_error_heatmap(error, save_path='error_heatmap.png')
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(
        error_map, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        aspect='auto'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error (MAE)', fontsize=12)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=12)
    ax.set_ylabel('Height (pixels)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差热力图已保存至: {save_path}")
    
    return fig


def plot_timeseries(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    missing_regions: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    title: str = "Pixel Timeseries Comparison",
    xlabel: str = "Frame Index",
    ylabel: str = "Pixel Value"
) -> plt.Figure:
    """
    =====================================================================
    绘制时序对比图
    =====================================================================
    
    比较不同方法在时间序列上的重建效果。
    
    参数:
    ------
    ground_truth : np.ndarray
        真实值序列，形状为 (T,)
    predictions : dict
        预测结果字典，键为方法名，值为预测序列
    missing_regions : list, optional
        缺失区域列表，每项为 (start, end) 元组
    save_path : str, optional
        保存路径
    title : str
        图表标题
        
    返回:
    ------
    plt.Figure
        
    示例:
    -----
    >>> gt = np.random.rand(8)
    >>> preds = {'DINEOF': np.random.rand(8), 'Spline': np.random.rand(8)}
    >>> fig = plot_timeseries(gt, preds, missing_regions=[(2, 4)])
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    frames = np.arange(len(ground_truth))
    
    # 绘制缺失区域背景
    if missing_regions:
        for start, end in missing_regions:
            ax.axvspan(start, end, alpha=0.2, color='red', label='Missing Region' if start == missing_regions[0][0] else '')
    
    # 绘制真实值曲线
    ax.plot(
        frames, ground_truth, 
        'k-', linewidth=3, 
        label='Ground Truth', 
        marker='o', markersize=8,
        zorder=10
    )
    
    # 绘制各方法预测曲线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['s', '^', 'v', '<', '>', 'D']
    
    for i, (method_name, pred_values) in enumerate(predictions.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax.plot(
            frames, pred_values,
            color=color, linewidth=2, linestyle='-',
            label=method_name, marker=marker, markersize=6,
            alpha=0.8, zorder=5
        )
    
    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 图例
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时序对比图已保存至: {save_path}")
    
    return fig


def plot_boxplot(
    data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Metrics Comparison",
    ylabel: str = "Value",
    metrics: Optional[List[str]] = None
) -> plt.Figure:
    """
    =====================================================================
    绘制箱线图
    =====================================================================
    
    比较不同方法的指标分布。
    
    参数:
    ------
    data : dict
        数据字典，键为方法名，值为指标值列表
    save_path : str, optional
        保存路径
    title : str
        图表标题
    ylabel : str
        Y轴标签
        
    返回:
    ------
    plt.Figure
        
    示例:
    -----
    >>> data = {
    ...     'DINEOF': [0.8, 0.85, 0.82],
    ...     'Proposed': [0.9, 0.92, 0.88]
    ... }
    >>> fig = plot_boxplot(data, save_path='boxplot.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据
    methods = list(data.keys())
    values = [data[m] for m in methods]
    
    # 绘制箱线图
    bp = ax.boxplot(
        values, 
        labels=methods,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
    )
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"箱线图已保存至: {save_path}")
    
    return fig


def plot_scatter(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Ground Truth vs Predictions",
    xlabel: str = "Ground Truth",
    ylabel: str = "Predictions",
    alpha: float = 0.5
) -> plt.Figure:
    """
    =====================================================================
    绘制散点图
    =====================================================================
    
    可视化预测值与真实值的相关性。
    
    参数:
    ------
    ground_truth : np.ndarray
        真实值数组
    predictions : np.ndarray
        预测值数组
    save_path : str, optional
        保存路径
    title : str
        图表标题
    alpha : float
        散点透明度
        
    返回:
    ------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制散点
    ax.scatter(
        ground_truth, predictions,
        alpha=alpha, s=20, c='blue'
    )
    
    # 绘制理想线 (y=x)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 计算相关系数
    correlation = np.corrcoef(ground_truth, predictions)[0, 1]
    ax.text(
        0.05, 0.95, f'Correlation: {correlation:.4f}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top'
    )
    
    # 图例和网格
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    ax.set_xlim(min_val - 0.05, max_val + 0.05)
    ax.set_ylim(min_val - 0.05, max_val + 0.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至: {save_path}")
    
    return fig


def plot_scatter_one_to_one(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "One-to-One Scatter Plot",
    methods: Optional[List[str]] = None,
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    =====================================================================
    绘制一对一散点图
    =====================================================================
    
    比较多个方法在同一组数据上的预测效果。
    
    参数:
    ------
    ground_truth : np.ndarray
        真实值
    predictions : dict or np.ndarray
        预测值，可以是字典(多方法)或数组(单方法)
    save_path : str, optional
        保存路径
    title : str
        图表标题
    methods : list, optional
        方法名列表
    colors : list, optional
        颜色列表
        
    返回:
    ------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 处理输入格式
    if isinstance(predictions, dict):
        pred_dict = predictions
        if methods is None:
            methods = list(pred_dict.keys())
    else:
        pred_dict = {'Prediction': predictions}
        if methods is None:
            methods = ['Prediction']
    
    # 默认颜色
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 计算数据范围
    all_values = [ground_truth.flatten()]
    for method in methods:
        if method in pred_dict:
            all_values.append(pred_dict[method].flatten())
    
    min_val = min([v.min() for v in all_values])
    max_val = max([v.max() for v in all_values])
    
    # 绘制理想线
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Ideal (y=x)')
    
    # 绘制各方法散点
    for i, method in enumerate(methods):
        if method in pred_dict:
            color = colors[i % len(colors)]
            ax.scatter(
                ground_truth.flatten(), 
                pred_dict[method].flatten(),
                alpha=0.4, s=15, c=color, label=method
            )
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    
    # 图例和网格
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"一对一散点图已保存至: {save_path}")
    
    return fig


def plot_comparison_grid(
    images: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Image Comparison",
    cmap: str = 'rgb'
) -> plt.Figure:
    """
    =====================================================================
    绘制图像对比网格
    =====================================================================
    
    并排展示多幅图像进行对比。
    
    参数:
    ------
    images : dict
        图像字典，键为标题，值为图像数组 (H, W) 或 (H, W, 3)
    save_path : str, optional
        保存路径
    title : str
        图表标题
    cmap : str
颜色映射
        
    返回:
    ------
    plt.Figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (name, img) in zip(axes, images.items()):
        if img.ndim == 2:
            im = ax.imshow(img, cmap=cmap if cmap != 'rgb' else 'gray')
        else:
            im = ax.imshow(img)
        
        ax.set_title(name, fontsize=12)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像对比网格已保存至: {save_path}")
    
    return fig


def plot_metrics_bar(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'SSIM',
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    ascending: bool = False
) -> plt.Figure:
    """
    =====================================================================
    绘制指标柱状图
    =====================================================================
    
    对比不同方法的某一指标。
    
    参数:
    ------
    metrics : dict
        指标字典，格式: {method: {metric: value}}
    metric_name : str
        要绘制的指标名
    save_path : str, optional
        保存路径
    title : str, optional
        图表标题
    ascending : bool
        是否升序排列
        
    返回:
    ------
    plt.Figure
    """
    # 提取指定指标
    values = {}
    for method, method_metrics in metrics.items():
        if metric_name in method_metrics:
            values[method] = method_metrics[metric_name]
    
    # 排序
    sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=not ascending)
    methods = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.bar(methods, vals, color=colors, alpha=0.8)
    
    # 在柱子上添加数值标签
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10
        )
    
    # 设置
    ax.set_title(title or f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标柱状图已保存至: {save_path}")
    
    return fig


# =====================================================================
# 主程序入口
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MALA 可视化工具测试")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "output_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 测试误差热力图
    print("\n1. 测试误差热力图...")
    error = np.random.rand(256, 256)
    fig = plot_error_heatmap(
        error, 
        save_path=os.path.join(output_dir, 'error_heatmap.png')
    )
    plt.close(fig)
    
    # 2. 测试时序对比图
    print("2. 测试时序对比图...")
    gt = np.random.rand(8)
    preds = {
        'DINEOF': gt + np.random.randn(8) * 0.1,
        'Spline': gt + np.random.randn(8) * 0.15,
        'Proposed': gt + np.random.randn(8) * 0.05
    }
    fig = plot_timeseries(
        gt, preds, 
        missing_regions=[(2, 4)],
        save_path=os.path.join(output_dir, 'timeseries.png')
    )
    plt.close(fig)
    
    # 3. 测试箱线图
    print("3. 测试箱线图...")
    data = {
        'DINEOF': np.random.rand(50) * 0.3 + 0.6,
        'Spline': np.random.rand(50) * 0.25 + 0.65,
        'Proposed': np.random.rand(50) * 0.2 + 0.75
    }
    fig = plot_boxplot(
        data, 
        save_path=os.path.join(output_dir, 'boxplot.png')
    )
    plt.close(fig)
    
    # 4. 测试散点图
    print("4. 测试散点图...")
    gt = np.random.rand(100)
    pred = gt + np.random.randn(100) * 0.1
    fig = plot_scatter(
        gt, pred,
        save_path=os.path.join(output_dir, 'scatter.png')
    )
    plt.close(fig)
    
    # 5. 测试指标柱状图
    print("5. 测试指标柱状图...")
    metrics = {
        'DINEOF': {'SSIM': 0.75, 'PSNR': 28.5, 'MAE': 15.2},
        'Spline': {'SSIM': 0.78, 'PSNR': 29.1, 'MAE': 14.8},
        'Proposed': {'SSIM': 0.85, 'PSNR': 32.5, 'MAE': 10.2}
    }
    fig = plot_metrics_bar(
        metrics, 'SSIM',
        save_path=os.path.join(output_dir, 'metrics_bar.png')
    )
    plt.close(fig)
    
    print(f"\n所有可视化测试完成! 结果保存在: {output_dir}/")
