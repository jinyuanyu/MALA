"""
学术论文级1:1散点图绘制工具（RGB图像+独立掩码版）
适用于Sentinel-2遥感影像重建质量评估

作者: [Your Name]
日期: 2024
用途: 对比重建影像与真值影像的三波段精度
输入: RGB彩色图像 + 独立掩码图像（白色=缺失区域，黑色=有效区域）
输出: 中文图表，反射率归一化后放大10000倍显示
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.stats import gaussian_kde
from PIL import Image
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')

# 1. 自动扫描系统已有中文字体
from matplotlib import font_manager
zh_fonts = [f for f in font_manager.fontManager.ttflist
            if 'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'Noto Sans CJK' in f.name]
if zh_fonts:                                # 优先用系统里第一个
    mpl.rcParams['font.family'] = zh_fonts[0].name
else:                                       # 兜底：用默认无衬线
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 2. 负号正常显示
mpl.rcParams['axes.unicode_minus'] = False

# 3. 统一字号（按需调）
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05


def load_rgb_image(file_path):
    """
    加载RGB图像文件（支持PNG, JPG, TIF等格式）
    """
    img = Image.open(file_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    data = np.array(img)
    return data


def load_mask_image(file_path):
    """
    加载掩码图像
    """
    img = Image.open(file_path)
    
    if img.mode != 'L':
        img = img.convert('L')
    
    mask_array = np.array(img)
    mask = mask_array < 128
    return mask


def normalize_to_reflectance(pixel_values):
    """
    将像素值归一化到反射率范围并放大10000倍
    """
    reflectance = (pixel_values / 255.0) * 10000.0
    return reflectance


def extract_valid_pixels(reference_img, reconstructed_img, mask):
    """
    提取有效像素（掩码指定的有效区域）并转换为反射率
    """
    if reference_img.shape[:2] != mask.shape:
        raise ValueError(f"图像尺寸 {reference_img.shape[:2]} 与掩码尺寸 {mask.shape} 不匹配")
    if reconstructed_img.shape[:2] != mask.shape:
        raise ValueError(f"图像尺寸 {reconstructed_img.shape[:2]} 与掩码尺寸 {mask.shape} 不匹配")
    
    ref_pixels_raw = reference_img[mask].astype(np.float64)
    rec_pixels_raw = reconstructed_img[mask].astype(np.float64)
    
    ref_pixels = normalize_to_reflectance(ref_pixels_raw)
    rec_pixels = normalize_to_reflectance(rec_pixels_raw)
    
    valid_idx = np.isfinite(ref_pixels).all(axis=1) & np.isfinite(rec_pixels).all(axis=1)
    ref_pixels = ref_pixels[valid_idx]
    rec_pixels = rec_pixels[valid_idx]
    
    valid_range = (ref_pixels >= 0) & (ref_pixels <= 10000) & \
                  (rec_pixels >= 0) & (rec_pixels <= 10000)
    valid_idx = valid_range.all(axis=1)
    ref_pixels = ref_pixels[valid_idx]
    rec_pixels = rec_pixels[valid_idx]
    
    return ref_pixels, rec_pixels


def calculate_statistics(ref_data, rec_data):
    """
    计算核心统计指标（只保留R²和RMSE）
    """
    n = len(ref_data)
    rmse = np.sqrt(np.mean((rec_data - ref_data) ** 2))
    r, _ = stats.pearsonr(ref_data, rec_data)
    r2 = r ** 2
    
    return {
        'N': n,
        'R2': r2,
        'RMSE': rmse
    }


def plot_academic_1to1_scatter(ref_img_path, rec_img_path, mask_img_path,
                               output_path='1to1_scatter.png',
                               band_names=None, band_wavelengths=None,
                               use_density=True, max_points=50000,
                               fig_width=7.5, fig_height=2.5):
    """
    绘制学术论文级1:1散点图（中文版）- 简化版
    """
    
    if band_names is None:
        band_names = ['B2', 'B3', 'B4']
    if band_wavelengths is None:
        band_wavelengths = ['490 nm', '560 nm', '665 nm']
    
    # 加载数据
    print(f"正在加载参考图像: {ref_img_path}")
    ref_img = load_rgb_image(ref_img_path)
    print(f"参考图像形状: {ref_img.shape}")
    
    print(f"正在加载重建图像: {rec_img_path}")
    rec_img = load_rgb_image(rec_img_path)
    print(f"重建图像形状: {rec_img.shape}")
    
    print(f"正在加载掩码图像: {mask_img_path}")
    mask = load_mask_image(mask_img_path)
    print(f"有效像素比例: {mask.sum() / mask.size * 100:.2f}%")
    
    # 提取有效像素
    ref_pixels, rec_pixels = extract_valid_pixels(ref_img, rec_img, mask)
    print(f"有效像素数量: {len(ref_pixels):,}")
    
    if len(ref_pixels) == 0:
        raise ValueError("没有有效像素！请检查掩码图像和输入图像")
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    all_stats = {}
    
    # 遍历三个波段
    for i, (ax, band_name, wavelength) in enumerate(zip(axes, band_names, band_wavelengths)):
        print(f"处理 {band_name} 波段...")
        
        # 提取当前波段数据
        ref_band = ref_pixels[:, i]
        rec_band = rec_pixels[:, i]
        
        # 随机采样（如果数据量过大）
        if len(ref_band) > max_points:
            idx = np.random.choice(len(ref_band), max_points, replace=False)
            ref_band_plot = ref_band[idx]
            rec_band_plot = rec_band[idx]
        else:
            ref_band_plot = ref_band
            rec_band_plot = rec_band
        
        # 计算统计量
        stats_dict = calculate_statistics(ref_band, rec_band)
        all_stats[band_name] = stats_dict
        
        # 确定坐标轴范围
        min_val = min(ref_band.min(), rec_band.min())
        max_val = max(ref_band.max(), rec_band.max())
        margin = (max_val - min_val) * 0.05
        plot_min = max(0, min_val - margin)
        plot_max = min(10000, max_val + margin)
        
        # 绘制散点图
        if use_density and len(ref_band_plot) > 1000:
            try:
                xy = np.vstack([ref_band_plot, rec_band_plot])
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                ref_sorted = ref_band_plot[idx]
                rec_sorted = rec_band_plot[idx]
                z_sorted = z[idx]
                
                scatter = ax.scatter(ref_sorted, rec_sorted, c=z_sorted, 
                                   s=3, cmap='viridis', alpha=0.6,
                                   edgecolors='none', rasterized=True)
            except:
                ax.scatter(ref_band_plot, rec_band_plot, 
                          s=1, alpha=0.3, c='steelblue',
                          edgecolors='none', rasterized=True)
        else:
            ax.scatter(ref_band_plot, rec_band_plot, 
                      s=1.5, alpha=0.4, c='steelblue',
                      edgecolors='none', rasterized=True)
        
        # 绘制1:1线
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 
               'r--', linewidth=1.5, label='1:1线', zorder=5, alpha=0.8)
        
        # 设置标签和标题
        if i == 0:
            ax.set_ylabel('重建影像反射率', fontsize=11, fontweight='bold')
        ax.set_xlabel('参考影像反射率', fontsize=11, fontweight='bold')
        
        # 标题：波段名称和波长
        title = f"{band_name}\n({wavelength})"
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        
        # 设置坐标轴
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect('equal', adjustable='box')
        
        # 网格
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, color='gray')
        
        # 简化的统计信息文本框（只显示R²和RMSE）
        textstr = f"$R^2$ = {stats_dict['R2']:.4f}\n"
        textstr += f"RMSE = {stats_dict['RMSE']:.1f}"
        
        # 文本框样式
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.9, linewidth=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=props, family='monospace')
        
        # 图例（仅第一个子图）
        if i == 0:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9,
                     edgecolor='gray', fancybox=False)
        
        # 刻度格式
        ax.ticklabel_format(style='plain', axis='both')
        
        # 打印统计信息
        print(f"  R² = {stats_dict['R2']:.4f}, RMSE = {stats_dict['RMSE']:.2f}")
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    print(f"\n保存图像到: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("完成!")
    
    return fig, all_stats


def print_statistics_table(all_stats):
    """
    打印格式化的统计表格 - 简化版
    """
    print("\n" + "="*50)
    print("统计指标汇总表")
    print("="*50)
    print(f"{'波段':<8} {'R²':<10} {'RMSE':<10} {'样本数':<12}")
    print("-"*50)
    
    for band_name, stats in all_stats.items():
        print(f"{band_name:<8} {stats['R2']:<10.4f} {stats['RMSE']:<10.2f} {stats['N']:<12,}")
    
    # 计算平均值
    avg_r2 = np.mean([s['R2'] for s in all_stats.values()])
    avg_rmse = np.mean([s['RMSE'] for s in all_stats.values()])
    
    print("-"*50)
    print(f"{'平均':<8} {avg_r2:<10.4f} {avg_rmse:<10.2f}")
    print("="*50)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="绘制 1:1 学术散点图")
    parser.add_argument("--reference-image-path", default='./rgb_S2_Daily_Mosaic/pseudo_color_doy_019.png')
    parser.add_argument("--reconstructed-image-path", default='./experiment_results/exp1_missing_types/thick_cloud/images/MALA/mala_inpainted_frame00.png')
    parser.add_argument("--mask-image-path", default='./experiment_results/exp1_missing_types/thin_cloud/images/mask_frame00.png')
    parser.add_argument("--output-figure-path", default='./1to1_scatter.png')
    args = parser.parse_args()

    fig, stats = plot_academic_1to1_scatter(
        ref_img_path=args.reference_image_path,
        rec_img_path=args.reconstructed_image_path,
        mask_img_path=args.mask_image_path,
        output_path=args.output_figure_path,
        band_names=['B2', 'B3', 'B4'],
        band_wavelengths=['490 nm', '560 nm', '665 nm'],
        use_density=True,
        max_points=10000,
        fig_width=7.5,
        fig_height=2.5
    )
    
    print_statistics_table(stats)
    plt.show()


if __name__ == "__main__":
    parse_args_and_run()
