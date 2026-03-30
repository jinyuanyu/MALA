import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 算法名称映射字典
ALGORITHM_NAMES = {
    'DINEOF': '数据插值（DINEOF）',
    'Proposed': '掩码自编码（MaskAE）',
    'Lama': 'LaMA',
    'Spline': '样条插值（Spline）',
    'Nearest_Neighbor': '最近邻插值（NN）',
    'EMAE': '方法一（EMAE）',
    'MALA': '方法二（MALA）'
}

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """加载图像"""
    if not os.path.exists(image_path):
        return None
    return cv2.imread(image_path, color_mode)

def calculate_error(original, processed, mask):
    """计算误差，仅在掩码区域"""
    if original is None or processed is None or mask is None:
        return None
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # 转换为浮点数计算
    original_float = original.astype(np.float32)
    processed_float = processed.astype(np.float32)
    
    # 计算归一化误差
    error = np.mean(np.abs(original_float - processed_float), axis=2) / 255.0 * 10000  # 归一化到0-10000范围

    # 创建掩码（255为掩码区域）
    mask_binary = (mask > 127).astype(np.uint8)
    
    # 仅保留掩码区域的误差
    error_masked = error * mask_binary
    
    return error_masked, mask_binary

def get_algorithm_display_name(algorithm_name):
    """获取算法的显示名称（中文）"""
    return ALGORITHM_NAMES.get(algorithm_name, algorithm_name)

def create_heatmap(error_data, mask_binary, output_path, frame_name, algorithm_name, 
                  rect_points=None, rect_label=None):
    """创建并保存误差热力图
    
    参数:
        error_data: 误差数据数组
        mask_binary: 二值掩码
        output_path: 输出路径
        frame_name: 帧名称
        algorithm_name: 算法名称
        rect_points: 矩形区域的四个点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 或 None
        rect_label: 矩形区域的标签文本
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取算法中文名称
    display_name = get_algorithm_display_name(algorithm_name)
    
    # 仅在掩码区域显示热力图
    mask_indices = mask_binary > 0
    
    if np.any(mask_indices):
        # 创建掩码后的误差数据，非掩码区域设为NaN（透明）
        error_masked = error_data.copy().astype(float)*2 # 放大1.3倍以增强对比度
        error_masked[~mask_indices] = np.nan
        
        # 使用'hot'颜色映射（黑->红->黄->白）
        # 为了得到黄到红的映射，我们使用YlOrRd（黄-橙-红）
        cmap = plt.cm.YlOrRd
        
        # 显示热力图
        im = ax.imshow(error_masked, cmap=cmap, interpolation='nearest')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('MAE ×10k', rotation=270, labelpad=15,fontsize=20, fontweight='bold')
        cbar.ax.tick_params(labelsize=14, width=1.5)      # 刻度数字大小
        for l in cbar.ax.get_yticklabels():
            l.set_fontweight('bold')  
            
        # 设置颜色条的范围
        im.set_clim(vmin=0, vmax=10000)
        
        # 如果提供了矩形区域点，绘制绿色框
        # 绘制绿色矩形框的修正版本
        if rect_points is not None and len(rect_points) == 4:
            # 检查坐标是否在图像范围内
            img_height, img_width = error_data.shape
            
            # 验证坐标有效性
            valid_points = []
            for x, y in rect_points:
                # 确保坐标在图像范围内
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                valid_points.append((x, y))
            
            # 将四个点连接成矩形
            points = valid_points + [valid_points[0]]  # 闭合矩形
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            # 绘制绿色矩形框
            ax.plot(x_coords, y_coords, 'g-', linewidth=3, label='关注区域')
            
            # 添加矩形标签
            if rect_label:
                # 在矩形中心位置添加标签
                center_x = np.mean(x_coords[:-1])
                center_y = np.mean(y_coords[:-1])
                ax.text(center_x, center_y, rect_label, 
                    fontsize=20, color='green', weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            alpha=0.8, edgecolor="green"))
            
            # 添加图例
            ax.legend(loc='upper right', fontsize=20, prop=dict(weight='bold'))
        
    else:
        # 如果没有掩码区域，显示空白图像
        ax.imshow(np.zeros_like(error_data), cmap='gray')
        ax.text(0.5, 0.5, 'No mask region found', transform=ax.transAxes, 
                ha='center', va='center', fontsize=18, color='red')

    # ax.set_title(f'误差热力图 - {display_name} ',fontsize=20, fontweight='bold')
    ax.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # 明确关闭figure

def process_experiment_results(base_dir="experiment_results", output_dir="error_heatmaps", 
                              rect_points=None, rect_label=None):
    """处理实验结果目录
    
    参数:
        base_dir: 基础目录路径
        output_dir: 输出目录路径
        rect_points: 矩形区域的四个点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 或 None
        rect_label: 矩形区域的标签文本
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if not base_path.exists():
        print(f"实验结果目录不存在: {base_path}")
        return
    
    # 创建输出目录
    output_path.mkdir(exist_ok=True)
    
    # 遍历所有实验目录
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        print(f"处理实验: {exp_dir.name}")
        
        # 遍历每个实验下的场景目录（如thin_cloud）
        for scene_dir in exp_dir.iterdir():
            if not  .is_dir():
                continue
                
            print(f"  处理场景: {scene_dir.name}")
            
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                print(f"    跳过，未找到images目录: {images_dir}")
                continue
            
            # 查找所有算法子目录
            algorithm_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
            
            if not algorithm_dirs:
                print(f"    未找到算法子目录在: {images_dir}")
                continue
            
            # 为当前场景创建输出目录
            scene_output_dir = output_path / exp_dir.name / scene_dir.name
            scene_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 处理每个算法目录
            for alg_dir in algorithm_dirs:
                print(f"    处理算法: {alg_dir.name}")
                
                # 为当前算法创建输出目录
                alg_output_dir = scene_output_dir / alg_dir.name
                alg_output_dir.mkdir(exist_ok=True)
                
                # 查找所有处理后的图像
                processed_images = list(alg_dir.glob("*_inpainted_frame*.png"))
                
                if not processed_images:
                    # 如果没有inpainted图像，查找其他可能的处理图像
                    processed_images = [f for f in alg_dir.glob("*.png") 
                                      if not f.name.startswith(('mask_', 'original_', 'masked_'))]
                
                for processed_img_path in processed_images:
                    # 提取帧编号
                    filename = processed_img_path.name
                    
                    # 尝试不同的命名模式来提取帧编号
                    frame_num = None
                    if "frame" in filename:
                        # 提取frame后的数字
                        import re
                        match = re.search(r'frame(\d+)', filename)
                        if match:
                            frame_num = match.group(1).zfill(2)
                    
                    if frame_num is None:
                        print(f"      无法提取帧编号: {filename}")
                        continue
                    
                    # 查找对应的原始图像和掩码
                    original_path = images_dir / f"original_frame{frame_num}.png"
                    mask_path = images_dir / f"mask_frame{frame_num}.png"
                    
                    if not original_path.exists():
                        print(f"      未找到原始图像: {original_path}")
                        continue
                        
                    if not mask_path.exists():
                        print(f"      未找到掩码图像: {mask_path}")
                        continue
                    
                    print(f"      处理帧 {frame_num}")
                    
                    # 加载图像
                    original = load_image(str(original_path))
                    processed = load_image(str(processed_img_path))
                    mask = load_image(str(mask_path))
                    
                    # 计算误差
                    result = calculate_error(original, processed, mask)
                    if result is None:
                        print(f"        计算误差失败")
                        continue
                        
                    error_data, mask_binary = result
                    
                    # 创建热力图
                    heatmap_filename = f"heatmap_frame{frame_num}.png"
                    heatmap_path = alg_output_dir / heatmap_filename
                    
                    create_heatmap(error_data, mask_binary, str(heatmap_path), 
                                 f"frame{frame_num}", alg_dir.name,
                                 rect_points, rect_label)
                    
                    print(f"        保存热力图: {heatmap_path}")
    
    print(f"处理完成！结果保存在: {output_path}")

def main(input_dir="experiment_results", output_dir="error_heatmaps", 
         rect_points=None, rect_label=None):
    """主函数，可以直接调用或通过命令行参数调用
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        rect_points: 矩形区域的四个点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 或 None
        rect_label: 矩形区域的标签文本
    """
    print("开始处理实验结果...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    if rect_points:
        print(f"矩形区域: {rect_points}")
        if rect_label:
            print(f"区域标签: {rect_label}")
    
    process_experiment_results(input_dir, output_dir, rect_points, rect_label)

def parse_args_and_run():
    """解析命令行参数并运行"""
    parser = argparse.ArgumentParser(description="生成掩码区域误差热力图")
    parser.add_argument("--input", "-i", default="experiment_results",
                       help="实验结果目录路径 (默认: experiment_results)")
    parser.add_argument("--output", "-o", default="error_heatmaps",
                       help="输出目录路径 (默认: error_heatmaps)")
    parser.add_argument("--rect-points", "-r", type=str,
                       help="矩形区域四个点，格式: 'x1,y1,x2,y2,x3,y3,x4,y4'")
    parser.add_argument("--rect-label", "-l", type=str,
                       help="矩形区域标签文本")
    
    args = parser.parse_args()
    
    # 解析矩形点参数
    rect_points = None
    if args.rect_points:
        try:
            coords = [float(x) for x in args.rect_points.split(',')]
            if len(coords) == 8:
                rect_points = [(coords[0], coords[1]), 
                             (coords[2], coords[3]),
                             (coords[4], coords[5]),
                             (coords[6], coords[7])]
            else:
                print("警告: 矩形点参数需要8个坐标值")
        except ValueError:
            print("警告: 矩形点参数格式错误")
    
    main(args.input, args.output, rect_points, args.rect_label)

# 使用示例函数
def example_usage():
    """使用示例"""
    # 示例矩形区域点（假设图像尺寸为512x512）
    # example_rect = [(100, 1550), (300, 1550), (300, 1750), (100, 1750)]# 左上、右上、右下、左下
    # example_label = "关注区域"
    
    # 调用主函数
    main("experiment_results", "error_heatmaps", example_rect)

# 检测运行环境
if __name__ == "__main__":
    # 如果在Jupyter中运行，直接调用main函数
    try:
        import sys
        if any('ipykernel' in arg for arg in sys.argv):
            # 在Jupyter环境中，使用默认参数或示例参数
            # 取消注释下面一行来使用示例矩形区域
            example_usage()
            # main()
        else:
            # 在命令行环境中，解析参数
            parse_args_and_run()
    except:
        # 如果出错，使用默认参数
        main()
else:
    # 如果作为模块导入，可以直接调用main函数
    pass