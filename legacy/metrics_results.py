import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import json
import argparse
from analysis.experiment import extract_frame_number, find_algorithm_output_images, iter_experiment_scenes

def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    """加载图像"""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path, color_mode)
    if img is not None and color_mode == cv2.IMREAD_COLOR:
        # OpenCV使用BGR，转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_mse(original, processed, mask):
    """计算掩码区域的MSE"""
    if original is None or processed is None or mask is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # 创建掩码（255为掩码区域）
    mask_binary = (mask > 127)
    
    if not np.any(mask_binary):
        return np.nan
    
    # 仅在掩码区域计算MSE
    original_masked = original[mask_binary]
    processed_masked = processed[mask_binary]
    
    mse = mean_squared_error(original_masked, processed_masked)
    return mse

def calculate_psnr(original, processed, mask, max_val=255.0):
    """计算掩码区域的PSNR"""
    mse = calculate_mse(original, processed, mask)
    
    if np.isnan(mse) or mse == 0:
        return np.inf if mse == 0 else np.nan
    
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def calculate_ssim(original, processed, mask):
    """计算掩码区域的SSIM"""
    if original is None or processed is None or mask is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # 创建掩码（255为掩码区域）
    mask_binary = (mask > 127)
    
    if not np.any(mask_binary):
        return np.nan
    
    try:
        # 对于彩色图像，计算每个通道的SSIM然后取平均
        if len(original.shape) == 3:
            ssim_values = []
            for channel in range(original.shape[2]):
                # 使用掩码创建感兴趣区域
                original_channel = original[:, :, channel]
                processed_channel = processed[:, :, channel]
                
                # 计算整个图像的SSIM，但主要关注掩码区域
                # 注意：SSIM需要一定的窗口大小，所以我们计算整个图像的SSIM
                ssim_val = ssim(original_channel, processed_channel, 
                               data_range=255, win_size=7)
                ssim_values.append(ssim_val)
            
            return np.mean(ssim_values)
        else:
            return ssim(original, processed, data_range=255, win_size=7)
            
    except Exception as e:
        print(f"SSIM calculation error: {e}")
        return np.nan

def extract_mask_region_for_ssim(original, processed, mask, padding=10):
    """提取掩码区域及周围区域用于SSIM计算"""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 找到掩码区域的边界
    mask_binary = (mask > 127)
    
    if not np.any(mask_binary):
        return None, None
    
    # 找到掩码区域的边界框
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 添加padding
    rmin = max(0, rmin - padding)
    rmax = min(original.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(original.shape[1], cmax + padding)
    
    # 提取区域
    if len(original.shape) == 3:
        original_region = original[rmin:rmax, cmin:cmax, :]
        processed_region = processed[rmin:rmax, cmin:cmax, :]
    else:
        original_region = original[rmin:rmax, cmin:cmax]
        processed_region = processed[rmin:rmax, cmin:cmax]
    
    return original_region, processed_region

def calculate_ssim_mask_region(original, processed, mask):
    """计算掩码区域的SSIM（改进版）"""
    if original is None or processed is None or mask is None:
        return np.nan
    
    # 确保图像尺寸一致
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # 提取掩码区域及周围区域
    original_region, processed_region = extract_mask_region_for_ssim(original, processed, mask)
    
    if original_region is None or processed_region is None:
        return np.nan
    
    # 确保区域足够大以计算SSIM
    if min(original_region.shape[:2]) < 7:
        return np.nan
    
    try:
        if len(original_region.shape) == 3:
            ssim_values = []
            for channel in range(original_region.shape[2]):
                ssim_val = ssim(original_region[:, :, channel], 
                               processed_region[:, :, channel], 
                               data_range=255, win_size=7)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            return ssim(original_region, processed_region, data_range=255, win_size=7)
    except Exception as e:
        print(f"SSIM calculation error: {e}")
        return np.nan

def calculate_metrics_for_frame(original_path, processed_path, mask_path):
    """计算单帧的所有指标"""
    # 加载图像
    original = load_image(str(original_path))
    processed = load_image(str(processed_path))
    mask = load_image(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if original is None or processed is None or mask is None:
        return None
    
    # 计算指标
    mse = calculate_mse(original, processed, mask)
    psnr = calculate_psnr(original, processed, mask)
    ssim_score = calculate_ssim_mask_region(original, processed, mask)
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_score
    }

def process_experiment_results(base_dir="experiment_results", output_dir="metrics_results"):
    """处理实验结果目录并计算所有指标"""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if not base_path.exists():
        print(f"实验结果目录不存在: {base_path}")
        return
    
    # 创建输出目录
    output_path.mkdir(exist_ok=True)
    
    # 存储所有结果
    all_results = []
    
    for exp_dir, scene_dir, images_dir in iter_experiment_scenes(base_path):
        print(f"处理实验: {exp_dir.name}")
        print(f"  处理场景: {scene_dir.name}")
        algorithm_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        if not algorithm_dirs:
            print(f"    未找到算法子目录在: {images_dir}")
            continue

        for alg_dir in algorithm_dirs:
            print(f"    处理算法: {alg_dir.name}")
            processed_images = find_algorithm_output_images(alg_dir)
            algorithm_results = []

            for processed_img_path in processed_images:
                filename = processed_img_path.name
                frame_num = extract_frame_number(filename)
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
                    
                    print(f"      计算帧 {frame_num} 的指标...")
                    
                    # 计算指标
                    metrics = calculate_metrics_for_frame(original_path, processed_img_path, mask_path)
                    
                    if metrics is not None:
                        result = {
                            'Experiment': exp_dir.name,
                            'Scene': scene_dir.name,
                            'Algorithm': alg_dir.name,
                            'Frame': f"frame{frame_num}",
                            'MSE': metrics['MSE'],
                            'PSNR': metrics['PSNR'],
                            'SSIM': metrics['SSIM']
                        }
                        
                        algorithm_results.append(result)
                        all_results.append(result)
                        
                        print(f"        MSE: {metrics['MSE']:.4f}, PSNR: {metrics['PSNR']:.2f}, SSIM: {metrics['SSIM']:.4f}")
                
                # 计算当前算法的平均指标
                if algorithm_results:
                    mse_values = [r['MSE'] for r in algorithm_results if not np.isnan(r['MSE'])]
                    psnr_values = [r['PSNR'] for r in algorithm_results if not np.isnan(r['PSNR']) and not np.isinf(r['PSNR'])]
                    ssim_values = [r['SSIM'] for r in algorithm_results if not np.isnan(r['SSIM'])]
                    
                    avg_result = {
                        'Experiment': exp_dir.name,
                        'Scene': scene_dir.name,
                        'Algorithm': alg_dir.name,
                        'Frame': 'AVERAGE',
                        'MSE': np.mean(mse_values) if mse_values else np.nan,
                        'PSNR': np.mean(psnr_values) if psnr_values else np.nan,
                        'SSIM': np.mean(ssim_values) if ssim_values else np.nan
                    }
                    
                    all_results.append(avg_result)
                    print(f"      算法 {alg_dir.name} 平均指标:")
                    print(f"        平均MSE: {avg_result['MSE']:.4f}")
                    print(f"        平均PSNR: {avg_result['PSNR']:.2f}")
                    print(f"        平均SSIM: {avg_result['SSIM']:.4f}")
    
    # 保存结果
    if all_results:
        # 转换为DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存为CSV
        csv_path = output_path / "metrics_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"结果已保存到: {csv_path}")
        
        # 保存为JSON（便于程序读取）
        json_path = output_path / "metrics_results.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"JSON结果已保存到: {json_path}")
        
        # 创建汇总统计
        summary_results = []
        for exp in df['Experiment'].unique():
            for scene in df[df['Experiment'] == exp]['Scene'].unique():
                scene_data = df[(df['Experiment'] == exp) & (df['Scene'] == scene) & (df['Frame'] == 'AVERAGE')]
                
                for _, row in scene_data.iterrows():
                    summary_results.append({
                        'Experiment': row['Experiment'],
                        'Scene': row['Scene'],
                        'Algorithm': row['Algorithm'],
                        'MSE': row['MSE'],
                        'PSNR': row['PSNR'],
                        'SSIM': row['SSIM']
                    })
        
        # 保存汇总结果
        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_csv_path = output_path / "metrics_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"汇总结果已保存到: {summary_csv_path}")
            
            # 打印汇总表格
            print("\n=== 算法性能汇总 ===")
            print(summary_df.round(4))
    else:
        print("未找到有效的结果数据")
    
    return all_results

def main(input_dir="experiment_results", output_dir="metrics_results"):
    """主函数"""
    print("开始计算图像质量指标...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("计算指标: MSE, PSNR, SSIM")
    print("-" * 50)
    
    results = process_experiment_results(input_dir, output_dir)
    
    print("-" * 50)
    print("计算完成！")
    return results

def parse_args_and_run():
    """解析命令行参数并运行"""
    parser = argparse.ArgumentParser(description="计算图像质量指标 (MSE, PSNR, SSIM)")
    parser.add_argument("--input", "-i", default="experiment_results",
                       help="实验结果目录路径 (默认: experiment_results)")
    parser.add_argument("--output", "-o", default="metrics_results",
                       help="输出目录路径 (默认: metrics_results)")
    
    args = parser.parse_args()
    main(args.input, args.output)

# 检测运行环境
if __name__ == "__main__":
    import sys

    if any('ipykernel' in arg for arg in sys.argv):
        main()
    else:
        parse_args_and_run()
else:
    # 如果作为模块导入，可以直接调用main函数
    pass
