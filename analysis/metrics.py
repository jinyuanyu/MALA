"""
MALA指标统计分析模块
====================

负责批量计算掩码区域的 MSE / PSNR / SSIM，并输出汇总结果。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from analysis.experiment import extract_frame_number, find_algorithm_output_images, iter_experiment_scenes


def load_image(image_path, color_mode=cv2.IMREAD_COLOR):
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path, color_mode)
    if img is not None and color_mode == cv2.IMREAD_COLOR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calculate_mse(original, processed, mask):
    if original is None or processed is None or mask is None:
        return np.nan

    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    mask_binary = mask > 127
    if not np.any(mask_binary):
        return np.nan

    return mean_squared_error(original[mask_binary], processed[mask_binary])


def calculate_psnr(original, processed, mask, max_val=255.0):
    mse_value = calculate_mse(original, processed, mask)
    if np.isnan(mse_value) or mse_value == 0:
        return np.inf if mse_value == 0 else np.nan
    return 20 * np.log10(max_val / np.sqrt(mse_value))


def extract_mask_region_for_ssim(original, processed, mask, padding=10):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_binary = mask > 127
    if not np.any(mask_binary):
        return None, None

    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - padding)
    rmax = min(original.shape[0], rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(original.shape[1], cmax + padding)

    if len(original.shape) == 3:
        return original[rmin:rmax, cmin:cmax, :], processed[rmin:rmax, cmin:cmax, :]
    return original[rmin:rmax, cmin:cmax], processed[rmin:rmax, cmin:cmax]


def calculate_ssim_mask_region(original, processed, mask):
    if original is None or processed is None or mask is None:
        return np.nan

    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    original_region, processed_region = extract_mask_region_for_ssim(original, processed, mask)
    if original_region is None or processed_region is None:
        return np.nan
    if min(original_region.shape[:2]) < 7:
        return np.nan

    try:
        if len(original_region.shape) == 3:
            values = []
            for channel in range(original_region.shape[2]):
                values.append(
                    ssim(
                        original_region[:, :, channel],
                        processed_region[:, :, channel],
                        data_range=255,
                        win_size=7,
                    )
                )
            return np.mean(values)
        return ssim(original_region, processed_region, data_range=255, win_size=7)
    except Exception as error:
        print(f"SSIM calculation error: {error}")
        return np.nan


def calculate_metrics_for_frame(original_path, processed_path, mask_path):
    original = load_image(str(original_path))
    processed = load_image(str(processed_path))
    mask = load_image(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if original is None or processed is None or mask is None:
        return None

    return {
        "MSE": calculate_mse(original, processed, mask),
        "PSNR": calculate_psnr(original, processed, mask),
        "SSIM": calculate_ssim_mask_region(original, processed, mask),
    }


def process_experiment_results(base_dir="experiment_results", output_dir="metrics_results"):
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    if not base_path.exists():
        print(f"实验结果目录不存在: {base_path}")
        return []

    output_path.mkdir(parents=True, exist_ok=True)
    all_results = []

    for exp_dir, scene_dir, images_dir in iter_experiment_scenes(base_path):
        print(f"处理实验: {exp_dir.name}")
        print(f"  处理场景: {scene_dir.name}")
        algorithm_dirs = [directory for directory in images_dir.iterdir() if directory.is_dir()]
        if not algorithm_dirs:
            print(f"    未找到算法子目录在: {images_dir}")
            continue

        for alg_dir in algorithm_dirs:
            print(f"    处理算法: {alg_dir.name}")
            algorithm_results = []

            for processed_img_path in find_algorithm_output_images(alg_dir):
                frame_num = extract_frame_number(processed_img_path.name)
                if frame_num is None:
                    print(f"      无法提取帧编号: {processed_img_path.name}")
                    continue

                original_path = images_dir / f"original_frame{frame_num}.png"
                mask_path = images_dir / f"mask_frame{frame_num}.png"

                if not original_path.exists():
                    print(f"      未找到原始图像: {original_path}")
                    continue
                if not mask_path.exists():
                    print(f"      未找到掩码图像: {mask_path}")
                    continue

                metrics = calculate_metrics_for_frame(original_path, processed_img_path, mask_path)
                if metrics is None:
                    print("      指标计算失败")
                    continue

                result = {
                    "Experiment": exp_dir.name,
                    "Scene": scene_dir.name,
                    "Algorithm": alg_dir.name,
                    "Frame": f"frame{frame_num}",
                    "MSE": metrics["MSE"],
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"],
                }
                algorithm_results.append(result)
                all_results.append(result)
                print(
                    f"      帧 {frame_num}: "
                    f"MSE={metrics['MSE']:.4f}, PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}"
                )

            if algorithm_results:
                mse_values = [item["MSE"] for item in algorithm_results if not np.isnan(item["MSE"])]
                psnr_values = [
                    item["PSNR"]
                    for item in algorithm_results
                    if not np.isnan(item["PSNR"]) and not np.isinf(item["PSNR"])
                ]
                ssim_values = [item["SSIM"] for item in algorithm_results if not np.isnan(item["SSIM"])]

                avg_result = {
                    "Experiment": exp_dir.name,
                    "Scene": scene_dir.name,
                    "Algorithm": alg_dir.name,
                    "Frame": "AVERAGE",
                    "MSE": np.mean(mse_values) if mse_values else np.nan,
                    "PSNR": np.mean(psnr_values) if psnr_values else np.nan,
                    "SSIM": np.mean(ssim_values) if ssim_values else np.nan,
                }
                all_results.append(avg_result)
                print(
                    f"      算法 {alg_dir.name} 平均指标: "
                    f"MSE={avg_result['MSE']:.4f}, PSNR={avg_result['PSNR']:.2f}, SSIM={avg_result['SSIM']:.4f}"
                )

    if not all_results:
        print("未找到有效的结果数据")
        return []

    dataframe = pd.DataFrame(all_results)
    csv_path = output_path / "metrics_results.csv"
    dataframe.to_csv(csv_path, index=False)
    print(f"结果已保存到: {csv_path}")

    json_path = output_path / "metrics_results.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)
    print(f"JSON结果已保存到: {json_path}")

    summary_results = []
    for experiment in dataframe["Experiment"].unique():
        for scene in dataframe[dataframe["Experiment"] == experiment]["Scene"].unique():
            scene_data = dataframe[
                (dataframe["Experiment"] == experiment)
                & (dataframe["Scene"] == scene)
                & (dataframe["Frame"] == "AVERAGE")
            ]
            for _, row in scene_data.iterrows():
                summary_results.append(
                    {
                        "Experiment": row["Experiment"],
                        "Scene": row["Scene"],
                        "Algorithm": row["Algorithm"],
                        "MSE": row["MSE"],
                        "PSNR": row["PSNR"],
                        "SSIM": row["SSIM"],
                    }
                )

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = output_path / "metrics_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"汇总结果已保存到: {summary_csv_path}")
        print("\n=== 算法性能汇总 ===")
        print(summary_df.round(4))

    return all_results


def main(input_dir="experiment_results", output_dir="metrics_results"):
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
    parser = argparse.ArgumentParser(description="计算图像质量指标 (MSE, PSNR, SSIM)")
    parser.add_argument("--input", "-i", default="experiment_results", help="实验结果目录路径")
    parser.add_argument("--output", "-o", default="metrics_results", help="输出目录路径")
    args = parser.parse_args()
    main(args.input, args.output)


__all__ = [
    "calculate_metrics_for_frame",
    "calculate_mse",
    "calculate_psnr",
    "calculate_ssim_mask_region",
    "load_image",
    "main",
    "parse_args_and_run",
    "process_experiment_results",
]

