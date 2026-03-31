"""
MALA误差热力图分析模块
======================

负责批量读取实验结果，计算掩码区域误差并生成热力图。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import configure_matplotlib_chinese, get_algorithm_display_name
from analysis.experiment import extract_frame_number, find_algorithm_output_images, iter_experiment_scenes


configure_matplotlib_chinese()


def load_image(image_path: str, color_mode=cv2.IMREAD_COLOR):
    if not os.path.exists(image_path):
        return None
    return cv2.imread(image_path, color_mode)


def calculate_error(original, processed, mask):
    if original is None or processed is None or mask is None:
        return None

    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    original_float = original.astype(np.float32)
    processed_float = processed.astype(np.float32)
    error = np.mean(np.abs(original_float - processed_float), axis=2) / 255.0 * 10000

    mask_binary = (mask > 127).astype(np.uint8)
    error_masked = error * mask_binary
    return error_masked, mask_binary


def create_heatmap(
    error_data,
    mask_binary,
    output_path: str,
    algorithm_name: str,
    rect_points=None,
    rect_label=None,
):
    fig, ax = plt.subplots(figsize=(12, 8))
    _ = get_algorithm_display_name(algorithm_name)
    mask_indices = mask_binary > 0

    if np.any(mask_indices):
        error_masked = error_data.copy().astype(float) * 2
        error_masked[~mask_indices] = np.nan

        im = ax.imshow(error_masked, cmap=plt.cm.YlOrRd, interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("MAE x10k", rotation=270, labelpad=15, fontsize=20, fontweight="bold")
        cbar.ax.tick_params(labelsize=14, width=1.5)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        im.set_clim(vmin=0, vmax=10000)

        if rect_points is not None and len(rect_points) == 4:
            img_height, img_width = error_data.shape
            valid_points = []
            for x, y in rect_points:
                valid_points.append((max(0, min(x, img_width - 1)), max(0, min(y, img_height - 1))))

            points = valid_points + [valid_points[0]]
            x_coords = [point[0] for point in points]
            y_coords = [point[1] for point in points]
            ax.plot(x_coords, y_coords, "g-", linewidth=3, label="关注区域")

            if rect_label:
                center_x = np.mean(x_coords[:-1])
                center_y = np.mean(y_coords[:-1])
                ax.text(
                    center_x,
                    center_y,
                    rect_label,
                    fontsize=20,
                    color="green",
                    weight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="green"),
                )

            ax.legend(loc="upper right", fontsize=20, prop=dict(weight="bold"))
    else:
        ax.imshow(np.zeros_like(error_data), cmap="gray")
        ax.text(
            0.5,
            0.5,
            "No mask region found",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color="red",
        )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_experiment_results(base_dir="experiment_results", output_dir="error_heatmaps", rect_points=None, rect_label=None):
    base_path = Path(base_dir)
    output_path = Path(output_dir)

    if not base_path.exists():
        print(f"实验结果目录不存在: {base_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for exp_dir, scene_dir, images_dir in iter_experiment_scenes(base_path):
        print(f"处理实验: {exp_dir.name}")
        print(f"  处理场景: {scene_dir.name}")
        algorithm_dirs = [directory for directory in images_dir.iterdir() if directory.is_dir()]
        if not algorithm_dirs:
            print(f"    未找到算法子目录在: {images_dir}")
            continue

        scene_output_dir = output_path / exp_dir.name / scene_dir.name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        for alg_dir in algorithm_dirs:
            print(f"    处理算法: {alg_dir.name}")
            alg_output_dir = scene_output_dir / alg_dir.name
            alg_output_dir.mkdir(exist_ok=True)

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

                original = load_image(str(original_path))
                processed = load_image(str(processed_img_path))
                mask = load_image(str(mask_path))

                result = calculate_error(original, processed, mask)
                if result is None:
                    print("      计算误差失败")
                    continue

                error_data, mask_binary = result
                heatmap_path = alg_output_dir / f"heatmap_frame{frame_num}.png"
                create_heatmap(
                    error_data,
                    mask_binary,
                    str(heatmap_path),
                    alg_dir.name,
                    rect_points=rect_points,
                    rect_label=rect_label,
                )
                print(f"      保存热力图: {heatmap_path}")

    print(f"处理完成！结果保存在: {output_path}")


def main(input_dir="experiment_results", output_dir="error_heatmaps", rect_points=None, rect_label=None):
    print("开始处理实验结果...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    if rect_points:
        print(f"矩形区域: {rect_points}")
        if rect_label:
            print(f"区域标签: {rect_label}")

    process_experiment_results(input_dir, output_dir, rect_points, rect_label)


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="生成掩码区域误差热力图")
    parser.add_argument("--input", "-i", default="experiment_results", help="实验结果目录路径")
    parser.add_argument("--output", "-o", default="error_heatmaps", help="输出目录路径")
    parser.add_argument("--rect-points", "-r", type=str, help="矩形区域四个点，格式: x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--rect-label", "-l", type=str, help="矩形区域标签文本")
    args = parser.parse_args()

    rect_points = None
    if args.rect_points:
        try:
            coords = [float(value) for value in args.rect_points.split(",")]
            if len(coords) == 8:
                rect_points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
            else:
                print("警告: 矩形点参数需要8个坐标值")
        except ValueError:
            print("警告: 矩形点参数格式错误")

    main(args.input, args.output, rect_points, args.rect_label)


__all__ = [
    "calculate_error",
    "create_heatmap",
    "load_image",
    "main",
    "parse_args_and_run",
    "process_experiment_results",
]

