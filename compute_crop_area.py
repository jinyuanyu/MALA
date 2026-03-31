# -*- coding: utf-8 -*-
"""
快速计算  >>>  PSNR + SSIM  <<<  同一时相成对对比
结果同时输出到控制台 + Excel + CSV
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path
import pandas as pd
import argparse
from utils.paths import resolve_data_path

ALG_CHINESE = {
    'DINEOF': '数据插值（DINEOF）',
    'Proposed': '掩码自编码（MaskAE）',
    'Lama': 'LaMA',
    'Spline': '样条插值（Spline）',
    'Nearest_Neighbor': '最近邻插值（NN）',
    'EMAE': '方法一（EMAE）',
    'MALA': '方法二（MALA）'
}
TARGET_ALGORITHMS = list(ALG_CHINESE.keys())

# ---------- 2. 工具 ----------
def compute_metrics(img_true, img_test):
    if img_true.shape != img_test.shape:
        img_test = cv2.resize(img_test, (img_true.shape[1], img_true.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    if np.all(img_true == 0) or np.all(img_test == 0):
        return 0, 0
    psnr = peak_signal_noise_ratio(img_true, img_test, data_range=255)
    ssim = structural_similarity(img_true, img_test, data_range=255)
    return psnr, ssim

def main(root_dir="E:/lama/error_heatmap_cropped_images", exp_name="exp1_missing_types", scene_name="thin_cloud"):
    root_dir = Path(resolve_data_path(root_dir))
    scene_path = root_dir / exp_name / scene_name
    orig_dir = scene_path / "original"
    alg_dirs = [scene_path / alg for alg in TARGET_ALGORITHMS if (scene_path / alg).is_dir()]
    out_dir = scene_path / "metrics_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_file = out_dir / f"{scene_name}_PSNR_SSIM.xlsx"
    csv_file = out_dir / f"{scene_name}_PSNR_SSIM.csv"

    orig_files = sorted(orig_dir.glob("*.png"))
    if not orig_files:
        raise FileNotFoundError("原始目录无 *.png")

    records = []
    for alg_dir in alg_dirs:
        alg_files = sorted(alg_dir.glob("*.png"))
        num = min(len(orig_files), len(alg_files))
        for i in range(num):
            orig = cv2.imread(str(orig_files[i]), cv2.IMREAD_GRAYSCALE)
            alg = cv2.imread(str(alg_files[i]), cv2.IMREAD_GRAYSCALE)
            if orig is None or alg is None:
                continue
            psnr, ssim = compute_metrics(orig, alg)
            records.append({
                'Frame': orig_files[i].name,
                'Algorithm': ALG_CHINESE.get(alg_dir.name, alg_dir.name),
                'PSNR(dB)': psnr,
                'SSIM': ssim
            })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存：\n  Excel → {excel_file}\n  CSV   → {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裁剪区域快速计算 PSNR / SSIM")
    parser.add_argument("--root-dir", default="E:/lama/error_heatmap_cropped_images")
    parser.add_argument("--exp-name", default="exp1_missing_types")
    parser.add_argument("--scene-name", default="thin_cloud")
    args = parser.parse_args()
    main(args.root_dir, args.exp_name, args.scene_name)
