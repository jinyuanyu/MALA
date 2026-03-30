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

# ---------- 1. 参数 ----------
ROOT_DIR      = Path("E:/lama/error_heatmap_cropped_images")
EXP_NAME      = "exp1_missing_types"
SCENE_NAME    = "thin_cloud"

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

# 输出文件目录（可改为你喜欢的任何地方）
OUT_DIR       = ROOT_DIR / EXP_NAME / SCENE_NAME / "metrics_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_FILE    = OUT_DIR / f"{SCENE_NAME}_PSNR_SSIM.xlsx"
CSV_FILE      = OUT_DIR / f"{SCENE_NAME}_PSNR_SSIM.csv"

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

# ---------- 3. 扫描 ----------
scene_path = ROOT_DIR / EXP_NAME / SCENE_NAME
orig_dir   = scene_path / "original"
alg_dirs   = [scene_path / alg for alg in TARGET_ALGORITHMS if (scene_path / alg).is_dir()]

orig_files = sorted(orig_dir.glob("*.png"))
if not orig_files:
    raise FileNotFoundError("原始目录无 *.png")

# ---------- 4. 逐帧计算并收集 ----------
records = []   # 每行：帧名、算法、PSNR、SSIM

for alg_dir in alg_dirs:
    alg_files = sorted(alg_dir.glob("*.png"))
    num       = min(len(orig_files), len(alg_files))
    for i in range(num):
        orig = cv2.imread(str(orig_files[i]), cv2.IMREAD_GRAYSCALE)
        alg  = cv2.imread(str(alg_files[i]),  cv2.IMREAD_GRAYSCALE)
        if orig is None or alg is None:
            continue
        psnr, ssim = compute_metrics(orig, alg)
        records.append({
            'Frame': orig_files[i].name,
            'Algorithm': ALG_CHINESE.get(alg_dir.name, alg_dir.name),
            'PSNR(dB)': psnr,
            'SSIM': ssim
        })

# ---------- 5. 生成 DataFrame ----------
df = pd.DataFrame(records)

# 6. 控制台打印
print(df.to_string(index=False))

# 7. 写文件
df.to_csv(CSV_FILE,   index=False, encoding='utf-8-sig')
# df.to_excel(EXCEL_FILE, index=False, encoding='utf-8-sig')
print(f"\n结果已保存：\n  Excel → {EXCEL_FILE}\n  CSV   → {CSV_FILE}")