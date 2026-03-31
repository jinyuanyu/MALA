"""
实验结果目录遍历工具
====================
"""

from __future__ import annotations

import re
from pathlib import Path


def iter_experiment_scenes(base_dir: str | Path):
    base_path = Path(base_dir)
    if not base_path.exists():
        return

    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        for scene_dir in sorted(exp_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            images_dir = scene_dir / "images"
            if images_dir.exists():
                yield exp_dir, scene_dir, images_dir


def find_algorithm_output_images(alg_dir: Path):
    processed_images = sorted(alg_dir.glob("*_inpainted_frame*.png"))
    if processed_images:
        return processed_images
    return sorted(
        file
        for file in alg_dir.glob("*.png")
        if not file.name.startswith(("mask_", "original_", "masked_"))
    )


def extract_frame_number(filename: str) -> str | None:
    if "frame" not in filename:
        return None
    match = re.search(r"frame(\d+)", filename)
    if not match:
        return None
    return match.group(1).zfill(2)
