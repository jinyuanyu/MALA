"""兼容转发入口，实际实现位于 legacy/compute_crop_area.py。"""

if __name__ == "__main__":
    import argparse
    from legacy.compute_crop_area import main

    parser = argparse.ArgumentParser(description="裁剪区域快速计算 PSNR / SSIM")
    parser.add_argument("--root-dir", default="E:/lama/error_heatmap_cropped_images")
    parser.add_argument("--exp-name", default="exp1_missing_types")
    parser.add_argument("--scene-name", default="thin_cloud")
    args = parser.parse_args()
    main(args.root_dir, args.exp_name, args.scene_name)
else:
    from legacy.compute_crop_area import *  # noqa: F401,F403
