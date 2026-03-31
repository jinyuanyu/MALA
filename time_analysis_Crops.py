"""兼容转发入口，实际实现位于 legacy/time_analysis_Crops.py。"""

if __name__ == "__main__":
    import argparse
    from legacy.time_analysis_Crops import main

    parser = argparse.ArgumentParser(description="裁剪区域时序分析")
    parser.add_argument("--cropped-root", default="E:/lama/landOcean_cropped_images")
    parser.add_argument("--exp-name", default="exp2_missing_ratios")
    parser.add_argument("--scene-name", default="10percent")
    parser.add_argument("--base-save-path", default="landOcean_region_timeseries_comparison")
    args = parser.parse_args()
    main(args.cropped_root, args.exp_name, args.scene_name, args.base_save_path)
else:
    from legacy.time_analysis_Crops import *  # noqa: F401,F403
