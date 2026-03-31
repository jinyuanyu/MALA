"""兼容转发入口，实际实现位于 legacy/crop_img.py。"""

if __name__ == "__main__":
    import argparse
    from legacy.crop_img import ImageAnalyzer
    from utils.paths import resolve_data_path

    parser = argparse.ArgumentParser(description="裁剪实验结果中的兴趣区域")
    parser.add_argument("--experiment-root", default="E:/lama/experiment_results")
    parser.add_argument("--output-dir", default="E:/lama/error_heatmap_cropped_images")
    parser.add_argument("--rect-points", default="100,1550,300,1550,300,1750,100,1750")
    args = parser.parse_args()

    coords = [int(value) for value in args.rect_points.split(",")]
    rect_points = [(coords[0], coords[1]), (coords[2], coords[3]), (coords[4], coords[5]), (coords[6], coords[7])]

    analyzer = ImageAnalyzer(resolve_data_path(args.experiment_root))
    analyzer.discover_experiments_and_algorithms()
    analyzer.visualize_cropped_region(rect_points)
    analyzer.crop_region_from_images(rect_points, output_dir=resolve_data_path(args.output_dir), save_marked_images=True)
else:
    from legacy.crop_img import *  # noqa: F401,F403
