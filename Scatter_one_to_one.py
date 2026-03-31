"""兼容转发入口，实际实现位于 legacy/Scatter_one_to_one.py。"""

if __name__ == "__main__":
    import argparse
    from legacy.Scatter_one_to_one import plot_academic_1to1_scatter, print_statistics_table, plt

    parser = argparse.ArgumentParser(description="绘制 1:1 学术散点图")
    parser.add_argument("--reference-image-path", default='./rgb_S2_Daily_Mosaic/pseudo_color_doy_019.png')
    parser.add_argument("--reconstructed-image-path", default='./experiment_results/exp1_missing_types/thick_cloud/images/MALA/mala_inpainted_frame00.png')
    parser.add_argument("--mask-image-path", default='./experiment_results/exp1_missing_types/thin_cloud/images/mask_frame00.png')
    parser.add_argument("--output-figure-path", default='./1to1_scatter.png')
    args = parser.parse_args()

    fig, stats = plot_academic_1to1_scatter(
        ref_img_path=args.reference_image_path,
        rec_img_path=args.reconstructed_image_path,
        mask_img_path=args.mask_image_path,
        output_path=args.output_figure_path,
        band_names=['B2', 'B3', 'B4'],
        band_wavelengths=['490 nm', '560 nm', '665 nm'],
        use_density=True,
        max_points=10000,
        fig_width=7.5,
        fig_height=2.5,
    )
    print_statistics_table(stats)
    plt.show()
else:
    from legacy.Scatter_one_to_one import *  # noqa: F401,F403
