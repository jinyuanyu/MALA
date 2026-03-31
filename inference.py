"""
MALA项目推理入口
================

当前文件只负责参数解析与对象装配，
具体推理逻辑已经模块化到 engine/ 目录。
"""

import argparse
import os

import torch

from engine.builders import build_dataloader, build_inference_dataset, build_model
from engine.config import DataConfig, InferenceConfig, ModelConfig, resolve_device
from engine.inference import run_inference, save_metrics_report


def parse_args():
    parser = argparse.ArgumentParser(description="MALA模型推理")

    parser.add_argument("--data_dir", type=str, required=True, help="推理数据目录路径")
    parser.add_argument("--ocean_mask_path", type=str, default=None, help="海洋掩码路径")
    parser.add_argument("--max_seq_len", type=int, default=8, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader工作线程数")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="random",
        choices=["random", "cloud", "strip", "mixed", "predefined"],
        help="掩码类型",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="掩码缺失比例")

    parser.add_argument("--model_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--img_size_h", type=int, default=224, help="图像高度")
    parser.add_argument("--img_size_w", type=int, default=224, help="图像宽度")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch大小")
    parser.add_argument("--embed_dim", type=int, default=768, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=12, help="注意力头数")
    parser.add_argument("--use_lama_init", action="store_true", help="是否使用LaMa初始修复")
    parser.add_argument("--use_mask_channel", action="store_true", help="是否使用掩码通道")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")

    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--save_images", action="store_true", help="是否保存重建图像")
    parser.add_argument("--save_visualization", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"使用设备: {device}")

    inference_config = InferenceConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        save_images=args.save_images,
        save_visualization=args.save_visualization,
        device=device,
    )
    data_config = DataConfig(
        data_dir=args.data_dir,
        ocean_mask_path=args.ocean_mask_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
    )
    model_config = ModelConfig(
        img_size_h=args.img_size_h,
        img_size_w=args.img_size_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_lama_init=args.use_lama_init,
        use_mask_channel=args.use_mask_channel,
        dropout=args.dropout,
    )

    os.makedirs(inference_config.output_dir, exist_ok=True)

    print("加载数据集...")
    dataset = build_inference_dataset(data_config)
    dataloader = build_dataloader(
        dataset=dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )
    print(f"推理样本数: {len(dataset)}")

    print("创建模型...")
    model = build_model(model_config, device)
    print(f"加载模型权重: {inference_config.model_path}")
    model.load_state_dict(torch.load(inference_config.model_path, map_location=device))

    print("开始推理...")
    metrics = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        save_images=inference_config.save_images,
        save_visualization=inference_config.save_visualization,
        output_dir=inference_config.output_dir,
    )

    print("\n" + "=" * 50)
    print("推理结果")
    print("=" * 50)
    print(f"SSIM: {metrics['SSIM']:.4f} ± {metrics['SSIM_std']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.2f} ± {metrics['PSNR_std']:.2f}")
    print(f"MAE:  {metrics['MAE']:.4f} ± {metrics['MAE_std']:.4f}")

    metrics_path = save_metrics_report(metrics, inference_config.output_dir)
    print(f"\n指标已保存到: {metrics_path}")


if __name__ == "__main__":
    main()
