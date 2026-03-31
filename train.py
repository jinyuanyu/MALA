"""
MALA项目训练入口
================

当前文件只负责参数解析与对象装配，
具体训练循环已经模块化到 engine/ 目录。
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from engine.builders import build_dataloader, build_model, build_train_dataset
from engine.config import DataConfig, ModelConfig, TrainConfig, resolve_device
from engine.trainer import create_scheduler, load_pretrained_weights, run_training


def parse_args():
    parser = argparse.ArgumentParser(description="MALA模型训练")

    parser.add_argument("--data_dir", type=str, required=True, help="训练数据目录路径")
    parser.add_argument("--ocean_mask_path", type=str, default=None, help="海洋掩码路径")
    parser.add_argument("--max_seq_len", type=int, default=8, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader工作线程数")

    parser.add_argument("--img_size_h", type=int, default=224, help="图像高度")
    parser.add_argument("--img_size_w", type=int, default=224, help="图像宽度")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch大小")
    parser.add_argument("--embed_dim", type=int, default=768, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=12, help="注意力头数")
    parser.add_argument("--use_lama_init", action="store_true", help="是否使用LaMa初始修复")
    parser.add_argument("--use_ocean_prior", action="store_true", help="是否使用海洋先验")
    parser.add_argument("--use_mask_channel", action="store_true", help="是否使用掩码通道")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")

    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--pretrained_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="周期性保存间隔")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"使用设备: {device}")

    data_config = DataConfig(
        data_dir=args.data_dir,
        ocean_mask_path=args.ocean_mask_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model_config = ModelConfig(
        img_size_h=args.img_size_h,
        img_size_w=args.img_size_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_lama_init=args.use_lama_init,
        use_ocean_prior=args.use_ocean_prior,
        use_mask_channel=args.use_mask_channel,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_path=args.pretrained_path,
        log_interval=args.log_interval,
        checkpoint_every=args.checkpoint_every,
        device=device,
    )

    print("加载数据集...")
    train_dataset = build_train_dataset(data_config)
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
    )
    print(f"训练样本数: {len(train_dataset)}")

    print("创建模型...")
    model = build_model(model_config, device)
    model = load_pretrained_weights(model, train_config.pretrained_path, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config.lr)
    scheduler = create_scheduler(optimizer)

    run_training(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=train_config,
        use_lama=model_config.use_lama_init,
    )


if __name__ == "__main__":
    main()
