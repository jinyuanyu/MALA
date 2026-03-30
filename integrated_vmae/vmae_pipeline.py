#!/usr/bin/env python3
"""Unified VMAE pipeline entrypoint.

Stages:
- preprocess: simulate missing data under different mask strategies
- train: model training from scratch
- finetune: continue training from pretrained weights
- infer: run inference with pretrained weights
- analyze: run experiment analysis and generate figures/reports
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from MAE_LaMa import Datasets_inference, VideoCompletionModel, inference_with_pretrained, train
from experiments import run_four_experiments
from visualizer import generate_all_visualizations, generate_comprehensive_report


def _parse_csv_str_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_csv_float_list(value: str) -> List[float]:
    out = []
    for token in _parse_csv_str_list(value):
        out.append(float(token))
    return out


def _parse_csv_int_list(value: str) -> List[int]:
    out = []
    for token in _parse_csv_str_list(value):
        out.append(int(token))
    return out


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detect_image_size(data_dir: str) -> tuple[int, int]:
    p = Path(data_dir)
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
    if not files:
        raise FileNotFoundError(f"No image files found under: {data_dir}")
    with Image.open(files[0]) as img:
        width, height = img.size
    return height, width


def _unnorm(img_t: torch.Tensor) -> torch.Tensor:
    return (img_t * 0.5 + 0.5).clamp(0, 1)


def _save_rgb_tensor(path: Path, img_t: torch.Tensor) -> None:
    img_t = _unnorm(img_t).detach().cpu()
    arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _save_mask_tensor(path: Path, mask_t: torch.Tensor) -> None:
    mask_t = mask_t.detach().cpu()
    if mask_t.ndim == 3:
        mask_t = mask_t.squeeze(0)
    arr = (mask_t.numpy() * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _build_model(args: argparse.Namespace, img_h: int, img_w: int) -> VideoCompletionModel:
    fine_tune_layers = _parse_csv_str_list(args.fine_tune_layers) if getattr(args, "fine_tune_layers", "") else []
    model = VideoCompletionModel(
        img_size_h=img_h,
        img_size_w=img_w,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        use_lama_init=args.use_lama_init,
        use_ocean_prior=args.use_ocean_prior,
        freeze_backbone=args.freeze_backbone,
        fine_tune_layers=fine_tune_layers,
        use_mask_channel=args.use_mask_channel,
        out_channels=args.out_channels,
        dropout=args.dropout,
    )
    return model


def _build_dataset(
    data_dir: str,
    max_seq_len: int,
    ocean_mask_path: Optional[str],
    mask_type: str,
    mask_ratio: float,
) -> Datasets_inference:
    return Datasets_inference(
        data_dir=data_dir,
        max_seq_len=max_seq_len,
        ocean_mask_path=ocean_mask_path,
        mask_type=mask_type,
        mask_ratio=mask_ratio,
    )


def run_preprocess(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    mask_types = _parse_csv_str_list(args.mask_types)
    mask_ratios = _parse_csv_float_list(args.mask_ratios)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] data_dir={args.data_dir}")
    for mask_type in mask_types:
        for mask_ratio in mask_ratios:
            ds = _build_dataset(
                data_dir=args.data_dir,
                max_seq_len=args.max_seq_len,
                ocean_mask_path=args.ocean_mask_path,
                mask_type=mask_type,
                mask_ratio=mask_ratio,
            )
            seq_count = min(args.num_sequences, len(ds))
            ratio_tag = f"{int(mask_ratio * 100):02d}percent"
            print(f"[preprocess] mask_type={mask_type}, mask_ratio={mask_ratio}, sequences={seq_count}")

            for seq_idx in range(seq_count):
                sample = ds[seq_idx]
                video = sample["video"]
                masked = sample["masked"]
                mask = sample["mask"]
                times = sample["times"]

                seq_dir = out_root / mask_type / ratio_tag / f"seq_{seq_idx:03d}"
                (seq_dir / "original").mkdir(parents=True, exist_ok=True)
                (seq_dir / "masked").mkdir(parents=True, exist_ok=True)
                (seq_dir / "mask").mkdir(parents=True, exist_ok=True)

                t_count = min(args.max_seq_len, video.shape[0])
                for t in range(t_count):
                    frame_id = int(times[t].item()) if t < len(times) else t
                    _save_rgb_tensor(seq_dir / "original" / f"original_frame{frame_id:04d}.png", video[t])
                    _save_rgb_tensor(seq_dir / "masked" / f"masked_frame{frame_id:04d}.png", masked[t])
                    _save_mask_tensor(seq_dir / "mask" / f"mask_frame{frame_id:04d}.png", mask[t])

    print(f"[preprocess] done. outputs -> {out_root}")


def _train_impl(args: argparse.Namespace, pretrained_path: Optional[str]) -> None:
    _set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = _detect_image_size(args.data_dir)

    print(f"[train] device={device}, image_size=({img_h},{img_w})")
    dataset = _build_dataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        ocean_mask_path=args.ocean_mask_path,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = _build_model(args, img_h=img_h, img_w=img_w).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        epochs=args.epochs,
        pretrained_path=pretrained_path,
    )

    out_model = Path(args.output_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)

    meta = {
        "stage": "finetune" if pretrained_path else "train",
        "data_dir": args.data_dir,
        "ocean_mask_path": args.ocean_mask_path,
        "mask_type": args.mask_type,
        "mask_ratio": args.mask_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": [img_h, img_w],
        "output_model": str(out_model),
        "pretrained_path": pretrained_path,
    }
    with open(out_model.with_suffix(out_model.suffix + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[train] saved model -> {out_model}")


def run_train(args: argparse.Namespace) -> None:
    _train_impl(args, pretrained_path=None)


def run_finetune(args: argparse.Namespace) -> None:
    if not args.pretrained_path:
        raise ValueError("finetune requires --pretrained-path")
    _train_impl(args, pretrained_path=args.pretrained_path)


def run_infer(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = _detect_image_size(args.data_dir)
    print(f"[infer] device={device}, image_size=({img_h},{img_w})")

    dataset = _build_dataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        ocean_mask_path=args.ocean_mask_path,
        mask_type=args.mask_type,
        mask_ratio=args.mask_ratio,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = _build_model(args, img_h=img_h, img_w=img_w).to(device)
    inference_with_pretrained(
        out_channels=args.out_channels,
        model_path=args.model_path,
        data_dir=args.data_dir,
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        input_seq_len=args.max_seq_len,
    )
    print("[infer] done. outputs are written by MAE_LaMa.inference_with_pretrained")


def run_analyze(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    img_h, img_w = _detect_image_size(args.data_dir)

    model = _build_model(args, img_h=img_h, img_w=img_w).to(device)
    if args.model_path and os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[analyze] loaded model weights: {args.model_path}")

    ratios = _parse_csv_int_list(args.ratios)
    mask_types = _parse_csv_str_list(args.mask_types)

    # Build dataloaders for experiment config.
    dataloaders = {}
    for ratio in ratios:
        ratio_float = ratio / 100.0
        ds = _build_dataset(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            ocean_mask_path=args.ocean_mask_path,
            mask_type="cloud",
            mask_ratio=ratio_float,
        )
        dataloaders[f"dataloader_{ratio}percent"] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    for mask_type in mask_types:
        ds = _build_dataset(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            ocean_mask_path=args.ocean_mask_path,
            mask_type=mask_type,
            mask_ratio=args.default_analyze_ratio,
        )
        dataloaders[f"dataloader_{mask_type}"] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    if "dataloader_thick_cloud" not in dataloaders:
        # Use cloud + default ratio as thick_cloud baseline.
        ds = _build_dataset(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
            ocean_mask_path=args.ocean_mask_path,
            mask_type="cloud",
            mask_ratio=args.default_analyze_ratio,
        )
        dataloaders["dataloader_thick_cloud"] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    missing_types_cfg = {}
    for key in ["thin_cloud", "thick_cloud", "strip", "mixed"]:
        dl_key = f"dataloader_{key}"
        if dl_key in dataloaders:
            missing_types_cfg[key] = dataloaders[dl_key]

    missing_ratios_cfg = {}
    for ratio in ratios:
        dl_key = f"dataloader_{ratio}percent"
        if dl_key in dataloaders:
            missing_ratios_cfg[ratio] = dataloaders[dl_key]

    experiment_config = {
        "missing_types": missing_types_cfg,
        "missing_ratios": missing_ratios_cfg,
    }

    print("[analyze] running experiments...")
    all_results = run_four_experiments(
        model=model,
        experiment_config=experiment_config,
        out_channels=args.out_channels,
        input_seq_len=args.max_seq_len,
    )

    if args.generate_figures:
        fig_paths = generate_all_visualizations(all_results)
        print(f"[analyze] figures generated: {fig_paths}")

    if args.generate_report:
        report_path = generate_comprehensive_report(all_results)
        print(f"[analyze] report generated: {report_path}")


def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-seq-len", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out-channels", type=int, default=4)
    parser.add_argument("--use-mask-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-lama-init", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ocean-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fine-tune-layers", type=str, default="decoder,mask_update_layer")


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=str, default="E:/lama/jet_S2_Daily_Mosaic/")
    parser.add_argument("--ocean-mask-path", type=str, default="E:/lama/S2_Daily_Mosaic_Masked/mask.png")
    parser.add_argument("--mask-type", type=str, default="cloud", choices=["random", "thin_cloud", "cloud", "strip", "mixed", "predefined"])
    parser.add_argument("--mask-ratio", type=float, default=0.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified VMAE pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu; default auto")

    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p_pre = sub.add_parser("preprocess", help="simulate missing data with different mask strategies")
    p_pre.add_argument("--data-dir", type=str, default="E:/lama/jet_S2_Daily_Mosaic/")
    p_pre.add_argument("--ocean-mask-path", type=str, default=None)
    p_pre.add_argument("--max-seq-len", type=int, default=8)
    p_pre.add_argument("--mask-types", type=str, default="thin_cloud,cloud,strip,mixed")
    p_pre.add_argument("--mask-ratios", type=str, default="0.1,0.3,0.5")
    p_pre.add_argument("--num-sequences", type=int, default=2)
    p_pre.add_argument("--output-dir", type=str, default="preprocessed_missing_data")
    p_pre.set_defaults(func=run_preprocess)

    # train
    p_train = sub.add_parser("train", help="train from scratch")
    _add_shared_data_args(p_train)
    _add_shared_model_args(p_train)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--output-model", type=str, default="checkpoints/vmae_trained.pth")
    p_train.set_defaults(func=run_train)

    # finetune
    p_ft = sub.add_parser("finetune", help="finetune from pretrained weights")
    _add_shared_data_args(p_ft)
    _add_shared_model_args(p_ft)
    p_ft.add_argument("--epochs", type=int, default=20)
    p_ft.add_argument("--batch-size", type=int, default=1)
    p_ft.add_argument("--num-workers", type=int, default=0)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--pretrained-path", type=str, required=True)
    p_ft.add_argument("--output-model", type=str, default="checkpoints/vmae_finetuned.pth")
    p_ft.set_defaults(func=run_finetune)

    # infer
    p_infer = sub.add_parser("infer", help="run inference")
    _add_shared_data_args(p_infer)
    _add_shared_model_args(p_infer)
    p_infer.add_argument("--model-path", type=str, default="fine_tuned_model.pth")
    p_infer.add_argument("--num-workers", type=int, default=0)
    p_infer.set_defaults(func=run_infer)

    # analyze
    p_an = sub.add_parser("analyze", help="run experiments and analysis")
    _add_shared_data_args(p_an)
    _add_shared_model_args(p_an)
    p_an.add_argument("--model-path", type=str, default="fine_tuned_model.pth")
    p_an.add_argument("--ratios", type=str, default="10,30,50,60")
    p_an.add_argument("--mask-types", type=str, default="thin_cloud,strip,mixed")
    p_an.add_argument("--default-analyze-ratio", type=float, default=0.3)
    p_an.add_argument("--num-workers", type=int, default=0)
    p_an.add_argument("--generate-figures", action=argparse.BooleanOptionalAction, default=True)
    p_an.add_argument("--generate-report", action=argparse.BooleanOptionalAction, default=True)
    p_an.set_defaults(func=run_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
