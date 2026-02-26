#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate segmentation config YAML for ISLES-2022 training.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--simclr-ckpt", type=Path, required=True)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("scripts/isles2022_segmentation.yml"),
        help="Path to write generated YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/isles2022_segmentation"),
        help="Directory where model checkpoints are saved during training.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory where training logs (e.g., wandb local logs) are saved.",
    )
    parser.add_argument("--gpu-device", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help='Lightning precision, e.g. "32", "16-mixed", "bf16-mixed".',
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch.set_float32_matmul_precision value.",
    )
    parser.add_argument(
        "--freeze-backbone",
        choices=["yes", "no"],
        default="yes",
        help="Freeze pretrained ViT backbone during segmentation finetuning.",
    )
    parser.add_argument("--run-name", type=str, default="isles2022_segmentation")
    parser.add_argument("--project-name", type=str, default="brainiac_isles2022_segmentation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_csv = args.train_csv.expanduser().resolve()
    val_csv = args.val_csv.expanduser().resolve()
    simclr_ckpt = args.simclr_ckpt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    output_config = args.output_config.expanduser().resolve()

    if not train_csv.exists():
        raise FileNotFoundError(f"train CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val CSV not found: {val_csv}")
    if not simclr_ckpt.exists():
        raise FileNotFoundError(f"SimCLR checkpoint not found: {simclr_ckpt}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        "data": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
        },
        "model": {
            "img_size": [96, 96, 96],
            "in_channels": 1,
            "out_channels": 1,
        },
        "pretrain": {
            "simclr_checkpoint_path": str(simclr_ckpt),
        },
        "training": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "sw_batch_size": args.sw_batch_size,
            "precision": args.precision,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "gradient_clip_val": args.gradient_clip_val,
            "matmul_precision": args.matmul_precision,
            "freeze": args.freeze_backbone,
        },
        "output": {
            "output_dir": str(output_dir),
        },
        "logger": {
            "save_dir": str(log_dir),
            "save_name": "isles_segmentation-{epoch:02d}-{val_dice:.4f}",
            "run_name": args.run_name,
            "project_name": args.project_name,
        },
        "gpu": {
            "visible_device": args.gpu_device,
        },
    }

    with output_config.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote config: {output_config}")
    print(f"output_dir:   {output_dir}")
    print(f"log_dir:      {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
