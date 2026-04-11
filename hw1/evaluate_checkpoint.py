"""CLI to load a trained checkpoint and evaluate it on a dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CNN_UNIT,
    DEFAULT_DATASET_PATH,
    DEFAULT_FC_UNIT,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RNN_UNIT,
    DEFAULT_SEQUENCE_LENGTH,
    HOP_SIZE,
)
from data_zoo import MAESTRO_small
from evaluate import evaluate_model, print_f1_metrics
from model_zoo import BasicOnsetsAndFrames, OffsetConditionedOnsetsAndFrames, OnsetsAndFrames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HW1 checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--data-path", type=str, default=None, help="Override dataset root path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test", "debug"])
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save evaluation metrics JSON.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gct731-hw1")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser.parse_args()


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def get_device(device_arg: str) -> torch.device:
    normalized_device_arg = "cuda" if device_arg == "auto" else device_arg
    device = torch.device(normalized_device_arg)
    if device.type != "cuda":
        raise ValueError(
            f"Unsupported device '{device_arg}'. Evaluation requires CUDA; "
            "use --device cuda or --device cuda:<index>."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but PyTorch cannot access one. "
            "Check that the NVIDIA driver is running and that a CUDA-enabled PyTorch build is installed."
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but PyTorch only sees {torch.cuda.device_count()} CUDA device(s)."
        )
    return device


def build_model_from_config(model_name: str, cnn_unit: int, fc_unit: int, rnn_unit: int) -> torch.nn.Module:
    if model_name == "basic":
        return BasicOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit)
    if model_name == "onsets-and-frames":
        return OnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    if model_name == "offset-conditioned-onsets-and-frames":
        return OffsetConditionedOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    raise ValueError(f"Unsupported model name: {model_name}")


def init_wandb(args: argparse.Namespace, config: Dict[str, Any]):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but `wandb` is not installed. Install with `pip install wandb`."
        ) from exc

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=config,
    )
    return wandb


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    checkpoint_config = checkpoint.get("config", {})
    if not checkpoint_config:
        checkpoint_config = load_json_if_exists(checkpoint_path.parent / "config.json")
    if not checkpoint_config:
        checkpoint_config = load_json_if_exists(checkpoint_path.parent / "args.json")
    model_hparams = checkpoint.get("model_hparams", {})
    cfg_model = checkpoint_config.get("model", {}) if isinstance(checkpoint_config.get("model", {}), dict) else {}
    cfg_data = checkpoint_config.get("data", {}) if isinstance(checkpoint_config.get("data", {}), dict) else {}

    model_name = checkpoint.get(
        "model_name",
        cfg_model.get("name", checkpoint_config.get("model", DEFAULT_MODEL_NAME)),
    )
    cnn_unit = model_hparams.get(
        "cnn_unit",
        cfg_model.get("cnn_unit", checkpoint_config.get("cnn_unit", DEFAULT_CNN_UNIT)),
    )
    fc_unit = model_hparams.get(
        "fc_unit",
        cfg_model.get("fc_unit", checkpoint_config.get("fc_unit", DEFAULT_FC_UNIT)),
    )
    rnn_unit = model_hparams.get(
        "rnn_unit",
        cfg_model.get("rnn_unit", checkpoint_config.get("rnn_unit", DEFAULT_RNN_UNIT)),
    )

    data_path = (
        args.data_path
        or cfg_data.get("path")
        or checkpoint_config.get("data_path")
        or DEFAULT_DATASET_PATH
    )
    sequence_length = (
        args.sequence_length
        if args.sequence_length is not None
        else cfg_data.get("sequence_length", checkpoint_config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    )

    device = get_device(args.device)
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Using model: {model_name}")
    print(f"Using split: {args.split}")
    print(f"Using dataset path: {data_path}")
    print(f"Using device: {device}")

    model = build_model_from_config(model_name, cnn_unit, fc_unit, rnn_unit).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = MAESTRO_small(
        path=data_path,
        groups=[args.split],
        sequence_length=sequence_length,
        hop_size=HOP_SIZE,
        random_sample=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    metrics = evaluate_model(model, dataloader, device, desc=f"Eval-{args.split}")
    eval_loss = float(metrics.get("metric/loss/frame_loss", 0.0) + metrics.get("metric/loss/onset_loss", 0.0))

    print(f"\neval_loss: {eval_loss:.6f}")
    print_f1_metrics(metrics)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "data_path": data_path,
        "sequence_length": sequence_length,
        "device": str(device),
        "eval_loss": eval_loss,
        "metrics": metrics,
    }

    output_json = Path(args.output_json) if args.output_json else checkpoint_path.parent / f"eval_{args.split}.json"
    save_json(output_json, result)
    print(f"Saved evaluation metrics to: {output_json}")

    wandb_module = init_wandb(args, result)
    if wandb_module is not None:
        wandb_module.log({"eval_loss": eval_loss, **metrics})
        wandb_module.finish()


if __name__ == "__main__":
    main()
