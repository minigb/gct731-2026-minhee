"""CLI for training HW1 piano transcription models."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CNN_UNIT,
    DEFAULT_DATASET_PATH,
    DEFAULT_FC_UNIT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RNN_UNIT,
    DEFAULT_SAVE_DIR,
    DEFAULT_SEED,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STEPS_PER_EPOCH,
    DEFAULT_WEIGHT_DECAY,
    HOP_SIZE,
)
from data_zoo import MAESTRO_small
from evaluate import print_f1_metrics
from model_zoo import BasicOnsetsAndFrames, OnsetsAndFrames
from trainer import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HW1 piano transcription model from the command line.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, choices=["basic", "onsets-and-frames"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--cnn-unit", type=int, default=DEFAULT_CNN_UNIT)
    parser.add_argument("--fc-unit", type=int, default=DEFAULT_FC_UNIT)
    parser.add_argument("--rnn-unit", type=int, default=DEFAULT_RNN_UNIT)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="validation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gct731-hw1")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_arg)


def build_model_from_config(model_name: str, cnn_unit: int, fc_unit: int, rnn_unit: int) -> torch.nn.Module:
    if model_name == "basic":
        return BasicOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit)
    return OnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    return build_model_from_config(
        model_name=args.model,
        cnn_unit=args.cnn_unit,
        fc_unit=args.fc_unit,
        rnn_unit=args.rnn_unit,
    )


def make_experiment_dir(args: argparse.Namespace) -> Path:
    if args.experiment_name:
        name = args.experiment_name
    else:
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = f"{args.model}_{now}"
    exp_dir = Path(args.save_dir) / name
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def build_experiment_config(args: argparse.Namespace, exp_dir: Path, device: torch.device) -> Dict[str, Any]:
    return {
        "experiment_name": exp_dir.name,
        "save_dir": str(exp_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": {
            "requested": args.device,
            "used": str(device),
        },
        "data": {
            "path": args.data_path,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "model": {
            "name": args.model,
            "cnn_unit": args.cnn_unit,
            "fc_unit": args.fc_unit,
            "rnn_unit": args.rnn_unit,
        },
        "optimization": {
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "wandb": {
            "enabled": args.use_wandb,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
            "group": args.wandb_group,
            "mode": args.wandb_mode,
            "tags": args.wandb_tags,
        },
    }


def init_wandb(args: argparse.Namespace, exp_dir: Path, experiment_config: Dict[str, Any]):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but `wandb` is not installed. Install with `pip install wandb`."
        ) from exc

    run_name = args.wandb_run_name or args.experiment_name or exp_dir.name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=experiment_config,
        dir=str(exp_dir),
    )
    return wandb


def save_checkpoint(
    path: Path,
    runner: Runner,
    args: argparse.Namespace,
    epoch: int,
    valid_loss: float,
    metrics: Dict[str, float],
    experiment_config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model,
            "model_hparams": {
                "cnn_unit": args.cnn_unit,
                "fc_unit": args.fc_unit,
                "rnn_unit": args.rnn_unit,
            },
            "optimizer_hparams": {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "model_state_dict": runner.model.state_dict(),
            "optimizer_state_dict": runner.optimizer.state_dict(),
            "scheduler_state_dict": runner.scheduler.state_dict(),
            "valid_loss": valid_loss,
            "metrics": metrics,
            "config": experiment_config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    exp_dir = make_experiment_dir(args)
    device = get_device(args.device)
    print(f"Experiment directory: {exp_dir}")
    print(f"Using model: {args.model}")
    print(f"Using device: {device}")

    train_dataset = MAESTRO_small(
        path=args.data_path,
        groups=[args.train_split],
        sequence_length=args.sequence_length,
        hop_size=HOP_SIZE,
        seed=args.seed,
        random_sample=True,
    )
    valid_dataset = MAESTRO_small(
        path=args.data_path,
        groups=[args.valid_split],
        sequence_length=args.sequence_length,
        hop_size=HOP_SIZE,
        seed=args.seed,
        random_sample=False,
    )

    pin_memory = torch.cuda.is_available() and (args.device != "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(args)
    runner = Runner(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        steps_per_epoch=args.steps_per_epoch,
        device=str(device),
    )

    experiment_config = build_experiment_config(args, exp_dir, runner.device)
    save_json(exp_dir / "args.json", vars(args))
    save_json(exp_dir / "config.json", experiment_config)
    wandb_module = init_wandb(args, exp_dir, experiment_config)

    best_valid_loss = float("inf")
    history: list[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        train_loss = runner.train_epoch(train_loader)
        valid_loss, metrics = runner.validate(valid_loader)
        current_lr = float(runner.optimizer.param_groups[0]["lr"])

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "learning_rate": current_lr,
        }
        row.update(metrics)
        history.append(row)

        print(f"train_loss: {train_loss:.6f}")
        print(f"valid_loss: {valid_loss:.6f}")
        print(f"learning_rate: {current_lr:.6e}")
        print_f1_metrics(metrics)

        if wandb_module is not None:
            wandb_module.log(row, step=epoch)

        if args.save_every_epoch:
            save_checkpoint(
                exp_dir / f"epoch_{epoch:03d}.pt",
                runner,
                args,
                epoch,
                valid_loss,
                metrics,
                experiment_config,
            )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                exp_dir / "best_model.pt",
                runner,
                args,
                epoch,
                valid_loss,
                metrics,
                experiment_config,
            )
            print(f"Saved new best model (valid_loss={valid_loss:.6f})")

        save_json(exp_dir / "history.json", {"epochs": history})

    print(f"\nTraining done. Best valid loss: {best_valid_loss:.6f}")
    print(f"Saved outputs to: {exp_dir}")

    if wandb_module is not None:
        if wandb_module.run is not None:
            wandb_module.run.summary["best_valid_loss"] = best_valid_loss
        wandb_module.finish()


if __name__ == "__main__":
    main()
