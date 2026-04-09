"""Hydra-based training entrypoint for HW1 piano transcription."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import hydra
from config import HOP_SIZE
from data_zoo import MAESTRO_small
from evaluate import print_f1_metrics
from model_zoo import BasicOnsetsAndFrames, OffsetConditionedOnsetsAndFrames, OnsetsAndFrames
from trainer import Runner


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
    if model_name == "onsets-and-frames":
        return OnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    if model_name == "offset-conditioned-onsets-and-frames":
        return OffsetConditionedOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_experiment_config(cfg: DictConfig, run_dir: Path, device: torch.device) -> Dict[str, Any]:
    return {
        "experiment_name": cfg.experiment.name,
        "save_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg.seed),
        "device": {
            "requested": str(cfg.device),
            "used": str(device),
        },
        "data": {
            "path": str(cfg.data.path),
            "train_split": str(cfg.data.train_split),
            "valid_split": str(cfg.data.valid_split),
            "sequence_length": int(cfg.data.sequence_length),
            "batch_size": int(cfg.data.batch_size),
            "num_workers": int(cfg.data.num_workers),
        },
        "model": {
            "name": str(cfg.model.name),
            "cnn_unit": int(cfg.model.cnn_unit),
            "fc_unit": int(cfg.model.fc_unit),
            "rnn_unit": int(cfg.model.rnn_unit),
        },
        "optimization": {
            "epochs": int(cfg.optimization.epochs),
            "steps_per_epoch": int(cfg.optimization.steps_per_epoch),
            "learning_rate": float(cfg.optimization.learning_rate),
            "weight_decay": float(cfg.optimization.weight_decay),
            "offset_loss_weight": float(cfg.optimization.offset_loss_weight),
        },
        "wandb": {
            "enabled": bool(cfg.wandb.enabled),
            "project": str(cfg.wandb.project),
            "entity": cfg.wandb.entity,
            "run_name": cfg.wandb.run_name,
            "group": cfg.wandb.group,
            "mode": str(cfg.wandb.mode),
            "tags": list(cfg.wandb.tags),
        },
    }


def init_wandb(cfg: DictConfig, run_dir: Path, experiment_config: Dict[str, Any]):
    if not cfg.wandb.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but `wandb` is not installed. Install with `pip install wandb`."
        ) from exc

    run_name = cfg.wandb.run_name or cfg.experiment.name
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode.name == "MULTIRUN" and cfg.wandb.append_job_num and cfg.wandb.run_name is None:
        run_name = f"{run_name}_{hydra_cfg.job.num}"

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        config=experiment_config,
        dir=str(run_dir),
    )
    return wandb


def save_checkpoint(
    path: Path,
    runner: Runner,
    cfg: DictConfig,
    epoch: int,
    valid_loss: float,
    metrics: Dict[str, float],
    experiment_config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_name": cfg.model.name,
            "model_hparams": {
                "cnn_unit": int(cfg.model.cnn_unit),
                "fc_unit": int(cfg.model.fc_unit),
                "rnn_unit": int(cfg.model.rnn_unit),
            },
            "optimizer_hparams": {
                "learning_rate": float(cfg.optimization.learning_rate),
                "weight_decay": float(cfg.optimization.weight_decay),
                "offset_loss_weight": float(cfg.optimization.offset_loss_weight),
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


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    run_dir = Path.cwd()
    set_seed(int(cfg.seed))
    device = get_device(str(cfg.device))

    print(f"Run directory: {run_dir}")
    print(f"Using model: {cfg.model.name}")
    print(f"Using device: {device}")

    train_dataset = MAESTRO_small(
        path=str(cfg.data.path),
        groups=[str(cfg.data.train_split)],
        sequence_length=int(cfg.data.sequence_length),
        hop_size=HOP_SIZE,
        seed=int(cfg.seed),
        random_sample=True,
    )
    valid_dataset = MAESTRO_small(
        path=str(cfg.data.path),
        groups=[str(cfg.data.valid_split)],
        sequence_length=int(cfg.data.sequence_length),
        hop_size=HOP_SIZE,
        seed=int(cfg.seed),
        random_sample=False,
    )

    pin_memory = torch.cuda.is_available() and str(cfg.device) != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=pin_memory,
    )

    model = build_model_from_config(
        model_name=str(cfg.model.name),
        cnn_unit=int(cfg.model.cnn_unit),
        fc_unit=int(cfg.model.fc_unit),
        rnn_unit=int(cfg.model.rnn_unit),
    )
    runner = Runner(
        model=model,
        lr=float(cfg.optimization.learning_rate),
        weight_decay=float(cfg.optimization.weight_decay),
        steps_per_epoch=int(cfg.optimization.steps_per_epoch),
        offset_loss_weight=float(cfg.optimization.offset_loss_weight),
        device=str(device),
    )

    experiment_config = build_experiment_config(cfg, run_dir, runner.device)
    save_json(run_dir / "config.json", experiment_config)
    save_json(
        run_dir / "config_resolved.json",
        OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    wandb_module = init_wandb(cfg, run_dir, experiment_config)

    best_valid_loss = float("inf")
    history: list[Dict[str, Any]] = []

    for epoch in range(1, int(cfg.optimization.epochs) + 1):
        print(f"\n[Epoch {epoch}/{int(cfg.optimization.epochs)}]")
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

        if cfg.save.save_every_epoch:
            save_checkpoint(
                run_dir / f"epoch_{epoch:03d}.pt",
                runner,
                cfg,
                epoch,
                valid_loss,
                metrics,
                experiment_config,
            )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                run_dir / "best_model.pt",
                runner,
                cfg,
                epoch,
                valid_loss,
                metrics,
                experiment_config,
            )
            print(f"Saved new best model (valid_loss={valid_loss:.6f})")

        save_json(run_dir / "history.json", {"epochs": history})

    print(f"\nTraining done. Best valid loss: {best_valid_loss:.6f}")
    print(f"Saved outputs to: {run_dir}")

    if wandb_module is not None:
        if wandb_module.run is not None:
            wandb_module.run.summary["best_valid_loss"] = best_valid_loss
        wandb_module.finish()


if __name__ == "__main__":
    main()
