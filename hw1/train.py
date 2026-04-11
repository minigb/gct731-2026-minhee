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


def parse_gpu_ids(gpu_ids_arg: Any) -> list[int] | None:
    if gpu_ids_arg is None:
        return None
    if isinstance(gpu_ids_arg, str):
        value = gpu_ids_arg.strip()
        if not value or value.lower() in {"none", "null"}:
            return None
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(gpu_id) for gpu_id in gpu_ids_arg]


def get_hydra_job_num(hydra_cfg: DictConfig) -> int | None:
    job_num = OmegaConf.select(hydra_cfg, "job.num")
    return None if job_num is None else int(job_num)


def validate_cuda_device(device: torch.device, available_gpu_count: int) -> torch.device:
    if device.type != "cuda":
        raise ValueError(
            f"Unsupported device '{device}'. This training script is configured to require CUDA; "
            "use device=cuda or device=cuda:<index> and fix the GPU/driver setup if CUDA is unavailable."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but PyTorch cannot access one. "
            "Check that the NVIDIA driver is running and that a CUDA-enabled PyTorch build is installed."
        )
    if device.index is not None and device.index >= available_gpu_count:
        raise RuntimeError(
            f"Requested {device}, but PyTorch only sees {available_gpu_count} CUDA device(s)."
        )
    return device


def get_device(device_arg: str, gpu_ids_arg: Any = None, hydra_job_num: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but PyTorch cannot access one. "
            "Check that the NVIDIA driver is running and that a CUDA-enabled PyTorch build is installed."
        )

    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count == 0:
        raise RuntimeError("CUDA GPU is required, but PyTorch reports 0 CUDA devices.")

    normalized_device_arg = "cuda" if device_arg == "auto" else device_arg
    requested_device = torch.device(normalized_device_arg)
    if requested_device.type != "cuda":
        raise ValueError(
            f"Unsupported device '{device_arg}'. This training script is configured to require CUDA; "
            "use device=cuda or device=cuda:<index>."
        )
    if requested_device.index is not None:
        return validate_cuda_device(requested_device, available_gpu_count)

    gpu_ids = parse_gpu_ids(gpu_ids_arg) or list(range(available_gpu_count))
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id when provided.")
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpu_count]
    if invalid_gpu_ids:
        raise RuntimeError(
            f"gpu_ids contains unavailable GPU id(s) {invalid_gpu_ids}; "
            f"PyTorch only sees {available_gpu_count} CUDA device(s)."
        )

    selected_gpu_id = gpu_ids[(hydra_job_num or 0) % len(gpu_ids)]
    return torch.device(f"cuda:{selected_gpu_id}")


def build_model_from_config(model_name: str, cnn_unit: int, fc_unit: int, rnn_unit: int) -> torch.nn.Module:
    if model_name == "basic":
        return BasicOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit)
    if model_name == "onsets-and-frames":
        return OnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    if model_name == "offset-conditioned-onsets-and-frames":
        return OffsetConditionedOnsetsAndFrames(cnn_unit=cnn_unit, fc_unit=fc_unit, rnn_unit=rnn_unit)
    raise ValueError(f"Unsupported model name: {model_name}")


def build_experiment_config(
    cfg: DictConfig,
    run_dir: Path,
    device: torch.device,
    resolved_data_path: Path,
) -> Dict[str, Any]:
    return {
        "experiment_name": cfg.experiment.name,
        "save_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(cfg.seed),
        "device": {
            "requested": str(cfg.device),
            "used": str(device),
            "gpu_ids": parse_gpu_ids(cfg.gpu_ids),
        },
        "data": {
            "path": str(resolved_data_path),
            "path_input": str(cfg.data.path),
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
        "scheduler": {
            "name": str(cfg.scheduler.name),
            "min_learning_rate": float(cfg.scheduler.get("min_learning_rate", 0.0)),
            "step_size": int(cfg.scheduler.get("step_size", 1000)),
            "gamma": float(cfg.scheduler.get("gamma", 0.98)),
        },
        "wandb_usage": {
            "enabled": bool(cfg.wandb_usage.enabled),
            "project": str(cfg.wandb_usage.project),
            "entity": cfg.wandb_usage.entity,
            "run_name": cfg.wandb_usage.run_name,
            "group": cfg.wandb_usage.group,
            "mode": str(cfg.wandb_usage.mode),
            "tags": list(cfg.wandb_usage.tags),
        },
    }


def resolve_data_path(data_path: str, launch_dir: Path) -> Path:
    """Resolves dataset path robustly even when Hydra changes cwd per run."""
    candidate = Path(data_path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (launch_dir / candidate).resolve()


def init_wandb(cfg: DictConfig, run_dir: Path, experiment_config: Dict[str, Any]):
    if not cfg.wandb_usage.enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging requested but `wandb` is not installed. Install with `pip install wandb`."
        ) from exc

    run_name = cfg.wandb_usage.run_name or cfg.experiment.name
    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode.name == "MULTIRUN" and cfg.wandb_usage.append_job_num and cfg.wandb_usage.run_name is None:
        run_name = f"{run_name}_{hydra_cfg.job.num}"

    wandb.init(
        project=cfg.wandb_usage.project,
        entity=cfg.wandb_usage.entity,
        name=run_name,
        group=cfg.wandb_usage.group,
        mode=cfg.wandb_usage.mode,
        tags=list(cfg.wandb_usage.tags),
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
            "scheduler_hparams": {
                "name": str(cfg.scheduler.name),
                "min_learning_rate": float(cfg.scheduler.get("min_learning_rate", 0.0)),
                "step_size": int(cfg.scheduler.get("step_size", 1000)),
                "gamma": float(cfg.scheduler.get("gamma", 0.98)),
            },
            "model_state_dict": runner.model.state_dict(),
            "optimizer_state_dict": runner.optimizer.state_dict(),
            "scheduler_state_dict": runner.scheduler.state_dict() if runner.scheduler is not None else None,
            "valid_loss": valid_loss,
            "metrics": metrics,
            "config": experiment_config,
        },
        path,
    )


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    run_dir = Path(hydra_cfg.runtime.output_dir)
    launch_dir = Path(hydra_cfg.runtime.cwd)
    set_seed(int(cfg.seed))
    job_num = get_hydra_job_num(hydra_cfg)
    device = get_device(str(cfg.device), cfg.gpu_ids, job_num)
    torch.cuda.set_device(device)
    data_path = resolve_data_path(str(cfg.data.path), launch_dir=launch_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Launch directory: {launch_dir}")
    print(f"Using model: {cfg.model.name}")
    print(f"Using device: {device}")
    print(f"Hydra job number: {job_num}")
    print(f"Resolved dataset path: {data_path}")

    train_dataset = MAESTRO_small(
        path=str(data_path),
        groups=[str(cfg.data.train_split)],
        sequence_length=int(cfg.data.sequence_length),
        hop_size=HOP_SIZE,
        seed=int(cfg.seed),
        random_sample=True,
    )
    valid_dataset = MAESTRO_small(
        path=str(data_path),
        groups=[str(cfg.data.valid_split)],
        sequence_length=int(cfg.data.sequence_length),
        hop_size=HOP_SIZE,
        seed=int(cfg.seed),
        random_sample=False,
    )

    pin_memory = device.type == "cuda"
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
        total_steps=int(cfg.optimization.epochs) * int(cfg.optimization.steps_per_epoch),
        scheduler_name=str(cfg.scheduler.name),
        scheduler_min_lr=float(cfg.scheduler.get("min_learning_rate", 0.0)),
        scheduler_step_size=int(cfg.scheduler.get("step_size", 1000)),
        scheduler_gamma=float(cfg.scheduler.get("gamma", 0.98)),
        offset_loss_weight=float(cfg.optimization.offset_loss_weight),
        progress_position=job_num or 0,
        progress_desc=f"Train job {job_num} ({device})" if job_num is not None else f"Train ({device})",
        device=str(device),
    )

    experiment_config = build_experiment_config(cfg, run_dir, runner.device, data_path)
    save_json(run_dir / "config.json", experiment_config)
    save_json(
        run_dir / "config_resolved.json",
        OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    wandb_module = init_wandb(cfg, run_dir, experiment_config)

    best_valid_loss = float("inf")
    history: list[Dict[str, Any]] = []
    step_logger = None
    if wandb_module is not None:
        step_logger = lambda step, metrics: wandb_module.log(metrics, step=step)

    for epoch in range(1, int(cfg.optimization.epochs) + 1):
        print(f"\n[Epoch {epoch}/{int(cfg.optimization.epochs)}]")
        train_loss = runner.train_epoch(train_loader, epoch=epoch, step_logger=step_logger)
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
            epoch_step = epoch * int(cfg.optimization.steps_per_epoch)
            wandb_module.log(row, step=epoch_step)

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
