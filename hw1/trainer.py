"""Training loop utilities for HW1 piano transcription."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_zoo import allocate_batch
from evaluate import evaluate_model, unpack_predictions


def cycle(iterable: Iterable):
    while True:
        for item in iterable:
            yield item


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        steps_per_epoch: int = 1000,
        total_steps: int | None = None,
        min_lr: float = 0.0,
        offset_loss_weight: float = 1.0,
        grad_clip: float = 3.0,
        progress_position: int = 0,
        progress_desc: str = "Train",
        device: str | None = None,
    ):
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda":
            raise ValueError("Runner requires a CUDA device. CPU execution is disabled for this project.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required, but PyTorch cannot access one. "
                "Check that the NVIDIA driver is running and that a CUDA-enabled PyTorch build is installed."
            )
        if self.device.index is not None and self.device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {self.device}, but PyTorch only sees {torch.cuda.device_count()} CUDA device(s)."
            )

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps or steps_per_epoch, eta_min=min_lr)
        self.criterion = nn.BCELoss().to(self.device)
        self.steps_per_epoch = steps_per_epoch
        self.offset_loss_weight = offset_loss_weight
        self.grad_clip = grad_clip
        self.progress_position = progress_position
        self.progress_desc = progress_desc

    def train_epoch(
        self,
        dataloader,
        epoch: int | None = None,
        step_logger: Callable[[int, Dict[str, float]], None] | None = None,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        loop = tqdm(
            range(1, self.steps_per_epoch + 1),
            desc=self.progress_desc,
            dynamic_ncols=True,
            leave=False,
            position=self.progress_position,
        )

        for batch_step, batch in zip(loop, cycle(dataloader)):
            self.optimizer.zero_grad()
            batch = allocate_batch(batch, self.device)

            model_output = self.model(batch["audio"])
            frame_pred, onset_pred, offset_pred = unpack_predictions(model_output)
            frame_loss = self.criterion(frame_pred, batch["frame"])
            onset_loss = self.criterion(onset_pred, batch["onset"])
            loss = onset_loss + frame_loss
            if offset_pred is not None and "offset" in batch:
                offset_loss = self.criterion(offset_pred, batch["offset"])
                loss = loss + self.offset_loss_weight * offset_loss

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            if step_logger is not None:
                global_step = ((epoch - 1) * self.steps_per_epoch + batch_step) if epoch is not None else batch_step
                step_metrics = {
                    "train/step_loss": loss_value,
                    "train/frame_loss": float(frame_loss.detach().item()),
                    "train/onset_loss": float(onset_loss.detach().item()),
                    "train/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                }
                if offset_pred is not None and "offset" in batch:
                    step_metrics["train/offset_loss"] = float(offset_loss.detach().item())
                step_logger(
                    global_step,
                    step_metrics,
                )

        return running_loss / self.steps_per_epoch

    def validate(self, dataloader) -> Tuple[float, Dict[str, float]]:
        avg_metrics = evaluate_model(self.model, dataloader, self.device, desc="Valid")
        valid_loss = float(avg_metrics.get("metric/loss/frame_loss", 0.0) + avg_metrics.get("metric/loss/onset_loss", 0.0))
        if "metric/loss/offset_loss" in avg_metrics:
            valid_loss += self.offset_loss_weight * float(avg_metrics["metric/loss/offset_loss"])
        return valid_loss, avg_metrics
