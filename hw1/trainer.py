"""Training loop utilities for HW1 piano transcription."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data_zoo import allocate_batch
from evaluate import evaluate_model


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
        grad_clip: float = 3.0,
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.98)
        self.criterion = nn.BCELoss().to(self.device)
        self.steps_per_epoch = steps_per_epoch
        self.grad_clip = grad_clip

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        running_loss = 0.0
        loop = tqdm(range(1, self.steps_per_epoch + 1), desc="Train", leave=False)

        for _, batch in zip(loop, cycle(dataloader)):
            self.optimizer.zero_grad()
            batch = allocate_batch(batch, self.device)

            frame_pred, onset_pred = self.model(batch["audio"])
            frame_loss = self.criterion(frame_pred, batch["frame"])
            onset_loss = self.criterion(onset_pred, batch["onset"])
            loss = onset_loss + frame_loss

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            loss_value = float(loss.detach().item())
            running_loss += loss_value
            loop.set_postfix_str(f"loss: {loss_value:.3e}")

        return running_loss / self.steps_per_epoch

    def validate(self, dataloader) -> Tuple[float, Dict[str, float]]:
        avg_metrics = evaluate_model(self.model, dataloader, self.device, desc="Valid")
        valid_loss = float(
            avg_metrics.get("metric/loss/frame_loss", 0.0) + avg_metrics.get("metric/loss/onset_loss", 0.0)
        )
        return valid_loss, avg_metrics
