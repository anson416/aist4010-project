# -*- coding: utf-8 -*-
# File: pytorch_pipeline.py

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import floating
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class PyTorchPipeline(object):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Union[LRScheduler, ReduceLROnPlateau]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._output_dir: Optional[Path] = None
        self._best_loss: Optional[float] = None
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._learning_rates: list[float] = []

    def start(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        save_full: bool = False,
    ) -> str:
        if val_dataloader is None and self._scheduler is not None:
            assert not isinstance(self._scheduler, ReduceLROnPlateau)

        if self._output_dir is None:
            self._output_dir = Path("train") / self.get_datetime()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}:", flush=True)

            # Retrieve learning rate for current epoch
            lr = self._optimizer.param_groups[0]["lr"]
            self._learning_rates.append(lr)

            # Compute losses for training and validation sets
            train_loss = self.train(train_dataloader)
            self._train_losses.append(train_loss)
            val_loss: Optional[float] = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self._val_losses.append(val_loss)

            # Update learning rate
            if self._scheduler is not None:
                self._scheduler.step(val_loss if isinstance(self._scheduler, ReduceLROnPlateau) else None)

            # Save current and best-performing (lowest loss) models
            self.save_checkpoint("checkpoint_last", save_full=save_full)
            if val_loss is not None and (self._best_loss is None or val_loss < self._best_loss):
                self._best_loss = val_loss
                self.save_checkpoint("checkpoint_best", save_full=save_full)

            # Plot training curves (losses and learning rates)
            self.plot()

            # Display epoch information
            print(
                f"Epoch {epoch}/{epochs} -",
                f"train_loss: {train_loss:.3f},",
                f"{f'val_loss: {val_loss:.3f},' if val_loss is not None else ''}",
                f"lr: {lr},",
                f"ETR: {self.get_etr(epoch, epochs, time.time() - start_time)}\n",
                flush=True,
            )

        return str(self._output_dir)

    def train(self, dataloader: DataLoader) -> floating:
        self._model.train()

        losses = []
        for X, y in tqdm(dataloader, desc="Training"):
            # Load a batch of data
            X, y = X.to(self._device), y.to(self._device)

            # Feedforward
            pred = self._model(X)

            # Loss
            loss = self._criterion(pred, y)
            losses.append(loss.item())

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return np.mean(losses)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> floating:
        self._model.eval()

        losses = []
        for X, y in tqdm(dataloader, desc="Validating"):
            # Load a batch of data
            X, y = X.to(self._device), y.to(self._device)

            # Feedforward
            pred = self._model(X)

            # Loss
            loss = self._criterion(pred, y)
            losses.append(loss.item())

        return np.mean(losses)

    def save_checkpoint(
        self,
        name: str = "checkpoint",
        save_full: bool = False,
    ) -> None:
        checkpoint = {"model_state_dict": self._model.state_dict()}
        if save_full:
            checkpoint.update(
                {
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "train_losses": self._train_losses,
                    "val_losses": self._val_losses,
                    "learning_rates": self._learning_rates,
                }
            )
        torch.save(checkpoint, self._output_dir / f"{name}.pt")

    def plot(self) -> None:
        x = list(range(1, len(self._train_losses) + 1))
        for curve in ("Loss", "Learning Rate"):
            plt.figure()

            if curve == "Loss":
                plt.plot(x, self._train_losses, "b-", label="Train")
                if len(self._val_losses) > 0:
                    plt.plot(x, self._val_losses, "r-", label="Val")
                plt.legend()
            else:
                plt.plot(x, self._learning_rates, "g-")

            plt.title(f"{curve} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(curve)
            plt.tight_layout()
            plt.savefig(self._output_dir / f"{curve.lower().replace(' ', '_')}.png", dpi=300)
            plt.close()

    @staticmethod
    def get_etr(progress: int, total: int, time_elapsed: float) -> str:
        assert total > 0, f"{total} > 0. `total` must be a positive integer."
        assert (
            0 < progress <= total
        ), f"0 < {progress} <= {total}. `progress` must be a positive integer not greater than `total`."

        etr = time_elapsed * ((total / progress) - 1)
        d = int(etr // 86400)
        etr -= d * 86400
        h = int(etr // 3600)
        etr -= h * 3600
        m = int(etr // 60)
        etr -= m * 60
        s = round(etr)
        return f"{f'{d}:' if d > 0 else ''}{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def get_datetime(sep: str = "-") -> str:
        return datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")
