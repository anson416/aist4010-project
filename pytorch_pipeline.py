# -*- coding: utf-8 -*-
# File: pytorch_pipeline.py

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        self.__model = model
        self.__criterion = criterion
        self.__optimizer = optimizer
        self.__scheduler = scheduler
        self.__device = device
        self.__output_dir: Optional[Path] = None
        self.__best_loss: Optional[float] = None
        self.__train_losses: list[float] = []
        self.__val_losses: list[float] = []
        self.__learning_rates: list[float] = []

    def train(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> None:
        if val_dataloader is None and self.__scheduler is not None:
            assert not isinstance(self.__scheduler, ReduceLROnPlateau)

        if self.__output_dir is None:
            self.__output_dir = Path("train") / self.get_datetime()
        self.__output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}:", flush=True)

            # Retrieve learning rate for current epoch
            lr = self.__optimizer.param_groups[0]["lr"]
            self.__learning_rates.append(lr)

            # Compute losses for training and validation sets
            train_loss = self.__train(train_dataloader)
            self.__train_losses.append(train_loss)
            val_loss: Optional[float] = None
            if val_dataloader is not None:
                val_loss = self.__validate(val_dataloader)
                self.__val_losses.append(val_loss)

            # Update learning rate
            if self.__scheduler is not None:
                self.__scheduler.step(
                    val_loss
                    if isinstance(self.__scheduler, ReduceLROnPlateau)
                    else None
                )

            # Save current and best-performing (lowest loss) models
            self.save_model("checkpoint_last")
            if val_loss is not None and (
                self.__best_loss is None or val_loss < self.__best_loss
            ):
                self.__best_loss = val_loss
                self.save_model("checkpoint_best")

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

    def __train(self, dataloader: DataLoader) -> float:
        self.__model.train()

        losses = []
        for X, y in tqdm(dataloader, desc="Training"):
            # Load a batch of data
            X, y = X.to(self.__device), y.to(self.__device)

            # Feedforward
            pred = self.__model(X)

            # Loss
            loss = self.__criterion(pred, y)
            losses.append(loss.item())

            # Backpropagation
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

        return np.mean(losses)

    @torch.no_grad()
    def __validate(self, dataloader: DataLoader) -> float:
        self.__model.eval()

        losses = []
        for X, y in tqdm(dataloader, desc="Validating"):
            # Load a batch of data
            X, y = X.to(self.__device), y.to(self.__device)

            # Feedforward
            pred = self.__model(X)

            # Loss
            loss = self.__criterion(pred, y)
            losses.append(loss.item())

        return np.mean(losses)

    def save_model(self, name: str = "checkpoint") -> None:
        torch.save(
            {
                "model_state_dict": self.__model.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict(),
                "train_losses": self.__train_losses,
                "val_losses": self.__val_losses,
                "learning_rates": self.__learning_rates,
            },
            self.__output_dir / f"{name}.pt",
        )

    def plot(self) -> None:
        x = list(range(1, len(self.__train_losses) + 1))
        for curve in ("Loss", "Learning Rate"):
            plt.figure()

            if curve == "Loss":
                plt.plot(x, self.__train_losses, "b-", label="Train")
                if len(self.__val_losses) > 0:
                    plt.plot(x, self.__val_losses, "r-", label="Val")
            else:
                plt.plot(x, self.__learning_rates, "g-")

            plt.legend()
            plt.title(f"{curve} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(curve)
            plt.tight_layout()
            plt.savefig(
                self.__output_dir / f"{curve.lower().replace(' ', '_')}.png",
                dpi=300,
            )
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
