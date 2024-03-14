# -*- coding: utf-8 -*-
# File: train.py

from typing import Optional

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from numpy import floating
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from lr_lambda import LRLambda
from model import aasr_tiny
from pytorch_pipeline import PyTorchPipeline
from utils import iter_files

BATCH_SIZE: int = 2
EPOCHS: int = 50
MAX_LR: float = 1e-4
MIN_LR: float = 1e-4
WARMUP: int = 0
EARLY_MIN: int = 0
WEIGHT_DECAY: float = 1e-3

IMG_SIZE: int = 64
MAX_SCALE: int = 8
STEP: float = 0.1
TRAIN_PCT: int = 0.2


class SRDataset(Dataset):
    DEFAULT_TRANSFORM = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])  # From image to tensor

    def __init__(
        self,
        root: str | list[str] | tuple[str, ...],
        transform: Optional[v2.Transform] = None,
    ) -> None:
        super().__init__()
        self.root = [root] if isinstance(root, str) else root
        self.transform = self.DEFAULT_TRANSFORM if transform is None else transform
        self.data = []
        for r in self.root:
            self.data.extend(iter_files(r, exts={".png", ".jpg"}, case_insensitive=True, recursive=True))

    def __getitem__(self, index: int) -> Tensor:
        return self.transform(Image.open(self.data[index]))

    def __len__(self) -> int:
        return len(self.data)


class TrainingPipeline(PyTorchPipeline):
    # Override
    def train(self, dataloader: DataLoader) -> floating:
        self._model.train()

        losses = []
        for batch in tqdm(dataloader, desc="Training"):
            batch = batch.to(self._device)
            # for scale in rng.choice(scales, size=(int(len(scales) * TRAIN_PCT), 2)).tolist():
            for scale in rng.choice(scales, size=(1, 2)).tolist():
                size = (int(IMG_SIZE * scale[0]), int(IMG_SIZE * scale[1]))
                y = K.RandomCrop(size, same_on_batch=False)(batch)
                y_aux = K.Resize((IMG_SIZE, IMG_SIZE))(y)
                x = input_aug(y_aux)

                # Feedforward
                pred, aux = self._model(x, size=size)

                # Loss
                loss = self._criterion(pred, y) + self._criterion(aux, y_aux)
                losses.append(loss.item())

                # Backpropagation
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

        return np.mean(losses)

    # Override
    @torch.no_grad()  # Turn off gradient descent
    def validate(self, dataloader: DataLoader) -> floating:
        self._model.eval()

        losses = []
        for X in tqdm(dataloader, desc="Validating"):
            # Load a batch of data
            X = X.to(self._device)

            # Feedforward
            pred = self._model(X)

            # Loss
            loss = self._criterion(pred, y)
            losses.append(loss.item())

        return np.mean(losses)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng()
scales = np.arange(1.0, MAX_SCALE + STEP, STEP)

train_transform = v2.Compose(
    [
        v2.RandomCrop(IMG_SIZE * MAX_SCALE, pad_if_needed=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.RandomAutocontrast(p=0.5),
        v2.RandomAdjustSharpness(2, p=0.5),
        v2.RandomEqualize(p=0.2),
        v2.RandomInvert(p=0.1),
        v2.RandomGrayscale(p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
val_transform = v2.Compose(
    [
        v2.CenterCrop(IMG_SIZE * MAX_SCALE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
input_aug = K.AugmentationSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.5, 0.5)),
    K.RandomGaussianNoise(),
    same_on_batch=False,
)

train_data = SRDataset("./data/train/DIV2K", transform=train_transform)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)
val_data = SRDataset("./data/valid", transform=val_transform)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, persistent_workers=False)

model = aasr_tiny()
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    LRLambda.linear(EPOCHS, MAX_LR, min_lr=MIN_LR, warmup=WARMUP, early_min=EARLY_MIN),
)

pipeline = TrainingPipeline(model, criterion, optimizer, scheduler=scheduler, device=device)
pipeline.start(EPOCHS, train_dataloader, val_dataloader=val_dataloader)
