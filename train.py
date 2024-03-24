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

from loss import FFT2DLoss, MeanGradientError, MultiScaleSSIMLoss
from model import *
from utils.file_ops import iter_files
from utils.lr_lambda import LRLambda
from utils.pytorch_pipeline import PyTorchPipeline

CONFIGS = {
    "levels": BASE,
    "block": "ConvNeXtBlock",
    "n_recurrent": 1,
    "use_channel_attention": True,
    "use_attention_gate": True,
    "add_bilinear": True,
    "downsampler": "conv2d",
    "upsampler": "pixelshuffle",
    "stochastic_depth_prob": 0.1,
}

TRAIN_DIR = "./data/train/DIV2K"
VAL_DIR = "./data/valid"

BATCH_SIZE: int = 16
EPOCHS: int = 20
MAX_LR: float = 1e-3
MIN_LR: float = 1e-3
WARMUP: int = 0
EARLY_MIN: int = 0
WEIGHT_DECAY: float = 1e-4

IMG_SIZE: int = 64
MAX_SCALE: int = 4
TRAIN_PCT: float = 0.05


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
    def train(
        self,
        dataloader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> floating:
        self._model.train()

        losses = []
        for batch in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Training", leave=False):
            batch = batch.to(self._device)
            for scale in train_scales[rng.choice(n := len(train_scales), max(int(n * TRAIN_PCT), 1), replace=False)]:
                size = (int(IMG_SIZE * scale[0]), int(IMG_SIZE * scale[1]))
                y = K.RandomCrop(size, same_on_batch=False)(batch)
                y_aux = K.Resize((IMG_SIZE, IMG_SIZE))(y)
                x = train_x_aug(y_aux)

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
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> floating:
        self._model.eval()

        losses = []
        for batch in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Validating", leave=False):
            batch = batch.to(self._device)
            for scale in val_scales:
                size = (int(IMG_SIZE * scale[0]), int(IMG_SIZE * scale[1]))
                y = K.CenterCrop(size)(batch)
                y_aux = K.Resize((IMG_SIZE, IMG_SIZE))(y)
                x = y_aux

                # Feedforward
                pred, aux = self._model(x, size=size)

                # Loss
                loss = self._criterion(pred, y) + self._criterion(aux, y_aux)
                losses.append(loss.item())

        return np.mean(losses)


class SRLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.6,
        gamma: float = 0.3,
        mu: float = 0.1,
    ) -> None:
        assert alpha >= 0.0
        assert beta >= 0.0
        assert gamma >= 0.0
        assert mu >= 0.0

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = MultiScaleSSIMLoss(data_range=1.0, win_size=5)
        self.mge = MeanGradientError(mode="mse")
        self.fft_loss = FFT2DLoss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return (
            self.alpha * self.l1_loss(input, target)
            + self.beta * self.ssim_loss(input, target)
            + self.gamma * self.mge(input, target)
            + self.mu * self.fft_loss(input, target)
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng()
train_scales = np.arange(1.0, MAX_SCALE + 0.1, 0.1)
train_scales = np.array(np.meshgrid(train_scales, train_scales)).T.reshape(-1, 2)
val_scales = np.arange(1.0, MAX_SCALE + 0.5, 0.5)
val_scales = np.array(np.meshgrid(val_scales, val_scales)).T.reshape(-1, 2)

train_y_aug = v2.Compose(
    [
        v2.RandomCrop(IMG_SIZE * MAX_SCALE, pad_if_needed=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply(nn.ModuleList([v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)]), p=0.2),
        v2.RandomAutocontrast(p=0.2),
        v2.RandomAdjustSharpness(2, p=0.2),
        v2.RandomEqualize(p=0.2),
        v2.RandomInvert(p=0.1),
        v2.RandomGrayscale(p=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
val_y_aug = v2.Compose(
    [
        v2.CenterCrop(IMG_SIZE * MAX_SCALE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)
train_x_aug = K.AugmentationSequential(
    K.RandomBoxBlur(kernel_size=(3, 3), p=0.125),
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2), p=0.125),
    K.RandomMedianBlur(kernel_size=(3, 3), p=0.125),
    K.RandomMotionBlur(kernel_size=3, angle=180, direction=1, p=0.125),
    K.RandomGaussianNoise(p=0.25),
    K.RandomSaltAndPepperNoise(p=0.25),
    # K.RandomJPEG(p=0.5),
    same_on_batch=False,
)

train_data = SRDataset(TRAIN_DIR, transform=train_y_aug)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)
val_data = SRDataset(VAL_DIR, transform=val_y_aug)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

model = AASR(**CONFIGS).to(device)
criterion = SRLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    LRLambda.cosine(EPOCHS, MAX_LR, min_lr=MIN_LR, warmup=WARMUP, early_min=EARLY_MIN),
)

pipeline = TrainingPipeline(model, criterion, optimizer, scheduler=scheduler, device=device, configs=CONFIGS)
pipeline.start(EPOCHS, train_dataloader, val_dataloader=val_dataloader)
