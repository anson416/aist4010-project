# -*- coding: utf-8 -*-
# File: train.py

import argparse
import random
from argparse import Namespace
from typing import Optional

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from kornia.constants import Resample
from numpy import floating
from PIL import Image
from torch import Tensor, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from loss import FFT2DLoss, MeanGradientError, MultiScaleSSIMLoss, SSIMLoss
from model import *
from utils.file_ops import iter_files
from utils.pytorch_pipeline import PyTorchPipeline


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=str, default="BASE")
    parser.add_argument("--block", type=str, default="ConvNeXtBlock")
    parser.add_argument("--n_recurrent", type=int, default=0)
    parser.add_argument("--use_channel_attention", action="store_true")
    parser.add_argument("--use_attention_gate", action="store_true")
    parser.add_argument("--concat_orig_interp", action="store_true")
    parser.add_argument("--downsampler", type=str, default="conv2d")
    parser.add_argument("--upsampler", type=str, default="pixelshuffle")
    parser.add_argument("--stochastic_depth_prob", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--max_scale", type=int, default=4)
    parser.add_argument("--train_pct", type=float, default=0.1)
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir", type=str, default="./data/valid")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


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
    RESAMPLES = (Resample.BILINEAR.name, Resample.BICUBIC.name)

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
            scales = rng.choice(n := len(train_scales), max(int(n * args.train_pct), 1), replace=False)
            for scale in train_scales[scales]:
                size = (int(args.img_size * scale[0]), int(args.img_size * scale[1]))
                y = K.RandomCrop(size, same_on_batch=False)(batch)
                y_aux = K.Resize((args.img_size, args.img_size), resample=random.choice(self.RESAMPLES))(y)
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
                size = (int(args.img_size * scale[0]), int(args.img_size * scale[1]))
                y = K.CenterCrop(size)(batch)
                y_aux = K.Resize((args.img_size, args.img_size), resample=Resample.BICUBIC.name)(y)
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
        beta: float = 0.8,
        gamma: float = 0.1,
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
        # self.ssim_loss = MultiScaleSSIMLoss(data_range=1.0, win_size=3)
        self.ssim_loss = SSIMLoss(data_range=1.0)
        self.mge = MeanGradientError()
        self.fft_loss = FFT2DLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (
            self.alpha * self.l1_loss(pred, target)
            + self.beta * self.ssim_loss(pred, target)
            + self.gamma * self.mge(pred, target)
            + self.mu * self.fft_loss(pred, target)
        )


args = parse_args()

configs = {
    "levels": getattr(aasr, args.levels.upper()),
    "block": args.block,
    "n_recurrent": args.n_recurrent,
    "use_channel_attention": args.use_channel_attention,
    "use_attention_gate": args.use_attention_gate,
    "concat_orig_interp": args.concat_orig_interp,
    "downsampler": args.downsampler,
    "upsampler": args.upsampler,
    "stochastic_depth_prob": args.stochastic_depth_prob,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng()
train_scales = np.arange(1.0, args.max_scale + 0.1, 0.1)
train_scales = np.array(np.meshgrid(train_scales, train_scales)).T.reshape(-1, 2)
val_scales = np.arange(1.0, args.max_scale + 0.3, 0.3)
val_scales = np.array(np.meshgrid(val_scales, val_scales)).T.reshape(-1, 2)

train_y_aug = v2.Compose(
    [
        v2.RandomCrop(args.img_size * args.max_scale, pad_if_needed=True),
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
        v2.CenterCrop(args.img_size * args.max_scale),
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

train_data = SRDataset(args.train_dir, transform=train_y_aug)
train_dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    persistent_workers=True,
)
val_data = SRDataset(args.val_dir, transform=val_y_aug)
val_dataloader = DataLoader(
    val_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    persistent_workers=True,
)

model = AASR(**configs).to(device)
criterion = SRLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=args.min_lr)

pipeline = TrainingPipeline(
    model,
    criterion,
    optimizer,
    scheduler=scheduler,
    device=device,
    configs=configs,
    name=args.name,
)
pipeline.start(args.epochs, train_dataloader, val_dataloader=val_dataloader)
