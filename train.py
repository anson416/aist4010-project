# -*- coding: utf-8 -*-
# File: train.py

import argparse
import json
import os
import random
from argparse import Namespace
from math import ceil, sqrt
from typing import Optional

import kornia.augmentation as K
import numpy as np
import torch
import torch.nn as nn
from kornia.constants import Resample
from numpy import floating
from PIL import Image
from torch import Tensor, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from loss import FFT2DLoss, MeanGradientError, MultiScaleSSIMLoss
from model import *
from test_ import test
from utils.file_ops import iter_files
from utils.num_ops import clamp
from utils.pytorch_pipeline import PyTorchPipeline


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        type=str,
        default="BASE",
        choices=("MOBILE", "TINY", "SMALL", "BASE", "LARGE", "XLARGE", "HUGE"),
    )
    parser.add_argument("--block", type=str, default="ConvNeXtBlock")
    parser.add_argument("--downsampler", type=str, default="conv2d", choices=("conv2d", "maxpool2d"))
    parser.add_argument(
        "--upsampler",
        type=str,
        default="pixelshuffle",
        choices=("bicubic", "bilinear", "convtranspose2d", "pixelshuffle"),
    )
    parser.add_argument("--super_upsampler", type=str, default="scale_aware", choices=("bicubic", "scale_aware"))
    parser.add_argument("--n_recurrent", type=int, default=0)
    parser.add_argument("--channel_attention", action="store_true")
    parser.add_argument("--scale_aware_adaption", action="store_true")
    parser.add_argument("--attention_gate", action="store_true")
    parser.add_argument("--concat_orig_interp", action="store_true")
    parser.add_argument("--stochastic_depth_prob", type=float, default=0.0)
    parser.add_argument("--init_weights", action="store_true")
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--aux_weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--max_scale", type=int, default=4)
    parser.add_argument("--asym_pct", type=float, default=0.05)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--val_dir", type=str, default="./data/valid")
    parser.add_argument("--test_dir", type=str, default="./data/test")
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
            indexes = rng.choice(n := len(train_asym_scales), clamp(int(n * args.asym_pct), 0, n), replace=False)
            for scale in np.concatenate((train_sym_scales, train_asym_scales[indexes]), axis=0):
                size = (round(args.img_size * scale[0]), round(args.img_size * scale[1]))
                y = K.RandomCrop(size, same_on_batch=False)(batch)
                y_aux = K.Resize(
                    [args.img_size] * 2,
                    resample=random.choice((Resample.BILINEAR.name, Resample.BICUBIC.name)),
                    antialias=random.choice((True, False)),
                )(y)
                x = train_x_aug(y_aux)

                # Feedforward
                pred, aux = self._model(x, size=size)

                # Loss
                loss = self._criterion(pred, y)
                if args.aux_weight > 0:
                    loss += args.aux_weight * self._criterion(aux, y_aux)
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
                size = (round(args.img_size * scale[0]), round(args.img_size * scale[1]))
                y = K.CenterCrop(size)(batch)
                y_aux = K.Resize([args.img_size] * 2, resample=Resample.BICUBIC.name, antialias=True)(y)
                x = y_aux

                # Feedforward
                pred, aux = self._model(x, size=size)

                # Loss
                loss = self._criterion(pred, y)
                if args.aux_weight > 0:
                    loss += args.aux_weight * self._criterion(aux, y_aux)
                losses.append(loss.item())

        return np.mean(losses)


class SRLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.5,
        eta: float = 0.01,
        mu: float = 0.01,
    ) -> None:
        assert alpha >= 0.0
        assert beta >= 0.0
        assert gamma >= 0.0
        assert eta >= 0.0
        assert mu >= 0.0
        assert alpha + beta + gamma + eta + mu > 0.0

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.mu = mu

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = MultiScaleSSIMLoss(data_range=1.0, win_size=3)
        self.mge = MeanGradientError()
        self.fft_loss = FFT2DLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = 0
        if self.alpha > 0.0:
            loss += self.alpha * self.l1_loss(pred, target)
        if self.beta > 0.0:
            loss += self.beta * self.mse_loss(pred, target)
        if self.gamma > 0.0:
            loss += self.gamma * self.ssim_loss(pred, target)
        if self.eta > 0.0:
            loss += self.eta * self.mge(pred, target)
        if self.mu > 0.0:
            loss += self.mu * self.fft_loss(pred, target)
        return loss


args = parse_args()
print(args)

configs = {
    "levels": getattr(aasr, args.levels.upper()),
    "block": args.block,
    "n_recurrent": args.n_recurrent,
    "channel_attention": args.channel_attention,
    "scale_aware_adaption": args.scale_aware_adaption,
    "attention_gate": args.attention_gate,
    "concat_orig_interp": args.concat_orig_interp,
    "downsampler": args.downsampler,
    "upsampler": args.upsampler,
    "super_upsampler": args.super_upsampler,
    "stochastic_depth_prob": args.stochastic_depth_prob,
    "init_weights": args.init_weights,
    "reduction": args.reduction,
    "n_experts": args.n_experts,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng()
train_scales = np.arange(1.0, args.max_scale + 0.1, 0.1)
train_sym_scales = np.column_stack((train_scales, train_scales))
train_asym_scales = np.array(np.meshgrid(train_scales, train_scales)).T.reshape(-1, 2)
train_asym_scales = np.array(list(set(map(tuple, train_asym_scales)) - set(map(tuple, train_sym_scales))))
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
    same_on_batch=False,
)

train_data = SRDataset(args.train_dir, transform=train_y_aug)
train_dataloader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_workers,
    persistent_workers=True,
)
val_data = SRDataset(args.val_dir, transform=val_y_aug)
val_dataloader = DataLoader(
    val_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_workers,
    persistent_workers=True,
)

model = AASR(**configs).to(device)
criterion = SRLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, eta=args.eta, mu=args.mu)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=ceil(args.epochs / 3), gamma=sqrt(args.min_lr / args.max_lr))

pipeline = TrainingPipeline(
    model,
    criterion,
    optimizer,
    scheduler=scheduler,
    device=device,
    name=args.name,
    configs=configs,
)
print(f"Trainable parameters: {pipeline.n_trainable_params:,}")
output_dir = pipeline.start(args.epochs, train_dataloader, val_dataloader=val_dataloader)

with open(os.path.join(output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

test(os.path.join(output_dir, "checkpoint_best.pt"), output_dir, test_dir=args.test_dir)
