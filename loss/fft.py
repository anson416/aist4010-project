# -*- coding: utf-8 -*-
# File: loss/fft.py

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FFT2DLoss(nn.Module):
    """
    References: [Focal Frequency Loss for Image Reconstruction and Synthesis](https://arxiv.org/abs/2012.12821),
    [Fourier Space Losses for Efficient Perceptual Image Super-Resolution](https://arxiv.org/abs/2106.00783)
    """

    def __init__(self, mode: Literal["mae", "mse"] = "mae") -> None:
        super().__init__()
        self.mode = mode

        self.diff = F.l1_loss if mode == "mae" else F.mse_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert len(input.shape) == 4
        assert len(target.shape) == 4

        return self.diff(self.fft2d(input), self.fft2d(target))

    @staticmethod
    def fft2d(img: Tensor) -> Tensor:
        return 20 * torch.log10(torch.fft.fftshift(torch.fft.fft2(img, norm="ortho")))
