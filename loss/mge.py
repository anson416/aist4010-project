# -*- coding: utf-8 -*-
# File: loss/mge.py

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MeanGradientError(nn.Module):
    """
    Reference: [Single Image Super Resolution based on a Modified U-net with Mixed Gradient Loss](https://arxiv.org/abs/1911.09428)
    """

    def __init__(
        self,
        channels: int = 3,
        mode: Literal["mae", "mse"] = "mae",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.mode = mode

        sobel_x = Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).repeat(channels, 1, 1, 1)
        sobel_y = Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).repeat(channels, 1, 1, 1)
        conv_x = lambda x: F.conv2d(x, sobel_x, padding="same", groups=channels)
        conv_y = lambda x: F.conv2d(x, sobel_y, padding="same", groups=channels)
        p = 1 + (mode == "mse")
        self.gmap = lambda x: (torch.abs(conv_x(x)) ** p + torch.abs(conv_y(x)) ** p) ** (1 / p)
        self.diff = F.l1_loss if mode == "mae" else F.mse_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert len(input.shape) == 4
        assert len(target.shape) == 4
        assert input.shape[1] == self.channels
        assert target.shape[1] == self.channels

        return self.diff(self.gmap(input), self.gmap(target))
