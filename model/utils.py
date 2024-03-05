# -*- coding: utf-8 -*-
# File: model/utils.py

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LayerNorm2d(nn.LayerNorm):
    """
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


# Global Response Normalization
class GRN(nn.Module):
    """
    Reference: [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)

    Source: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        Gx = x.norm(p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChannelAttention(nn.Module):
    """
    Reference: [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

    Source: https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        reduced = channels // reduction
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.attention(x)
