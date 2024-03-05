# -*- coding: utf-8 -*-
# File: model/bottlenecks.py

from typing import Any, Type

import torch
from torch import Tensor, nn
from torchvision.ops.stochastic_depth import StochasticDepth

from .utils import GRN, ChannelAttention, LayerNorm2d


class RecurrentAttentionBlock(nn.Module):
    """
    Reference: [Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation](https://arxiv.org/abs/1802.06955)

    Source: https://github.com/navamikairanda/R2U-Net/blob/main/r2unet.py
    """

    def __init__(
        self,
        block: Type[nn.Module],
        channels: int,
        n_recurrent: int = 0,
        attention: bool = False,
        reduction: int = 16,
        stochastic_depth_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.block = block
        self.channels = channels
        self.n_recurrent = n_recurrent
        self.attention = attention
        self.stochastic_depth_prob = stochastic_depth_prob

        self.bottleneck = block(channels, **kwargs)
        self.channel_attention = ChannelAttention(channels, reduction=reduction) if attention else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x: Tensor) -> Tensor:
        out = self.bottleneck(x)
        for _ in range(self.n_recurrent):
            out = self.bottleneck(x + out)
        out = self.channel_attention(out)
        return x + self.stochastic_depth(out)


class ResNetBlockV2(nn.Module):
    """
    Reference: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvNeXtBlock(nn.Module):
    """
    Reference: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    """

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.layer_scale_init_value = layer_scale_init_value

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=1e-6),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale_init_value)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) * self.layer_scale


class ConvNeXtBlockV2(nn.Module):
    """
    Reference: [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)

    Source: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=1e-6),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            GRN(expanded),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
