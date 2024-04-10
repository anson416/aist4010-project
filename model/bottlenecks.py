# -*- coding: utf-8 -*-
# File: model/bottlenecks.py

from typing import Any

import torch
from torch import Tensor, nn

from .utils import GRN, LayerNorm2d


class ResNetBlockV2(nn.Module):
    """
    NOTE: The residual connection is missing, as this module is supposed to be
    used indirectly by `utils.RecurrentAttentionBlock()`.

    Reference: [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.layer_norm = layer_norm
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            LayerNorm2d(channels, eps=self.eps) if layer_norm else nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same", bias=layer_norm),
            LayerNorm2d(channels, eps=self.eps) if layer_norm else nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, expanded, kernel_size=1, bias=layer_norm),
            LayerNorm2d(expanded, eps=self.eps) if layer_norm else nn.BatchNorm2d(expanded),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvNeXtBlock(nn.Module):
    """
    NOTE: The residual connection is missing, as this module is supposed to be
    used indirectly by `utils.RecurrentAttentionBlock()`.

    Reference: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        layer_scale_init_value: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.layer_scale_init_value = layer_scale_init_value
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=self.eps),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale_init_value)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) * self.layer_scale


class ConvNeXtBlockV2(nn.Module):
    """
    NOTE: The residual connection is missing, as this module is supposed to be
    used indirectly by `utils.RecurrentAttentionBlock()`.

    Reference: [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    Source: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.expansion = expansion
        self.eps: float = kwargs.pop("eps", 1e-6)

        expanded = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding="same", groups=channels),
            LayerNorm2d(channels, eps=self.eps),
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            GRN(expanded),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EDSRBlock(nn.Module):
    """
    NOTE: The residual connection is missing, as this module is supposed to be
    used indirectly by `utils.RecurrentAttentionBlock()`.

    Reference: [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)
    Source: https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/common.py
    """

    def __init__(self, channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.channels = channels

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
