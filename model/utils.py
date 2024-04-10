# -*- coding: utf-8 -*-
# File: model/utils.py

import math
from typing import Any, Optional, Type

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops.stochastic_depth import StochasticDepth


class LayerNorm2d(nn.LayerNorm):
    """
    Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ChannelModification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modification = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.modification(x)


class Concatenation(nn.Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, *inputs: Tensor) -> Tensor:
        return torch.cat(inputs, dim=self.dim)


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

        reduced = max(channels // reduction, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.attention(x)


class ScaleAwareAdaption(nn.Module):
    """
    Reference: [Learning A Single Network for Scale-Arbitrary Super-Resolution](https://arxiv.org/abs/2004.03791)
    Source: https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR/blob/master/model/arbrcan.py
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        n_experts: int = 4,
        dropout: float = 0.5,
    ) -> None:
        assert n_experts >= 1

        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_experts = n_experts

        self.routing = nn.Sequential(
            nn.Linear(2, n_experts * 8),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_experts * 8, n_experts),
            nn.Softmax(dim=1),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding="same", groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=kernel_size, padding="same"),
            nn.Sigmoid(),
        )
        self.weight_pool = nn.Parameter(Tensor(n_experts, channels, 1, kernel_size, kernel_size))
        nn.init.trunc_normal_(self.weight_pool, std=0.02)
        self.bias_pool = nn.Parameter(Tensor(n_experts, channels))
        nn.init.constant_(self.bias_pool, 0)

    def forward(self, x: Tensor, scale_h: float, scale_w: float) -> Tensor:
        scale_h = torch.ones(1, 1).to(x.device) / scale_h
        scale_w = torch.ones(1, 1).to(x.device) / scale_w

        routing_weights = self.routing(torch.cat((scale_h, scale_w), 1)).view(self.n_experts, 1, 1)

        fused_weight = (self.weight_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, 1, self.kernel_size, self.kernel_size)
        fused_bias = (self.bias_pool.view(self.n_experts, -1, 1) * routing_weights).sum(0)
        fused_bias = fused_bias.view(-1)

        adapted = F.conv2d(x, fused_weight, fused_bias, padding="same", groups=self.channels)

        return x + adapted * self.mask(x)


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
        scale_aware: bool = False,
        layer_norm: bool = False,
        stochastic_depth_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.block = block
        self.channels = channels
        self.n_recurrent = n_recurrent
        self.attention = attention
        self.scale_aware = scale_aware
        self.layer_norm = layer_norm
        self.stochastic_depth_prob = stochastic_depth_prob
        self.reduction: int = kwargs.pop("reduction", 16)
        self.n_experts: int = kwargs.pop("n_experts", 4)
        self.eps: float = kwargs.pop("eps", 1e-6)

        self.bottleneck = block(channels, layer_norm=layer_norm, eps=self.eps, **kwargs)
        if attention:
            self.channel_attention = ChannelAttention(channels, reduction=self.reduction)
        if scale_aware:
            self.scale_aware_adaption = ScaleAwareAdaption(channels, n_experts=self.n_experts)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(
        self,
        x: Tensor,
        scale_h: Optional[float] = None,
        scale_w: Optional[float] = None,
    ) -> Tensor:
        if self.scale_aware:
            assert scale_h is not None and scale_w is not None

        out = self.bottleneck(x)

        for _ in range(self.n_recurrent):
            out = self.bottleneck(x + out)

        if self.attention:
            out = self.channel_attention(out)

        out = x + self.stochastic_depth(out)

        if self.scale_aware:
            out = self.scale_aware_adaption(out, scale_h, scale_w)

        return out


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


class AttentionGate(nn.Module):
    """
    Reference: [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
    Source: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.attention = nn.Sequential(
            nn.GELU(),
            ChannelModification(channels, 1),
            LayerNorm2d(1, eps=eps),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, g: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        assert g.shape[1] == self.channels

        return x * self.attention(x + g)


def grid_sample(
    x: Tensor,
    offset: Tensor,
    out_h: int,
    out_w: int,
) -> Tensor:
    """
    Reference: [Learning A Single Network for Scale-Arbitrary Super-Resolution](https://arxiv.org/abs/2004.03791)
    Source: https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR/blob/master/model/arbrcan.py
    """

    b, _, h, w = x.size()
    scale_h, scale_w = out_h / h, out_w / w

    # generate grids
    grid = np.meshgrid(range(out_w), range(out_h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale_w - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale_h - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
    offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
    grid = grid + torch.cat((offset_0, offset_1), 1)
    grid = grid.permute(0, 2, 3, 1)

    # sampling
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    return output


class ScaleAwareUpsampler(nn.Module):
    """
    Reference: [Learning A Single Network for Scale-Arbitrary Super-Resolution](https://arxiv.org/abs/2004.03791)
    Source: https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR/blob/master/model/arbrcan.py
    """

    def __init__(
        self,
        channels: int,
        n_experts: int = 4,
        reduction: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_experts = n_experts
        self.reduction = reduction
        self.eps = eps

        self.reduced = max(channels // reduction, 1)

        # experts
        self.weight_compress = nn.Parameter(Tensor(n_experts, self.reduced, channels, 1, 1))
        nn.init.kaiming_uniform_(self.weight_compress, a=math.sqrt(5))
        self.weight_expand = nn.Parameter(Tensor(n_experts, channels, self.reduced, 1, 1))
        nn.init.kaiming_uniform_(self.weight_expand, a=math.sqrt(5))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.GELU(),
        )

        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(64, n_experts, kernel_size=1),
            nn.Sigmoid(),
        )
        # offset head

        self.offset = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: Tensor, out_h: int, out_w: int) -> Tensor:
        b, _, h, w = x.shape
        scale_h, scale_w = out_h / h, out_w / w

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [
            torch.arange(0, out_h, 1).unsqueeze(0).float().to(x.device),
            torch.arange(0, out_w, 1).unsqueeze(0).float().to(x.device),
        ]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale_h) - (torch.floor((coor_hr[0] + 0.5) / scale_h + self.eps)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale_w) - (torch.floor((coor_hr[1] + 0.5) / scale_w + self.eps)) - 0.5

        input = torch.cat(
            (
                torch.ones_like(coor_h).expand([-1, out_w]).unsqueeze(0) / scale_w,
                torch.ones_like(coor_h).expand([-1, out_w]).unsqueeze(0) / scale_h,
                coor_h.expand([-1, out_w]).unsqueeze(0),
                coor_w.expand([out_h, -1]).unsqueeze(0),
            ),
            dim=0,
        ).unsqueeze(0)

        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(self.n_experts, out_h * out_w).transpose(0, 1)  # (h*w) * n

        weight_compress = self.weight_compress.view(self.n_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(1, out_h, out_w, self.reduced, self.channels)

        weight_expand = self.weight_expand.view(self.n_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(1, out_h, out_w, self.channels, self.reduced)

        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, out_h, out_w)  ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)  ## b * h * w * c * 1

        ## spatially varying filtering
        out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
        out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0
