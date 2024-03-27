# -*- coding: utf-8 -*-
# File: model/aasr.py

# import sys
from collections.abc import Sequence
from typing import Any, Literal, Optional, Type

from torch import Tensor, nn

from . import bottlenecks
from .utils import (
    AttentionGate,
    ChannelModification,
    Concatenation,
    LayerNorm2d,
    RecurrentAttentionBlock,
)

__all__ = [
    "AASR",
    "TINY",
    "SMALL",
    "BASE",
    "LARGE",
    "XLARGE",
    "HUGE",
]

TINY = ((16, 2), (32, 2))  # ~39K parameters
SMALL = ((32, 4), (64, 6), (128, 4))  # ~1M parameters
BASE = ((64, 3), (128, 3), (256, 9), (512, 3))  # ~18M parameters
LARGE = ((64, 3), (128, 3), (256, 27), (512, 3))  # ~28M parameters
XLARGE = ((96, 3), (192, 3), (384, 9), (768, 3))  # ~41M parameters
HUGE = ((96, 3), (192, 3), (384, 9), (768, 27), (1536, 3))  # ~308M parameters


class Downsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        mode: Literal["conv2d", "maxpool2d"] = "conv2d",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.mode = mode

        if scale == 1:
            self.downsampler = nn.Identity()
        else:
            if mode == "conv2d":
                self.downsampler = nn.Sequential(
                    LayerNorm2d(in_channels, eps=1e-6),
                    nn.Conv2d(in_channels, out_channels, kernel_size=scale, stride=scale),
                )
            elif mode == "maxpool2d":
                self.downsampler = nn.Sequential(
                    LayerNorm2d(in_channels, eps=1e-6),
                    nn.MaxPool2d(kernel_size=scale, stride=scale),
                    ChannelModification(in_channels, out_channels),
                )
            else:
                raise ValueError(f'Unknown value "{mode}" for `mode`.')

    def forward(self, x: Tensor) -> Tensor:
        return self.downsampler(x)


class Upsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        mode: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.mode = mode

        if scale == 1:
            self.upsampler = nn.Identity()
        else:
            if mode in {"bicubic", "bilinear"}:
                self.upsampler = nn.Sequential(
                    nn.Upsample(scale_factor=scale, mode=mode),
                    ChannelModification(in_channels, out_channels),
                )
            elif mode == "convtranspose2d":
                self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
            elif mode == "pixelshuffle":
                self.upsampler = nn.Sequential(
                    ChannelModification(in_channels, out_channels * scale**2),
                    nn.PixelShuffle(scale),
                )
            else:
                raise ValueError(f'Unknown value "{mode}" for `mode`.')

    def forward(self, x: Tensor) -> Tensor:
        return self.upsampler(x)


class Collector(nn.Module):
    def __init__(
        self,
        levels: Sequence[tuple[int, int]],
        level: int,
        use_attention: bool = False,
        downsampler: Literal["conv2d", "maxpool2d"] = "conv2d",
        upsampler: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.levels = levels
        self.level = level
        self.use_attention = use_attention
        self.downsampler = downsampler
        self.upsampler = upsampler

        channels = levels[level][0]
        self.concat = Concatenation(dim=1)
        self.from_encoder = nn.ModuleList(
            [
                Downsampler(
                    levels[i][0],
                    channels,
                    2 ** (level - i),
                    mode=downsampler,
                )
                for i in range(level + 1)
            ]
        )
        self.from_decoder = nn.ModuleList(
            [
                Upsampler(
                    levels[i][0],
                    channels,
                    2 ** (i - level),
                    mode=upsampler,
                )
                for i in range(len(levels) - 1, level, -1)
            ]
        )
        if use_attention:
            self.conv = ChannelModification((level + 1) * channels, channels)
            self.attention = AttentionGate(channels)
        self.collector = ChannelModification((len(levels) - level * use_attention) * channels, channels)

    def forward(self, encoded: list[Tensor], decoded: list[Tensor]) -> Tensor:
        assert len(encoded) == len(self.from_encoder)
        assert len(decoded) == len(self.from_decoder)

        encoded = [self.from_encoder[i](encoded[i]) for i in range(len(encoded))]
        decoded = [self.from_decoder[i](decoded[i]) for i in range(len(decoded))]
        if self.use_attention:
            gated = self.attention(self.conv(self.concat(*encoded)), decoded[-1])
            out = self.collector(self.concat(gated, *decoded))
        else:
            out = self.collector(self.concat(*encoded, *decoded))

        return out


class AASR(nn.Module):
    def __init__(
        self,
        levels: Sequence[tuple[int, int]] = SMALL,
        in_channels: int = 3,
        out_channels: int = 3,
        block: str | Type[nn.Module] = "ConvNeXtBlock",
        n_recurrent: int = 1,
        use_channel_attention: bool = True,
        use_attention_gate: bool = True,
        concat_orig_interp: bool = True,
        downsampler: Literal["conv2d", "maxpool2d"] = "conv2d",
        upsampler: Literal["bicubic", "bilinear", "convtranspose2d", "pixelshuffle"] = "pixelshuffle",
        stochastic_depth_prob: float = 0.0,
        init_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        assert len(levels) >= 2
        assert in_channels >= 1
        assert out_channels >= 1
        assert n_recurrent >= 0
        assert 0.0 <= stochastic_depth_prob <= 1.0

        super().__init__()
        self.block = getattr(bottlenecks, block) if isinstance(block, str) else block
        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_recurrent = n_recurrent
        self.use_channel_attention = use_channel_attention
        self.use_attention_gate = use_attention_gate
        self.concat_orig_interp = concat_orig_interp
        self.downsampler = downsampler
        self.upsampler = upsampler
        self.stochastic_depth_prob = stochastic_depth_prob
        self.init_weights = init_weights
        self.reduction: int = kwargs.pop("reduction", 16)
        self.kwargs = kwargs

        self.concat = Concatenation(dim=1)
        self.stem = self.__make_stem()
        self.encoder = self.__make_encoder()
        self.decoder = self.__make_decoder()
        self.output = self.__make_output()
        self.auxiliary = self.__make_auxiliary()

        if init_weights:
            self.initialize_weights()

    def forward(
        self,
        x: Tensor,
        size: Optional[int | tuple[int, int]] = None,
        scale: float | tuple[float, float] = 1.0,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(scale, float):
            scale = (scale, scale)

        assert len(x.shape) == 4
        assert x.shape[1] == self.in_channels
        assert x.shape[2] % (1 << (len(self.levels) - 1)) == 0
        assert x.shape[3] % (1 << (len(self.levels) - 1)) == 0

        # Instantiate a bicubic upsampler for later use
        bicubic = (
            nn.Upsample(scale_factor=scale, mode="bicubic") if size is None else nn.Upsample(size=size, mode="bicubic")
        )

        out = self.stem(x)

        # Encode
        encoded: list[Tensor] = []
        for encoder in self.encoder:
            for i, enc in enumerate(encoder):  # `i` should be at most 1
                out = enc(out)
                if i == 0:
                    encoded.append(out)

        # Decode
        decoded = [encoded.pop()]
        for decoder in self.decoder:
            for i, dec in enumerate(decoder):
                if i == 0:
                    out = dec(encoded, decoded)
                else:
                    out = dec(out)
                    decoded.append(out)
                    encoded.pop()

        auxiliary = self.auxiliary(out)
        out = self.output(self.concat(bicubic(x), bicubic(out)) if self.concat_orig_interp else bicubic(out))

        return out, auxiliary

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __make_stem(self) -> nn.Module:
        return ChannelModification(self.in_channels, self.levels[0][0])

    def __make_encoder(self) -> nn.ModuleList:
        encoder = nn.ModuleList()

        total_blocks = sum(lvl[1] for lvl in self.levels)
        block_id = 1
        for idx, (channels, n_blocks) in enumerate(self.levels):
            blocks: list[nn.Module] = []
            for _ in range(n_blocks):
                blocks.append(
                    RecurrentAttentionBlock(
                        self.block,
                        channels,
                        n_recurrent=self.n_recurrent,
                        use_attention=self.use_channel_attention,
                        reduction=self.reduction,
                        stochastic_depth_prob=self.stochastic_depth_prob * block_id / total_blocks,
                        **self.kwargs,
                    )
                )
                block_id += 1
            level = nn.ModuleList([nn.Sequential(*blocks)])  # Only one element at the last level

            # Downsample
            if idx != len(self.levels) - 1:  # Except the last level
                level.append(Downsampler(channels, self.levels[idx + 1][0], 2, mode=self.downsampler))

            encoder.append(level)

        return encoder

    def __make_decoder(self) -> nn.ModuleList:
        decoder = nn.ModuleList()

        for idx, (channels, _) in enumerate(self.levels[-2::-1], start=2):
            level = nn.ModuleList(
                [
                    Collector(
                        self.levels,
                        len(self.levels) - idx,
                        use_attention=self.use_attention_gate,
                        downsampler=self.downsampler,
                        upsampler=self.upsampler,
                    ),
                    RecurrentAttentionBlock(
                        self.block,
                        channels,
                        n_recurrent=self.n_recurrent,
                        use_attention=self.use_channel_attention,
                        reduction=self.reduction,
                        **self.kwargs,
                    ),
                ]
            )
            decoder.append(level)

        return decoder

    def __make_output(self) -> nn.Sequential:
        return nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                self.concat_orig_interp * self.in_channels + self.levels[0][0],
                self.levels[0][0],
                kernel_size=3,
                padding="same",
            ),
            nn.GELU(),
            nn.Conv2d(
                self.levels[0][0],
                self.out_channels,
                kernel_size=3,
                padding="same",
            ),
        )

    def __make_auxiliary(self) -> nn.Sequential:
        return nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                self.levels[0][0],
                self.levels[0][0],
                kernel_size=3,
                padding="same",
            ),
            nn.GELU(),
            nn.Conv2d(
                self.levels[0][0],
                self.out_channels,
                kernel_size=3,
                padding="same",
            ),
        )
