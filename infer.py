# -*- coding: utf-8 -*-
# File: infer.py

import argparse
from typing import Optional

import torch
from PIL import Image
from torchvision.transforms import v2

from model import AASR
from utils.pytorch_pipeline import PyTorchPipeline


def upscale(
    model_path: str,
    img_path: str,
    size: Optional[int | tuple[int, int]] = None,
    scale: float | tuple[float, float] = 1.0,
) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    model = AASR(**checkpoint["configs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    img = to_tensor(Image.open(img_path)).unsqueeze(0)

    pred, _ = model(img, scale=scale) if size is None else model(img, size=size)

    return v2.ToPILImage()(pred.squeeze())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Path to AASR model (.pt).",
    )
    parser.add_argument(
        "source",
        type=str,
        help="Path to source image or video.",
    )
    parser.add_argument(
        "-size",
        "--size",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Output size (width, height). "
            "If one value is provided, the output size will be (`size`, `size`). "
            "If multiple values are provided, the output size will be (`size[0]`, `size[1]`). "
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "-scale",
        "--scale",
        nargs="+",
        type=float,
        default=1.0,
        help=(
            "Factor by which the original size will be multiplied. "
            "If one value is provided, both sides will be multiplied by `scale`. "
            "If multiple values are provided, the original width will be multiplied by `scale[0]` "
            "while the original height will be multiplied by `scale[1]`. "
            "Defaults to 1."
        ),
    )
    args = parser.parse_args()

    size = None if args.size is None else args.size[0] if len(args.size) == 1 else tuple(args.size[1::-1])
    scale = args.scale[0] if len(args.scale) == 1 else tuple(args.scale[1::-1])
    upscaled = upscale(args.model, args.source, size, scale)
    upscaled.save(f"upscaled_{PyTorchPipeline.get_datetime()}.png")
