# -*- coding: utf-8 -*-
# File: test.py

import argparse
import json
import os
from argparse import Namespace
from statistics import mean
from typing import Optional

import torch
from kornia.color import rgb_to_ycbcr
from PIL import Image
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from tqdm import tqdm

from infer import Upscaler
from utils.file_ops import iter_files
from utils.pytorch_pipeline import PyTorchPipeline


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to AASR model (.pt).")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("result_json", type=str)
    parser.add_argument("--test_dir", type=str, default="./data/test")
    parser.add_argument("--scales", nargs="+", type=float, default=(2, 3, 4))
    parser.add_argument("--border", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def test(
    model_path: str,
    output_dir: str,
    result_json: str,
    test_dir: str = "./data/test",
    scales: list[float] = [2, 3, 4],
    border: Optional[int] = 4,
    save: bool = False,
) -> None:
    if border is not None and border <= 0:
        raise ValueError("`border` should be either None or an integer greater than 0.")

    os.makedirs(output_dir, exist_ok=True)
    if save:
        upscaled_dir = os.path.join(output_dir, "upscaled_images")
        os.makedirs(upscaled_dir, exist_ok=True)

    device = PyTorchPipeline.get_device()
    upscaler = Upscaler(model_path, device=device)
    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    to_pil = v2.ToPILImage()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    all_psnr_rgb, all_psnr_y, all_ssim_rgb, all_ssim_y = [], [], [], []
    result = {"mean": {"PSNR_RGB": 0, "PSNR_Y": 0, "SSIM_RGB": 0, "SSIM_Y": 0}}
    for dataset in tqdm(os.listdir(test_dir), desc="Testing"):
        path = os.path.join(test_dir, dataset)
        if not os.path.isdir(path):
            continue

        result[dataset] = {s: {"PSNR_RGB": [], "PSNR_Y": [], "SSIM_RGB": [], "SSIM_Y": []} for s in scales}
        images = list(iter_files(path, exts={".jpg", ".png"}, case_insensitive=True, recursive=True))
        for f in tqdm(images, desc=dataset, leave=False):
            target: Tensor = to_tensor(Image.open(f).convert(mode="RGB"))
            target = target.unsqueeze(0).to(device)
            target_h, target_w = target.shape[-2:]
            target_rgb = target if border is None else target[:, :, border:-border, border:-border]
            for scale in scales:
                img_h, img_w = target_h // scale, target_w // scale
                img = v2.Resize((img_h, img_w), interpolation=InterpolationMode.BICUBIC, antialias=True)(target)
                upscaled = upscaler(img_tensor=img, size=(target_h, target_w))

                if save:
                    final_dir = os.path.join(upscaled_dir, dataset, f"X{scale}")
                    os.makedirs(final_dir, exist_ok=True)
                    to_pil(upscaled.squeeze()).save(os.path.join(final_dir, f.name))

                upscaled_rgb = upscaled if border is None else upscaled[:, :, border:-border, border:-border]
                upscaled_y = rgb_to_ycbcr(upscaled)[:, :1]
                target_y = rgb_to_ycbcr(target)[:, :1]

                result[dataset][scale]["PSNR_RGB"].append(psnr(upscaled_rgb, target_rgb).item())
                result[dataset][scale]["PSNR_Y"].append(psnr(upscaled_y, target_y).item())
                result[dataset][scale]["SSIM_RGB"].append(ssim(upscaled_rgb, target_rgb).item())
                result[dataset][scale]["SSIM_Y"].append(ssim(upscaled_y, target_y).item())

        for scale in scales:
            psnr_rgb = mean(result[dataset][scale]["PSNR_RGB"])
            result[dataset][scale]["PSNR_RGB"] = psnr_rgb
            all_psnr_rgb.append(psnr_rgb)

            psnr_y = mean(result[dataset][scale]["PSNR_Y"])
            result[dataset][scale]["PSNR_Y"] = psnr_y
            all_psnr_y.append(psnr_y)

            ssim_rgb = mean(result[dataset][scale]["SSIM_RGB"])
            result[dataset][scale]["SSIM_RGB"] = ssim_rgb
            all_ssim_rgb.append(ssim_rgb)

            ssim_y = mean(result[dataset][scale]["SSIM_Y"])
            result[dataset][scale]["SSIM_Y"] = ssim_y
            all_ssim_y.append(ssim_y)

        result["mean"]["PSNR_RGB"] = mean(all_psnr_rgb)
        result["mean"]["PSNR_Y"] = mean(all_psnr_y)
        result["mean"]["SSIM_RGB"] = mean(all_ssim_rgb)
        result["mean"]["SSIM_Y"] = mean(all_ssim_y)

        with open(os.path.join(output_dir, result_json), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    test(
        args.model_path,
        args.output_dir,
        args.result_json,
        test_dir=args.test_dir,
        scales=args.scales,
        border=args.border,
        save=args.save,
    )
