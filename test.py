# -*- coding: utf-8 -*-
# File: test.py

import json
import os
from statistics import mean

import torch
from kornia.color import rgb_to_ycbcr
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from tqdm import tqdm

from infer import Upscaler
from utils.file_ops import iter_files
from utils.pytorch_pipeline import PyTorchPipeline

MODEL_PATH = "/Users/anson/Downloads/checkpoint_last.pt"
TEST_DIR = "./data/test"
SCALES = (2, 3, 4)

device = PyTorchPipeline.get_device()
upscaler = Upscaler(MODEL_PATH, device=device)
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
result = {}
for dataset in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, dataset)

    if not os.path.isdir(path):
        continue

    result[dataset] = {s: {"PSNR_RGB": [], "PSNR_Y": [], "SSIM_RGB": [], "SSIM_Y": []} for s in SCALES}
    images = list(iter_files(path, exts={".jpg", ".png"}, case_insensitive=True, recursive=True))
    for f in tqdm(images, leave=False):
        target = to_tensor(Image.open(f).convert(mode="RGB")).unsqueeze(0).to(device)
        target_h, target_w = target.shape[-2:]
        for scale in SCALES:
            img_h, img_w = target_h // scale, target_w // scale
            img = v2.Resize((img_h, img_w), interpolation=InterpolationMode.BICUBIC)(target)
            upscaled = upscaler(img_tensor=img, size=(target_h, target_w))

            upscaled_rgb = upscaled[:, :, 4:-4, 4:-4]
            upscaled_y = rgb_to_ycbcr(upscaled[:, :, 4:-4, 4:-4])[:, :1]
            target_rgb = target[:, :, 4:-4, 4:-4]
            target_y = rgb_to_ycbcr(target[:, :, 4:-4, 4:-4])[:, :1]

            result[dataset][scale]["PSNR_RGB"].append(psnr(upscaled_rgb, target_rgb).item())
            result[dataset][scale]["PSNR_Y"].append(psnr(upscaled_y, target_y).item())
            result[dataset][scale]["SSIM_RGB"].append(ssim(upscaled_rgb, target_rgb).item())
            result[dataset][scale]["SSIM_Y"].append(ssim(upscaled_y, target_y).item())

    for scale in SCALES:
        result[dataset][scale]["PSNR_RGB"] = mean(result[dataset][scale]["PSNR_RGB"])
        result[dataset][scale]["PSNR_Y"] = mean(result[dataset][scale]["PSNR_Y"])
        result[dataset][scale]["SSIM_RGB"] = mean(result[dataset][scale]["SSIM_RGB"])
        result[dataset][scale]["SSIM_Y"] = mean(result[dataset][scale]["SSIM_Y"])

    with open("test-result.json", "w") as f:
        json.dump(result, f, indent=2)
