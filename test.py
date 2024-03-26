# -*- coding: utf-8 -*-
# File: test.py

import os
from statistics import mean

import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from tqdm import tqdm

from infer import Upscaler
from utils.file_ops import iter_files

MODEL_PATH = "/Users/anson/Downloads/checkpoint_last.pt"
TEST_DIR = "./data/test"
SCALES = (2, 3, 4)

upscaler = Upscaler(MODEL_PATH)
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
psnr = PeakSignalNoiseRatio(data_range=1.0)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
result = {}
for dataset in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, dataset)
    if os.path.isdir(path):
        result[dataset] = {s: ([], []) for s in SCALES}
        images = list(iter_files(path, exts={".jpg", ".png"}, case_insensitive=True, recursive=True))
        for f in tqdm(images, leave=False):
            target = to_tensor(Image.open(f).convert(mode="RGB")).unsqueeze(0)
            target_h, target_w = target.shape[-2:]
            for scale in SCALES:
                img_h, img_w = target_h // scale, target_w // scale
                img = v2.Resize((img_h, img_w), interpolation=InterpolationMode.BICUBIC)(target)
                upscaled = upscaler(img_tensor=img, size=(target_h, target_w))
                result[dataset][scale][0].append(psnr(target, upscaled).item())
                result[dataset][scale][1].append(ssim(target, upscaled).item())

    for scale in SCALES:
        result[dataset][scale][0] = mean(result[dataset][scale][0])
        result[dataset][scale][1] = mean(result[dataset][scale][1])

print(result)
