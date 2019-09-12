#!/usr/bin/env python
import argparse
import utils
from PIL import Image
import numpy as np
from os.path import join
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--data", type=str, default="output", help="path to load data images")
parser.add_argument("--gt", type=str, help="path to load gt images")
parser.add_argument("--view", type=str, help="path to save data gt images")

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.view):
    os.makedirs(opt.view)

datas = utils.load_all_image(opt.data)
datas.sort()


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


psnrs = []
for data_p in tqdm(datas):
    data = Image.open(data_p)
    gt = Image.open(join(opt.gt, data_p.split('/')[-1][:-3]+'jpg'))
    w, h = data.size
    new_im = Image.new('RGB', (w * 2, h))
    new_im.paste(data, (0, 0))
    new_im.paste(gt, (w, 0))
    new_im.save(os.path.join(opt.view, data_p.split('/')[-1]))

    data = np.asarray(data).astype(float) / 255.0
    gt = np.asarray(gt).astype(float) / 255.0
    psnr = output_psnr_mse(data, gt)
    psnrs.append(psnr)
print("mean PSNR:", np.mean(psnrs))
