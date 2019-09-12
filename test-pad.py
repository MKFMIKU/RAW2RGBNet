import argparse
import os
from importlib import import_module

from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
from skimage.io import imsave

import utils
import gc

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="name of model for this training"
)
parser.add_argument("--checkpoint", type=str, required=True, help="path to load model checkpoint")
parser.add_argument("--output", type=str, required=True, help="path to save output images")
parser.add_argument("--data", type=str, required=True, help="path to load data images")

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

model = import_module('models.' + opt.model.lower()).make_model(opt)
model.load_state_dict(torch.load(opt.checkpoint)['state_dict_model'])
model = model.cuda()
model = model.eval()

images = utils.load_all_image(opt.data)
images.sort()


def add_noise(x, noise='.'):
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def infer(im):
    w, h = im.size
    pad_w = 8 - w % 8
    pad_h = 8 - h % 8
    padding = 100

    im_pad = transforms.Pad(padding=(pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2), padding_mode='reflect')(im)
    im_pad = np.asarray(im_pad)
    im_pad = add_noise(im_pad, 'G1')
    im_pad_th = transforms.ToTensor()(im_pad)
    im_pad_th = im_pad_th.unsqueeze(0).cuda()
    _, _, _, ww = im_pad_th.shape
    im_pad_th_l, im_pad_th_r = im_pad_th[:, :, :, :ww//2 + padding], im_pad_th[:, :, :, ww//2-padding:]
    with torch.no_grad():
        torch.cuda.empty_cache()
        im_pad_th_l = model(im_pad_th_l)
        torch.cuda.empty_cache()
        im_pad_th_r = model(im_pad_th_r)
    pad_th = (im_pad_th_l[:, :, :, -padding * 2:] + im_pad_th_r[:, :, :, :padding * 2]) / 2
    output = torch.cat((im_pad_th_l[:, :, :, :-padding*2], pad_th, im_pad_th_r[:, :, :, padding*2:]), dim=-1)
    output = output.squeeze(0).cpu()
    output = torch.clamp(output, 0., 1.)
    output = transforms.ToPILImage()(output)
    return output


for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    img = Image.open(im_path)
    output = infer(img)
    output.save(os.path.join(opt.output, filename))
