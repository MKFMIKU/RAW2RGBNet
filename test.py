import argparse
import os
from importlib import import_module

from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

import utils

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
model = model.eval()
model = model.cuda()


def infer(im):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_back = transforms.Compose([
        transforms.ToPILImage(),
    ])
    im = transform(im)
    im = im.unsqueeze(0)
    im = im.cuda()

    with torch.no_grad():
        output = model(im)
    output = output.cpu().data[0]
    output = torch.clamp(output, 0, 1)
    output = transform_back(output)
    return output


images = utils.load_all_image(opt.data)
images.sort()

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    img = Image.open(im_path)
    output = infer(img)
    output.save(os.path.join(opt.output, filename))
