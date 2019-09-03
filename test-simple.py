import argparse
import os
from importlib import import_module

from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms.functional as F
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

def infer_(im, model, k=0):
    torch.cuda.empty_cache()
    model.eval()

    to_pil = transforms.ToPILImage()
    im = to_pil(im)

    h, w = im.size[0], im.size[1]
    p1, p2 = (4 - h % 4) % 4, (4 - w % 4) % 4

    transform = transforms.Compose([
        transforms.Pad((p1, p2, 0, 0), padding_mode='edge'), ### left, top, right and bottom
        transforms.ToTensor()
    ])

    if k == 0:
        im_augs = [
            transform(im),
        ]
    elif k == 1:
        im_augs = [
            transform(F.hflip(im)),
        ]
    elif k == 2:
        im_augs = [
            transform(F.vflip(im)),
        ]   
    elif k == 3:
        im_augs = [
            transform(F.hflip(F.vflip(im))),
        ]              
    
    im_augs = torch.stack(im_augs)   
    im_augs = im_augs.cuda()
    with torch.no_grad():
        output_augs = model(im_augs) 
    output_augs = output_augs[:, :, p2:, p1:]
    return output_augs

def infer(im, model, path):
    model.load_state_dict(torch.load(path)['state_dict_model']) 
    model = model.cuda()
    model = model.eval()

    to_tensor = transforms.ToTensor()
    im = to_tensor(im) # 4, 1509, 2065
    h, w = im.shape[-2], im.shape[-1]
    r = w // 2

    im_xy = [(0, r), (r//2, r//2+r), (r, w)] # (0, 1032), (516, 1548), (1032, 2065)
    output_augs = torch.Tensor()
    for k in range(1):
        outs = []
        for x, y in im_xy:
            inp = im[:, :, x:y]
            torch.cuda.empty_cache()
            outs.append(infer_(inp, model, k=k) ) # 1, 3, 1509, 1032
            torch.cuda.empty_cache()
        feature = torch.cat((outs[0], outs[2]), dim=-1) #  1, 3, 1509, 2065
        feature[:, :, :, im_xy[1][0]:im_xy[1][1]] = (feature[:, :, :, im_xy[1][0]:im_xy[1][1]] + outs[1]) / 2 # 1, 3, 1509, 1032
        feature = feature.cpu()
        output_augs = torch.cat((output_augs, feature), dim=0)
        del feature
        gc.collect()
        torch.cuda.empty_cache()

    output_augs = np.transpose(output_augs.cpu().numpy(), (0, 2, 3, 1))
    # output_augs = [
    #     output_augs[0],
    #     np.fliplr(output_augs[1]),
    #     np.flipud(output_augs[2]),
    #     np.fliplr(np.flipud(output_augs[3]))
    # ]
    # return np.mean(output_augs, axis=0)
    return output_augs[0]

images = utils.load_all_image(opt.data)
images.sort()

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    img = Image.open(im_path)

    # output = infer(img)
    # c1 = '/data1/kangfu/Checkpoints/RAW2RGB/_mix_ednet_light_data_shuttle_32_8_10_128/54.pth'
    # c2 = '/data1/kangfu/Checkpoints/RAW2RGB/_mix_ednet_light_increase_32_8_12_128/42.pth'
    # c3 = '/data1/kangfu/Checkpoints/RAW2RGB/_mix_ednet_light_increase_32_8_12_128/43.pth'
    output_aug = [infer(img, model, opt.checkpoint),
                  # infer(img, model, '/data1/kangfu/Checkpoints/RAW2RGB/_mix_ednet_light_data_shuttle_32_10_12_128_futher/81.pth'),
                  # infer(img, model, c3)
                  ]
    output_aug = np.round(np.mean(output_aug, axis=0) * 255.).astype(np.uint8)
    output_aug = np.clip(output_aug, 0, 255)
    imsave(os.path.join(opt.output, filename), output_aug)
