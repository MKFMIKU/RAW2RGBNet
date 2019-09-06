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
from os.path import join
from os import listdir
import shutil
import utils

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])

# CUDA_VISIBLE_DEVICES=0 python val.py --model mix3_deep_encoder_decoder --checkpoint ./96.pth --output ~/haoyu/val_results

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument(
    "--model", 
    required=True,
    type=str,
    help="name of model for this training"
)
parser.add_argument("--checkpoint", type=str, required=True, help="path to load model checkpoint")
parser.add_argument("--output", type=str, required=True, help="path to save output images")

dataset_dir = '/home/kangfu/ram_data/RAW2RGB/'
data_dir = join(dataset_dir, "RAW")
label_dir = join(dataset_dir, "RGB")
data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]
label_filenames.sort()
data_filenames.sort()
data_filenames = data_filenames[::200]
label_filenames = label_filenames[::200]
label_filenames.sort()
data_filenames.sort()

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

model = import_module('models.' + opt.model.lower()).make_model(opt)

def infer(im, model, path, gpu=True):
    if gpu:
        model.load_state_dict(torch.load(path)['state_dict_model']) 
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(path, map_location='cpu')['state_dict_model'])
    model = model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    im_augs = [
        transform(im),
    ]

    im_augs = torch.stack(im_augs) # 1, 4, 1512, 2068
    if gpu: im_augs = im_augs.cuda()
    with torch.no_grad():
        output_augs = model(im_augs) # 1, 3, 1512, 2068
    output_augs = np.transpose(output_augs.cpu().numpy(), (0, 2, 3, 1))
    output_augs = [
        output_augs[0],
    ]
    return output_augs[0]

for i, im_path in tqdm(enumerate(data_filenames)):
    filename = im_path.split('/')[-1]

    raw_path = im_path
    rgb_path = label_filenames[i]

    raw_path2 = os.path.join(opt.output, filename[:-4]+'_raw.png')
    rgb_path2 = os.path.join(opt.output, filename[:-4]+'_rgb.jpg')

    print(raw_path, 'to', raw_path2)
    print(rgb_path, 'to', rgb_path2)
    shutil.copy(raw_path,  raw_path2)
    shutil.copy(rgb_path,  rgb_path2)

    img = Image.open(im_path)

    # output = infer(img)
    output_aug = [infer(img, model, opt.checkpoint),]
    output_aug = np.round(np.mean(output_aug, axis=0) * 255.).astype(np.uint8)
    output_aug = np.clip(output_aug, 0, 255)
    imsave(os.path.join(opt.output, filename), output_aug)