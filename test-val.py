import argparse
import os
from importlib import import_module
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import utils
import numpy as np
import torchvision.transforms.functional as F

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
images = images[::200]


def infer(im):
    to_tensor = transforms.ToTensor()
    im_augs = [
        to_tensor(im),
        to_tensor(F.hflip(im)),
        to_tensor(F.vflip(im)),
        to_tensor(F.hflip(F.vflip(im))),
    ]
    im_augs = torch.stack(im_augs)
    im_augs = im_augs.cuda()
    with torch.no_grad():
        output_augs = model(im_augs)
    output_augs = np.transpose(output_augs.cpu().numpy(), (0, 2, 3, 1))
    output_augs = [
        output_augs[0],
        np.fliplr(output_augs[1]),
        np.flipud(output_augs[2]),
        np.fliplr(np.flipud(output_augs[3])),
    ]
    output = np.mean(output_augs, axis=0) * 255.
    output = output.round()
    output[output >= 255] = 255
    output[output <= 0] = 0
    output = Image.fromarray(output.astype(np.uint8), mode='RGB')
    return output


for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    img = Image.open(im_path)
    img = infer(img)
    img.save(os.path.join(opt.output, filename))
