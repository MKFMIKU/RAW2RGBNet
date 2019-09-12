import argparse
import os
from PIL import Image
import utils
import numpy as np

parser = argparse.ArgumentParser(description="Test Script")
parser.add_argument("--output", type=str, required=True, help="path to save output images")
parser.add_argument("--datas", type=str, required=True, help="path to load datas images")

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.output):
    os.makedirs(opt.output)

datas = opt.datas.split(',')
data_images = [utils.load_all_image(data) for data in datas]
[data_image.sort() for data_image in data_images]
for i, p in enumerate(data_images[0]):
    filename = p.split('/')[-1]
    image_paths = [ps[i] for ps in data_images]
    images = [Image.open(ip) for ip in image_paths]
    images_np = [np.asarray(image) for image in images]
    output = np.mean(images_np, axis=0)
    output = output.round()
    output[output >= 255] = 255
    output[output <= 0] = 0
    output = Image.fromarray(output.astype(np.uint8), mode='RGB')
    output.save(os.path.join(opt.output, filename))
