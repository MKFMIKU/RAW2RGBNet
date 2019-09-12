from PIL import Image
import torch.utils.data as data
from os import listdir
from os.path import join
import random
import numpy as np
import torch


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


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def get_patch(*args, patch_size):
    if patch_size == 0:
        return args
    ih, iw = args[0].shape[:2]
    ix = random.randrange(0, iw - patch_size + 1)
    iy = random.randrange(0, ih - patch_size + 1)

    ret = [*[a[iy:iy + patch_size, ix:ix + patch_size, :] for a in args]]

    return ret


def augment(*args, hflip=True, rot=False):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(a) for a in args]


def np2Tensor(*args, rgb_range=1.):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]


class RAW2RGBData(data.Dataset):
    def __init__(self, dataset_dir, patch_size=0, test=False):
        super(RAW2RGBData, self).__init__()
        self.patch_size = patch_size
        self.test = test
        data_dir = join(dataset_dir, "RAW")
        label_dir = join(dataset_dir, "RGB")

        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]

        label_filenames.sort()
        data_filenames.sort()

        # data_filenames = data_filenames[:1200]
        # label_filenames = label_filenames[:1200]

        data_filenames = data_filenames[::200] if test else list(set(data_filenames) - set(data_filenames[::200]))
        label_filenames = label_filenames[::200] if test else list(set(label_filenames) - set(label_filenames[::200]))
        label_filenames.sort()
        data_filenames.sort()

        self.data_filenames = data_filenames
        self.label_filenames = label_filenames

    def __getitem__(self, index):
        data = np.asarray(Image.open(self.data_filenames[index]))
        add_noise(data, 'G1')
        label = np.asarray(Image.open(self.label_filenames[index]))

        data, label = get_patch(data, label, patch_size=self.patch_size)
        if not self.test:
            data, label = augment(data, label)
        data, label = np2Tensor(data, label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)
