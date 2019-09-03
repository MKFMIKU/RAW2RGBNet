from PIL import Image
import torch.utils.data as data
from os import listdir
from os.path import join
import random
from torchvision import transforms


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


class RAW2RGBData(data.Dataset):
    def __init__(self, dataset_dir, div=None, transform=None, test=False):
        super(RAW2RGBData, self).__init__()
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

        self.transform = transform

    def __getitem__(self, index):
        data = Image.open(self.data_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.transform:
            seed = random.randint(0, 2 ** 32)

            random.seed(seed)
            data = self.transform(data)
            random.seed(seed)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)
