import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from random import shuffle
from os import listdir
from os.path import join
import numpy as np


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


class RAW2RGBInputIterator(object):
    def __init__(self, dataset_dir, batch_size, div=88000, test=False):
        self.batch_size = batch_size
        data_dir = join(dataset_dir, "RAW")
        label_dir = join(dataset_dir, "RGB")

        data_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        label_filenames = [join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)]

        data_filenames.sort()
        label_filenames.sort()

        data_filenames = data_filenames[div:] if test else data_filenames[:div]
        label_filenames = label_filenames[div:] if test else label_filenames[:div]

        data_label_filenames = list(zip(data_filenames, label_filenames))
        shuffle(data_label_filenames)
        data_filenames, label_filenames = zip(*data_label_filenames)

        self.data_filenames = data_filenames
        self.label_filename = label_filenames

    def __iter__(self):
        self.i = 0
        self.n = len(self.data_filenames)
        return self

    def __next__(self):
        batch_data = []
        batch_label = []
        for _ in range(self.batch_size):
            data_path = self.data_filenames[self.i]
            label_path = self.label_filename[self.i]
            f_data = open(data_path, 'rb')
            f_label = open(label_path, 'rb')
            batch_data.append(np.frombuffer(f_data.read(), dtype=np.uint8))
            batch_label.append(np.frombuffer(f_label.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return batch_data, batch_label

    next = __next__


class HybridTrainPipe(Pipeline):
    def __init__(self, dataset_dir, batch_size, num_threads, device_id, crop, dali_cpu=False, local_rank=0, test=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=666)
        self.raw2rgbit = iter(RAW2RGBInputIterator(dataset_dir, batch_size, test=test))
        dali_device = "gpu"
        self.input_data = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.data_decode = ops.ImageDecoder(device="mixed")
        self.label_decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW)
        self.coin = ops.CoinFlip(probability=0.5)

    def iter_setup(self):
        data, label = self.raw2rgbit.next()
        self.feed_input(self.data, data)
        self.feed_input(self.label, label)

    def define_graph(self):
        rng = self.coin()
        self.data = self.input_data()
        self.label = self.input_label()
        data_im = self.data_decode(self.data)
        label_im = self.label_decode(self.label)
        return data_im, label_im
