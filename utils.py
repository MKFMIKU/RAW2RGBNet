import os
from os import listdir
from os.path import join
from os.path import exists
import torch
import random
from torch.autograd import Variable
from torch.nn import init


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.tif'])


def load_all_image(path):
    return [join(path, x) for x in listdir(path) if is_image_file(x)]


def save_checkpoint(model, discriminator, epoch, model_folder):
    if not exists(model_folder):
        os.makedirs(model_folder)
    model_out_path = "%s/%d.pth" % (model_folder, epoch)

    state_dict_model = model.module.state_dict()

    for key in state_dict_model.keys():
        state_dict_model[key] = state_dict_model[key].cpu()

    if discriminator:
        state_dict_discriminator = discriminator.module.state_dict()
        for key in state_dict_discriminator.keys():
            state_dict_discriminator[key] = state_dict_discriminator[key].cpu()

        torch.save({"epoch": epoch,
                    "state_dict_model": state_dict_model,
                    "state_dict_discriminator": state_dict_discriminator}, model_out_path)
    else:
        torch.save({"epoch": epoch,
                    "state_dict_model": state_dict_model}, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
