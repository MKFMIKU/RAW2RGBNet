# coding=utf-8
import argparse
import sys
import os
from importlib import import_module

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data_noise import RAW2RGBData

from tqdm import tqdm

from utils import save_checkpoint, plot_grad_flow, init_weights

parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("--name", required=True, type=str, help="name for training version")
parser.add_argument("--div", type=int, default=88800, help="division of train && test data. Default=88000")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size. Default=64")
parser.add_argument("--threads", type=int, default=8, help="threads for data loader to use. Default=8")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay. Default=1000")
parser.add_argument("--resume", default="", type=str, help="path to checkpoint. Default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number. Default=1")
parser.add_argument("--n-epoch", type=int, default=2000, help="number of epochs to train. Default=2000")
parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda?")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate. Default=1e-4")
parser.add_argument("--size", type=int, default=64, help="size that crop image into")
parser.add_argument(
    "--model",
    required=True,
    type=str,
    help="name of model for this training"
)
parser.add_argument(
    "--data_root",
    required=True,
    type=str,
    help="path to load train datasets"
)
parser.add_argument(
    "--checkpoint",
    required=True,
    type=str,
    help="path to save checkpoints"
)

opts = parser.parse_args()
print(opts)

writer = SummaryWriter(comment=opts.name)
writer.add_text("command", " ".join(sys.argv))
KWAI_SEED = 666
torch.manual_seed(KWAI_SEED)
np.random.seed(KWAI_SEED)


cuda = opts.cuda
cudnn.benchmark = True

train_dataset = RAW2RGBData(opts.data_root, patch_size=opts.size)
test_datasets = RAW2RGBData(opts.data_root, test=True)

training_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=opts.batchSize,
    pin_memory=True,
    shuffle=True,
    num_workers=opts.threads,
)

testing_data_loader = DataLoader(
    dataset=test_datasets,
    batch_size=1,
    num_workers=1,
)

model = import_module('models.' + opts.model.lower()).make_model(opts)
model_define_r = open(os.path.join("models", opts.model.lower() + ".py"), 'r')
model_define = model_define_r.read()
writer.add_text("models", model_define)
model_define_r.close()
criterion = nn.L1Loss()

# init_weights(model, 'orthogonal')

if opts.resume:
    if os.path.isfile(opts.resume):
        print("======> loading checkpoint at '{}'".format(opts.resume))
        checkpoint = torch.load(opts.resume)
        model.load_state_dict(checkpoint["state_dict_model"], strict=False)
    else:
        print("======> founding no checkpoint at '{}'".format(opts.resume))

if cuda:
    model = nn.DataParallel(model).cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opts.lr, betas=(0.9, 0.999))
# optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999))
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.decay_epoch, gamma=0.1)

for epoch in range(opts.start_epoch, opts.n_epoch + 1):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    pbar = tqdm(training_data_loader)
    output = None
    for iteration, batch in enumerate(pbar):
        data, label = batch[0], batch[1]
        data = data.cuda() if opts.cuda else data.cpu()
        label = label.cuda() if opts.cuda else label.cpu()

        model.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            pbar.set_description("Epoch[{}]({}/{}): Loss: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss.item())
            )
            writer.add_scalar("l1loss", loss.item(), iteration+(epoch-1)*len(training_data_loader))
    lr_scheduler.step(epoch=epoch)
    writer.add_image("output", make_grid(output, range=[0., 1.]), epoch)
    save_checkpoint(model, opts.name, None, epoch, opts.checkpoint)
    if epoch % 1 == 0:
        mean_psnr = 0
        model.eval()
        for iteration, batch in enumerate(testing_data_loader, 1):
            data, label = batch[0], batch[1]
            data = data.cuda() if opts.cuda else data.cpu()
            label = label.cuda() if opts.cuda else label.cpu()

            with torch.no_grad():
                output = model(data)
            output = torch.clamp(output, 0.0, 1.0)
            mse = F.mse_loss(output, label)
            psnr = 10 * np.log10(1.0 / mse.item())
            mean_psnr += psnr
        mean_psnr /= len(testing_data_loader)
        writer.add_scalar("mean_psnr", mean_psnr, epoch)
        print("Vaild  epoch %d psnr: %f" % (epoch, mean_psnr))
writer.close()
