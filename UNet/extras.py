import torch
import torchvision
import numpy as np
import random

from plot import *

# CHANGED
def eval_net(net, dataset, criterion, dir, tb_val_writer, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0].astype(np.float32)
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu and torch.cuda.is_available(): # CHANGED
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img) # CHANGED
        tot += criterion(mask_pred, torch.argmax(true_mask, 1).long()).item()
        if i == 0:
            plot_img_and_mask(  img.cpu().squeeze().numpy(),
                                torch.argmax(true_mask, 1).cpu().numpy(),
                                torch.argmax(mask_pred, 1).cpu().squeeze().numpy(),
                                tot, dir,
                                tb_val_writer)
    return tot / (i + 1)

def augmentation(imgs, masks, hflip=True, vflip=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5

    def _augment(img):
         if hflip: img = img[:, :, ::-1]
         if vflip: img = img[:, ::-1, :]
         return img

    _imgs = np.asarray([_augment(img) for img in imgs])
    _masks = np.asarray([_augment(mask) for mask in masks])

    return _imgs, _masks
