import torch
import torchvision
import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score

from plot import *
from utils2 import *
from init import *

# NEW
def balanced_score(target, prediction):

    t = target.flatten()
    p = prediction.flatten()
    w = get_weights('test', target).flatten()

    assert t.shape == p.shape == w.shape, str(target.shape) + str(prediction.shape) + str(weights.shape)
    indexes = np.array([])
    for i, v in enumerate(t):
        if v == -1:
            indexes = np.append(indexes, i)
    if indexes.size == t.size:
        return 0
    t = np.delete(t, indexes)
    p = np.delete(p, indexes)
    w = np.delete(w, indexes)

    return balanced_accuracy_score(t, p, w)

# CHANGED
def eval_net(net, val, dir, tb_val_writer, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(val):
        img = b[0].astype(np.float32)
        true_mask = b[1]
        train_mask = loss_mask(true_mask, type = 'test')

        img = torch.from_numpy(img).unsqueeze(0)

        if gpu and torch.cuda.is_available(): # CHANGED
            img = img.cuda()

        mask_pred = net(img) # CHANGED
        mask_pred = torch.argmax(mask_pred, 1).squeeze()
        mask_pred = mask_pred.data.cpu().detach().numpy()
        tot += balanced_score(train_mask, mask_pred)
        # tot += criterion(mask_pred, torch.argmax(true_mask, 1).long()).item()
        if i == 0:
            plot_img_and_mask(  img.cpu().squeeze().numpy(),
                                plot_mask(true_mask),
                                mask_pred,
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

# gets the mask for loss or accuracy
def loss_mask(balanced_mask, type):

    assert type == 'train' or type == 'test'
    considered_value = 1 if type == 'train' else -1

    loss_mask = np.zeros((balanced_mask.shape[1], balanced_mask.shape[2]))
    _balanced_mask = np.transpose(balanced_mask, axes=[1,2,0]) #hwc
    for r, row in enumerate(_balanced_mask):
        for c, col in enumerate(row):
            assert col.shape == (balanced_mask.shape[0],), '(' + str(r) + ',' + str(c) + ')'
            con = np.where(col == considered_value)[0]
            loss_mask[r, c] = con[0] if con.size != 0 else -1
    return loss_mask.astype(np.int)

# asi te devuelve la classe o menos 1 si no esta etiquetado, en este caso el type
# importa para lo del considered_value en loss_mask (podriamos poner test pero entonces
# le pasariamos -abs(balanced_mask))
def plot_mask(balanced_mask, type='train'):
    return loss_mask(np.squeeze(abs(balanced_mask)), type)
