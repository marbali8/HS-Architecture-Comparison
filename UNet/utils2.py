from utils1 import *
from init import *

import os
import numpy as np

# CHANGED lo de dins i el nom pq necessito un per tallar amb .npy
def get_ids_npy(dir_img, dir_masks):
    """Returns a list of the ids"""
    images = [f.split('.')[0] for f in os.listdir(dir_img) if '.npy' in f]
    masks = [f.split('_mask.')[0] for f in os.listdir(dir_masks) if '.npy' in f]
    return (f for f in list(set(images).intersection(set(masks))))

# CHANGED adaptat a npy
def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for i, id in enumerate(ids): #for id, pos in ids:

        f = np.load(dir + id + suffix)
        yield f #get_square(im, pos) CHANGED

# CHANGED per adaptar a npy
def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.npy') # CHANGED to npy

    imgs_normalized = map(normalize, imgs)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.npy') # CHANGED to npy

    #if [i for i in masks if i.shape[2] != net_classes] != []:
        #masks = test_random_masks(ids, dir_mask, '_mask.npy', scale, net_classes)

    #assert [i for i in maskss if i.shape[2] != net_classes] == [], "wrong number of classes"

    return zip(imgs_normalized, masks) # ajunta en una tupla cada img amb la mask

# gets the weights in the format for loss or accuracy
def get_weights(type, loss_mask=None, im = ""):
    weights = [f for f in os.listdir(dir_mask + "npy/") if im in f and "_class_weights" in f]
    w = np.load(dir_mask + "npy/" + weights[0])

    assert type == 'train' or type == 'test'
    if type == 'train':
        if float("Inf") in np.unique(w):
            for i, v in enumerate(w):
                 w[i] = 0 if v == float("Inf") else v
        return w
    else:
        assert loss_mask.any() != None
        new_w = np.zeros(loss_mask.shape)
        for i, p in np.ndenumerate(loss_mask):
            new_w[i[0], i[1]] = 0 if w[int(p)] == float("Inf") else w[int(p)]
        return new_w
