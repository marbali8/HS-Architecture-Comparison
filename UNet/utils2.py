from utils1 import *

import os
import numpy as np

# CHANGED lo de dins i el nom pq necessito un per tallar amb .tif
def get_ids_npy(dir_img, dir_masks):
    """Returns a list of the ids"""
    images = [f.split('.')[0] for f in os.listdir(dir_img) if '.npy' in f]
    masks = [f.split('_mask.')[0] for f in os.listdir(dir_masks) if '.npy' in f]
    return (f for f in list(set(images).intersection(set(masks))))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    """Fa això perquè parteix cada imatge (id) en 2 (n=2)"""
    return ((di, u) for di in ids for u in range(n))

# CHANGED adaptat a npy
def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids: #for id, pos in ids:

        f = np.load(dir + id + suffix)

        #im = resize_and_crop(f, scale=scale)
        #if first:
          #print("img after resize_and_crop", im.shape) # formato HWC
          #first = False
        yield f #get_square(im, pos) CHANGED

# NEW
#def test_random_masks(ids, dir, suffix, scale, net_classes):
    #for id, pos in ids:
        #f = np.load(dir + id + suffix)

        #f = np.random.randint(2, size=(f.shape[0], f.shape[1], net_classes))

        #im = resize_and_crop(f, scale=scale)
        #yield get_square(im, pos)

# CHANGED per adaptar a npy
def get_imgs_and_masks(ids, dir_img, dir_mask, scale, net_channels, net_classes):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.npy', scale) # CHANGED to npy

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.npy', scale) # CHANGED to npy

    #if [i for i in masks if i.shape[2] != net_classes] != []:
        #masks = test_random_masks(ids, dir_mask, '_mask.npy', scale, net_classes)

    #assert [i for i in maskss if i.shape[2] != net_classes] == [], "wrong number of classes"

    return zip(imgs_normalized, masks) # ajunta en una tupla cada img amb la mask


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = np.load(dir_img + id + '.npy') # CHANGED to npy
    mask = np.load(dir_mask + id + '_mask.npy') # CHANGED to npy
    return np.array(im), np.array(mask)
