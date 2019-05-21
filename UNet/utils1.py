import random
import numpy as np


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]) # si abans era shape (a,b,c) ara és (c,a,b)

def resize_and_crop(img, final_height=None):
    w = img.shape[1] # CHANGED from size to shape bc in npy size is a number
    h = img.shape[0] # CHANGED from size to shape bc in npy size is a number
    newW = int(w * 1)
    newH = int(h * 1)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height
    img = np.resize(img, (newH, newW, img.shape[2])) # CHANGED from PIL resize to npy resize
    img = img[diff // 2 : diff // 2 + newH - diff // 2, 0: newW, :]
    return np.array(img, dtype=np.float32)

# de batch_size en batch_size, va tornant les tuples de (img, mask)
def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable): # i de 0 a #imatges-1
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

# CHANGED els range i dimensions
def merge_masks(img1, img2, full_w):
    h = img1.shape[1] #chw
    #print(img1.shape, img2.shape, full_w, h)
    new = np.zeros((img1.shape[0], h, full_w), np.float32)
    new[:, :, : full_w // 2 + 1] = img1[:, :, :full_w // 2 + 1]
    new[:, :, -(full_w // 2 + 1):] = img2[:, :, -(full_w // 2 + 1):]

    return new


    #h = img1.shape[1] #chw
    #range = min(h, full_w // 2)
    #print(img1.shape, img2.shape, full_w, range)
    #new = np.zeros((h, full_w), np.float32)
    #new[:, : range - 1] = img1[:, : range - 1]
    #new[:, range - 1 : 2*range - 1] = img2[:, range - 1 : 2*range - 1]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
