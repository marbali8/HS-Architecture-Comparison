from init import *

import rasterio
import os
import math
import numpy as np
from matplotlib import pyplot as plt

def isPower(num, of = 2, inc = 0):
    n = of**inc
    if n < num:
        return isPower(num, inc = inc + 1)
    elif n == num:
        return True
    else:
        return False

# NEW from get_ids_npy
def get_ids_tif(dir_img, dir_masks):
    """Returns a list of the ids"""
    images = [f.split('.')[0] for f in os.listdir(dir_img) if '.tif' in f]
    masks = [f.split('_mask.')[0] for f in os.listdir(dir_masks) if '_mask.tif' in f]
    return (f for f in list(set(images).intersection(set(masks))))

# based on https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
def weights_and_samples_balanced(mask, val_percent=VAL_PERCENT):

    samples_per_class = [(mask == i).sum() for i in range(NET_CLASSES)]

    N = sum(samples_per_class) # N pixels
    nsamples = math.floor(N*(1-val_percent))

    weight_per_class = samples_per_class/N
    samples_per_class = np.floor(weight_per_class*nsamples).astype(int)

    return weight_per_class, samples_per_class

# vull masks per cada imatge que poden portar les weights
def new_split_train_val(image, mask, val_percent=VAL_PERCENT):

    assert len(mask.shape) == 2
    weights, max_samples = weights_and_samples_balanced(mask, val_percent)
    assert len(max_samples) == NET_CLASSES
    train_mask = np.zeros((NET_CLASSES, mask.shape[0], mask.shape[1]), dtype=int)

    indexes = [(x, y) for (x, y), _ in np.ndenumerate(mask)]
    np.random.shuffle(indexes)
    labeled = [0] * NET_CLASSES

    # (NET_CLASSES, MASK_H, MASK_W) but now has values of 1 (pixel of that class),
    # 0 (pixel of other class) or -1 (pixel of that class not considered
    # for training, just for testing)
    for i in indexes:
        label = mask[i[0], i[1]]
        if labeled[label] < max_samples[label]:
            train_mask[label, i[0], i[1]] = 1
            labeled[label] += 1
        else:
            train_mask[label, i[0], i[1]] = -1

    return train_mask, weights

# NEW per pasar de tif a npy tallades
def doit():

    assert isPower(WIDTH_CUTS)
    f_width = WIDTH_CUTS
    f_height = f_width

    if not os.path.exists(dir_img + "npy/"):
        os.makedirs(dir_img + "npy/")
    if not os.path.exists(dir_mask + "npy/"):
        os.makedirs(dir_mask + "npy/")

    for c,im in enumerate(get_ids_tif(dir_img, dir_mask)):

        im_path = dir_img + im + '.tif'
        ma_path = dir_mask + im + '_mask.tif'
        image = rasterio.open(im_path)
        mask = rasterio.open(ma_path)
        assert image is not None, "image not found"
        cols = image.width
        rows = image.height

        assert image.count == INPUT_BANDS, "check the number of bands"
        assert mask.count == 1, "check the number of classes"

        train_mask, weights = new_split_train_val(image.read(), mask.read(1))

        print("Started cutting image", c+1, "/", len(list(get_ids_tif(dir_img, dir_mask))))
        num = 1
        for j in range(0, cols - f_width + 1, COL_STRIDE):
            for i in range(0, rows - f_height + 1, ROW_STRIDE):

                #num = j*n_row+i+1
                image_array = np.zeros((image.count, f_height, f_width)) # CHW

                # image
                for b in range(image.count):
                    band = image.read(b+1)
                    # assert band != None
                    image_array[b,:,:] = band[i : i + f_height, j : j + f_width]

                # mask
                mask_array = train_mask[:, i : i + f_height, j : j + f_width]

                #if j == 0:
                  #img_array = image_array[:,:,50:53]/np.amax(image_array[:,:,50:53])
                  #plt.imshow(img_array)
                  #plt.show()

                np.save(dir_img + "npy/" + im + '_' + str(num), image_array)
                np.save(dir_mask + "npy/" + im + '_' + str(num) + '_mask', mask_array)

                if num % 10 == 0:
                    print(str(num), end=' ', flush = True)

                num = num + 1

        print(flush = False)
        np.save(dir_mask + "npy/" + im + "_class_weights", weights)
        print("Finished cutting image", c+1, "(", num, "subimages saved )")

if __name__ == "__main__":
    doit()
