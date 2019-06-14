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
    images = [('recorte1.npy').split('.')[0]]
    masks = [f.split('_mask.')[0] for f in os.listdir(dir_masks) if '_mask.tif' in f]
    return (f for f in list(set(images).intersection(set(masks))))

# based on https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
def weights_and_samples_balanced(mask):

    samples_per_class = [(mask == i).sum() for i in range(NET_CLASSES)]

    N = sum(samples_per_class) # N pixels

    weight_per_class = N/samples_per_class

    return weight_per_class, samples_per_class

# vull masks per cada imatge que poden portar les weights
def new_split_train_val(image, mask, val_percent=VAL_PERCENT):

    assert len(mask.shape) == 2
    train_mask = np.zeros((NET_CLASSES, mask.shape[0], mask.shape[1]), dtype=int)

    indexes_t = os.listdir('/work-nfs/mbalibrea/data/TRAIN/maspalomas/ROIs/Entrenamiento/1')
    indexes_v = os.listdir('/work-nfs/mbalibrea/data/TRAIN/maspalomas/ROIs/Evaluación')

    # (NET_CLASSES, MASK_H, MASK_W) but now has values of 1 (pixel of that class),
    # 0 (pixel of other class) or -1 (pixel of that class not considered
    # for training, just for testing)
    for i in indexes_t:
        indexes = np.load('/work-nfs/mbalibrea/data/TRAIN/maspalomas/ROIs/Entrenamiento/1/'+i)
        label = int(i.split('.')[0])

        for index in indexes:
            train_mask[label, index[0], index[1]] = 1

    for i in indexes_v:
        indexes = np.load('/work-nfs/mbalibrea/data/TRAIN/maspalomas/ROIs/Evaluación/'+i)
        label = int(i.split('.')[0])

        for index in indexes:
            train_mask[label, index[0], index[1]] = -1


    # hago abs() porque solo lo quiero para calcular las weights
    weights, max_samples = weights_and_samples_balanced(abs(train_mask))
    assert len(max_samples) == NET_CLASSES

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

        im_path = dir_img + im + '.npy'
        ma_path = dir_mask + im + '_mask.tif'
        image = np.load(im_path)
        mask = rasterio.open(ma_path)
        assert image is not None, "image not found"
        cols = mask.width
        rows = mask.height

        #assert image.count == INPUT_BANDS, "check the number of bands"
        assert mask.count == 1, "check the number of classes"

        train_mask, weights = new_split_train_val(image, mask.read(1))

        print("Started cutting image", c+1, "/", len(list(get_ids_tif(dir_img, dir_mask))))
        num = 1
        for j in range(0, cols - f_width + 1, COL_STRIDE):
            for i in range(0, rows - f_height + 1, ROW_STRIDE):

                #num = j*n_row+i+1
                image_array = np.zeros((image.shape[0], f_height, f_width)) # CHW

                # image
                for b in range(image.shape[0]):
                    band = image[b,:,:]
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
        print("Finished cutting image", c+1, "(", num-1, "subimages saved )")

if __name__ == "__main__":
    doit()
