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
        assert mask.count == NET_CLASSES or mask.count == 1, "check the number of classes"

        print("Started cutting image", c+1, "/", len(list(get_ids_tif(dir_img, dir_mask))))
        #n_col = math.floor(cols/f_width + f_width - COL_STRIDE)
        #n_row = math.floor(rows/f_height + f_height - ROW_STRIDE)
        num = 1
        for j in range(0, cols - f_width + 1, COL_STRIDE):
            for i in range(0, rows - f_height + 1, ROW_STRIDE):

                #num = j*n_row+i+1
                image_array = np.zeros((image.count, f_height, f_width))

                for b in range(image.count):
                    band = image.read(b+1)
                    # assert band != None
                    image_array[b,:,:] = band[i : i + f_height, j : j + f_width]

                if mask.count == 1:
                    aux = mask.read(1)[i : i + f_height, j : j + f_width]
                    mask_array = np.zeros((NET_CLASSES, f_height, f_width))
                    for (x,y), p in np.ndenumerate(aux):
                      mask_array[p, x, y] = 1

                else: # mask.RasterCount == NET_CLASSES per l'assert d'abans
                  mask_array = mask.read()

                image_array = np.transpose(image_array,(1,2,0))   # Format HWC
                #mask_array = np.transpose(mask_array,(1,2,0))   # Format HWC

                #if j == 0:
                  #img_array = image_array[:,:,50:53]/np.amax(image_array[:,:,50:53])
                  #plt.imshow(img_array)
                  #plt.show()

                np.save(dir_img + "npy/" + im + '_' + str(num), image_array)
                np.save(dir_mask + "npy/" + im + '_' + str(num) + '_mask', mask_array)
                num = num + 1
        print("Finished cutting image", c+1, "(", num, "subimages saved )")

if __name__ == "__main__":
    doit()
