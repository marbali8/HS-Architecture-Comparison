import numpy as np
# import rasterio
# import xml.etree.ElementTree as ET
# from matplotlib.path import Path
#
# img = rasterio.open('/imatge/mbalibrea/Documents/data/recorte1_smv95_v2.tif')
# img = rasterio.open('/imatge/mbalibrea/Documents/data/recorte1.tif')
#
# tree = ET.parse('/imatge/mbalibrea/Documents/data/ROIs/ROIs/Entrenamiento/1/CASI/Mar.xml')
# root = tree.getroot()
#
# for roi in range(1, len(root[0][0])):
#     coords = root[0][0][roi][0][0][0].text.replace('\n', '').split(' ')
#     coords = [c for c in coords if c != '']
#     coords = [float(c) for c in coords]
#     coords = [coords[i:i+2] for i in range(0,len(coords),2)]
#     print(coords[0][0], coords[0][1], img.index(coords[0][0], coords[0][1]))
#
# x, y = np.meshgrid(np.arange(300), np.arange(300)) # make a canvas with coordinates
# x, y = x.flatten(), y.flatten()
# points = np.vstack((x,y)).T
#
# p = Path(tupVerts) # make a polygon
# grid = p.contains_points(points)
# mask = grid.reshape(300,300) # now you have a mask with points inside a polygon
#
# balanced_mask = np.zeros((128,128,15))
# for r in range(balanced_mask.shape[0]):
#     for c in range(balanced_mask.shape[1]):
#         ch = np.random.randint(balanced_mask.shape[2])
#         bornot = 1 if np.random.rand() > 0.5 else -1
#         balanced_mask[r,c,ch] = bornot
#
# mask = np.zeros((128,128))
# for r in range(balanced_mask.shape[0]):
#     for c in range(balanced_mask.shape[1]):
#         max = np.argmax(abs(balanced_mask[r, c, :]))
#         mask[r, c] = max*balanced_mask[r, c, max]

# import os
# def to_cropped_imgs(ids):
#     for id in ids:
#         f = np.load('/Users/marbalibrea/Desktop/' + id)
#         yield f
#
# def get_imgs_and_masks(ids):
#     """Return all the couples (img, mask)"""
#
#     imgs = to_cropped_imgs(ids)
#     masks = to_cropped_imgs(ids)
#     return zip(imgs, masks)
#
# def batch(iterable, batch_size):
#     """Yields lists by batch"""
#     b = []
#     for i, t in enumerate(iterable): # i de 0 a #imatges-1
#         b.append(t)
#         if (i + 1) % batch_size == 0:
#             yield b
#             b = []
#
#     if len(b) > 0:
#         yield b
#
# def split_train_val(dataset):
#     dataset = list(dataset)
#     return dataset, dataset
#
# def eval_net(things, e):
#     for i, b in enumerate(things):
#         print("eval", e, ":", b[0].shape)
#
# path_img = '/Users/marbalibrea/Desktop/'
# ids = (f for f in os.listdir(path_img) if '.npy' in f)
# iddataset = split_train_val(ids)
#
# for e, epoch in enumerate(range(10)):
#
#     train = get_imgs_and_masks(iddataset[0])
#     val = get_imgs_and_masks(iddataset[1])
#
#     for i, b in enumerate(batch(train, 3)):
#         print(i, b[0][0].shape)
#         imgs = np.array([i[0] for i in b])
#         masks = np.array([i[1] for i in b])
#
#     eval_net(val, e)

import rasterio
import numpy as np
import os
mask = rasterio.open("/work-nfs/mbalibrea/data/TRAIN/maspalomas/masks/recorte1_mask.tif")
assert len(mask.shape) == 2
train_mask = np.zeros((13, mask.shape[0], mask.shape[1]), dtype=int)

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

train_mask = abs(train_mask)
samples_per_class = [(train_mask[i] == 1).sum() for i in range(13)]
N = sum(samples_per_class) # N pixels
weights = N/samples_per_class
np.save("/work-nfs/mbalibrea/data/TRAIN/maspalomas/masks/" + "npy/recorte1_class_weights", weights)
