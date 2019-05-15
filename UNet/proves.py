import os
from init import *
import random
import numpy as np

global dir_img, dir_mask

images = [f.split('.')[0] for f in os.listdir(dir_img+'npy/') if '.npy' in f]
masks = [f.split('_mask.')[0] for f in os.listdir(dir_mask+'npy/') if '.npy' in f]

ids = [f for f in list(set(images).intersection(set(masks)))]

print("length ids", len(ids))

#ids = [(di, u) for di in ids for u in range(2)]

#print("length ids 2", len(ids))

dataset = list(ids)
length = len(dataset)
n = int(length * 0.05)
random.shuffle(dataset)
iddataset = {'train': dataset[:-n], 'val': dataset[-n:]}

print("length train", len(iddataset['train']))

imgs = [np.load(dir_img + 'npy/' + id + '.npy') for id in ids]

# need to transform from HWC to CHW
imgs_switched = [np.transpose(img, axes=[2, 0, 1]) for img in imgs]
imgs_normalized = [img/255 for img in imgs_switched]
print("length imgs", len(imgs_normalized))

masks = [np.load(dir_mask + 'npy/' + id + '_mask.npy') for id in ids]
print("length masks", len(masks))

train = zip(imgs_normalized, masks)

def batch(train):
    b = []
    for i, t in enumerate(train):
        b.append(t)
        if (i + 1) % 5 == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

for i, b in enumerate(batch(train)):
    print(i, len(b), type(b))
