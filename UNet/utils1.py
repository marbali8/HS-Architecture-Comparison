import random
import numpy as np
from math import floor

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]) # si abans era shape (a,b,c) ara és (c,a,b)

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

def normalize(x):
    return x / 255
