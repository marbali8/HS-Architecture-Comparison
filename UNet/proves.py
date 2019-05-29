import numpy as np
import rasterio
import xml.etree.ElementTree as ET
from matplotlib.path import Path

img = rasterio.open('/imatge/mbalibrea/Documents/data/recorte1_smv95_v2.tif')
img = rasterio.open('/imatge/mbalibrea/Documents/data/recorte1.tif')

tree = ET.parse('/imatge/mbalibrea/Documents/data/ROIs/ROIs/Entrenamiento/1/CASI/Mar.xml')
root = tree.getroot()

for roi in range(1, len(root[0][0])):
    coords = root[0][0][roi][0][0][0].text.replace('\n', '').split(' ')
    coords = [c for c in coords if c != '']
    coords = [float(c) for c in coords]
    coords = [coords[i:i+2] for i in range(0,len(coords),2)]
    print(coords[0][0], coords[0][1], img.index(coords[0][0], coords[0][1]))

x, y = np.meshgrid(np.arange(300), np.arange(300)) # make a canvas with coordinates
x, y = x.flatten(), y.flatten()
points = np.vstack((x,y)).T

p = Path(tupVerts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(300,300) # now you have a mask with points inside a polygon
