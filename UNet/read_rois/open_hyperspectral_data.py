import rasterio
from rasterio import features
from shapely.geometry import shape, mapping, MultiPolygon
import os
import numpy as np
import matplotlib.pyplot as plt
from handle_rois import *

def open_raster(path_raster):
    raster = rasterio.open(path_raster)
    geotransform = raster.transform
    return geotransform, raster.shape

def open_rois(rois_path):
    rois_props=[]
    data = list(get_regions(rois_path))
    rois_props=[data[i]['properties'] for i in range(len(data))]
    shapes = [shape(r['geometry']) for r in data]
    return shapes, rois_props

# path_raster = '/imatge/mbalibrea/Documents/data/TRAIN/maspalomas/images/recorte1.tif'
path_raster = '/Users/marbalibrea/Downloads/segmentacion_mar/recorte1.tif'
# path_rois = os.path.join(my_path,'../data/TRAIN/maspalomas/ROIs/ROIs/Entrenamiento/1/CASI/')
path_rois = '/Users/marbalibrea/Downloads/segmentacion_mar/ROIs/Evaluación/'
rois = [roi for roi in os.listdir(path_rois) if ".xml" in roi]
geotransform, outshape = open_raster(path_raster)

for r, roi in enumerate(rois):

    multipolygons, rois_props = open_rois(path_rois+roi)
    roisarrays = {}
    for i in range(len(multipolygons)):
        #print(multipolygons[i])
        name = rois_props[i]['name']
        polygonseq = multipolygons[i].geoms
        c = np.zeros(outshape)
        for polygon in polygonseq:
            b = mapping(polygon)
            a = features.rasterize([b], out_shape = outshape, transform = geotransform, fill=0)
            c=c+a
        roisarrays[name] = c
        if not os.path.exists(path_rois + "/new"):
            os.makedirs(path_rois + "/new")
        np.save(os.path.join(path_rois + 'new', name), c)
        # plt.imsave(os.path.join(path_rois + 'new', name+'.png'), c)
    print(r)
