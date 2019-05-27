from init import *
from unet import *
from utils1 import *
from utils2 import *
from plot import *

import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def predict_img(net,
                full_img,
                scale_factor=1, # CHANGED default scale a 1
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net = net.double()
    net.eval()
    img_height = full_img.shape[1] # CHANGED PIL to npy
    img_width = full_img.shape[0] # CHANGED PIL to npy

    #img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(full_img)

    #left_square, right_square = split_img_into_squares(img)

    #left_square = hwc_to_chw(left_square)
    #right_square = hwc_to_chw(right_square)
    img = hwc_to_chw(img)

    #X_left = torch.from_numpy(left_square).unsqueeze(0)
    #X_right = torch.from_numpy(right_square).unsqueeze(0)
    X = torch.from_numpy(img).unsqueeze(0).double()

    if use_gpu and torch.cuda.is_available(): # CHANGED
        # X_left = X_left.cuda()
        # X_right = X_right.cuda()
        X = X.cuda()

    with torch.no_grad():
        #output_left = net(X_left)
        #output_right = net(X_right)
        output = net(X)
        # NEW pq ara es una imatge de [NET_CLASSES, H, W] i jo vull [1, H, W]
        # on cada pixel sigui el num de la classe
        output = torch.argmax(output, 1)

        #left_mask_np = left_probs.squeeze()
        #right_mask_np = right_probs.squeeze()
        mask_np = output.squeeze()

        #print(left_mask_np.shape, right_mask_np.shape)

    #full_mask = merge_masks(left_mask_np, right_mask_np, img_width)
    full_mask = mask_np

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), mask_np)

    return full_mask # > out_threshold CHANGED pq ara es segmentacio

def get_output_filenames(input, output, extension=''):
    in_files = input
    out_files = []
    if len(output) == 0 or (len(output) == 1 and output[0] == None):
        for f in in_files:
            pathsplit = os.path.splitext(f)
            ext = ('.' + extension) if extension != '' else pathsplit[1]
            out_files.append("{}_OUT{}".format(pathsplit[0], ext))
    elif len(in_files) != len(output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

#import gdal # CHANGED
import rasterio # CHANGED
from PIL import Image
import numpy as np # fins que no he posat això, em donava errors sense sentit xd

# CHANGED substitut del parser
class ArgsPredict:

  # CHANGED default scale a 1
  def __init__(self, input, output='', model=dir_docs + 'MODEL.pth', cpu=True, viz=True, no_save=True, no_crf=True, mask_th=0.5, scale=1):
      self.model = model
      self.cpu = cpu
      self.viz = viz
      self.no_save = no_save
      self.no_crf = no_crf
      self.mask_threshold = mask_th
      self.scale = scale
      self.input = input
      self.output = None if output == '' else output


def predict(input, model):
    args = ArgsPredict(input=input, model=model)
    # CHANGED
    out_files = get_output_filenames([args.input], [args.output], extension='jpg')

    net = UNet(n_channels=INPUT_BANDS, n_classes=NET_CLASSES)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate([args.input]):
        print("\nPredicting image {} ...".format(fn))

        # CHANGED per si es np o tif + per imprimir tambe el target

        img = rasterio.open(fn).read()
        target = rasterio.open(fn.replace("/images/", "/masks/").replace(".", "_mask."))
        target = target.read()
        img = np.transpose(img,(1,2,0)) # CHANGED chw to hwc

        if img.any() == None or target.any() == None:
          img = np.load(fn)
          target = np.load(fn.replace("/images/", "/masks/").replace(".", "_mask."))

          if img.any() == None or target.any() == None:
            print("could not open image")
            exit(1)

        if img.shape[1] > img.shape[0]: # CHANGED per npy
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(np.transpose(img,(2,0,1)), target, mask, dir=args.model)

        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray(mask) # mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))

if __name__ == "__main__":
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_120543_E4000B20R01P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_125022_E3500B10R0005P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_133257_E4000B10R0005P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_142132_E3000B20R001P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_145504_E3500B20R001P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_153404_E4000B20R001P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_161833_E2500B20R01P18Au2OsgdLce/MODEL.pth')
    predict(dir_img + '19920612_AVIRIS_IndianPine_Site3.tif', model = dir_project + 'runs/May25_164612_E3500B20R01P18Au2OsgdLce/MODEL.pth')
