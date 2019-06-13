from init import *
from unet import *
from unet3d import *
from utils1 import *
from utils2 import *
from plot import *
from extras import *

import os

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import rasterio

def predict_img(net,
                full_img,
                scale_factor=1, # CHANGED default scale a 1
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net = net.double()
    net.eval()

    img = normalize(full_img)

    X = torch.from_numpy(img).unsqueeze(0).double()

    if use_gpu and torch.cuda.is_available(): # CHANGED
        X = X.cuda()

    with torch.no_grad():
        output = net(X)
        # NEW pq ara es una imatge de [NET_CLASSES, H, W] i jo vull [1, H, W]
        # on cada pixel sigui el num de la classe
        output = torch.argmax(output, 1)

        mask_np = output.squeeze()

        #print(left_mask_np.shape, right_mask_np.shape)

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

# CHANGED substitut del parser
class ArgsPredict:

  # CHANGED default scale a 1
  def __init__(self, input, output='', model = dir_runs + 'MODEL.pth', cpu=True, viz=True, no_save=True, no_crf=True, mask_th=0.5, scale=1):
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

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        model = torch.load(args.model)
        l = [1 for key in model.keys() if 'encoders' in key]
        if len(l) > 0:
            net = UNet3D(INPUT_BANDS, NET_CLASSES)
        else:
            net = UNet(n_channels=INPUT_BANDS, n_classes=NET_CLASSES)
        net.cuda()
        net.load_state_dict(model)
    else:
        model = torch.load(args.model, map_location='cpu')
        l = [1 for key in model.keys() if 'encoders' in key]
        if len(l) > 0:
            net = UNet3D(INPUT_BANDS, NET_CLASSES)
        else:
            net = UNet(n_channels=INPUT_BANDS, n_classes=NET_CLASSES)
        net.cpu()
        net.load_state_dict(model)
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate([args.input]):
        print("\nPredicting image {} ...".format(fn))

        # CHANGED per si es np o tif + per imprimir tambe el target

        if ".tif" in fn:
            img = rasterio.open(fn).read() # chw
            # here target is MxN
            target = rasterio.open(fn.replace("/images/", "/masks/").replace(".", "_mask."))
            target = target.read(1)
            train_mask = target
        elif ".npy" in fn:
            img = np.load(fn)
            # here target is CxMxN
            target = np.load(fn.replace("/images/", "/masks/").replace(".", "_mask."))
            # not using loss_mask because i dont care about which pixels
            train_mask = loss_mask(-abs(target), type = 'test')
            target = train_mask
        else:
            print("could not open image")
            exit(1)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))

            target1 = torch.from_numpy(target).squeeze()
            mask1 = mask

            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(mask1.float(), torch.argmax(target1, 1)).item()
            # print(target1.shape, mask1.shape)
            loss = balanced_score(train_mask, mask1)

            plot_img_and_mask(img, target, mask, loss, dir=args.model)

        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray(mask) # mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))

if __name__ == "__main__":
    predict(dir_img + 'npy/recorte1_1.npy', model = dir_runs + 'Jun13_014614_E4B20+R01P128x128S64x64Au2OsgdLceACbal/MODEL.pth')
