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

# NEW from get_ids_npy
def get_ids_tif(dir_img, dir_masks):
    """Returns a list of the ids"""
    images = [('recorte1.npy').split('.')[0]]
    masks = [f.split('_mask.')[0] for f in os.listdir(dir_masks) if '_mask.tif' in f]
    return (f for f in list(set(images).intersection(set(masks))))

# based on https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
def weights_and_samples_balanced(mask):

    samples_per_class = [(mask[i] == 1).sum() for i in range(NET_CLASSES)]

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
    args = ArgsPredict(input=input, model=model, cpu=False)
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

    image = np.load(input)
    target = rasterio.open(input.replace("/images/", "/masks/").replace(".npy", "_mask.tif"))

    train_mask, weights = new_split_train_val(image, target.read(1))

    f_width = WIDTH_CUTS
    f_height = f_width
    cols = target.width
    rows = target.height

    t_samples_per_class = [0] * NET_CLASSES
    p_samples_per_class = [0] * NET_CLASSES

    print("Started")
    big_image = np.zeros((3, 5108, 7856))
    num = 1
    for j in range(0, cols - f_width + 1, f_width):
        for i in range(0, rows - f_height + 1, f_height):

            #num = j*n_row+i+1
            image_array = np.zeros((image.shape[0], f_height, f_width)) # CHW

            # image
            for b in range(image.shape[0]):
                band = image[b,:,:]
                # assert band != None
                image_array[b,:,:] = band[i : i + f_height, j : j + f_width]

            # mask
            mask_array = train_mask[:, i : i + f_height, j : j + f_width]

            mask = predict_img(net=net,
                               full_img=image_array,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               use_dense_crf=not args.no_crf,
                               use_gpu=not args.cpu)

            t_mask = loss_mask(mask_array, type = 'test')

            mask1 = mask.data.cpu().detach().numpy()
            assert mask1.shape == t_mask.shape
            for (m,n), v in np.ndenumerate(mask1):
                pred = int(mask1[m,n])
                big_image[:, i+m, j+n] = rgb_maspalomas[pred]

            if num % 50 == 0:
                print(str(num), end=' ', flush = True)

            num = num + 1
    print(flush = False)

    fig = plt.figure()
    big_image = np.transpose(big_image, axes=[1,2,0]).astype(int)
    plt.imshow(big_image)
    model_name = args.model.split('/')[-1].split('.')[0]
    dir = '/'.join(args.model.split('/')[:-1]) + '/'
    path = dir + "FULLpredictrecorte1" + model_name + ".png"
    plt.savefig(path)

if __name__ == "__main__":
    predict(dir_img + 'recorte1.npy', model = dir_runs + 'Jun23_211802_E80B30+R001P128x128S64x64Au2OsgdLceACbal/MODEL_79.pth')
    predict(dir_img + 'recorte1.npy', model = dir_runs + 'Jun24_160815_E80B30+R001P128x128S64x64Au3d2OsgdLceACbal/MODEL_79.pth')
