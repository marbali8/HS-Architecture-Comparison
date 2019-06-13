from init import *
from unet import *
from unet3d import *
from utils1 import *
from utils2 import *
from extras import *

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import math
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter

def train_net(net, dir,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              gpu=True,
              augment=True):

    tb_train_writer = SummaryWriter(dir)

    path_img = dir_img+'npy/'
    path_mask = dir_mask+'npy/'
    ids = get_ids_npy(path_img, path_mask)
    iddataset = gettwo(ids)

    tb_train_writer.add_text('info', str(epochs) + ' epochs;\n' +
                  str(batch_size) + ' batch size;\n' +
                  str(lr) + ' lr;\n' +
                  # str(len(iddataset['train'])) + ' training size;\n' +
                  # str(len(iddataset['val'])) + ' validation size;\n' +
                  str(WIDTH_CUTS) + ' width input patches;\n' +
                  net.__class__.__name__ + ' 2 deep; SGD, CrossEntropyLoss'
                  )
    # posar SGD parameters de momentum i tal?

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, #len(iddataset['train']),
               # math.ceil(len(iddataset['train'])/batch_size),
               # len(iddataset['val']),
               str(gpu)))

    # N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # esta loss calcula pixel a pixel i despres fa la mitja
    weights = torch.from_numpy(get_weights('train'))
    if gpu and torch.cuda.is_available():
        weights = weights.cuda()
    criterion = nn.NLLLoss(weight=weights.float(), ignore_index=-1) # CHANGED

    epoch_loss = np.zeros(epochs) # NEW

    # cada epoca, pasara les mateixes imatges per la xarxa
    for e, epoch in enumerate(range(epochs)):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # reset the generators
        train = get_imgs_and_masks(iddataset[0], path_img, path_mask)
        val = get_imgs_and_masks(iddataset[1], path_img, path_mask)

        # i de 0 a ceil(length(train)/batch_size)-1
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            imgs, true_masks = augmentation(imgs, true_masks)
            # now true_masks will be [BATCHSIZE, W, H]
            train_masks = np.array([loss_mask(mask, type='train') for mask in true_masks])

            imgs = torch.from_numpy(imgs)
            #print("shape of batch before net", imgs.shape) # (BATCHSIZE, NET_CHANNELS, W, H)
            train_masks = torch.from_numpy(train_masks).squeeze()

            if gpu and torch.cuda.is_available(): # CHANGED
                imgs = imgs.cuda()
                train_masks = train_masks.cuda()

            masks_pred = net(imgs) # shape (BATCHSIZE, NET_CLASSES, W, H)
            #print("shape of batch after net", masks_pred.shape)
            #per fer plt.imshow he de fer masks_pred.cpu()

            #print(masks_pred.shape, true_masks.shape)

            loss = criterion(masks_pred, train_masks.long())
            epoch_loss[e] += loss.item()

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if tb_train_writer == None:
            print(str(i+1), end=' ', flush = True)

        print(flush = False)
        print('Epoch finished ! Loss: {}'.format(epoch_loss[e] / i))

        if tb_train_writer == None:
            plt.title('Loss (' + str(e+1) + '/' + str(len(epoch_loss)) + ' epochs)')
            plt.xlabel('epochs')
            plt.ylabel('cross entropy loss')
            plt.plot(range(len(epoch_loss)), epoch_loss / i)
            plt.savefig(dir + 'loss.png', bbox_inches='tight')
            plt.close()
        else:
            tb_train_writer.add_scalar('train loss', epoch_loss[e] / i, epoch)

        if 1:
            val_loss = eval_net(net, val, dir, tb_train_writer, gpu)
            print('Validation Coeff: {}'.format(val_loss))
            tb_train_writer.add_scalar('validation coeff', val_loss / i, epoch)
    torch.save(net.state_dict(), dir + '/MODEL.pth')
    print("Saved model")
    tb_train_writer.close()

# CHANGED substitut del parser
class ArgsTrain:

  # CHANGED tret img_scale
  def __init__(self, epochs=5, batchsize=10, lr=0.1, gpu=True, load=False, augment=True):
      self.epochs = epochs
      self.batchsize = batchsize
      self.lr = lr
      self.gpu = gpu
      self.load = load
      self.augment = augment

# CHANGED tret img_scale
def train(net, epochs=5, batchsize=10, lr=0.1, gpu=True, load=False, augment=True):

    args = ArgsTrain(epochs=epochs, batchsize=batchsize, lr=lr, gpu=gpu, load=load, augment=True)
    assert net == 'unet' or net == 'unet3d'
    if net == 'unet':
        net = UNet(INPUT_BANDS, NET_CLASSES)
    else:
        net = UNet3D(INPUT_BANDS, NET_CLASSES)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu and torch.cuda.is_available(): # CHANGED
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    net_name = "u3d2" if "3D" in net.__class__.__name__ else "u2"
    dir = dir_runs + time.strftime("%b%d_%H%M%S", time.localtime())
    dir += '_E' + str(args.epochs) + 'B' + str(args.batchsize) + ('+' if augment else '')
    dir += 'R' + str(args.lr).split('.')[-1]
    dir += 'P' + str(WIDTH_CUTS) + 'x' + str(WIDTH_CUTS)
    dir += 'S' + str(COL_STRIDE) + 'x' + str(ROW_STRIDE) + 'A' + net_name
    dir += 'O' + 'sgd' + 'L' + 'ce' + 'AC' + 'bal'

    try:
        train_net(net=net, dir=dir,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  augment=True)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir_runs + 'INTERRUPTED.pth')
        print('\nSaved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    train(net = 'unet', epochs=4, batchsize=20, lr=0.01)
    train(net = 'unet3d', epochs=4, batchsize=20, lr=0.01)
    # train(net = 'unet', epochs=1000, batchsize=20, lr=0.01)
    # train(net = 'unet3d', epochs=1000, batchsize=20, lr=0.01)
    # train(net = 'unet', epochs=1000, batchsize=30, lr=0.01)
    # train(net = 'unet3d', epochs=1000, batchsize=30, lr=0.01)
    # train(net = 'unet', epochs=3000, batchsize=20, lr=0.001)
    # train(net = 'unet3d', epochs=3000, batchsize=20, lr=0.001)
    # train(net = 'unet', epochs=3000, batchsize=30, lr=0.001)
    # train(net = 'unet3d', epochs=3000, batchsize=30, lr=0.001)
