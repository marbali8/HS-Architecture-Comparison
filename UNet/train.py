from init import *
from unet import *
from utils1 import *
from utils2 import *

import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    global dir_project, dir_img, dir_mask
    path_img = dir_img+'npy/'
    path_mask = dir_mask+'npy/'
    ids = get_ids_npy(path_img, path_mask)
    #ids = split_ids(ids) # CHANGED ja no parteixo les imatges en 2

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
            ({} batches needed each epoch)
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               math.ceil(len(iddataset['train'])/batch_size),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # esta loss fa logsofmax i calcula pixel a pixel i despres fa la mitja
    criterion = nn.CrossEntropyLoss() # CHANGED

    epoch_loss = np.zeros(epochs) # NEW

    # cada epoca, pasara les mateixes imatges per la xarxa
    for e, epoch in enumerate(range(epochs)):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], path_img, path_mask, img_scale, net.in_channels, net.classes)
        val = get_imgs_and_masks(iddataset['val'], path_img, path_mask, img_scale, net.in_channels, net.classes)

        # i de 0 a ceil(length(train)/batch_size)-1
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            #print("shape of batch before net", imgs.shape) # (BATCHSIZE, NET_CHANNELS, W/2, H/2)
            true_masks = torch.from_numpy(true_masks).squeeze()

            if gpu and torch.cuda.is_available(): # CHANGED
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs) # shape (BATCHSIZE, NET_CLASSES, W, H)
            #print("shape of batch after net", masks_pred.shape)
            #per fer plt.imshow he de fer masks_pred.cpu()

            #print(masks_pred.shape, true_masks.shape)

            #print("shape of true masks", true_masks.shape, "are they binary?", len(torch.unique(true_masks)) == 2)
            #print("shape of pred masks", masks_pred.shape, "are they binary?", len(torch.unique(masks_pred)) == 2)
            loss = criterion(masks_pred, torch.argmax(true_masks, 1).long())
            epoch_loss[e] += loss.item()

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(str(i+1), end=' ', flush = True)

        print(flush = False)
        print('Epoch finished ! Loss: {}'.format(epoch_loss[e] / i))

        plt.title('Loss (' + str(e+1) + '/' + str(len(epoch_loss)) + ' epochs)')
        plt.xlabel('epochs')
        plt.ylabel('cross entropy loss')
        plt.plot(range(len(epoch_loss)), epoch_loss / i)
        plt.savefig(dir_docs + 'loss.png', bbox_inches='tight')
        plt.close()

        # FIXME de moment comentat pq no funciona
        #if 1:
            #val_dice = eval_net(net, val, gpu)
            #print('Validation Dice Coeff: {}'.format(val_dice))
    torch.save(net.state_dict(), dir_docs + 'MODEL.pth')
    print("Saved model")

# CHANGED substitut del parser
class ArgsTrain:

  # CHANGED img_scale a 1
  def __init__(self, epochs=5, batchsize=10, lr=0.1, gpu=True, load=False, img_scale=1):
      self.epochs = epochs
      self.batchsize = batchsize
      self.lr = lr
      self.gpu = gpu
      self.load = load
      self.scale = img_scale

# CHANGED img_scale a 1
def train(epochs=5, batchsize=10, lr=0.1, gpu=True, load=False, img_scale=1):

    global INPUT_BANDS, NET_CLASSES

    args = ArgsTrain(epochs=epochs, batchsize=batchsize, lr=lr, gpu=gpu, load=load, img_scale=img_scale)
    net = UNet(INPUT_BANDS, NET_CLASSES)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu and torch.cuda.is_available(): # CHANGED
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        global dir_project
        torch.save(net.state_dict(), dir_docs + 'INTERRUPTED.pth')
        print('\nSaved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    train(epochs=50, batchsize=20)
