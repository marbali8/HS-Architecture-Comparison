from init import *
from unet import *
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
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              gpu=False):

    dir = dir_project + 'runs/' + time.strftime("%b%d_%H%M%S", time.localtime())
    dir += '_E' + str(epochs) + 'B' + str(batch_size) + 'R' + str(lr).split('.')[-1]
    dir += 'P' + str(WIDTH_CUTS) + 'A' + 'u2' + 'O' + 'sgd' + 'L' + 'ce'
    tb_train_writer = SummaryWriter(dir)

    path_img = dir_img+'npy/'
    path_mask = dir_mask+'npy/'
    ids = get_ids_npy(path_img, path_mask)
    #ids = split_ids(ids) # CHANGED ja no parteixo les imatges en 2

    iddataset = split_train_val(ids, val_percent)

    tb_train_writer.add_text('info', str(epochs) + ' epochs;\n' +
                  str(batch_size) + ' batch size;\n' +
                  str(lr) + ' lr;\n' +
                  str(len(iddataset['train'])) + ' training size;\n' +
                  str(len(iddataset['val'])) + ' validation size;\n' +
                  str(WIDTH_CUTS) + ' width input patches;\n'
                  'UNET 2 deep; SGD, CrossEntropyLoss'
                  )
    # posar SGD parameters de momentum i tal?

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
            ({} batches needed each epoch)
        Validation size: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               math.ceil(len(iddataset['train'])/batch_size),
               len(iddataset['val']), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # esta loss fa logsoftmax i calcula pixel a pixel i despres fa la mitja
    criterion = nn.CrossEntropyLoss() # CHANGED

    epoch_loss = np.zeros(epochs) # NEW

    # cada epoca, pasara les mateixes imatges per la xarxa
    for e, epoch in enumerate(range(epochs)):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], path_img, path_mask)
        val = get_imgs_and_masks(iddataset['val'], path_img, path_mask)

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
            if tb_train_writer == None:
                print(str(i+1), end=' ', flush = True)

        print(flush = False)
        print('Epoch finished ! Loss: {}'.format(epoch_loss[e] / i))

        if tb_train_writer == None:
            plt.title('Loss (' + str(e+1) + '/' + str(len(epoch_loss)) + ' epochs)')
            plt.xlabel('epochs')
            plt.ylabel('cross entropy loss')
            plt.plot(range(len(epoch_loss)), epoch_loss / i)
            plt.savefig(dir_docs + 'loss.png', bbox_inches='tight')
            plt.close()
        else:
            tb_train_writer.add_scalar('train loss', epoch_loss[e] / i, epoch)

        if 1:
            val_loss = eval_net(net, val, criterion, tb_train_writer, gpu)
            print('Validation Coeff: {}'.format(val_loss))
            tb_train_writer.add_scalar('validation loss', val_loss / i, epoch)
    torch.save(net.state_dict(), dir + '/MODEL.pth')
    print("Saved model")
    tb_train_writer.close()

# CHANGED substitut del parser
class ArgsTrain:

  # CHANGED tret img_scale
  def __init__(self, epochs=5, batchsize=10, lr=0.1, gpu=True, load=False):
      self.epochs = epochs
      self.batchsize = batchsize
      self.lr = lr
      self.gpu = gpu
      self.load = load

# CHANGED tret img_scale
def train(epochs=5, batchsize=10, lr=0.1, gpu=True, load=False):

    args = ArgsTrain(epochs=epochs, batchsize=batchsize, lr=lr, gpu=gpu, load=load)
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
                  gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), dir + '/INTERRUPTED.pth')
        print('\nSaved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == "__main__":
    train(epochs=4000, batchsize=20, lr=0.01)
    train(epochs=3500, batchsize=10, lr=0.0005)
    train(epochs=4000, batchsize=10, lr=0.0005)
    train(epochs=3000, batchsize=20, lr=0.001)
    train(epochs=3500, batchsize=20, lr=0.001)
    train(epochs=4000, batchsize=20, lr=0.001)
    train(epochs=2500, batchsize=20, lr=0.01)
    train(epochs=3500, batchsize=20, lr=0.01)
