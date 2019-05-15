from unet_extras import *

import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = n_channels
        self.classes = n_classes
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        #self.prob_to_bin = prob_to_bin()

    def forward(self, x):
        #print("first", x.shape)
        x1 = self.inc(x)
        #print("inc", x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #print("down1", x2.shape)
        #print("down2", x3.shape)
        #print("down3", x4.shape)
        #print("down4", x5.shape)
        x = self.up1(x5, x4)
        #print("up1", x.shape)
        x = self.up2(x, x3)
        #print("up2", x.shape)
        x = self.up3(x, x2)
        #print("up3", x.shape)
        x = self.up4(x, x1)
        #print("up4", x.shape)
        x = self.outc(x)
        #print("outc", x.shape)
        #x = torch.sigmoid(x) # CHANGED per avis de deprecated i tret pq ja ho fa a la nova loss
        #print("sigmoid/out", x.shape)
        return x #self.prob_to_bin(x)
