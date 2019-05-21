import torch
import torchvision
from torch.autograd import Function #, Variable
import numpy as np

from plot import *

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# CHANGED
def eval_net(net, dataset, tb_val_writer, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0].astype(np.float32)
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu and torch.cuda.is_available(): # CHANGED
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img) # CHANGED
        tot += dice_coeff(mask_pred.double(), true_mask.double()).item()
        if i == 0:
            plot_img_and_mask(  img.cpu().squeeze().numpy(),
                                torch.argmax(true_mask, 1).cpu().numpy(),
                                torch.argmax(mask_pred, 1).cpu().squeeze().numpy(),
                                tb_val_writer)
    return tot / (i + 1)
