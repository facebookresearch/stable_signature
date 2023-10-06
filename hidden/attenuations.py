# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import functional

class JND(nn.Module):
    """ Same as in https://github.com/facebookresearch/active_indexing """
    
    def __init__(self, preprocess = lambda x: x):
        super(JND, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_lum = [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.FloatTensor(kernel_lum).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
        self.weight_lum = nn.Parameter(data=kernel_lum, requires_grad=False)

        self.preprocess = preprocess
    
    def jnd_la(self, x, alpha=1.0):
        """ Luminance masking: x must be in [0,255] """
        la = F.conv2d(x, self.weight_lum, padding=2) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum]/127)) + 3
        la[~mask_lum] = 3/128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117):
        """ Contrast masking: x must be in [0,255] """
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    def heatmaps(self, x, clc=0.3):
        """ x must be in [0,1] """
        x = 255 * self.preprocess(x)
        x = 0.299 * x[...,0:1,:,:] + 0.587 * x[...,1:2,:,:] + 0.114 * x[...,2:3,:,:]
        la =  self.jnd_la(x)
        cm = self.jnd_cm(x)
        return (la + cm - clc * torch.minimum(la, cm))/255 # b 1 h w
    