# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT


class unet_3D_Condition(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_Condition, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0] + 2, filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1] + 2, filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2] + 2, filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3] + 2, filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4] + 2, filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3] + 2, filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2] + 2, filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1] + 2, filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0]+2, n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, condition=1):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        # expand dim for broadcast
        condition = condition.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        #concat condition
        B,C,D,W,H = maxpool1.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        maxpool1 = torch.cat((c,maxpool1),dim=1)
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        #concat condition
        B,C,D,W,H = maxpool2.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        maxpool2 = torch.cat((c,maxpool2),dim=1)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        #concat condition
        B,C,D,W,H = maxpool3.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        maxpool3 = torch.cat((c,maxpool3),dim=1)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        #concat condition
        B,C,D,W,H = maxpool4.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        maxpool4 = torch.cat((c,maxpool4),dim=1)

        center = self.center(maxpool4)
        center = self.dropout1(center)

        #concat condition
        B,C,D,W,H = center.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        center = torch.cat((c,center),dim=1)

        up4 = self.up_concat4(conv4, center)
        
        #concat condition
        B,C,D,W,H = up4.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        up4 = torch.cat((c,up4),dim=1)
        
        up3 = self.up_concat3(conv3, up4)

        #concat condition
        B,C,D,W,H = up3.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        up3 = torch.cat((c,up3),dim=1)

        up2 = self.up_concat2(conv2, up3)

        #concat condition
        B,C,D,W,H = up2.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        up2 = torch.cat((c,up2),dim=1)

        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        #concat condition
        B,C,D,W,H = up1.shape
        c = (torch.ones(B,2,D,W,H).cuda()*condition)
        up1 = torch.cat((c,up1),dim=1)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
