import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

class res18GConv(nn.Module):
    def __init__(self, num_class = 1000):
        super(res18GConv,self).__init__()
        res18 = models.resnet18(pretrained = True)
        self.res18_out = nn.Sequential(*list(res18.children())[:-2])

        # for params in self.res18_out.parameters():
        #     params.required_grad = False


        self.expand = nn.Conv2d(512, num_class, 1, bias=False)
        self.expand_InsNorm0 = nn.InstanceNorm2d(num_class, affine=True)
        self.expand_relu0 = nn.ReLU(inplace=True)
        self.expand_drop0 = nn.Dropout2d(p = 0.3)

        self.conv2dT2 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1, bias=False)
        self.InsNorm2 = nn.BatchNorm2d(num_class, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d(p = 0.3)

        self.conv2dT3 = nn.ConvTranspose2d(num_class, num_class, kernel_size=4, stride=2, padding=1, bias=False, groups=num_class)
        self.InsNorm3 = nn.BatchNorm2d(num_class, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgPool = nn.AdaptiveAvgPool2d(1)


    def forward(self,x):
        res_out = self.res18_out(x)

        expand = self.expand(res_out)
        expand = self.expand_InsNorm0(expand)
        expand = self.expand_relu0(expand)
        expand = self.expand_drop0(expand)

        upSample2 = self.conv2dT2(expand)
        upSample2_norm = self.InsNorm2(upSample2)
        upSample2_relu = self.relu2(upSample2_norm)
        upSample2_drop = self.drop2(upSample2_relu)

        upSample3 = self.conv2dT3(upSample2_drop)
        upSample3_norm = self.InsNorm3(upSample3)
        upSample3_relu = self.relu3(upSample3_norm)

        result = self.avgPool(upSample3_relu)
        result = result.view(result.size(0), -1)

        return result, upSample3_relu, upSample2_drop

# test = res18GConv(200)
# dummy_input = torch.randn(1, 3, 64, 64)
# result, u1, u2, u3 = test.forward(dummy_input)
# print(u1.size())
# print(u2.size())
# print(u3.size())

