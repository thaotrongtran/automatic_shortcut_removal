import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck

class Unet_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True))
        layers = []
        downsample = nn.Sequential(
          nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
          nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        layers.append(Bottleneck(64,64, downsample=downsample))
        for _ in range(0, 4):
            layers.append(Bottleneck(256, 64))
        self.blocks = nn.Sequential(*layers)
        self.conv_end = nn.Sequential( nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, padding=0),
                                  nn.ReLU(inplace=True))
        #Reference source code for initialization of Batch Norm and Conv2d https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x: Tensor) -> Tensor:
        orig = x
        x = self.conv_input(x)
        x = self.blocks(x)
        x = self.conv_end(x)
        x = orig + x
        return x