# coding:utf-8

from torchvision.models.resnet import resnet18

A = resnet18()

import torch

a = torch.Tensor(12)
b = a
c = b + torch.Tensor(2)
import torch.nn.functional as F


F.selu()
