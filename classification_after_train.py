#!/usr/bin/env python
# coding: utf-8

# # 신경망 깊게 쌓아 컬러 데이터셋에 적용하기
# Convolutional Neural Network (CNN) 을 쌓아올려 딥한 러닝을 해봅시다.

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import numpy as np


# ## ResNet 모델 만들기

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.linear = nn.Linear(12544, num_classes) # 64

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out



if __name__ == '__main__':
    # ## load camera data
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    compose = transforms.Compose([transforms.Resize((448, 448))])

    file_name = 'data/img/26-1b.JPG'
    r_im = Image.open(file_name)
    r_im = compose(r_im)
    data = np.array(r_im)
    data = np.expand_dims(data, axis=0)
    data = data.transpose((0, 3, 1, 2))
    data = torch.FloatTensor(data).to(DEVICE)

    # load model
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load('save/resnet_classification.pt'))

    output = model(data)
    pred = output.max(1, keepdim=True)[1]
    print('prediction label is', pred.item()+1)



