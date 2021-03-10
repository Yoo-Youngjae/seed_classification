#!/usr/bin/env python
# coding: utf-8

# # 신경망 깊게 쌓아 컬러 데이터셋에 적용하기
# Convolutional Neural Network (CNN) 을 쌓아올려 딥한 러닝을 해봅시다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
csv_file_name = 'data/dataset.csv'

# ## 하이퍼파라미터

EPOCHS = 60
BATCH_SIZE = 32
image_size = (448, 448, 3) # 224, 224


class SeedDataset(Dataset):
    def __init__(self, dataframe, root='data/img'):
        super(SeedDataset, self).__init__()
        self.dataframe = dataframe
        self.root = root
        self.compose = transforms.Compose([transforms.Resize((448, 448))])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        file_name, label = self.dataframe.loc[i]['name'], self.dataframe.loc[i]['label']
        img_dir = self.root + file_name +'.JPG'
        r_im = Image.open(img_dir)
        r_im = self.compose(r_im)
        r_im = np.array(r_im)
        data = torch.FloatTensor(r_im)
        return data, label

    def get_sample(self, size):
        file_names, labels = self.dataframe.loc[0:size-1]['name'], self.dataframe.loc[0:4]['label']
        img_dir = self.root + file_names[0] +'.JPG'
        r_im = Image.open(img_dir)
        r_im = self.compose(r_im)
        data = [np.array(r_im)]

        for file_name in file_names[1:]:
            img_dir = self.root + file_name +'.JPG'
            r_im = Image.open(img_dir)
            r_im = self.compose(r_im)
            r_im = np.array(r_im)
            data = np.concatenate((data, [r_im]), axis=0)

        data = torch.FloatTensor(data)
        return data, torch.tensor(labels.values.astype(np.float32))


def get_data(file_name):
    df = pd.read_csv(file_name)
    train_set, test_set = train_test_split(df, test_size=0.2)  # split ratio => train : test =  8 : 2
    train_set.index = [i for i in range(len(train_set.index))]
    test_set.index = [i for i in range(len(test_set.index))]
    return train_set, test_set

# ## 데이터셋 불러오기

trainset, testset = get_data(csv_file_name)

trainset = SeedDataset(trainset, root='data/img/')
testset = SeedDataset(testset, root='data/img/')

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE, shuffle=True)


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


# ## 준비

model = ResNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

print(model)


# ## 학습하기

def train(model, train_loader, optimizer, epoch):
    model.train()
    idx = 0
    for data, target in tqdm(train_loader):
        data = np.array(data)
        data = data.transpose((0, 3, 1, 2))
        data = torch.FloatTensor(data).to(DEVICE)
        target = target - 1

        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('loss',loss.item())
        idx += 1

# ## 테스트하기

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = np.array(data)
            data = data.transpose((0, 3, 1, 2))
            data = torch.FloatTensor(data).to(DEVICE)
            target = target - 1

            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 훈련이 되는지 확인해봅시다!
best_accuracy = 0
for epoch in range(1, EPOCHS + 1):
    scheduler.step()
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch, test_loss, test_accuracy))
    if best_accuracy < test_accuracy:
        torch.save(model.state_dict(), 'save/resnet_classification.pt')
        best_accuracy = test_accuracy




