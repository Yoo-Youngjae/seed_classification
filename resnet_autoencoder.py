#!/usr/bin/env python
# coding: utf-8

# # 오토인코더로 이미지의 특징을 추출하기

import torch
from torch import nn, optim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# 하이퍼파라미터
EPOCH = 1
BATCH_SIZE = 64
dimension = 3
image_size = (224, 224, 3)
feature_size = 1000

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)
# csv_file_name = 'data/example.csv'
csv_file_name = 'data/dataset.csv'
class SeedDataset(Dataset):
    def __init__(self, dataframe, root='data/img'):
        super(SeedDataset, self).__init__()
        self.dataframe = dataframe
        self.root = root
        self.compose = transforms.Compose([transforms.Resize((224, 224))])

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





trainset = pd.read_csv(csv_file_name)
trainset = SeedDataset(trainset, root='data/img/')

train_loader = DataLoader(
    trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 3
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        global dimension

        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, dimension),   # 입력의 특징을 3차원으로 압축합니다
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimension, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()



resnet18 = models.resnet18(pretrained=True)  # in (18, 34, 50, 101, 152)
resnet18 = resnet18.to(DEVICE)

def train(autoencoder, train_loader):
    autoencoder.train()
    idx = 0
    # try:
    for x, label in tqdm(train_loader):
        seed_arr = np.array(x) / 255.0
        x = seed_arr.transpose((0, 3, 1, 2))
        x = torch.FloatTensor(x).to(DEVICE)
        feature_x = resnet18(x)
        x = feature_x.view(-1, 1000).to(DEVICE)
        y = feature_x.view(-1, 1000).to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss',loss.item())
        idx += 1
    # except Exception as e:
    #     # print(e)
    #     pass

for epoch in range(1, EPOCH+1):
    train(autoencoder, train_loader)


# # 잠재변수 들여다보기

# 잠재변수를 3D 플롯으로 시각화
seed_im, labels = trainset.get_sample(300)
seed_arr = np.array(seed_im) / 255.0
seed_arr = seed_arr.transpose((0, 3, 1, 2))
seed_arr = torch.FloatTensor(seed_arr).to(DEVICE)
view_data_x = resnet18(seed_arr)
view_data_x = view_data_x.view(-1, 1000).to(DEVICE)

labels = labels.numpy()
test_x = view_data_x.to(DEVICE)
encoded_data, _ = autoencoder(test_x)
encoded_data = encoded_data.to("cpu")


CLASSES = {
    0: 'Species-0',
    1: 'Species-1',
    2: 'Species-2',
    3: 'Species-3',
    4: 'Species-4',
    5: 'Species-5',
    6: 'Species-6',
    7: 'Species-7',
    8: 'Species-8'
}

if dimension == 2:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()


    for x, y, s in zip(X, Y, labels):
        name = CLASSES[s]
        print(name)
        color = cm.rainbow(int(255*s/9))
        ax.text(x, y, name, backgroundcolor=color)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    plt.show()
else: # dimension == 3
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy()


    for x, y, z, s in zip(X, Y, Z, labels):
        name = CLASSES[s]
        color = cm.rainbow(int(255*s/9))
        ax.text(x, y, z, name, backgroundcolor=color)


    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()



