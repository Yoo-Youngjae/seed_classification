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
from PIL import Image


# 하이퍼파라미터
EPOCH = 10
BATCH_SIZE = 1
dimension = 3
image_size = (28, 28, 3)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)
csv_file_name = 'data/example.csv'

class SeedDataset(Dataset):
    def __init__(self, dataframe, root='data/img'):
        self.dataframe = dataframe
        self.root = root

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        file_name, label = self.dataframe.loc[i]['name'], self.dataframe.loc[i]['label']
        img_dir = self.root + file_name
        r_im = Image.open(img_dir).resize((image_size[0], image_size[1]))
        r_im = np.array(r_im)
        data = torch.FloatTensor(r_im)
        label = torch.FloatTensor(label)
        return data, label

    def get_sample(self, size):
        file_names, labels = self.dataframe.loc[0:size-1]['name'], self.dataframe.loc[0:4]['label']
        img_dir = self.root + file_names[0]
        r_im = Image.open(img_dir).resize((image_size[0], image_size[1]))
        data = [np.array(r_im)]

        for file_name in file_names[1:]:
            img_dir = self.root + file_name
            r_im = Image.open(img_dir).resize((image_size[0], image_size[1]))
            r_im = np.array(r_im)
            data = np.concatenate((data, [r_im]), axis=0)

        data = torch.FloatTensor(data)
        return data, torch.tensor(labels.values.astype(np.float32))





trainset = pd.read_csv(csv_file_name)
trainset = SeedDataset(trainset, root='data/img/')

train_loader = torch.utils.data.DataLoader(
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 2
)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        global dimension

        self.encoder = nn.Sequential(
            nn.Linear(image_size[0]*image_size[1]*image_size[2], 128),
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
            nn.Linear(128, image_size[0]*image_size[1]*image_size[2]),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
criterion = nn.MSELoss()


# 원본 이미지를 시각화 하기 (첫번째 열)
view_data, _ = trainset.get_sample(5)
view_data = view_data.view(-1, image_size[0]*image_size[1]*image_size[2])
view_data = view_data.type(torch.FloatTensor)/255.


def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, image_size[0]*image_size[1]*image_size[2]).to(DEVICE)
        y = x.view(-1, image_size[0]*image_size[1]*image_size[2]).to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step %10 == 0:
            print('step',step, 'loss',loss.item())


for epoch in range(1, EPOCH+1):
    train(autoencoder, train_loader)

    # 디코더에서 나온 이미지를 시각화 하기 (두번째 열)
    test_x = view_data.to(DEVICE)
    _, decoded_data = autoencoder(test_x)

    # 원본과 디코딩 결과 비교해보기
    f, a = plt.subplots(2, 5, figsize=(5, 2))
    print("[Epoch {}]".format(epoch))
    for i in range(5):
        img = np.reshape(np.array(view_data[i]),(image_size[0], image_size[1],image_size[2]))
        a[0][i].imshow(img)
        a[0][i].set_xticks(()); a[0][i].set_yticks(())

    for i in range(5):
        img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (image_size[0], image_size[1],image_size[2]))
        a[1][i].imshow(img)
        a[1][i].set_xticks(()); a[1][i].set_yticks(())
    plt.show()


# # 잠재변수 들여다보기

# 잠재변수를 3D 플롯으로 시각화
view_data, labels = trainset.get_sample(200)
view_data = view_data.view(-1, image_size[0] * image_size[1] *image_size[2])
view_data = view_data.type(torch.FloatTensor)/255.
labels = labels.numpy()
test_x = view_data.to(DEVICE)
encoded_data, _ = autoencoder(test_x)
encoded_data = encoded_data.to("cpu")


CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

if dimension == 2:
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()


    for x, y, s in zip(X, Y, labels):
        name = CLASSES[s]
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




