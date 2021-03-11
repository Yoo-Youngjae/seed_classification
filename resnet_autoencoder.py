#!/usr/bin/env python
# coding: utf-8

# # 오토인코더로 이미지의 특징을 추출하기

import torch
from torch import nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


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


# 하이퍼파라미터
EPOCH = 20
BATCH_SIZE = 64
dimension = 2
feature_size = 1000

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# DEVICE = 1
print("Using Device:", DEVICE)
# csv_file_name = 'data/example.csv'
csv_file_name = 'data/dataset.csv'
test_file_name = 'data/test_set.csv'

resnet18 = models.resnet18(pretrained=True)  # in (18, 34, 50, 101, 152)
resnet18 = resnet18.to(DEVICE)

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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100),   # 입력의 특징을 3차원으로 압축합니다
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_size),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력합니다
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(autoencoder, train_loader):
    autoencoder.train()
    idx = 0

    for x, label in tqdm(train_loader):
        x = np.array(x)
        x = x.transpose((0, 3, 1, 2))
        x = torch.FloatTensor(x).to(DEVICE)
        feature_x = resnet18(x)
        x = feature_x.view(-1, 1000).to(DEVICE)
        y = feature_x.view(-1, 1000).to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss', loss.item())
        idx += 1


def evaluate(autoencoder, train_loader, epoch):
    autoencoder.eval()
    first = True
    base_data = None
    labels = []
    # tsne = TSNE(n_components=dimension)
    svd = TruncatedSVD(n_components=dimension)
    for x, label in tqdm(train_loader):
        x = np.array(x)
        x = x.transpose((0, 3, 1, 2))
        x = torch.FloatTensor(x).to(DEVICE)
        feature_x = resnet18(x)
        test_x = feature_x.view(-1, 1000).to(DEVICE)
        encoded_data, _ = autoencoder(test_x)
        encoded_data = encoded_data.to("cpu").detach().numpy()
        encoded_data = svd.fit_transform(encoded_data)
        if first:
            first = False
            base_data = encoded_data
        else:
            base_data = np.concatenate((base_data, encoded_data), axis=0)
        labels += label.tolist()
    encoded_data = base_data

    colormap = ['red', 'orange', '#0077BB', 'green', 'blue', 'indigo', 'purple', '#EE3377', 'pink', 'saddlebrown']
    if dimension == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
        X = encoded_data[:, 0]
        Y = encoded_data[:, 1]
        names = []

        for x, y, s in zip(X, Y, labels):
            name = CLASSES[s]
            color = colormap[s % 10]
            if name not in names:
                ax.scatter(x, y, label=name, c=color)
                names.append(name)
            else:
                ax.scatter(x, y, c=color)

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        plt.legend()
        plt.savefig('save/image/2d_' + str(epoch) + '.png', dpi=300)
        plt.show()
    else:  # dimension == 3
        fig = plt.figure(figsize=(10, 8))
        ax = Axes3D(fig)
        X = encoded_data[:, 0]
        Y = encoded_data[:, 1]
        Z = encoded_data[:, 2]
        names = []

        for x, y, z, s in zip(X, Y, Z, labels):
            name = CLASSES[s]
            color = colormap[s % 10]
            if name not in names:
                ax.scatter(x, y, z, c=color, label=name)
                names.append(name)
            else:
                ax.scatter(x, y, c=color)

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        plt.legend()
        plt.savefig('save/image/3d_'+str(epoch)+'.png', dpi=300)
        plt.show()



trainset = pd.read_csv(csv_file_name)
trainset = SeedDataset(trainset, root='data/img/')

testset = pd.read_csv(test_file_name)
testset = SeedDataset(testset, root='data/img/')

train_loader = DataLoader(
    trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 5
)
test_loader = DataLoader(
    testset,
    batch_size  = 40,
    shuffle     = True,
    num_workers = 5
)

autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0005)
criterion = nn.MSELoss()



#########################

for epoch in range(1, EPOCH+1):
    train(autoencoder, test_loader)
    evaluate(autoencoder, test_loader, epoch)







