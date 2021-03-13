#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import cv2 as cv


CLASSES = {
    1: 'Species-1',
    2: 'Species-2',
    3: 'Species-3',
    4: 'Species-4',
    5: 'Species-5',
    6: 'Species-6',
    7: 'Species-7',
    8: 'Species-8',
    9: 'Species-9'
}


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

def draw_clustering(model, test_loader):
    model.eval()
    dimension = 2
    tsne = TSNE(n_components=dimension)
    # pca = PCA(n_components=dimension)
    # svd = TruncatedSVD(n_components=dimension)
    first = True
    base_data = None
    labels = []

    for x, label in tqdm(test_loader):
        x = np.array(x) / 255.0
        x = x.transpose((0, 3, 1, 2))
        test_x = torch.FloatTensor(x).to(DEVICE)
        encoded_data = model(test_x)
        encoded_data = encoded_data.to("cpu").detach().numpy()
        encoded_data = tsne.fit_transform(encoded_data)
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
        plt.savefig('save/classification_img/2d.png', dpi=300)
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
            color = colormap[s]
            if name not in names:
                ax.scatter(x, y, z, c=color, label=name)
                names.append(name)
            else:
                ax.scatter(x, y, c=color)

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(Z.min(), Z.max())
        plt.legend()
        plt.savefig('save/classification_img/3d.png', dpi=300)
        plt.show()

# ## load camera data
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# load model
model = ResNet().to(DEVICE)
model.load_state_dict(torch.load('save/resnet_classification.pt'))

def predict_fn(test_xs):
    # data = np.expand_dims(test_xs, axis=0)
    data = test_xs.transpose((0, 3, 1, 2))
    data = torch.FloatTensor(data).to(DEVICE)
    output = model(data)
    return output

if __name__ == '__main__':

    compose = transforms.Compose([transforms.Resize((448, 448))])

    file_name = 'data/img/26-4b.JPG'
    r_im = Image.open(file_name)
    r_im = compose(r_im)
    data_arr = np.array(r_im)
    data = np.expand_dims(data_arr, axis=0)
    data = data.transpose((0, 3, 1, 2))
    data = torch.FloatTensor(data).to(DEVICE)



    output = model(data)
    print(output)
    pred = output.max(1, keepdim=True)[1]
    print('prediction label is', pred.item()+1)


    # # to visualize test_set.csv
    # test_file_name = 'data/test_set.csv'
    # testset = pd.read_csv(test_file_name)
    # testset = SeedDataset(testset, root='data/img/')
    #
    # test_loader = DataLoader(
    #     testset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4
    # )
    # draw_clustering(model, test_loader)
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(data_arr, predict_fn, hide_color=0, top_labels=2, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=2, hide_rest=False)
    cv.imwrite('limeTest.jpg', mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))

    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask).astype('uint8'))
    plt.show()




