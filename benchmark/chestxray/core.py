from torchvision import datasets, transforms
from torchvision.transforms.functional import rgb_to_grayscale
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader

import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader,Dataset
import requests, zipfile, io, os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict

class CustomImageDataset(Dataset):
    def __init__(self, labels, img_paths, transform=None, target_transform=None):
        self.img_labels = labels
        self.img_paths = img_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(img_path)
        if image.shape[0]!=1:
            image = rgb_to_grayscale(image)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, num_groups=3, skewness = 0.5,seed=0):
        super(TaskGen, self).__init__(benchmark='chestxray',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      num_groups=num_groups,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/chestxray/data',
                                      seed=seed
                                      )
        self.num_classes = 2
        self.save_data = self.XYData_to_json

    def load_data(self):
        filepath = os.path.join(self.rawdata_path,'archive.zip')
        assert os.path.isfile(filepath), "Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia to download data and place it inside data folder"
        z = zipfile.ZipFile(filepath,'r')
        z.extractall(self.rawdata_path)

        path, y = defaultdict(list), defaultdict(list)
        for split in ['train','val', 'test']:
            for label, label_text in enumerate(['NORMAL', 'PNEUMONIA']):
                split_label_dir = os.path.join(self.rawdata_path, 'chest_xray', split, label_text)
                for f in os.listdir(split_label_dir):
                    path[split].append(os.path.join(split_label_dir,f))
                    y[split].append(label)

        self.train_data = CustomImageDataset(y['train']+y['val'], path['train']+path['val'], transform=transforms.Compose([transforms.Resize([224,224 ]), transforms.ConvertImageDtype(torch.float), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = CustomImageDataset(y['test'], path['test'], transform=transforms.Compose([transforms.Resize([224,224 ]), transforms.ConvertImageDtype(torch.float), transforms.Normalize((0.1307,), (0.3081,))]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return

class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

