from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader

import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader,Dataset
import requests, zipfile, io, os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

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
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, num_groups=3, skewness = 0.5,seed=0):
        super(TaskGen, self).__init__(benchmark='xray',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      num_groups=num_groups,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/xray/data',
                                      seed=seed
                                      )
        self.num_classes = 4
        self.save_data = self.XYData_to_json

    def load_data(self):
        filepath = os.path.join(self.rawdata_path,'archive.zip')
        assert os.path.isfile(filepath), "Go to https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion to download data and place it inside data folder"
        z = zipfile.ZipFile(filepath,'r')
        z.extractall(self.rawdata_path)

        df = pd.read_csv(os.path.join(self.rawdata_path, './train_df.csv'))
        df = df[df['Target'].apply(lambda x: x.strip().isnumeric())]
        labels = df.Target.apply(int).values.tolist()
        label_freq = dict(Counter(labels))
        selected_labels = [k for k,v in label_freq.items() if v>=80]
        filtered_image_paths, filtered_labels = [], []
        for label, image_path in zip(labels, df.image_path.values):
            if label in selected_labels:
                filtered_labels.append(selected_labels.index(label))
                filtered_image_paths.append(os.path.join(self.rawdata_path,image_path))
        #image_paths = [os.path.join(self.,i) for i in df.image_path.values]
        path_train, path_test, y_train, y_test = train_test_split(filtered_image_paths, filtered_labels,
                                                            stratify=filtered_labels, 
                                                            test_size=0.25,
                                                            random_state=self.seed)
        self.train_data = CustomImageDataset(y_train, path_train, transform=transforms.Compose([transforms.Resize([96,96 ]), transforms.ConvertImageDtype(torch.float), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = CustomImageDataset(y_test, path_test, transform=transforms.Compose([transforms.Resize([96,96 ]), transforms.ConvertImageDtype(torch.float), transforms.Normalize((0.1307,), (0.3081,))]))

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

