import torch.nn as nn
from utils.fmodule import FModule
from benchmark.medmnist_params import params
from ..core import BENCHMARK

class Model(FModule):
    def __init__(self, num_classes=params[BENCHMARK]['n_labels'], base_dim=32):
        super(Model, self).__init__()
        self.in_channels = params[BENCHMARK]['n_channels']
        self.num_classes = num_classes
        # convolutional layers 
        self.b012_conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, base_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.b12_conv2 = nn.Sequential(
        #     nn.Conv2d(base_dim, 2*base_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(2*base_dim, 2*base_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        # self.b2_conv3 = nn.Sequential(
        #     nn.Conv2d(2*base_dim, 4*base_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(4*base_dim, 4*base_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b012_fc = nn.Linear(base_dim , num_classes)
        # self.b0_fc = nn.Sequential(
        #     nn.Linear(base_dim , 4*base_dim ),
        #     nn.BatchNorm1d( 4*base_dim ),
        #     nn.ReLU(inplace=True)
        # ) 
        # self.b1_fc = nn.Sequential(
        #     nn.Linear(2*base_dim , 4*base_dim ),
        #     nn.BatchNorm1d( 4*base_dim ),
        #     nn.ReLU(inplace=True)
        # ) 

    def forward(self, x, n=3):
        x = self.b012_conv1(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.b012_fc(x) 
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)
