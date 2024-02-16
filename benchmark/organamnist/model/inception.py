import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule
from benchmark.medmnist_params import params
from ..core import BENCHMARK

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)

class Model(FModule):
    def __init__(self, aux_logits=True, num_classes=params[BENCHMARK]['n_labels'], base_dim=32):
        super(Model, self).__init__()
        self.b012_conv1 = nn.Sequential(
            ConvBlock(
                in_channels=params[BENCHMARK]['n_channels'], 
                out_chanels=32,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
            ),
            ConvBlock(32, 96 , kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b12_conv2 = nn.Sequential(
            InceptionBlock(96, 32, 48, 64, 8, 16, 16),
            InceptionBlock(128, 64, 64, 96, 16, 48, 32)
        )

        self.b2_conv3 = nn.Sequential(
            InceptionBlock(240, 96, 46, 104, 8, 24, 32),
            InceptionBlock(256, 80, 56, 112, 12, 32, 32)
        )


        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b012_fc = nn.Linear(256, num_classes)
        self.b0_fc = nn.Sequential(
            nn.Linear(96, 256),
            nn.BatchNorm1d( 256),
            nn.ReLU(inplace=True)
        ) 
        self.b1_fc = nn.Sequential(
            nn.Linear(240, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        ) 
    
    def forward(self, x, n=3):
        x = self.b012_conv1(x)

        if n==0:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b0_fc(x) 
            x = self.b012_fc(x) 
            return x

        x = self.b12_conv2(x)
        if n==1:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b1_fc(x) 
            x = self.b012_fc(x) 
            return x

        x = self.b2_conv3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.b012_fc(x) 
        return x

    def pred_and_rep(self, x, n=3):

        x = self.b012_conv1(x)

        if n==0:
            e = self.avg_pool(x)
            e = e.view(e.size(0), -1)
            e = self.b0_fc(e) 
            o = self.b012_fc(e) 
            return o, [e]

        x = self.b12_conv2(x)

        if n==1:
            e = self.avg_pool(x)
            e = e.view(e.size(0), -1)
            e = self.b1_fc(e)
            o = self.b012_fc(e) 
            return o, [e]

        x = self.b2_conv3(x)
        e = self.avg_pool(x)
        e = e.view(e.size(0), -1)
        o = self.b012_fc(e)
        return o, [e]

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)
