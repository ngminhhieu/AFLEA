"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch.nn as nn
from utils.fmodule import FModule
from benchmark.medmnist_params import params
from ..core import BENCHMARK

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Model(FModule):
    def __init__(self, block=BasicBlock, num_block=[1,2,2,2], num_classes=params[BENCHMARK]['n_labels'], base_dim=32):
        super().__init__()
        self.in_channels = base_dim
        self.b0123_conv1 = nn.Sequential(
            nn.Conv2d(params[BENCHMARK]['n_channels'], base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.b123_conv2_x = self._make_layer(block, base_dim, num_block[0], 1)
        self.b23_conv3_x = self._make_layer(block, 2*base_dim, num_block[1], 2)
        self.b3_conv4_x = self._make_layer(block, 4*base_dim, num_block[2], 2)
        # self.b2_conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b0123_fc = nn.Linear(4*base_dim * block.expansion, num_classes)
        self.b0_fc = nn.Sequential(
            nn.Linear(base_dim * block.expansion, 4*base_dim * block.expansion),
            nn.BatchNorm1d( 4*base_dim * block.expansion),
            nn.ReLU(inplace=True)
        ) 
        self.b1_fc = nn.Sequential(
            nn.Linear(base_dim * block.expansion, 4*base_dim * block.expansion),
            nn.BatchNorm1d( 4*base_dim * block.expansion),
            nn.ReLU(inplace=True)
        ) 
        self.b2_fc = nn.Sequential(
            nn.Linear(2*base_dim * block.expansion, 4*base_dim * block.expansion),
            nn.BatchNorm1d( 4*base_dim * block.expansion),
            nn.ReLU(inplace=True)
        ) 

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, n=3):
        x = self.b0123_conv1(x)
        if n==0:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b0_fc(x) 
            x = self.b0123_fc(x) 
            return x        
        x = self.b123_conv2_x(x)
        # x = self.b012_conv3_x(x)
        if n==1:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b1_fc(x) 
            x = self.b0123_fc(x) 
            return x

        x = self.b23_conv3_x(x)
        if n==2:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b2_fc(x) 
            x = self.b0123_fc(x) 
            return x

        x = self.b3_conv4_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.b0123_fc(x) 
        return x

    def pred_and_rep(self, x, n=3):

        x = self.b0123_conv1(x)
        if n==0:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            e = self.b0_fc(x) 
            o = self.b0123_fc(e) 
            return o, [e]
        x = self.b123_conv2_x(x)
        # x = self.b012_conv3_x(x)
        if n==1:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            e = self.b1_fc(x) 
            o = self.b0123_fc(e) 
            return o, [e]

        x = self.b23_conv3_x(x)
        if n==2:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            e = self.b2_fc(x) 
            o = self.b0123_fc(e) 
            return o, [e]

        x = self.b3_conv4_x(x)
        x = self.avg_pool(x)
        e = x.view(x.size(0), -1)
        o = self.b0123_fc(e) 
        return o, [e]

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)