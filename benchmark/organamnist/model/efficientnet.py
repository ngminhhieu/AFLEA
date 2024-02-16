import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

import torch.nn as nn
from utils.fmodule import FModule
from benchmark.medmnist_params import params
from ..core import BENCHMARK

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

def _RoundChannels(c, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_c = max(min_value, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c

def _RoundRepeats(r):
    return int(math.ceil(r))

def _DropPath(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)

    return x

def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)

def _Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

def _Conv1x1Bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.non_linear1 = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.non_linear2 = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.non_linear1(self.se_reduce(y))
        y = self.non_linear2(self.se_expand(y))
        y = x * y

        return y

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                _BatchNorm(expand_channels),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            nn.Conv2d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride,
                kernel_size//2,
                groups=expand_channels,
                bias=False
            ),
            _BatchNorm(expand_channels),
            Swish()
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            _BatchNorm(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            return x + _DropPath(self.conv(x), self.drop_path_rate, self.training)
        else:
            return self.conv(x)

class Model(FModule):
    

    def __init__(self, param=(1.0, 1.0, 28, 0.2),num_classes=params[BENCHMARK]['n_labels'], stem_channels=32, feature_size=128, drop_connect_rate=0.2,**kwargs):
        super(Model, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.n_blocks = 0
        self.config = [
            #(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
            [32,  32,  3, 1, 1, 0.25, 1],
            [32,  64,  3, 2, 6, 0.25, 2],
            [64,  128,  5, 2, 6, 0.25, 2]
        ]
            # scaling width
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = _RoundChannels(stem_channels*width_coefficient)
            for conf in self.config:
                conf[0] = _RoundChannels(conf[0]*width_coefficient)
                conf[1] = _RoundChannels(conf[1]*width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = _RoundRepeats(conf[6]*depth_coefficient)

        # scaling resolution
        input_size = param[2]

        # stem convolution
        self.b012_conv1 = _Conv3x3Bn(params[BENCHMARK]['n_channels'], stem_channels, 2)

        # total #blocks
        self.total_blocks = 0
        for conf in self.config:
            self.total_blocks += conf[6]

        self.b012_conv2 = self._make_block(self.config[0])
        self.b12_conv3 = self._make_block(self.config[1])
        self.b2_conv4 = self._make_block(self.config[2])

        # last several layers
        self.b0_head_conv = _Conv1x1Bn(self.config[0][1], feature_size)
        self.b1_head_conv = _Conv1x1Bn(self.config[1][1], feature_size)
        self.b2_head_conv = _Conv1x1Bn(self.config[2][1], feature_size)
        #self.avgpool = nn.AvgPool2d(input_size//32, stride=1)
        self.dropout = nn.Dropout(param[3])
        self.b012_fc = nn.Linear(feature_size, num_classes)

        self._initialize_weights()

    def _make_block(self, config):
        in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats = config 
        blocks = []
        drop_rate = self.drop_connect_rate * (self.n_blocks / self.total_blocks)
        blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
        self.n_blocks += 1
        for _ in range(repeats-1):
            drop_rate = self.drop_connect_rate * (self.n_blocks / self.total_blocks)
            blocks.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
            self.n_blocks += 1
        return nn.Sequential(*blocks)

    def forward(self, x, n=3):
        x = self.b012_conv1(x)
        x = self.b012_conv2(x)

        if n==0:
            x = self.b0_head_conv(x)
            x = torch.mean(x, (2, 3))
            x = self.b012_fc(x) 
            return x

        x = self.b12_conv3(x)
        if n==1:
            x = self.b1_head_conv(x)
            x = torch.mean(x, (2, 3))
            x = self.b012_fc(x) 
            return x

        x = self.b2_conv4(x)
        x = self.b2_head_conv(x)
        x = torch.mean(x, (2, 3))
        x = self.b012_fc(x) 
        return x

    def pred_and_rep(self, x, n=3):

        x = self.b012_conv1(x)
        x = self.b012_conv2(x)

        if n==0:
            x = self.b0_head_conv(x)
            e = torch.mean(x, (2, 3))
            x = self.b012_fc(e) 
            return x, [e]

        x = self.b12_conv3(x)
        if n==1:
            x = self.b1_head_conv(x)
            e = torch.mean(x, (2, 3))
            x = self.b012_fc(e) 
            return x, [e]

        x = self.b2_conv4(x)
        x = self.b2_head_conv(x)
        e = torch.mean(x, (2, 3))
        x = self.b012_fc(e) 
        return x, [e]
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net_param = {
        # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5)
    }

    param = net_param['efficientnet-b0']
    net = EfficientNet(param)
    x_image = Variable(torch.randn(1, 3, param[2], param[2]))
    y = net(x_image)