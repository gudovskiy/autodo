import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


_bn_momentum = 0.1

__all__ = ['EncoderWideResNet', 'SupConWideResNet', 'SupCeWideResNet']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.act = nn.CELU(inplace=True) #self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.act(self.bn1(x))))
        out = self.conv2(self.act(self.bn2(out)))
        out += self.shortcut(x)

        return out


class EncoderWideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout_rate=0.0):
        super(EncoderWideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.act = nn.CELU(inplace=True)
        # self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        y = out.view(out.size(0), -1)
        return y


model_dict = {
    'wresnet28_10': [EncoderWideResNet, 640],
}


class SupConWideResNet(nn.Module):
    """projection head"""
    def __init__(self, name='wresnet28_10', num_classes=10, head='mlp', feat_dim=128):
        super(SupConWideResNet, self).__init__()
        _, dim_in = model_dict[name]
        self.cls = nn.Linear(dim_in, num_classes)
        if head == 'linear':
            self.scl = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.scl = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.BatchNorm1d(dim_in),
                nn.CELU(inplace=True),
                nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))
        
    def forward(self, x):
        y, z = self.cls(x), F.normalize(self.scl(x), dim=1)
        return [y, z]


class SupCeWideResNet(nn.Module):
    """classifier"""
    def __init__(self, name='wresnet28_10', num_classes=10):
        super(SupCeWideResNet, self).__init__()
        _, dim_in = model_dict[name]
        self.cls = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        y = self.cls(x)
        return [y]

