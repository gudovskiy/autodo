# Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['EncoderResNet', 'SupConResNet', 'SupCeResNet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = nn.CELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.act = nn.CELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out

class EncoderResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(EncoderResNet, self).__init__()        
        self.dataset = dataset
        self.act = nn.CELU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dataset == 'MNIST':
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            #
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)            
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
        elif self.dataset.startswith('CIFAR'):
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            #
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)            
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
        elif dataset == 'ImageNet':
            blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
            #
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        # init parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        if self.dataset == 'ImageNet':
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        #
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


model_dict = {
    'MNIST-18':     64,
    'ImageNet-18':  512,
    'ImageNet-34':  512,
    'ImageNet-50':  2048,
    'ImageNet-101': 2048,
}


class SupConResNet(nn.Module):
    """projection head"""
    def __init__(self, dataset, depth, num_classes=1000, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        name = dataset+'-'+str(depth)
        dim_in = model_dict[name]
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


class SupCeResNet(nn.Module):
    """classifier"""
    def __init__(self, dataset, depth, num_classes=1000):
        super(SupCeResNet, self).__init__()
        name = dataset+'-'+str(depth)
        dim_in = model_dict[name]
        self.cls = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        y = self.cls(x)
        return [y]

