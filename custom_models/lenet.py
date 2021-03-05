import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_models.lpf import *

__all__ = ['EncoderLeNet', 'SupConLeNet', 'SupCeLeNet']


class Swish(nn.Module):
    """Applies the element-wise function :math:`f(x) = x / ( 1 + exp(-x))`
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = nn.Swish()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, x):
        return x*torch.sigmoid(x)


class wsConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(wsConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def wsBatchNorm2d(num_features, norm_layer='NO', num_groups=2):
    if   'NO' in norm_layer:
        return nn.Identity()
    elif 'BN' in norm_layer:
        return nn.BatchNorm2d(num_features=num_features)
    elif 'IN' in norm_layer:
        return nn.InstanceNorm2d(num_features=num_features)
    elif 'GN' in norm_layer:
        return nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    else:
        print('Wrong norm layer!')


class EncoderLeNet(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None, LPF=False, ACT='CELU'):
        super(EncoderLeNet, self).__init__()
        #
        if 'WS' in norm_layer:
            self.conv1 = wsConv2d( 1, 10, kernel_size=5)
            self.conv2 = wsConv2d(10, 20, kernel_size=5)
        else:
            self.conv1 = nn.Conv2d( 1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #
        if LPF:
            self.mpool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.downs1 = Downsample(filt_size=3, stride=2, channels=10)
            self.downs2 = Downsample(filt_size=3, stride=2, channels=20)
        else:
            self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.downs1 = nn.Identity()
            self.downs2 = nn.Identity()
        #
        if   ACT == 'RELU':
            self.act = nn.ReLU(inplace=True)
        elif ACT == 'CELU':
            self.act = nn.CELU(inplace=True)
        elif ACT == 'GELU':
            self.act = nn.GELU()
        elif ACT == 'SWISH':
            self.act = Swish()
        else:
            print('{} is not supported activation!'.format(ACT))
        # norm layer
        self.bn1 = wsBatchNorm2d(10, norm_layer=norm_layer, num_groups=2)
        self.bn2 = wsBatchNorm2d(20, norm_layer=norm_layer, num_groups=4)

    def forward(self, x):
        x1 = self.downs1(self.mpool(self.act(self.bn1(self.conv1(x)))))
        x2 = self.downs2(self.mpool(self.act(self.bn2(self.conv2(x1)))))
        y = x2.view(x2.size(0), -1)
        return y


model_dict = {
    'lenet': [EncoderLeNet, 320],
}


class SupConLeNet(nn.Module):
    """projection head"""
    def __init__(self, name='lenet', num_classes=10, head='mlp', feat_dim=128):
        super(SupConLeNet, self).__init__()
        _, dim_in = model_dict[name]
        self.cls = nn.Linear(dim_in, num_classes)
        if head == 'linear':
            self.scl = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.scl = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.CELU(inplace=True),
                nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError('head not supported: {}'.format(head))
        
    def forward(self, x):
        y, z = self.cls(x), F.normalize(self.scl(x), dim=1)
        return [y, z]


class SupCeLeNet(nn.Module):
    """classifier"""
    def __init__(self, name='lenet', num_classes=10):
        super(SupCeLeNet, self).__init__()
        _, dim_in = model_dict[name]
        self.cls = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        y = self.cls(x)
        return [y]

