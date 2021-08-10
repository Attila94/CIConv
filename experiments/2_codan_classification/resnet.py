#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:06:52 2020 by Attila Lengyel - attila@lengyel.nl
"""

import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, model_urls
from torchvision.models.utils import load_state_dict_from_url

from ciconv2d import CIConv2d

# Add custom weights to model urls dict
model_urls['w_resnet18'] = 'https://gitlab.tudelft.nl/attilalengyel/ciconv/-/raw/master/model_zoo/w_resnet18.pth'
model_urls['w_resnet101'] = 'https://gitlab.tudelft.nl/attilalengyel/ciconv/-/raw/master/model_zoo/w_resnet101.pth'

class ResNet(nn.Module):
    def __init__(self, block, layers, invariant=None, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, k=3, scale=0.0, return_features=False):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.return_features = return_features

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # Custom layers
        self.invariant = invariant
        if invariant:
            self.ciconv = CIConv2d(invariant, k=k, scale=scale)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Default ResNet
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        if self.invariant:
            x = self.ciconv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.return_features:
            return x1,x2,x3,x4
        else:
            x = self.avgpool(x4)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, invariant=None, num_classes=1000, **kwargs):
    model = ResNet(block, layers, invariant=invariant, num_classes=num_classes, **kwargs)

    if pretrained:
        arch_try = invariant.lower()+'_'+arch if invariant else arch
        try:
            # Check if pre-trained color invariants weights are available
            print('Loading {} weights...'.format(arch_try))
            state_dict = load_state_dict_from_url(model_urls[arch_try], progress=progress)
        except:
            # Else load pre-trained RGB weights
            print('{} weights not found, loading {} weights...'.format(arch_try,arch))
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            state_dict['conv1.weight'] = torch.sum(state_dict['conv1.weight'], dim=1, keepdim=True)
        if num_classes != 1000:
            print('Skipping fc layer parameters.')
            del state_dict['fc.weight']
            del state_dict['fc.bias']
        # Load weights and print result
        r = model.load_state_dict(state_dict, strict=False)
        print(r)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet151(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model with NConv layer
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet('resnet151', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


if __name__ == '__main__':
    print('Printing ResNet model definition, then exiting.')
    m = resnet101(pretrained=True, num_classes=10)
    print(m)
