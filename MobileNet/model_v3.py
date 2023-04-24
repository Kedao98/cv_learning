import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channels),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channel = self._make_divisible(in_channel // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channel, squeeze_channel, 1)
        self.fc2 = nn.Conv2d(squeeze_channel, in_channel, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x

    @staticmethod
    def _make_divisible(ch, divisor=8, min_ch=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_ch is None:
            min_ch = divisor
        new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_ch < 0.9 * ch:
            new_ch += divisor
        return new_ch
