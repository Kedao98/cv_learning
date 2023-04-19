import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(BasicResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.down_sample = down_sample

    def forward(self, x):
        identity = self.down_sample(x) if self.down_sample is not None else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = F.relu(x, inplace=True)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.down_sample = down_sample

    def forward(self, x):
        identity = self.down_sample(x) if self.down_sample is not None else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity
        x = F.relu(x, inplace=True)

        return x


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(3, self.in_channel, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, channel, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel, channel, down_sample=down_sample, stride=stride)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.max_pool(x)

        for i in range(1, 5):
            x = getattr(self, f"layer{i}")(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    @classmethod
    def initialize_model_for_learning(cls):
        import os
        model = cls(BasicResidualBlock, [3, 4, 6, 3], num_classes=5, include_top=True)
        model_save_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_pth = os.path.join(model_save_dir, 'best.pth')
        model_dict_json = os.path.join(model_save_dir, 'class_indices.json')
        return model, model_save_pth, model_dict_json

    @classmethod
    def resnet34(cls, num_classes, include_top=True):
        return cls(BasicResidualBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    @classmethod
    def resnet50(cls, num_classes, include_top=True):
        return cls(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    @classmethod
    def resnet101(cls, num_classes, include_top=True):
        return cls(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

