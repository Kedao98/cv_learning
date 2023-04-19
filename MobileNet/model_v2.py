import os

import torch
import torch.nn as nn


class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        hidden_channel = in_channels * expand_ratio
        self.use_shortcut = bool((stride == 1) and (in_channels == out_channels))

        layers = []
        if expand_ratio != 1:
            # 1x1 conv(up sample)
            layers.append(ConvBNRelu(in_channels, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 dw conv
            ConvBNRelu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 conv(down sample)
            nn.Conv2d(hidden_channel, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobilenetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobilenetV2, self).__init__()

        input_channel = self._make_divisible(32 * alpha, divisor=round_nearest)
        last_channel = self._make_divisible(1280 * alpha, divisor=round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # conv1 layer
        features = [ConvBNRelu(3, input_channel, stride=2)]
        # inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c * alpha, divisor=round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNRelu(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

    @classmethod
    def initialize_model_for_learning(cls):
        import os
        model = cls(num_classes=5)
        model_save_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_pth = os.path.join(model_save_dir, 'best.pth')
        model_dict_json = os.path.join(model_save_dir, 'class_indices.json')
        return model, model_save_pth, model_dict_json

    @staticmethod
    def wget_pth():
        import os
        model_save_dir = os.path.dirname(os.path.abspath(__file__))

        url = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"
        os.system(f"wget {url} -O {os.path.join(model_save_dir, 'best.pth')}")
