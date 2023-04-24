import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        branch_features = out_channel // 2
        if stride == 1:
            self.branch1 = nn.Sequential()

        elif stride == 2:
            self.branch1 = nn.Sequential(
                # dw conv
                nn.Conv2d(in_channel, in_channel, stride=stride, kernel_size=3, groups=in_channel, padding=1, bias=False),
                nn.BatchNorm2d(in_channel),
                # pw conv
                nn.Conv2d(in_channel, branch_features, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel if stride > 1 else branch_features, branch_features, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # dw conv
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, groups=branch_features, padding=1, bias=False),
            nn.BatchNorm2d(branch_features),
            # pw conv
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:

        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
        x = x.view(batch_size, groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batch_size, -1, height, width)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes):
        super(ShuffleNetV2, self).__init__()

        # input RGB image
        in_channel, out_channel = 3, stages_out_channels[0]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        in_channel = out_channel
        for i in range(len(stages_repeats)):
            name, repeats, out_channel = f"stage{i+2}", stages_repeats[i], stages_out_channels[1:][i]
            seq = [InvertedResidual(in_channel, out_channel, 2)] + \
                  [InvertedResidual(out_channel, out_channel, 1) for _ in range(1, repeats)]
            setattr(self, name, nn.Sequential(*seq))
            in_channel = out_channel

        out_channel = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(out_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(1, 5):
            x = getattr(self, f"stage{i}")(x)
        x = self.conv5(x)
        x = x.mean((2, 3))  # global pool
        x = self.fc(x)
        return x

    @classmethod
    def initialize_model_for_learning(cls):
        import os
        model = cls(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024], num_classes=5)
        model_save_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_pth = os.path.join(model_save_dir, 'best.pth')
        model_dict_json = os.path.join(model_save_dir, 'class_indices.json')
        return model, model_save_pth, model_dict_json


if __name__ == '__main__':
    model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024], num_classes=5)
    print(model(torch.randn(1, 3, 224, 224)).shape)
