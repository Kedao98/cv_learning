import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(InvertedResidual, self).__init__()

        self.stride = stride

        if stride == 1:
            self.branch1 = nn.Sequential()

        elif stride == 2:
            self.branch1 = nn.Sequential(
                # dw conv
                nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=3, groups=in_channel, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                # pw conv
                nn.Conv2d(out_channel, out_channel, kernel_size=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

        branch_features = in_channel // 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel if stride > 1 else branch_features, branch_features, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.concat((self.branch1(x), self.branch2(x)), dim=1)

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


