import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = Conv(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            Conv(in_channels, ch3x3red, kernel_size=1),
            Conv(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            Conv(in_channels, ch5x5red, kernel_size=1),
            Conv(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        outputs = [getattr(self, f"branch{i}")(x) for i in range(1, 5)]
        return torch.cat(outputs, dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = Conv(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.header = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            Conv(in_channels, 128, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 1024),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.header(x)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, use_aux=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.use_aux = use_aux

        self.conv1 = Conv(3, 64, kernel_size=7, padding=3, stride=2)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = Conv(64, 64, kernel_size=1)
        self.conv3 = Conv(64, 192, kernel_size=3, padding=1)

        self.max_pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.use_aux:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
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

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.max_pool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.max_pool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.max_pool3(x)
        # N x 480 x 14 x 14
        x_4a = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x_4a)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x_4d = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x_4d)
        # N x 832 x 14 x 14
        x = self.max_pool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avg_pool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.use_aux:   # eval model lose this layer
            aux1 = self.aux1(x_4a)
            aux2 = self.aux2(x_4d)
            return x, aux2, aux1
        return x

    @classmethod
    def initialize_model_for_learning(cls):
        import os
        model = cls(num_classes=5, use_aux=True, init_weights=True)
        model_save_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_pth = os.path.join(model_save_dir, 'best.pth')
        model_dict_json = os.path.join(model_save_dir, 'class_indices.json')
        return model, model_save_pth, model_dict_json


if __name__ == '__main__':
    m = AuxClassifier(512, 10)
    img = torch.randn(1, 512, 14, 14)
    print(m.forward(img).shape)
