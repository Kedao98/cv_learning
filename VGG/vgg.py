import os.path

import torch
import torch.nn as nn


vgg_cfg = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
}


class VGG(nn.Module):
    def __init__(self, cfg: list, class_num: int, init_weight: bool = True):
        super().__init__()
        self.cfg = cfg
        self.class_num = class_num
        self.backbone = self.construct_backbone()
        self.header = self.construct_header()
        if init_weight:
            self._initialize_weights()

    def construct_backbone(self):
        backbone = nn.Sequential()
        in_channels = 3
        for layer in self.cfg:
            if layer == 'M':
                backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(layer, int):
                backbone.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                backbone.append(nn.ReLU(True))
                in_channels = layer
        return backbone

    def construct_header(self):
        header = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),

            nn.Linear(2048, self.class_num),
        )
        return header

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N 3 224 224
        x = self.backbone(x)
        # N 512 7 7
        x = torch.flatten(x, start_dim=1)
        # N 512 * 7 * 7
        x = self.header(x)
        return x

    @classmethod
    def initialize_model_for_learning(cls):
        model = cls(vgg_cfg['vgg16'], class_num=5, init_weight=True)
        model_save_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_pth = os.path.join(model_save_dir, 'best.pth')
        model_dict_json = os.path.join(model_save_dir, 'class_indices.json')
        return model, model_save_pth, model_dict_json


if __name__ == '__main__':
    model, pth, js = VGG.initialize_model_for_learning()
    img = torch.randn(1, 3, 224, 224)
    print(model.forward(img).shape)
    print(pth)
    print(js)
