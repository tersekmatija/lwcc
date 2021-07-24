from ..util.functions import weights_check

import torch.nn as nn
import torch


def make_model(model_weights):
    available_weights = ["SHA", "SHB", "QNRF"]

    if model_weights not in available_weights:
        raise ValueError("Weights {} not available for CSRNet. Available weights: {}".format(model_weights,
                                                                                             available_weights))
    weights_path = weights_check("DM-Count", model_weights)

    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(torch.load(weights_path, map_location ='cpu')["model"])

    return model


class VGG(nn.Module):

    def get_name(self):
        return "DM-Count"

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1) , nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.interpolate(x, scale_factor = 2, mode='bilinear', align_corners=True)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        # B, C, H, W = mu.size()
        # mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # mu_normed = mu / (mu_sum + 1e-6)
        return mu  # , mu_normed


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}
