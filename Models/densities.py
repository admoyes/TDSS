import torch
from torch import nn
from .blocks import ConvBlock


class DensityNet(nn.Module):

    def __init__(self, feature_channels, n_stains):
        super(DensityNet, self).__init__()

        self.main = nn.Sequential(
            ConvBlock(feature_channels, feature_channels // 2, downsample=False),
            nn.Conv2d(feature_channels // 2, n_stains, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.main(z)