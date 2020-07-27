import torch
from torch import nn, optim
from .u_net import UNet
from .blocks import ConvBlock
from itertools import chain
from tqdm import tqdm

class FeatureNet(nn.Module):

    def __init__(self, **kwargs):
        super(FeatureNet, self).__init__()

        in_channels = kwargs["input_channels"]
        multiplier = kwargs["multiplier"]

        self.first = ConvBlock(in_channels, multiplier, downsample=False)
        self.u_net = UNet(**kwargs)

    def forward(self, x):
        z = self.first(x)
        return self.u_net(z)

class ReconstructionLayer(nn.Module):

    """ Used to help train the FeatureNet. Maps from features to reconstruction of tissue image. """

    def __init__(self, multiplier, output_channels):
        super(ReconstructionLayer, self).__init__()

        self.main = nn.Conv2d(multiplier, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, z):
        return self.main(z)
