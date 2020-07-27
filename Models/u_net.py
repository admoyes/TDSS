import torch
from torch import nn
from .encoder import Encoder
from .decoder import Decoder


class UNet(nn.Module):

    def __init__(self, **kwargs):
        super(UNet, self).__init__()

        self.encoder = Encoder(**kwargs)

        self.decoder = Decoder(**kwargs)

    def forward(self, x):
        encoder_activations = self.encoder(x)
        decoded = self.decoder(encoder_activations)

        return encoder_activations, decoded