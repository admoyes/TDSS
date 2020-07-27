import torch
from torch import nn
from torch.nn import functional as F


class Upsample(nn.Module):

    def __init__(self, input_channels, output_channels, drop_rate=0.0, batch_norm=False, activation=nn.LeakyReLU, bias=True):
        super(Upsample, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        parts = []

        self.drop_rate = drop_rate

        # upsample
        parts.append(nn.UpsamplingNearest2d(scale_factor=2))

        # convolution
        parts.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=bias))

        # norm
        if batch_norm:
            parts.append(nn.BatchNorm2d(output_channels))       

        # activation
        if activation is not None:
            parts.append(activation())

        # put parts into a sequential module
        self.main = nn.Sequential(*parts)

    def forward(self, x):
        
        z = x

        # dropout
        if self.drop_rate > 0.0:
            z = F.dropout2d(z, p=self.drop_rate)

        out = self.main(z)

        return out



class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, drop_rate=0.0, batch_norm=True, activation=nn.LeakyReLU, downsample=True, bias=True):
        super(ConvBlock, self).__init__()

        parts = []

        # dropout
        if drop_rate > 0.0:
            parts.append(nn.Dropout2d(drop_rate))

        if downsample:
            parts.append(nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=bias))
        else:
            parts.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=bias))


        # batch normalisation
        if batch_norm:
            parts.append(nn.BatchNorm2d(output_channels))                    

        # activation
        if activation is not None:
            parts.append(activation())

        # put parts into a sequential module
        self.main = nn.Sequential(*parts)

    def forward(self, x):
        return self.main(x)