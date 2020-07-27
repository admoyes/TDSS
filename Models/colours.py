import torch
from torch import nn
from .blocks import ConvBlock


class ColourNet(nn.Module):

    def __init__(self, input_channels, n_stains, alpha):
        super(ColourNet, self).__init__()

        self.n_stains = n_stains
        self.alpha = alpha

        self.image_level_colours = nn.Parameter(torch.rand(n_stains, 3))
        self.register_parameter("image_level_colours", self.image_level_colours)
        
        self.offset_net = nn.Sequential(
            ConvBlock(input_channels, input_channels // 2, downsample=False),
            ConvBlock(input_channels // 2, 3 * n_stains, downsample=False, activation=None)
        )

    def forward(self, z):        

        # estimate offsets from the image_level_colours
        pixel_level_offsets = self.offset_net(z)
        batch_size, _, patch_size, _ = pixel_level_offsets.size()
        
        # reshape pixel_level_offsets to make addition with image_level_colours easier
        pixel_level_offsets = pixel_level_offsets.view(batch_size, self.n_stains, 3, patch_size, patch_size).clamp(min=-self.alpha, max=self.alpha)

        # reshape image_level_colours to make addition easier
        image_level_colours = self.image_level_colours.view(1, self.n_stains, 3, 1, 1)

        # add offsets and average colours ( and clamp to ensure the colours are non-negative)
        corrected_colours = (pixel_level_offsets + image_level_colours).clamp(min=0.0)

        return self.image_level_colours, pixel_level_offsets, corrected_colours
