import torch
from torch import nn
from .colours import ColourNet
from .densities import DensityNet


class TDSS(nn.Module):

    def __init__(self, feature_net, feature_channels, n_stains, alpha):
        super(TDSS, self).__init__()

        self.feature_net = feature_net
        self.colour_net = ColourNet(feature_channels, n_stains, alpha)
        self.density_net = DensityNet(feature_channels, n_stains)
        self.n_stains = n_stains

    def reconstruct_stains(self, colours, densities):
        # reconstruct each stain separately
        stains = []
        
        for i in range(self.n_stains):
            colours_i = colours[:, i, ...]
            densities_i = densities[:, i, ...].unsqueeze(1)
            stain_i = (colours_i * densities_i)
            stains.append(stain_i)
        
        return stains

    def forward(self, x):

        # get pixel-wise features
        _, z = self.feature_net(x)

        # get colours
        _, _, pixel_level_colours = self.colour_net(z)

        # get densities
        densities = self.density_net(z)

        # get individual stains
        reconstructed_stains = self.reconstruct_stains(pixel_level_colours, densities)
        
        return reconstructed_stains, densities



