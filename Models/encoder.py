import torch
from torch import nn
from .blocks import ConvBlock


class Encoder(nn.Module):

    def __init__(self, num_blocks=4, multiplier=16, drop_rate=0.0, **kwargs):
        super(Encoder, self).__init__()


        # create convolution blocks
        blocks = []
        input_channels = multiplier
        output_channels = 2 * multiplier

        for block_id in range(num_blocks):
            block = ConvBlock(input_channels, output_channels, drop_rate=drop_rate)
            blocks.append(block)

            # if this isn't the last block, update input_channels and output_channels
            if block_id < num_blocks - 1:
                input_channels = output_channels
                output_channels *= 2

        # put blocks into a module list so we can iterate them and the optimiser is aware of them
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        activations = []

        # iterate layers and save output of each
        for block_id, block in enumerate(self.blocks):
            x = block(x)
            activations.append(x)

        return activations

