import torch
from torch import nn
from torch.nn import functional as F
from .blocks import Upsample, ConvBlock


class Decoder(nn.Module):

    def __init__(self, num_blocks=4, multiplier=16, drop_rate=0.0, **kwargs):
        super(Decoder, self).__init__()
        self.multiplier = multiplier


        """ 
        Decoder accepts a *list* of activations (from each layer of the decoder).
        For each block in the decoder, we will concatenate the output from the previous decoder block (if it exists)
        as well as the output from the appropriate encoder block.
        """

        blocks = []
        input_channels = multiplier * (2 ** num_blocks)
        output_channels = input_channels // 2

        for block_id in range(num_blocks):

            """
            If this isn't the first decoder block then we are concatenating the outputs from the previous decoder
            layer and the corresponding encoder layer, so the true input channels is doubled.
            """
            if block_id > 0:
                actual_input_channels = 2 * input_channels
            else:
                actual_input_channels = input_channels

            # create block
            up_block = Upsample(actual_input_channels, output_channels, drop_rate=drop_rate)
            blocks.append(up_block)

            # if this isn't the last layer, update input_channels and output_channels
            if block_id < num_blocks - 1:
                input_channels = output_channels
                output_channels = output_channels // 2
            else:
                final_block = nn.Sequential(
                    nn.Conv2d(output_channels, multiplier, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(True)
                )
                blocks.append(final_block)

        # put blocks into a module list so we can iterate them and the optimiser is aware of them
        self.blocks = nn.ModuleList(blocks)


    def forward(self, activations):
        # input is list of activations from each encoder block

        # placeholder
        last_decoder_activation = None

        # iterate decoder blocks
        for block_id, block in enumerate(self.blocks[:-1]):

            # get the last activation from activations (and remove it from the list)
            last_encoder_activation = activations[::-1][block_id]

            # if this is the first block, we aren't concatenating.
            if block_id == 0:
                block_input = last_encoder_activation
            else:
                # concatenate last decoder activation and corresponding encoder activation
                block_input = torch.cat([
                    last_encoder_activation,
                    last_decoder_activation
                ], dim=1)

            last_decoder_activation = block(block_input)

        # final conv
        last_decoder_activation = self.blocks[-1](last_decoder_activation)
        return last_decoder_activation

       