import torch
from matplotlib import pyplot as plt
import pandas as pd
import seaborn
import os
from PIL import Image
import numpy as np


def absorption_criterion(individual_stains):
    losses = [torch.norm(stain, dim=1, p=2).mean() for stain in individual_stains]
    return torch.stack(losses).mean()


def plot_loss(items, output_path):

    """
    create a line plot of the selected loss

    Parameters:
        items (list): contains the loss values for each timestep in the format (epoch, batch_id, loss value)
        output_path (str): location to save the plot
    """

    df = pd.DataFrame(items, columns=["epoch", "batch_id", "value"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    seaborn.lineplot(x="epoch", y="value", data=df, ax=ax)
    fig.savefig(output_path)
    plt.close(fig)


def RGB2OpticalDensity(im):
    """ Convert an RGB image to the optical density space. Assumes that transforms.ToTensor() has been applied beforehand.
    
    Args:
        im (torch.FloatTensor): RGB Image to be converted. Must be in range [0., 1.].

    Returns:
        (torch.FloatTensor): Image in optical density space
    """

    # clamp values to avoid divide by zero error
    im = im.clamp(min=1e-8)

    # convert to optical density
    im_od = -torch.log(im)

    return im_od


def OpticalDensity2RGB(im_od):
    """ Convert an image in the optical density space to the RGB space. 
    
    Args:
        im_od (torch.FloatTensor): optical density image.

    Returns:
        (torch.FloatTensor): Image in the RGB space (range [0., 1.])
    """

    # clamp values to avoid divide by zero
    im_od = im_od.clamp(min=1e-8)

    im_rgb = torch.exp(-im_od)

    return im_rgb


def mkdir(p):

    try:
        os.makedirs(p)
    except OSError:
        pass

    return p



def save_image(item, path):

    """ Converts a given torch tensor into the correct format and saves it to the path 

    Args:
        item (torch.FloatTensor): image data from tensor in format [n_channels, patch_size, patch_size]
        path (str): location to save the image
    """

    size = len(item.size())

    # make sure item is on the CPU
    item = item.cpu()

    # detach if needed
    if item.requires_grad_:
        item = item.detach()

    # check if it needed to be permuted
    if size == 3:
        item = item.permute(1, 2, 0).contiguous()

    # normalise to [0, 255]
    item_norm = (item.numpy() * 255).astype(np.uint8)

    # save it
    Image.fromarray(item_norm).save(path)