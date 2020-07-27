import toml
import os
import torch
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
from itertools import chain
from PIL import Image

config = toml.load("./config.toml")

from Models import FeatureNet, TDSS, ReconstructionLayer
from utils import absorption_criterion, plot_loss, RGB2OpticalDensity, OpticalDensity2RGB, mkdir, save_image
from Data import PatchDataset
from torch.utils.data import DataLoader
from torchvision import utils as vutils


"""
[0] Datasets, dataloaders, data transforms
"""

# check if a custom transform is needed
if bool(config["training_data"]["use_custom_transform"]):
    from Data import CustomTransform as TrainingTransform
else:
    from Data import DefaultTransform as TrainingTransform


INCIDENT_LIGHT = torch.FloatTensor(config["data"]["incident_light"]).to(torch.device(config["device"]))

# build the testing dataset
testing_dataset = PatchDataset(config["testing_data"]["path_to_patches"], TrainingTransform())

# build the testing data loader
testing_dataloader = DataLoader(
    testing_dataset,
    batch_size=int(config["testing_data"]["batch_size"]),
    shuffle=True,
    num_workers=int(config["testing_data"]["num_workers"])
)


"""
[1] Feature extraction.
    - if force_train is true, the feature extraction model must be trained
    - if force_train is false and the path_to_state has been provided, the state must be loaded, training is skipped
    - if force_train is false and the path_to_state has not been provided, raise an error
"""
feature_config = config["feature_extraction"]
feature_net = FeatureNet(**feature_config).to(torch.device(config["device"]))
feature_net_state_path = config["feature_extraction"]["path_to_state"]

assert len(feature_net_state_path) > 0 and os.path.exists(feature_net_state_path), "feature_extraction.path_to_state is not valid."

feature_net.load_state_dict(torch.load(feature_net_state_path))


"""
[2] Stain Separation
"""
tdss = TDSS(
    feature_net,
    int(config["feature_extraction"]["multiplier"]),
    int(config["stain_separation"]["number_of_stains"]),
    float(config["stain_separation"]["alpha"])
).to(torch.device(config["device"])).eval()
tdss_net_state_path = config["stain_separation"]["path_to_state"]

assert len(tdss_net_state_path) > 0 and os.path.exists(tdss_net_state_path), "stain_separation.path_to_state is not valid"

tdss.load_state_dict(torch.load(tdss_net_state_path))

output_root = "./output/testing/"
count = 0
for batch_id, data in enumerate(testing_dataloader):

    data = data.to(torch.device(config["device"]))

    # normalise by the incident light
    data = (data / INCIDENT_LIGHT.view(1, 3, 1, 1)).clamp(max=1.0)

    # convert to optical density
    data_od = RGB2OpticalDensity(data)

    # get stains, densities
    with torch.no_grad():
        reconstructed_stains, densities = tdss(data_od)

    # add individual stains together to form reconstruction of x
    reconstruction = torch.stack(reconstructed_stains, dim=0).sum(dim=0)
    reconstruction_rgb = OpticalDensity2RGB(reconstruction)

    #print("data", data.size())

    # save each of the required items
    for i in range(data.size(0)):

        item_root_dir = mkdir(os.path.join(output_root, str(count)))

        # data
        data_path = os.path.join(item_root_dir, "data.png")
        save_image(data[i, ...], data_path)

        # reconstruction
        reconstruction_rgb_path = os.path.join(item_root_dir, "reconstruction.png")
        save_image(reconstruction_rgb[i, ...], reconstruction_rgb_path)

        # stains
        for s, stain in enumerate(reconstructed_stains):
            stain_path = os.path.join(item_root_dir, "stain_{}.png".format(s))
            stain_rgb = OpticalDensity2RGB(stain)
            save_image(stain_rgb[i, ...], stain_path)

        # densities
        for j in range(densities.size(1)):
            density = densities[i, j, ...]
            density_path = os.path.join(item_root_dir, "density_{}.png".format(j))
            save_image(density, density_path)
            

        count += 1