# Unsupervised Deep Learning for Stain Separation and Artifact Detection in Histopathology Images
This repository contains a PyTorch implementation of the stain separation model proposed in [this paper](https://link.springer.com/chapter/10.1007/978-3-030-52791-4_18). This paper was presented at [MIUA2020](https://miua2020.com/) and won the *best paper award*. A video summary of this paper can be found [here](https://www.youtube.com/watch?v=HQ6kL6eVdig).

The TDSS model uses a U-Net convolutional neural network to capture texture, structure and colour features from tissue images. This information is then used to estimate the stain colours and densities. This model (TDSS) is based on a modified approach to Non-negative Matrix Factorisation where a stain colour matrix is estimated for every pixel location. It was found that allowing the stain colours to vary at each pixel location, improved stain separation results are found.

## Getting Started
- Our local environment is a typical conda environment
- Make sure you have a proper nvidia driver if using a GPU
- Install [PyTorch](https://pytorch.org/get-started/locally/) along with `torchvision` and `cudatoolkit` (if using GPU)
- Install `pillow`, `pandas`, `seaborn`, `toml`, `tqdm` and `matplotlib`
	- Specific versions can be found in the `requirements.txt` file

### Training
- Ensure the `path_to_patches` paramter in `config.toml` under `training_data` points to a directory that is populated with tissue image patches
	- The model was originally trained with 128x128 pixel patches but this is not hard coded (See config)
- Run `python main.py`
- Model output will be output at `./output/training/`

### Testing
- Again, ensure the `path_to_patches` parameter in `config.toml` under `testing_data` is populated with tissue image patches.
- Run `python test.py`
- Model output will be saved to disk at `./output/testing/*`

	- Configuration is done in `config.toml`
- To test the model, run `python test.py`

### Recommendations
- An accurate estimation of the incident light is needed to properly perform stain separation.
    - This can be done by manually selecting pixels that **do not** contain tissue in your preferred image editing software (e.g. [GIMP](https://www.gimp.org/)) and looking at the *histogram* section which will let you see the average RGB intensity values. 
    - Once you have these values, make sure the `incident_light` parameter under `data` in the `config.toml` file is updated accordingly.

## Citation
If you use this code in your research, please cite the [paper](https://link.springer.com/chapter/10.1007/978-3-030-52791-4_18):

```
@inproceedings{moyes2020unsupervised,
  title={Unsupervised Deep Learning for Stain Separation and Artifact Detection in Histopathology Images},
  author={Moyes, Andrew and Zhang, Kun and Ji, Ming and Zhou, Huiyu and Crookes, Danny},
  booktitle={Annual Conference on Medical Image Understanding and Analysis},
  pages={221--234},
  year={2020},
  organization={Springer}
}
```

