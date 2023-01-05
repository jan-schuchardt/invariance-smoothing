# Invariance-Aware Randomized Smoothing Certificates

<p align="left">
<img src="https://www.cs.cit.tum.de/fileadmin/_processed_/3/e/csm_invariance_main_fig_7a0b96296c.png", width="50%">

This is the official reference implementation of 

["Invariance-Aware Randomized Smoothing Certificates"](https://openreview.net/forum?id=5TfqL2gWdV9)  
Jan Schuchardt, and Stephan Günnemann, NeurIPS 2022.

## Requirements
Our main requirements are numpy, torch, torchvision and the ["Slurm Experiment Management Library (SEML)"](https://github.com/TUM-DAML/seml).

To install the requirements, execute
```
conda env create -f requirements.yaml
```

You also need to download reference implementations for different point cloud classifiers we use in our experiments,
which can be found in the `reference implementations` subfolder.
```
git submodule init
git submodule update
```

## Installation
You can install this package via `pip install .`

## Data
To reproduce our experiments, you need two datasets: MNIST and the ModelNet40 point cloud dataset.

MNIST can be installed via torchvision:
```
from torchvision.datasets import MNIST

mnist_dir = 'your_folder_here'

MNIST('./data', train=True, download=True)
MNIST('./data', train=False, download=True)
```

For the ModelNet40 dataset, download the original and pre-processed files linked in the [Pointnet_Pointnet2_pytorch repository](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

## Usage
In the `demos` subfolder, you can find two notebooks that explain and recreate the four different types of experiments from our paper.

In order to reproduce all experiments, you will need need to execute the scripts in `seml/scripts` using the config files provided
in `seml/configs`.  
There are four different scripts:
* train.py: Used for training the base classifiers for randomized smoothing
* sample_votes.py: Used for randomly sampling predictions from a base classifier under the smoothing distribution
* eval_forward.py: Used for evaluating the certificates, i.e. how robust predictions are under a specific form of perturbation
* eval_inverse.py: Used for evaluating the inverse certificates, i.e. the minimum clean prediction probability required for certifying robustness (does not require training or sampling).

After that, you can use the notebooks in `plotting` to recreate the figures from the paper.

For more details on which config files and plotting notebooks to use for recreating which figure from the paper, please consult REPRODUCE.MD

## Licenses

This repository includes code from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [DGCNN](https://github.com/WangYueFt/dgcnn) and 
[Rotation-invariant-deep-pointcloud-analysis](https://github.com/SILI1994/rotation-invariant-pointcloud-analysis), which are made available under [MIT license](https://opensource.org/licenses/MIT).


## Cite
Please cite our paper if you use this code in your own work:

```
@InProceedings{Schuchardt2022_Invariance,
  author = {Schuchardt, Jan and G{\"u}nnemann, Stephan},
  title     = {Invariance-Aware Randomized Smoothing Certificates},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
}
```
