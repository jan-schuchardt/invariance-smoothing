"""This module contains functionality for loading the point cloud datasets

get_dataset: Main function for loading specific dataset.

ToPointCloud: Maps image datasets like MNIST to 2D pointclouds

RemapTargets: Changes labels of dataset, used to merge class 6 and 9 in MNIST.
"""

import os
import sys

import h5py
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.utils import _log_api_usage_once

from pointcloud_invariance_smoothing.utils import dotdict

sys.path.append('../reference_implementations/Pointnet_Pointnet2_pytorch/data_utils')
from ModelNetDataLoader import ModelNetDataLoader


def get_dataset(name: str, data_folder: str,
                val_percentage: float = 0.2) -> tuple[Dataset, Dataset, Dataset]:
    """Returns specified dataset, with fixed percentage of train set used for validation"""
    name = name.lower()
    assert name in ['modelnet40', 'mnist', 'scanobjectnn']

    if name == 'modelnet40':

        data_args = {
            'num_point': 1024,
            'use_uniform_sample': True,
            'use_normals': False,
            'num_category': 40
        }

        data_args = dotdict(data_args)

        data_train = ModelNetDataLoader(root=data_folder, args=data_args,
                                        split='train', process_data=True)

        data_test = ModelNetDataLoader(root=data_folder, args=data_args,
                                       split='test', process_data=True)

    elif name == 'mnist':

        target_transform = RemapTargets({
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: 5,  6: 6, 7: 7, 8: 8, 9: 6
        })

        data_train = MNIST(data_folder, train=True,
                           transform=T.Compose([T.ToTensor(), ToPointCloud()]),
                           target_transform=target_transform)

        data_test = MNIST(data_folder, train=False,
                          transform=T.Compose([T.ToTensor(), ToPointCloud()]),
                          target_transform=target_transform)

    elif name == 'scanobjectnn':
        data_train = ScanObjectNN(root=data_folder, split='train')
        data_test = ScanObjectNN(root=data_folder, split='test')

    n_val = int(val_percentage * len(data_train))
    n_train = len(data_train) - n_val

    data_train, data_val = random_split(data_train, (n_train, n_val),
                                        generator=torch.Generator().manual_seed(0))

    return data_train, data_val, data_test


class ToPointCloud(nn.Module):
    """Layer that converts input greyscale image to 2D point cloud.

    Implements the pre-processing steps described in the appendix of
    the PointNet Paper (https://arxiv.org/abs/1612.00593), i.e.
    thresholding grey-scale values, and then either subsampling or
    padding the resulting point cloud.

    Args:
        threshold: Value between 0 and 1.0 used for thresholding gray-scale values
        n_points: Number of points that resulting point cloud should have
    """

    def __init__(self, threshold=0.5, n_points=256):
        super(ToPointCloud, self).__init__()
        _log_api_usage_once(self)

        self.threshold = threshold
        self.n_points = n_points

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not torch.all((image <= 1) & (image >= 0)):
            raise ValueError('Only support pixel values in [0, 1]')

        # Only support grey-scale (i.e. single-channel) images
        assert image.ndim == 3 and image.shape[0] == 1
        image = image[0]

        height, width = image.shape

        # Convert pixel indices to 2D coordinates
        coordinates = torch.torch.stack(torch.meshgrid(
                                            torch.linspace(0, 1, height),
                                            torch.linspace(0, 2, width)))

        threshold_mask = image >= self.threshold
        point_cloud = coordinates[:, threshold_mask].T  # N_above_threshold x 2

        # Normalize
        point_cloud -= point_cloud.mean(dim=0)
        max_len = torch.linalg.norm(point_cloud, dim=1).max()
        if max_len > 0:
            point_cloud /= max_len

        if len(point_cloud) < self.n_points:
            # PointNet does max-pooling, so can just pad with same value without changing prediction
            padding_value = point_cloud[0].unsqueeze(0)

            point_cloud = torch.cat(
                [point_cloud,
                    torch.repeat_interleave(padding_value, self.n_points - len(point_cloud), dim=0)
                 ],
                dim=0)

        elif len(point_cloud) > self.n_points:
            # If two many points, subsample uniformly at random
            subsample_idcs = torch.randperm(len(point_cloud))[:self.n_points]
            point_cloud = point_cloud[subsample_idcs]

        return point_cloud


class ScanObjectNN(Dataset):
    def __init__(self, root, split='train', pca_preprocessed=True):
        self.data_list = []
        self.split = split

        if split not in ['train', 'test']:
            raise ValueError('Only have train and test set')

        data_dir = os.path.join(root,
                                'pca' if pca_preprocessed else 'ori',
                                split)

        for file_name in open(os.path.join(data_dir, f'{split}_list.txt')):
            self.data_list.append(os.path.join(data_dir, file_name).rstrip())

    def __getitem__(self, ind):
        file = h5py.File(self.data_list[ind] + '.h5', 'r', swmr=True)
        data = file['data'][:]
        pointcloud = data[:1024, :]
        data, label = torch.from_numpy(pointcloud), torch.from_numpy(file['label'][:])
        file.close()
        return data, int(label)

    def __len__(self):
        return len(self.data_list)


class RemapTargets(nn.Module):
    """Module that replaces each label with a specific other label.

    We use this to combine class 6 and class 9 from MNIST into a single class.

    Args:
        remap_dict: A dictionary, with each key being the original label
            and the value being the new label
    """
    def __init__(self, remap_dict: dict[int, int]):
        super(RemapTargets, self).__init__()
        _log_api_usage_once(self)

        self.remap_dict = remap_dict

    def forward(self, x: int) -> int:
        return self.remap_dict[x]
