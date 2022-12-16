"""This module contains the main training loop"""

from random import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointcloud_invariance_smoothing.models.invariance_wrappers import (
    PCAEnsembleWrapper, SinglePCAWrapper, StackingPCAWrapper)
from pointcloud_invariance_smoothing.models.pointnet import (
    PointNetClassifier, pointnet_loss)


def train(model, data_train, data_val,
          num_epochs, batch_size,
          learning_rate, weight_decay, regularization_weight,
          scheduler_stepsize, scheduler_gamma,
          rotate, scale, add_noise,
          scale_limits, std,
          rotate_validation, add_noise_validation,
          device, num_workers):
    """Main training loop, for meaning of parameters see seml/scripts/train.py.

    Returns:
        Four arrays of length num_epochs, tracking train loss, val loss,
        train accuracy, val accuracy
        and two state dicts containing model with best validation loss and model
        with best validation accuracy.
    """

    train_loader = DataLoader(data_train, drop_last=True, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(data_val, drop_last=False, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_stepsize,
                                                gamma=scheduler_gamma)

    loss_fn = pointnet_loss

    loss_trace_train = []
    acc_trace_train = []

    loss_trace_val = []
    acc_trace_val = []

    best_loss_val = None
    best_acc_val = None

    best_model_loss = None
    best_model_acc = None

    for _ in tqdm(range(num_epochs)):
        model.train()

        per_batch_losses = []
        per_batch_accs = []

        for X, targets in train_loader:
            optimizer.zero_grad()

            X, targets = X.to(device), targets.to(device)

            if rotate:
                X = randomly_rotate(X, device)

            if scale:
                X = randomly_scale(X, *scale_limits, device)

            if add_noise:
                X = add_gaussian_noise(X, std, device)

            logits, trans_feat = model(X)

            loss = loss_fn(logits, targets.long(), trans_feat, regularization_weight)
            loss.backward()

            optimizer.step()

            pred = logits.detach().max(dim=1)[1]
            n_correct = (pred == targets.long()).sum().cpu()
            per_batch_accs.append(int(n_correct) / batch_size)

            per_batch_losses.append(float(loss.detach().cpu()))

        loss_trace_train.append(np.mean(per_batch_losses))
        acc_trace_train.append(np.mean(per_batch_accs))

        scheduler.step()

        model.eval()

        per_batch_losses = []
        per_batch_accs = []

        for X, targets in val_loader:
            X, targets = X.to(device), targets.to(device)

            if rotate_validation:
                X = randomly_rotate(X, device)
                raise ValueError()

            if add_noise_validation:
                X = add_gaussian_noise(X, std, device)
            else:
                if add_noise:
                    raise ValueError()

            logits, _ = model(X)

            loss = loss_fn(logits, targets.long())

            pred = logits.detach().max(dim=1)[1]
            n_correct = (pred == targets.long()).sum().cpu()
            per_batch_accs.append(int(n_correct) / batch_size)

            per_batch_losses.append(float(loss.detach().cpu()))

        loss_trace_val.append(np.mean(per_batch_losses))
        acc_trace_val.append(np.mean(per_batch_accs))

        print(loss_trace_val[-1], acc_trace_val[-1])

        if isinstance(model, (PCAEnsembleWrapper, SinglePCAWrapper, StackingPCAWrapper)):
            state_dict = model.base_model.state_dict()
        else:
            state_dict = model.state_dict()

        if best_loss_val is None or loss_trace_val[-1] < best_loss_val:
            best_loss_val = loss_trace_val[-1]
            best_model_loss = state_dict
        if best_acc_val is None or acc_trace_val[-1] < best_acc_val:
            best_acc_val = acc_trace_val[-1]
            best_model_acc = state_dict

    return (loss_trace_train, loss_trace_val,
            acc_trace_train, acc_trace_val,
            best_model_loss, best_model_acc)


def add_gaussian_noise(x: torch.Tensor, std: float, device: torch.device) -> torch.Tensor:

    return x + std * torch.randn(x.shape).to(device)


def randomly_scale(x: torch.Tensor, scale_min: float, scale_max: float,
                   device: torch.device) -> torch.Tensor:
    """Scale each point cloud in batch by a different factor sampled from uniform distribution"""

    scale = torch.rand(len(x)) / (scale_max - scale_min) + scale_min
    scale = scale[:, None, None].to(device)

    return x * scale


def randomly_rotate(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Randomly rotate each point cloud in batch using uniformly sampled rotation matrix"""
    n_dim = x.shape[-1]
    batch_size = x.shape[0]

    rotation_matrices = generate_random_rotations(n_dim, batch_size).to(device)

    return torch.bmm(x, rotation_matrices.transpose(2, 1))


def generate_random_rotations(n_dim: int, n_samples: int) -> torch.Tensor:
    """Return n_samples different rotation matrices of shape n_dim x n_dim"""
    assert n_dim in [2, 3]

    if n_dim == 2:
        angles = np.random.uniform(0, 2 * np.pi, n_samples)

        rotation_matrices = np.empty((n_samples, n_dim, n_dim))
        rotation_matrices[:, 0, 0] = np.cos(angles)
        rotation_matrices[:, 0, 1] = -1 * np.sin(angles)
        rotation_matrices[:, 1, 0] = np.sin(angles)
        rotation_matrices[:, 1, 1] = np.cos(angles)

    if n_dim == 3:
        rotation_matrices = Rotation.random(n_samples).as_matrix()

    return torch.Tensor(rotation_matrices)
