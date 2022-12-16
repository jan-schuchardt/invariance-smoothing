"""This module contains functionality for evaluating the smoothed classifier

sample_votes_datapoint: Count how often specific input is classified as each class.

sample_votes_dataset: Apply randomized smoothing to multiple datapoints from dataset.
"""

from multiprocessing.sharedctypes import Value

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn as nn


def sample_votes_datapoint(
        X: torch.Tensor, model: nn.Module,
        std: float, n_samples: int, batch_size: int,
        device: torch.device) -> np.ndarray:
    """Apply model to multiple samples from smoothing distribution, count preds per class.

    Args:
        X: Input point cloud of shape N x D
        model: Classifier
        std: Smoothing standard deviation
        n_samples: Number of samples to classify
        batch_sizes: Number of samples to classify simultaneously
        device: Torch device (cpu or cuda device)

    Returns:
        Numpy array with one entry per class, each entry containing the number of times
        this class was predicted.
    """

    assert X.ndim == 2

    X = X.unsqueeze(0).to(device)

    n_samples_remaining = n_samples

    votes = None

    while n_samples_remaining > 0:
        n_samples_batch = min(n_samples_remaining, batch_size)

        X_pert = X + std * torch.randn(n_samples_batch, *X.shape[1:]).to(device)

        logits, _ = model(X_pert)
        logits = logits.cpu()
        assert logits.ndim == 2

        if votes is None:
            n_classes = logits.shape[-1]
            votes = torch.zeros(n_classes, dtype=torch.long)

        pred = logits.max(dim=1)[1]

        votes = votes.scatter_add(0, pred, torch.ones_like(pred))

        n_samples_remaining -= n_samples_batch

    assert votes.sum() == n_samples

    return votes.detach().cpu().numpy()


def sample_votes_dataset(
        dataset: Dataset, n_datapoints: int, model: nn.Module,
        std: float, n_samples_pred: int, n_samples_cert: int,
        batch_size: int, device: torch.device) -> dict:
    """Obtain prediction and certification MC samples for data in dataset.

    Args:
        Dataset: Set of point clouds with associated ground truth class
        n_datapoints: Will only obtain MC samples for random subset of size n_datapoints
            from entire dataset
        model: Classifier
        std: Smoothing standard deviation
        n_samples_pred: Number of samples for prediction
        n_samples_cert: Number of samples for certification
        batch_sizes: Number of samples to classify simultaneously
        device: Torch device (cpu or cuda device)

    Returns:
        Dictionary with five entries:
            1.) 'datapoint_idx': Array of length n_datapoints, containing indices
                of points we obtained Monte Carlo samples for
            2.) 'targets: Array of legnth n_datapoints, containing corresponding
                ground truth labels
            3.) 'votes_pred': Array of shape n_datapoints x N_classes, with entry
                n, c indicating how many times datapoint n was classified as class c
                under Gaussian noise (each row sums up to n_samples_pred)
            4.) 'accuracy' (float): The accuracy of the smoothed predictions (i.e. argmax along
                dim 1 of votes_pred).
            5.) votes_cert: Array of shape n_datapoints x N_classes, with entry
                n, c indicating how many times datapoint n was classified as class c
                under Gaussian noise (each row sums up to n_samples_cert)

    """

    model.eval()

    n_datapoints = min(n_datapoints, len(dataset))

    idx = np.random.permutation(len(dataset))[:n_datapoints]

    votes_pred = []
    votes_cert = []
    targets = []
    accuracy = 0

    for i in tqdm(idx):
        X, target = dataset[i]
        target = int(target)
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)

        with torch.no_grad():

            if n_samples_pred > 0:
                vp = sample_votes_datapoint(X, model, std, n_samples_pred, batch_size, device)
                votes_pred.append(vp)
                accuracy += (np.argmax(vp) == target) / n_datapoints
            if n_samples_cert > 0:
                vc = sample_votes_datapoint(X, model, std, n_samples_cert, batch_size, device)
                votes_cert.append(vc)

        targets.append(target)

    sample_dict = {
        'datapoint_idx': np.array(idx),
        'targets': np.array(targets)
    }

    if n_samples_pred > 0:
        sample_dict['votes_pred'] = np.vstack(votes_pred)
        sample_dict['accuracy'] = accuracy

    if n_samples_cert > 0:
        sample_dict['votes_cert'] = np.vstack(votes_cert)

    return sample_dict
