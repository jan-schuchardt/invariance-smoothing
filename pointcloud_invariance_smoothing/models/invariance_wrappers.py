"""This module contains PCA-based wrappers to make point cloud classifiers rotation-invariant.

SinglePCAWrapper: For use in training. Feeds a single PCA-projected sample into model

StackingPCAWrapper: For PointNetAttention. Feeds all possible PCA projections into model at once.

EnsemblePCAWrapper: Feeds all possible PCA projections into model, averages logits.

For more details on what "all possible PCA projections" are, see discussion
in "A Closer Look at Rotation-Invariant Deep Point Cloud Analysis" by Li et al.
"""

import itertools
from math import factorial
from readline import set_completer_delims

import numpy as np
import torch
import torch.nn as nn

from pointcloud_invariance_smoothing.models.dgcnn import DGCNNClassifier
from pointcloud_invariance_smoothing.models.pointnet import PointNetClassifier


class SinglePCAWrapper(nn.Module):
    """Performs a single PCA projection of the input data and passes it to the base model.

    Args:
        base_model: The classifier to wrap
        random_sign: Whether to randomize the sign of each dimension after PCA projection
        random_order: Whether to randomize the order of the dimensions after PCA projection
        n_point_dim: Number of point cloud dimensions (not number of points)
        n_feat_dim: Should be 0, was meant for attributed point clouds
    """

    def __init__(self, base_model: nn.Module,
                 random_sign: bool = False, random_order:bool = False,
                 n_point_dim: int = 3, n_feat_dim: int = 0):
        super(SinglePCAWrapper, self).__init__()

        self.base_model = base_model
        self.random_sign = random_sign
        self.random_order = random_order
        self.n_point_dim = n_point_dim
        self.n_feat_dim = n_feat_dim

        if n_feat_dim != 0:
            raise NotImplementedError('Pointclouds with features not supported yet')

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        assert D == self.n_point_dim

        device = x.device

        x_centered = x - x.mean(dim=1, keepdims=True)
        cov = torch.bmm(x_centered.transpose(2, 1) / np.sqrt(N), x_centered / np.sqrt(N))

        cov = cov.cpu()

        _, evecs = torch.linalg.eigh(cov)
        evecs = evecs.to(device)

        if self.random_sign:
            signs = torch.randint(0, 2, (B, 1, D)).float().to(device)
            signs[signs == 0] = -1
            evecs *= signs

        if self.random_order:
            for b in range(B):
                evecs[b] = evecs[b, :, torch.randperm(D)]

        x = torch.bmm(x_centered, evecs)

        return self.base_model(x)


class StackingPCAWrapper(nn.Module):
    """Stacks results of all possible PCA projections, for use with PointNetAttention model.

    If all eigenvalues are distinct, we can ignore the order-ambiguity, i.e. only have to
    consider different eigenvalue signs.
    But if multiple eigenvalues are similar, we have to stack all possible permutations
    of the corresponding dimensions.
    See discussion in "A Closer Look at Rotation-Invariant Deep Point Cloud Analysis"
    by Li et al.

    Args:
        base_model: The classifier to wrap
        n_point_dim: Number of point cloud dimensions (not number of points)
        n_feat_dim: Should be 0, was meant for attributed point clouds
        proper_rotations_only: Only stack results of projection with determinant 1
        random_order: Randomize stacking order of the different projection results
        stack_ambiguous_orders: Stack all permutations when eigenvalues are not distinct.
    """

    def __init__(self, base_model: nn.Module, n_point_dim: int = 3, n_feat_dim: int = 0,
                 proper_rotations_only: bool = True,
                 random_order: bool = False,
                 stack_ambiguous_orders: bool = True
                 ):
        super(StackingPCAWrapper, self).__init__()

        self.base_model = base_model
        self.n_point_dim = n_point_dim
        self.n_feat_dim = n_feat_dim
        self.proper_rotations_only = proper_rotations_only

        self.random_order = random_order
        self.stack_ambiguous_orders = stack_ambiguous_orders

        if n_feat_dim != 0:
            raise NotImplementedError('Pointclouds with features not supported yet')

        if self.random_order and self.stack_ambiguous_orders:
            raise ValueError('Cannot both randomize order and stack all ambiguous orders')

    def _stack_sign_ambiguities(self, evecs):
        B, D = evecs.shape[:2]

        stacked_evecs = torch.empty((B, D, 0))

        for sign_vector in itertools.product([-1, 1], repeat=D):

            sign_vector = torch.Tensor(sign_vector)
            if self.proper_rotations_only and torch.prod(sign_vector) == -1:
                continue

            if self.random_order:

                stacked_evecs = torch.cat((stacked_evecs,
                                          evecs[:, :, torch.randperm(D)] * sign_vector),
                                          dim=2)

            else:

                stacked_evecs = torch.cat((stacked_evecs,
                                          evecs * sign_vector),
                                          dim=2)

        return stacked_evecs  # B x D x 2^D (if not proper_rotations_only)

    def _stack_order_and_sign_ambiguities(self, evecs):
        B, D = evecs.shape[:2]

        stacked_evecs = torch.empty((B, D, 0))

        for permutation in itertools.permutations(range(D)):
            evecs_permuted = evecs.detach().clone()[:, :, permutation]

            stacked_evecs = torch.cat(
                            (stacked_evecs,
                             self._stack_sign_ambiguities(evecs_permuted)),
                            dim=2)

        return stacked_evecs

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        assert D == self.n_point_dim

        device = x.device

        x_centered = x - x.mean(dim=1, keepdims=True)
        cov = torch.bmm(x_centered.transpose(2, 1) / np.sqrt(N), x_centered / np.sqrt(N))

        cov = cov.cpu()

        evals, evecs = torch.linalg.eigh(cov)  # B x D x D

        order_ambiguity_mask = torch.zeros(B, dtype=torch.bool)

        for i, j in itertools.combinations(range(D), 2):
            order_ambiguity_mask |= torch.isclose(evals[:, i], evals[:, j])

        if not self.stack_ambiguous_orders:
            order_ambiguity_mask[:] = False

        if order_ambiguity_mask.sum() < B:
            stacked_poses = self._stack_sign_ambiguities(
                                evecs[~order_ambiguity_mask]).to(device)

            x = torch.bmm(x_centered[~order_ambiguity_mask], stacked_poses)  # B x N x (P * d)
            x = x.view(B, N, -1, D)  # B x N x P x D

            logits_no_order_ambiguity, _ = self.base_model(x)

            n_classes = logits_no_order_ambiguity.shape[-1]

        if order_ambiguity_mask.sum() > 0:
            stacked_poses = self._stack_order_and_sign_ambiguities(
                                    evecs[order_ambiguity_mask]).to(device)

            x = torch.bmm(x_centered[order_ambiguity_mask], stacked_poses)  # B x N x (P * d)
            x = x.view(B, N, -1, D)  # B x N x P x D

            logits_order_ambiguity, _ = self.base_model(x)

            n_classes = logits_order_ambiguity.shape[-1]

        logits = torch.zeros((B, n_classes)).to(device)

        if order_ambiguity_mask.sum() < B:
            logits[~order_ambiguity_mask] = logits_no_order_ambiguity

        if order_ambiguity_mask.sum() > 0:
            logits[order_ambiguity_mask] = logits_order_ambiguity

        return logits, None


class PCAEnsembleWrapper(nn.Module):
    """Applies wrapped model to all possible results of PCA projection, then averages logits.

    If all eigenvalues are distinct, we can ignore the order-ambiguity, i.e. only have to
    consider different eigenvalue signs.
    But if multiple eigenvalues are similar, we have to consider all possible permutations
    of the corresponding dimensions.
    See discussion in "A Closer Look at Rotation-Invariant Deep Point Cloud Analysis"
    by Li et al.

    Args:
        base_model: The classifier to wrap
        n_point_dim: Number of point cloud dimensions (not number of points)
        n_feat_dim: Should be 0, was meant for attributed point clouds
    """

    def __init__(self, base_model: nn.Module,
                 n_point_dim: int = 3, n_feat_dim: int = 0):
        super(PCAEnsembleWrapper, self).__init__()

        self.base_model = base_model
        self.n_point_dim = n_point_dim
        self.n_feat_dim = n_feat_dim

        if n_feat_dim != 0:
            raise NotImplementedError('Pointclouds with features not supported yet')

        if not isinstance(self.base_model, (PointNetClassifier, DGCNNClassifier)):
            raise NotImplementedError('Only PointNet and DGCNN supported')

    def _average_sign_ambiguities(self, x_centered, evecs):
        logits = None

        device = x_centered.device
        D = x_centered.shape[-1]

        for sign_vector in itertools.product([-1, 1], repeat=D):
            sign_vector = torch.Tensor(sign_vector).to(device)

            x = torch.bmm(x_centered, evecs * sign_vector)

            if logits is None:
                logits, _ = self.base_model(x)
            else:
                logits += self.base_model(x)[0]

        return logits / (2 ** D)

    def _average_order_and_sign_ambiguities(self, x_centered, evecs):
        logits = None

        D = x_centered.shape[-1]

        for permutation in itertools.permutations(range(D)):
            evecs_permuted = evecs.detach().clone()[:, :, permutation]

            if logits is None:
                logits = self._average_sign_ambiguities(x_centered, evecs_permuted)
            else:
                logits += self._average_sign_ambiguities(x_centered, evecs_permuted)

        return logits / factorial(D)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        assert D == self.n_point_dim

        device = x.device

        x_centered = x - x.mean(dim=1, keepdims=True)
        cov = torch.bmm(x_centered.transpose(2, 1) / np.sqrt(N), x_centered / np.sqrt(N))

        cov = cov.cpu()

        evals, evecs = torch.linalg.eigh(cov)
        evecs = evecs.to(device)
        evals = evals.to(device)

        order_ambiguity_mask = torch.zeros(B, dtype=torch.bool).to(device)

        for i, j in itertools.combinations(range(D), 2):
            order_ambiguity_mask |= torch.isclose(evals[:, i], evals[:, j])

        if order_ambiguity_mask.sum() < B:
            logits_no_order_ambiguity = self._average_sign_ambiguities(
                                            x_centered[~order_ambiguity_mask],
                                            evecs[~order_ambiguity_mask])

            n_classes = logits_no_order_ambiguity.shape[-1]

        if order_ambiguity_mask.sum() > 0:
            logits_order_ambiguity = self._average_order_and_sign_ambiguities(
                                            x_centered[order_ambiguity_mask],
                                            evecs[order_ambiguity_mask])

            n_classes = logits_order_ambiguity.shape[-1]

        logits = torch.zeros((B, n_classes)).to(device)

        if order_ambiguity_mask.sum() < B:
            logits[~order_ambiguity_mask] = logits_no_order_ambiguity

        if order_ambiguity_mask.sum() > 0:
            logits[order_ambiguity_mask] = logits_order_ambiguity

        return logits, None
