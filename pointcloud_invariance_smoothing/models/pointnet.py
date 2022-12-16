"""This module contains slightly modified version of the PointNet model
from the reference implementation, to allow compatibility with
our invariance wrappers.
It also contains the attention-based model for rotation-invariant classification
proposed in "Endowing deep 3d models with rotation invariance based on principal component analysis"
by Xiao et al.
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../reference_implementations/Pointnet_Pointnet2_pytorch/models')
from pointnet_utils import STNkd, feature_transform_reguliarzer


class TNet(STNkd):
    def __init__(self, n_input_dim, n_point_dim):
        super(TNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(n_input_dim, 64, 1)
        self.fc3 = nn.Linear(256, n_point_dim * n_point_dim)

        self.k = n_point_dim


class PointNetEncoder(nn.Module):
    def __init__(self, n_point_dim=3, n_feat_dim=0, input_tnet=False, feature_tnet=False):
        super(PointNetEncoder, self).__init__()

        self.n_point_dim = n_point_dim
        self.n_feat_dim = n_feat_dim
        self.n_input_dim = n_point_dim + n_feat_dim
        self.input_tnet = input_tnet
        self.feature_tnet = feature_tnet

        if self.input_tnet:
            self.stn = TNet(self.n_input_dim, self.n_point_dim)

        self.conv1 = torch.nn.Conv1d(self.n_input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_tnet:
            self.fstn = TNet(64, 64)

    def forward(self, x):
        B, D, N = x.size()

        if self.input_tnet:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if self.n_feat_dim > 0:
                feature = x[:, :, self.n_point_dim:]
                x = x[:, :, :self.n_point_dim]
            x = torch.bmm(x, trans)
            if self.n_feat_dim > 0:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_tnet:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, trans, trans_feat


class PointNetClassifier(nn.Module):

    def __init__(self, n_classes=40, n_point_dim=3, n_feat_dim=0,
                 input_tnet=False, feature_tnet=False):

        super(PointNetClassifier, self).__init__()

        self.feat = PointNetEncoder(n_point_dim=n_point_dim, n_feat_dim=n_feat_dim,
                                    input_tnet=input_tnet, feature_tnet=feature_tnet)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)

        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x
        return x, trans_feat


def pointnet_loss(pred, target, trans_feat=None, regularization_weight=0.001):
    loss = F.cross_entropy(pred, target)

    if trans_feat is not None:
        regularization = feature_transform_reguliarzer(trans_feat)
        loss += regularization_weight * regularization

    return loss


class PointNetAttentionClassifier(nn.Module):

    def __init__(self, n_classes=40, n_point_dim=3, n_feat_dim=0,
                 n_attention_weight_dims=1024,
                 n_attention_value_dims=1024,
                 input_tnet=False, feature_tnet=False):

        super(PointNetAttentionClassifier, self).__init__()

        self.feat = PointNetEncoder(n_point_dim=n_point_dim, n_feat_dim=n_feat_dim,
                                    input_tnet=input_tnet, feature_tnet=feature_tnet)

        self.n_attention_weight_dims = n_attention_weight_dims
        self.n_attention_value_dims = n_attention_value_dims

        self.query_transform = nn.Linear(1024, n_attention_weight_dims)
        self.key_transform = nn.Linear(1024, n_attention_weight_dims)
        self.value_transform = nn.Linear(1024, n_attention_value_dims)

        self.fc1 = nn.Linear(n_attention_value_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def self_attention(self, x):
        x_q = self.query_transform(x)  # B x P x Dq
        x_k = self.key_transform(x).transpose(1, 2)  # B x Dq x P

        attention_logits = torch.bmm(x_q, x_k) / np.sqrt(self.n_attention_weight_dims)  # B x P x P
        attention_weights = F.softmax(attention_logits, dim=2)

        x_v = self.value_transform(x)  # B x P x Dv

        return torch.bmm(attention_weights, x_v)

    def forward(self, x):
        B, N, P, D = x.shape
        x = x.permute(0, 2, 3, 1)  # B x P x D x N
        x = x.reshape(-1, D, N)  # (B * P) x D x N

        x, _, trans_feat = self.feat(x)  # (B*P) x 1024
        x = x.view(B, P, -1)  # B x P x 1024

        x = self.self_attention(x)  # B x P x Dv

        x = torch.mean(x, dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = x
        return x, trans_feat
