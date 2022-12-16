"""This module contains slightly modified version of the DGCNN model
from the reference implementation, to allow compatibility with
our invariance wrappers."""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

sys.path.append('../reference_implementations/dgcnn/pytorch')

from model import get_graph_feature


class TransformNet(nn.Module):
    # Mix of invariance repo and poitnnet repo
    def __init__(self):
        super(TransformNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.transform = nn.Linear(256, 3 * 3)

    def forward(self, x):
        bs = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        x = self.transform(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            bs, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class DGCNNEncoder(nn.Module):
    # taken from dgcnn repo
    def __init__(self, n_neighbors, use_input_tnet=False, n_emb_dims=1024):
        super(DGCNNEncoder, self).__init__()
        self.n_neighbors = n_neighbors
        self.n_emb_dims = n_emb_dims

        self.use_input_tnet = use_input_tnet

        if use_input_tnet:
            self.input_tnet = TransformNet()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(320, n_emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.shape[0]

        if self.use_input_tnet:
            x0 = get_graph_feature(x, k=self.n_neighbors)  # (bs, 3, n_points) -> (bs, 3*2, n_points, k)
            t = self.input_tnet(x0)  # (bs, 3, 3)
            x = x.transpose(2, 1)  # (bs, 3, n_points) -> (bs, n_points, 3)
            x = torch.bmm(x, t)  # (bs, n_points, 3) * (bs, 3, 3) -> (bs, n_points, 3)
            x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.n_neighbors)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_neighbors)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_neighbors)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.n_neighbors)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)

        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        return x


class DGCNNClassifier(nn.Module):
    def __init__(self,
                 n_neighbors=20, n_emb_dims=1024,
                 n_classes=40, n_point_dim=3, n_feat_dim=0,
                 input_tnet=False):

        if n_point_dim != 3 or n_feat_dim != 0:
            raise NotImplementedError('DGCNN currently only allows 3D coordinates w/o features')

        super(DGCNNClassifier, self).__init__()

        self.encoder = DGCNNEncoder(n_neighbors, input_tnet, n_emb_dims)

        self.linear1 = nn.Linear(n_emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.transpose(2, 1)

        x = self.encoder(x)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, None
