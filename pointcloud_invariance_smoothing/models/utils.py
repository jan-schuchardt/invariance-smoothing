from multiprocessing.sharedctypes import Value
from random import random
from typing import Optional

import torch.nn as nn

from pointcloud_invariance_smoothing.models.dgcnn import DGCNNClassifier
from pointcloud_invariance_smoothing.models.invariance_wrappers import (
    PCAEnsembleWrapper, SinglePCAWrapper, StackingPCAWrapper)
from pointcloud_invariance_smoothing.models.pointnet import (
    PointNetAttentionClassifier, PointNetClassifier)


def get_model(model_type: str, model_params: dict,
              invariance_wrapper_params: Optional[dict] = None) -> nn.Module:
    """Yields chosen model and invariance wrapper

    Args:
        model_type: String from ['pointnet', 'dgcnn', 'pointnet_attention'],
        model_params: See seml/scripts/train.py for explanation of parameters
        invariance_wrapper_params: See seml/scripts/train.py for explanation of parameters.
            For pointnet_attention, the invariance_wrapper_params['wrapper_type'] must be
            stacking_pca
    """
    model_type = model_type.lower()
    assert model_type in ['pointnet', 'dgcnn', 'pointnet_attention']

    if model_type == 'pointnet':
        model = PointNetClassifier(n_classes=model_params['n_classes'],
                                   n_point_dim=model_params['n_point_dim'],
                                   n_feat_dim=model_params['n_feat_dim'],
                                   input_tnet=model_params['input_tnet'],
                                   feature_tnet=model_params['feature_tnet'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'single_pca':
            return SinglePCAWrapper(model,
                                    invariance_wrapper_params['random_sign'],
                                    invariance_wrapper_params['random_order'],
                                    n_point_dim=model_params['n_point_dim'],
                                    n_feat_dim=model_params['n_feat_dim'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'ensemble_pca':
            return PCAEnsembleWrapper(model,
                                      n_point_dim=model_params['n_point_dim'],
                                      n_feat_dim=model_params['n_feat_dim'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'no_wrapper':
            return model

        else:
            raise NotImplementedError('Currently only support \
                    SinglePCAWrapper and PCAEnsembleWrapper')

    if model_type == 'dgcnn':
        model = DGCNNClassifier(n_neighbors=model_params['n_neighbors'],
                                n_emb_dims=model_params['n_emb_dims'],
                                n_point_dim=model_params['n_point_dim'],
                                n_feat_dim=model_params['n_feat_dim'],
                                input_tnet=model_params['input_tnet'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'single_pca':
            return SinglePCAWrapper(model,
                                    invariance_wrapper_params['random_sign'],
                                    invariance_wrapper_params['random_order'],
                                    n_point_dim=model_params['n_point_dim'],
                                    n_feat_dim=model_params['n_feat_dim'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'ensemble_pca':
            return PCAEnsembleWrapper(model,
                                      n_point_dim=model_params['n_point_dim'],
                                      n_feat_dim=model_params['n_feat_dim'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'no_wrapper':
            return model

        else:
            raise NotImplementedError('Currently only support \
                    SinglePCAWrapper and PCAEnsembleWrapper')

    if model_type == 'pointnet_attention':
        model = PointNetAttentionClassifier(
                                   n_classes=model_params['n_classes'],
                                   n_point_dim=model_params['n_point_dim'],
                                   n_feat_dim=model_params['n_feat_dim'],
                                   input_tnet=model_params['input_tnet'],
                                   feature_tnet=model_params['feature_tnet'],
                                   n_attention_weight_dims=model_params['n_attention_weight_dims'],
                                   n_attention_value_dims=model_params['n_attention_value_dims'])

        if invariance_wrapper_params['wrapper_type'].lower() == 'stacking_pca':
            return StackingPCAWrapper(
                model,
                n_point_dim=model_params['n_point_dim'],
                n_feat_dim=model_params['n_feat_dim'],
                proper_rotations_only=invariance_wrapper_params['proper_rotations_only'],
                random_order=invariance_wrapper_params['random_order'],
                stack_ambiguous_orders=invariance_wrapper_params['stack_ambiguous_orders']
            )

        else:
            raise ValueError('Attention model must be used with STackingPCAWrapper')
