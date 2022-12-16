"""This script is used for training point cloud classifier for later use
as base classifiers in randomized smoothing.
Meant to be executed with seml, using the config files in seml/configs/.
"""


import numpy as np
import torch
from sacred import Experiment

import seml
from pointcloud_invariance_smoothing.data import get_dataset
from pointcloud_invariance_smoothing.models.utils import get_model
from pointcloud_invariance_smoothing.training import train
from pointcloud_invariance_smoothing.utils import set_seed

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
            )

    dataset = {
        'name': 'modelnet40',  # modelnet40 or mnist
        'data_folder': '/nfs/shared/modelnet/modelnet40_normal_resampled',
        'val_percentage': 0.2,  # Percentage of train set to use for validation
    }

    model = {
        'model_type': 'pointnet',  # pointnet, dgcnn or pointnet_attention
        'model_params': {
            'n_classes': 40,  # Number of classes, 40 for modelnet, 9 for mnist
            'n_point_dim': 3,  # Number of dimensions. 2 for mnist, 3 for modelnet40
            'n_feat_dim': 0,  # keep this at 0
            'input_tnet': True,  # Whether to use TNet from PointNet Paper on inputs
            'feature_tnet': True,  # Whetehr to use TNet in latent space (see PointNet)
            'n_neighbors': 20,  # Number of neighbors to use for DGCNN
            'n_emb_dims': 1024,  # Number of latent dimensions before classification head
            'n_attention_weight_dims': 1024,  # Attention dimensions for pointnet_attention
            'n_attention_value_dims': 1024  # Attention dimenions for pointnet_attention
        },
        'invariance_wrapper_params': {
            'wrapper_type': 'single_pca',  # single_pca, ensemble_pca or None
            'random_sign': False,  # Randomly choose sign of eigenvalue in PCA
            'random_order': False,  # Randomly shuffle dimensions after PCA
            'proper_rotations_only': True,  # Only consider PCA projections with determinant +1
            'stack_ambiguous_orders': False  # Consider all possible orders of PCA dimensions, when multiple eigenvalues are identical
        }
    }

    training_params = {
        'num_epochs': 256,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0,
        'regularization_weight': 0.001,
        'scheduler_stepsize': 20,
        'scheduler_gamma': 0.7,
        'num_workers': 8,
        'rotate': False,  # Randomly rotate training data
        'scale': False,  # Randomly scale training data
        'add_noise': False,  # Add gaussian noise to training data
        'rotate_validation': False,  # Randomly rotate validation data
        'add_noise_validation': True,  # Add gaussian noise to validation data
        'scale_limits': [0.8, 1.25],  # Random scaling is uniformly sampled from this range
        'std': 0.1  # Standard deviation for Gaussian noise
    }

    seed = 0
    save_dir = '/home/jan/tmp/save'


get_model = ex.capture(get_model, prefix='model')

get_dataset = ex.capture(get_dataset, prefix='dataset')


@ex.automain
def main(_config, seed, save_dir, training_params):

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = get_model()
    model.to(device)

    data_train, data_val, _ = get_dataset()

    (losses_train, losses_val,
     accs_train, accs_val,
     best_state_loss, best_state_acc) = train(model,
                                              data_train, data_val,
                                              device=device, **training_params)

    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    dict_to_save = {'losses_train': losses_train,
                    'losses_val': losses_val,
                    'accs_train': accs_train,
                    'accs_val': accs_val,
                    'state_dict_best_loss': best_state_loss,
                    'state_dict_best_acc': best_state_acc}

    torch.save(dict_to_save,
               f'{save_dir}/{db_collection}_{run_id}')

    return {
        'best_loss_train': np.min(losses_train),
        'best_loss_val': np.min(losses_val),
        'best_acc_train': np.max(accs_train),
        'best_acc_val': np.max(accs_val),
        'best_loss_epoch_val': np.argmin(losses_val),
        'best_acc_epoch_val': np.argmax(accs_val)
    }
