"""This script is used for obtaining Monte Carlo samples from a specific model.
Meant to be executed with seml, using the config files in seml/configs/.
You should first run seml/scripts/train.py to train your model.
After executing this script, you can calculate certificates from the Monte Carlo
samples using seml/scripts/eval_forward.py
"""


import torch
from sacred import Experiment

import seml
from pointcloud_invariance_smoothing.data import get_dataset
from pointcloud_invariance_smoothing.models.invariance_wrappers import (
    PCAEnsembleWrapper, SinglePCAWrapper, StackingPCAWrapper)
from pointcloud_invariance_smoothing.models.utils import get_model
from pointcloud_invariance_smoothing.smoothing import sample_votes_dataset
from pointcloud_invariance_smoothing.utils import dict_to_dot, set_seed

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

    save_dir = '/home/jan/tmp/save'
    seed = 0

    dataset = {
        'name': 'modelnet40',  # modelnet40 or mnist
        'data_folder': '/nfs/shared/modelnet/modelnet40_normal_resampled',
        'val_percentage': 0.2,  # Percentage of train set to use for validation
    }

    # Used for retrieving trained model weights
    train_loading = {
        'collection': None,  # database you used for storing train data
        'exp_id': None,  # Can manually specify SEML experiment id
        'restrictions': None,  # Restrictions for query, in dot notation
        'load_best_acc': True,  # If False, get weights with lowest validation loss.
        'find_std': False  # Find model trained on Gaussian noise matching sample_params.std
    }

    sample_params = {
        'std': 0.1,  # Smoothing standard deviation
        'n_datapoints': 100,  # Number of point clouds to obtain Monte Carlo samples for
        'n_samples_cert': 0,  # Number of MC samples to take per point cloud for prediction
        'n_samples_pred': 1000,  # Number of MC samples to take per point cloud for certification
        'batch_size': 512
    }

    overwrite_invariance_wrapper = False  # Use a different PCA wrapper than at train time

    invariance_wrapper_params = {
        'wrapper_type': 'ensemble_pca',  # single_pca, ensemble_pca or None
        'random_sign': False,  # Randomize eigenvalue sign in PCA
        'random_order': False  # Randomize order of dimensions after PCA projection
    }


@ex.capture(prefix='train_loading')
def load_train_data(collection, exp_id, restrictions,
                    find_std=False, load_best_acc=True,
                    std=-1):

    if exp_id is None and restrictions is None:
        raise ValueError('You must provide either an exp-id or a restriction dict')
    if collection is None:
        raise ValueError('You must a collection to load trained model from')
    print(collection)
    mongodb_config = seml.database.get_mongodb_config()
    coll = seml.database.get_collection(collection, mongodb_config)

    if exp_id is not None:
        train_config = coll.find_one({'_id': exp_id}, ['config'])['config']
    else:
        coll_filter = restrictions.copy()
        coll_filter = {'config.' + k: v for k, v in dict_to_dot(coll_filter)}

        if find_std:
            coll_filter['config.training_params.std'] = std

        exps = list(coll.find(coll_filter, ['config']))
        if len(exps) == 0:
            raise ValueError("Find yielded no results.")
        elif len(exps) > 1:
            raise ValueError(f"Find yielded more than one result: {exps}")
        else:
            exp_id = exps[0]['_id']
            train_config = exps[0]['config']

    return train_config, exp_id


get_dataset = ex.capture(get_dataset, prefix='dataset')
sample_votes_dataset = ex.capture(sample_votes_dataset, prefix='sample_params')


@ex.automain
def main(_config, seed, save_dir, sample_params, train_loading,
         overwrite_invariance_wrapper, invariance_wrapper_params):

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_config, train_exp_id = load_train_data(std=sample_params['std'])

    model_config = train_config['model']

    if overwrite_invariance_wrapper:
        model_config['invariance_wrapper_params'] = invariance_wrapper_params

    train_file = torch.load(f'{train_config["save_dir"]}'
                            f'/{train_loading["collection"]}_{train_exp_id}')

    model = get_model(**model_config)

    model.to(device)
    if train_loading['load_best_acc']:
        state_dict_file = train_file['state_dict_best_acc']
    else:
        state_dict_file = train_file['state_dict_best_loss']

    if isinstance(model, (SinglePCAWrapper, StackingPCAWrapper, PCAEnsembleWrapper)):
        model.base_model.load_state_dict(state_dict_file)
    else:
        model.load_state_dict(state_dict_file)

    _, _, data_test = get_dataset()

    run_id = _config['overwrite']
    save_file = f'{save_dir}/{train_loading["collection"]}_{train_exp_id}_{run_id}'

    sample_dict = sample_votes_dataset(dataset=data_test, model=model, device=device)

    torch.save(sample_dict, save_file)

    if 'accuracy' in sample_dict.keys():
        return {'save_file': save_file, 'accuracy': sample_dict['accuracy']}
    else:
        return {'save_file': save_file}
