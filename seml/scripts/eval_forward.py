"""This script is used for evaluating our randomized smoothing certificates
and baselines, using the Monte Carlo samples we obtained from specific
point cloud classification datasets.
Meant to be executed with seml, using the config files in seml/configs/.
You should first run seml/scripts/train.py to train your model and then
seml/scripts/sample_votes.py to obtain Monte Carlo samples.
"""


from multiprocessing.sharedctypes import Value

import seml
import torch
from pointcloud_invariance_smoothing.data import get_dataset
from pointcloud_invariance_smoothing.evaluation.eval_forward_certs import \
    eval_dataset_threatmodel
from pointcloud_invariance_smoothing.models.invariance_wrappers import (
    PCAEnsembleWrapper, SinglePCAWrapper, StackingPCAWrapper)
from pointcloud_invariance_smoothing.models.utils import get_model
from pointcloud_invariance_smoothing.smoothing import sample_votes_dataset
from pointcloud_invariance_smoothing.utils import dict_to_dot, set_seed
from sacred import Experiment

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

    std = 0.1  # Smoothing standard deviations

    n_datapoints = 3  # Number of pointclouds from dataset to evaluate on
    n_samples_per_datapoint = 1  # Number of randomly perturbed inputs to generate per point cloud

    dataset = {
        'name': 'modelnet40',  # modelnet40 or mnist
        'data_folder': '/nfs/shared/modelnet/modelnet40_normal_resampled',
        'val_percentage': 0.2,  # Percentage of train set to use for validation
    }

    # Used for retrieving Monte Carlo samples
    pred_sample_loading = {
        'collection': None,  # database you used for storing Monte Carlo samples
        'exp_id': None,  # Can manually specify SEML experiment id
        'restrictions': None,  # Restrictions for query, in dot notation
        'find_std': False  # Find MC samples obtained from Gaussian noise matching sample_params.std
    }

    # Parameters for generating (randomly) perturbed inputs
    threat_model_params = {
        'norms_delta_params': {  # Perturbation norms
            'values': [0.05],
            'linspace': False,  # If False, use norms from 'values'
            'min': 0,  # start of linspace
            'max': 0,  # end of linspace
            'steps': 10  # number of points in linspace
        },
        'distribution_params': {
            'parallel': False,  # If True, scale input s.t. perturbation norm is reached
                                # If False, randomly sample perturbation of specified norm
            'angle': 0.8,  # Random rotation in degrees, applied to perturbed input if 'parallel' is False
            'axis_parallel': False  # If False, rotate around one of the main axes.
                                    # If True, rotate around randomly sampled axis.
        }
    }

    certificate_params = {
        'baseline': True,  # Evaluate black-box randomized smoothing
        'preprocessing': True,  # Evaluate orbit-based certificates
        'tight': True,  # Evaluate tight invariance-aware certificates
        'preprocess_translation': True,  # Center data before certification (i.e. assume translation invariance)
        'n_samples_clean': 10000,  # Number of MC samples to find classification threshold in Algorithm 1
        'n_samples_pert': 10000,  # Number of MC samples to bound perturbed prediction probability in Algorithm 1
        'alpha': 0.001,  # Significance level for Monte Carlo certification procedure
        'elemental': True,  # If True, assume invariance to rotation around main axis.
                            # If False, assume invariance to arbitrary rotations
        'quadrature_method': 'clenshaw_curtis',  # quadpy quadrature method to use
        'quadrature_degree': 20  # Degree for quadrature method
    }


@ex.capture(prefix='pred_sample_loading')
def load_pred_samples(collection, exp_id, restrictions,
                      find_std=False,
                      std=-1):

    if exp_id is None and restrictions is None:
        raise ValueError('You must provide either an exp-id or a restriction dict')
    if collection is None:
        raise ValueError('You must provide a collection to load predictions from')
    print(collection)
    mongodb_config = seml.database.get_mongodb_config()
    coll = seml.database.get_collection(collection, mongodb_config)

    if exp_id is not None:
        exp = coll.find_one({'_id': exp_id})
        sample_config = exp['config']
        sample_results = torch.load(exp['result']['save_File'])
    else:
        coll_filter = restrictions.copy()
        coll_filter = {'config.' + k: v for k, v in dict_to_dot(coll_filter)}

        if find_std:
            coll_filter['config.sample_params.std'] = std

        exps = list(coll.find(coll_filter))
        if len(exps) == 0:
            raise ValueError("Find yielded no results.")
        elif len(exps) > 1:
            raise ValueError(f"Find yielded more than one result: {exps}")
        else:
            exp_id = exps[0]['_id']
            sample_config = exps[0]['config']
            sample_results = torch.load(exps[0]['result']['save_file'])

    return sample_config, exp_id, sample_results


get_dataset = ex.capture(get_dataset, prefix='dataset')


@ex.automain
def main(_config, seed, save_dir, std, dataset, pred_sample_loading,
         threat_model_params, certificate_params,
         n_datapoints, n_samples_per_datapoint):

    set_seed(seed)

    pred_sample_config, pred_sample_exp_id, pred_sample_results = load_pred_samples(std=std)

    if not dataset['name'] == pred_sample_config['dataset']['name']:
        raise ValueError('Datasets for sampling and certification must be the same')

    _, _, data_test = get_dataset()

    cert_results_dict = eval_dataset_threatmodel(
        std, data_test, pred_sample_results,
        threat_model_params, certificate_params,
        n_datapoints, n_samples_per_datapoint
    )

    run_id = _config['overwrite']
    save_file = f'{save_dir}/{pred_sample_loading["collection"]}_{pred_sample_exp_id}_{run_id}'

    torch.save(cert_results_dict, save_file)

    return {'save_file': save_file}
