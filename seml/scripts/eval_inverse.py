"""This script is used for evaluating our randomized smoothing certificates
and baselines in an inverse manner.
I.e. given certificate parameters, find the smallest clean prediction probability
such that robustness can be certified.
Meant to be executed with seml, using the config files in seml/configs/.
For these experiments, it is not necessary to train a model or obtain Monte Carlo
samples on a specific dataset."""


import numpy as np
import seml
import torch
from pointcloud_invariance_smoothing.evaluation.eval_inverse_certs import (
    eval_multiple_norms_multiple_inner_cross,
    eval_multiple_norms_single_inner_cross)
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

    save_dir = '/nfs/homedirs/schuchaj/test'
    seed = 0

    sigma = 0.1  # Smoothing standard deviation
    n_samples_clean = 10000  # Number of samples to determine classificaiton treshold in Algorithm 3
    n_samples_pert = 10000  # Nubmer of samples to bound required clean prediction probability in Algorithm 3
    alpha = 0.001  # Significance level for Monte Carlo certification procedure

    # Clean data norms ||X|| to evaluate on
    norms_clean_params = {
        'linspace': False,  # If False, use clean norms specified in 'values'
        'values': [0.01, 0.05, 0.1, 0.15, 0.2],
        'min': 0,  # Start of linspace
        'max': 1,  # End of linspace
        'steps': 10  # Number of points in linspace to evaluate on
    }

    # Perturbation norms ||Delta|| to evaluate on
    norms_delta_params = {
        'linspace': False,  # If False, use clean norms specified in 'values'
        'values': [0.05, 0.1, 0.2],
        'min': 0,  # Start of linspace
        'max': 1,  # End of linspace
        'steps': 10  # Number of points in linspace to evaluate on
    }

    # Inner product between clean pointcloud and perturbation matrix
    # before (eps_1) and after (eps_2) rotation of clean point cloud by 90°
    inner_cross_params = {
        'multiple_inner_cross': False,  # Evaluate on linspace of eps_1 and eps_2 within feasible range (like Fig. 4)
        'inner': 1,  # Value for eps_1 to use if multiple_inner_cross is False
        'cross': 0,  # Value for eps_2 to use if multiple_inner_cross is False
        'relative_inner_cross': True,  # To be used if multiple_inner_cross is False
                                       # Interpret inner, cross as tilde(eps)_1 and tilde(eps)_2 from paper
        'steps_inner': 21,  # Number of linspace steps to use for eps_1 when multiple_inner_cross is True
        'steps_cross': 21  # Number of linspace steps to use for eps_2 when multiple_inner_cross is True
    }


@ex.automain
def main(_config, save_dir, seed,
         sigma, n_samples_clean, n_samples_pert, alpha,
         norms_clean_params, norms_delta_params, inner_cross_params):

    set_seed(seed)

    if norms_clean_params['linspace']:
        norms_clean = np.linspace(norms_clean_params['min'],
                                  norms_clean_params['max'],
                                  norms_clean_params['steps'])
    else:
        norms_clean = norms_clean_params['values']

    if norms_delta_params['linspace']:
        norms_delta = np.linspace(norms_delta_params['min'],
                                  norms_delta_params['max'],
                                  norms_delta_params['steps'])
    else:
        norms_delta = norms_delta_params['values']

    if inner_cross_params['multiple_inner_cross']:

        eval_dict = eval_multiple_norms_multiple_inner_cross(
            sigma,
            norms_clean, norms_delta,
            inner_cross_params['steps_inner'], inner_cross_params['steps_cross'],
            n_samples_clean,
            n_samples_pert,
            alpha
        )

    else:
        eval_dict = eval_multiple_norms_single_inner_cross(
            sigma,
            norms_clean, norms_delta,
            inner_cross_params['inner'], inner_cross_params['cross'],
            inner_cross_params['relative_inner_cross'],
            n_samples_clean,
            n_samples_pert,
            alpha
        )

    run_id = _config['overwrite']
    db_collection = _config['db_collection']

    torch.save(eval_dict,
               f'{save_dir}/{db_collection}_{run_id}')

    return {'save_file': f'{save_dir}/{db_collection}_{run_id}'}
