import numpy as np
from pointcloud_invariance_smoothing.certificates.blackbox import \
    forward_cert_cohen
from pointcloud_invariance_smoothing.certificates.rotation_2d import \
    forward_2d_rotation_cert
from pointcloud_invariance_smoothing.evaluation.eval_forward_certs import (
    calc_p_cert_baseline, calc_p_cert_preprocessing, calc_p_cert_tight_2d)
from statsmodels.stats.proportion import proportion_confint

n_samples_certification = 10000


def get_smoothed_prediction(votes_pred):
    return votes_pred.argmax()


def get_clean_prediction_prob_lower(smoothed_pred, votes_cert, alpha=0.01):
    n_samples = votes_cert.sum()

    # Clopper-Pearson confidence bounds
    return proportion_confint(votes_cert[smoothed_pred], n_samples, alpha, method='beta')[0]


# This function determines for a single pointcloud whether the prediction is certifiably robust
def certified_correct_scaling(x, target, votes_pred, votes_cert, norm_delta, sigma,
                              method='tight', alpha=0.01):

    if method not in ['tight', 'orbit']:
        raise ValueError('Only support "tight" and "orbit" certificate.')

    smoothed_pred = get_smoothed_prediction(votes_pred)

    if smoothed_pred != target:
        return False

    clean_prediction_prob_lower = get_clean_prediction_prob_lower(
                                    smoothed_pred, votes_cert, alpha=alpha)

    # Tight certificate
    if method == 'tight':
        perturbed_prediction_prob_lower = forward_2d_rotation_cert(
            sigma, np.linalg.norm(x), norm_delta,
            np.linalg.norm(x) * norm_delta,  # eps_1
            0,  # eps_2
            n_samples_certification, n_samples_certification,
            clean_prediction_prob_lower, alpha)

        if isinstance(perturbed_prediction_prob_lower, tuple):
            perturbed_prediction_prob_lower = perturbed_prediction_prob_lower[0]

    # Orbit-based certificate
    else:
        perturbed_prediction_prob_lower = forward_cert_cohen(
            sigma, norm_delta, clean_prediction_prob_lower)

    return perturbed_prediction_prob_lower > 0.5


# This function determines for a single pointcloud and perturbed pointcloud
#  whether the prediction is certifiably robust
def certified_correct(x, x_pert, target, votes_pred, votes_cert, norm_delta, sigma,
                      method='tight', alpha=0.01):

    if method not in ['tight', 'orbit', 'black-box']:
        raise ValueError('Only support "tight", "orbit" and "black-box" certificate.')

    smoothed_pred = get_smoothed_prediction(votes_pred)

    if smoothed_pred != target:
        return False

    clean_prediction_prob_lower = get_clean_prediction_prob_lower(
                                    smoothed_pred, votes_cert, alpha=alpha)

    certificate_params = {  # This is used internally by certification functions
        'preprocess_translation': True,  # If True, get SE(D) instead of SO(D) certificates
        'n_samples_clean': n_samples_certification,
        'n_samples_pert': n_samples_certification,
        'alpha': alpha
    }

    # Tight certificate
    if method == 'tight':
        perturbed_prediction_prob_lower = calc_p_cert_tight_2d(
            sigma, x, x_pert, clean_prediction_prob_lower, certificate_params
        )

        if isinstance(perturbed_prediction_prob_lower, tuple):
            perturbed_prediction_prob_lower = perturbed_prediction_prob_lower[0]

    elif method == 'orbit':
        perturbed_prediction_prob_lower = calc_p_cert_preprocessing(
            sigma, x, x_pert, clean_prediction_prob_lower, certificate_params
        )

    else:
        perturbed_prediction_prob_lower = calc_p_cert_baseline(
            sigma, x, x_pert, clean_prediction_prob_lower, certificate_params
        )

    return perturbed_prediction_prob_lower > 0.5
