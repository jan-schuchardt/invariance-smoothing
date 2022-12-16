"""This module implements the cert and inverse cert for rotation around coordinate axes.

inverse_elemental_rotation_cert: Smallest prediction probability s.t. robustness can be certified.

forward_elemental_rotation_cert: Lower bound on perturbed prediction probability.

sample_single_axis_likelihood_ratio_logs: Sample likelihood ratios along rotation axis.
"""

from typing import Optional

import numpy as np
from statsmodels.stats.proportion import proportion_confint

from .rotation_2d import sample_2d_likelihood_ratio_logs
from .utils import forward_mc_estimate, sample_multivariate_normal


def inverse_elemental_rotation_cert(sigma: float,
                                    norm_delta_rot_axis: float,
                                    norm_clean_other: float, norm_delta_other: float,
                                    inner: float, cross: float,
                                    n_samples_clean: int, n_samples_pert: int,
                                    alpha: float = 0.001) -> float:
    """Yields smallest probability for certification, assuming invariance to elemental rotation.

    This method uses the inverse Monte Carlo certification procedure
    specified in Appendix I, i.e. adjust threshold s.t. perturbed prediction probability is 0.5.
    and then bound corresponding clean prediction probability.
    Holm correction is used to ensure that these two bounds, together
    with the bound for p_lower hold with significance alpha.

    Args:
        sigma: Smoothing standard deviation.
        norm_delta_rot_axis: Norm of perturbation along rotation axis.
        norm_clean_other: Norm of unperturbed point cloud in the other two axes.
        norm_delta_other: Norm of perturbation in the other two axes.
        inner: Frobenius inner product between perturbed and unperturbed pointcloud
            in other two axes.
        cross: Frobenius inner product  between perturbed and unperturbed pointcloud
            after rotation by 90 degrees in other two axes.
        n_samples_clean: Number of samples used to bound required clean prediction probability.
        n_samples_pert: Number of samples used to bound classification threshold.
        alpha: Significance between 0 and 1.

    Returns:
        A float between 0 and 1 corresponding to the smallest prediction probability.
    """

    ratios_clean = sample_2d_likelihood_ratio_logs(True, sigma,
                                                   norm_clean_other, norm_delta_other,
                                                   inner, cross, n_samples_clean)

    ratios_clean += sample_single_axis_likelihood_ratio_logs(
        True, sigma, norm_delta_rot_axis, n_samples_clean
    )

    ratios_pert = sample_2d_likelihood_ratio_logs(False, sigma,
                                                  norm_clean_other, norm_delta_other,
                                                  inner, cross, n_samples_pert)

    ratios_pert += sample_single_axis_likelihood_ratio_logs(
        False, sigma, norm_delta_rot_axis, n_samples_pert
    )

    cdf_pert_lower = proportion_confint(np.arange(1, n_samples_pert + 1),
                                        n_samples_pert, alpha=2 * alpha / 3, method='beta')[0]

    threshold = np.sort(ratios_pert)[(cdf_pert_lower > 0.5).argmax()]

    p_clean_needed_upper = proportion_confint(
            (ratios_clean <= threshold).sum(),
            n_samples_clean, alpha=2 * alpha / 2, method='beta')[1]

    if p_clean_needed_upper < 0.5:
        # catch numerical issues (and super rare sample edge cases)
        assert p_clean_needed_upper > 0.01
        # We know that p_clean must be at least 0.5 to ensure that p_perturbed is g.e.q. than 0.5.
        p_clean_needed_upper = 0.5

    return p_clean_needed_upper


def forward_elemental_rotation_cert(
        sigma: float,
        norm_delta_rot_axis: float,
        norm_clean_other: float, norm_delta_other: float,
        inner: float, cross: float,
        n_samples_clean: int, n_samples_pert: int,
        p_lower: float, alpha: float = 0.001,
        alpha_clean: Optional[float] = None,
        alpha_pert: Optional[float] = None) -> tuple[float, float, float]:
    """Lower-bounds prediction probability, assuming invariance to rotation around single axis.

    This method uses the Monte Carlo certification procedure specified in Appendix F.5,
    i.e. using samples to bound classification threshold and then uses second round of samples
    to bound the perturbed prediction probability given this threshold.
    By default, Holm correction is used to ensure that these two bounds, together
    with the bound for p_lower hold with significance alpha.

    Args:
        sigma: Smoothing standard deviation.
        norm_delta_rot_axis: Norm of perturbation along rotation axis.
        norm_clean_other: Norm of unperturbed point cloud in the other two axes.
        norm_delta_other: Norm of perturbation in the other two axes.
        inner: Frobenius inner product between perturbed and unperturbed pointcloud
            in other two axes.
        cross: Frobenius inner product  between perturbed and unperturbed pointcloud
            after rotation by 90 degrees in other two axes.
        n_samples_clean: Number of samples used to bound classification threshold.
        n_samples_pert: Number of samples used to obtain lower bound on prediciton probability.
        p_lower: Clean prediction probability.
        alpha: Significance between 0 and 1.
        alpha_clean: If not None overrides the significance for the threshold bound.
        alpha_pert: If not None overrides the significance for the perturbed probability bound.

    Returns:
        Three floats, which return the tight lower bound using:
        1.) the most pessimistic threshold and prediction probability at the significance level,
        2.) the raw Monte Carlo estimates for the threshold and the prediction probability,
        3.) the most optimistic threshold and prediction probability.
    """

    if alpha_clean is None:
        alpha_clean = alpha / 2
    if alpha_pert is None:
        alpha_pert = alpha / 3

    if norm_delta_rot_axis == 0 and norm_delta_other == 0:
        return p_lower

    # Likelihood ratios are 2D rotation invariance ratios for non-rotation axes multiplied with
    # Gaussian likelihood ratios along rotation axis.

    ratios_clean = sample_2d_likelihood_ratio_logs(True, sigma,
                                                   norm_clean_other, norm_delta_other,
                                                   inner, cross, n_samples_clean)

    if norm_delta_other == 0:
        ratios_clean *= 0

    if norm_delta_rot_axis > 0:
        ratios_clean += sample_single_axis_likelihood_ratio_logs(
            True, sigma, norm_delta_rot_axis, n_samples_clean
        )

    ratios_pert = sample_2d_likelihood_ratio_logs(False, sigma,
                                                  norm_clean_other, norm_delta_other,
                                                  inner, cross, n_samples_pert)

    if norm_delta_other == 0:
        ratios_pert *= 0

    if norm_delta_rot_axis > 0:
        ratios_pert += sample_single_axis_likelihood_ratio_logs(
            False, sigma, norm_delta_rot_axis, n_samples_pert
        )

    p_cert_estimates = [
        forward_mc_estimate(ratios_clean, ratios_pert, p_lower, estimate_type,
                            alpha_clean, alpha_pert)
        for estimate_type in ['lower', 'mean', 'upper']
    ]

    return tuple(p_cert_estimates)


def sample_single_axis_likelihood_ratio_logs(
        clean: bool, sigma: float, norm_delta: float, n_samples: int) -> np.ndarray:
    """Samples logarithm of likelihood ratios along the rotation axis.

    These likelihood ratios are the same as in classic randomized smoothing by Cohen et al.

    Args:
        clean: If True, sample from clean smoothing distribution.
            Otherwise from perturbed  distribution.
        norm_delta: Norm of perturbation along rotation axis.
        n_samples: Number of likelihood ratios to sample.

    Returns:
        Array of length n_samples containing logarithms of sampled likelihood ratios.
    """

    if clean:
        mean = - (1 / (2 * (sigma ** 2))) * (norm_delta ** 2)
    else:
        mean = (1 / (2 * (sigma ** 2))) * (norm_delta ** 2)

    std = norm_delta / sigma

    ratio_log = np.random.normal(0, 1, n_samples) * std + mean

    return ratio_log
