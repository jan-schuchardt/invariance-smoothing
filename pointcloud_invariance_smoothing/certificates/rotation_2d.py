"""This module implements the certificate and inverse certificate for 2D rotation invariance.

inverse_2d_rotation_cert: Smallest prediction probability s.t. robustness can be certified.

forward_2d_rotation_cert: Lower bound on perturbed prediction probability.

sample_2d_likelihood_ratio_logs: Sample likelihood ratios (for finding worst-case classifier).
"""

from typing import Optional

import gmpy2
import numpy as np
from scipy.special import ive as exp_modified_bessel
from statsmodels.stats.proportion import proportion_confint

from .utils import forward_mc_estimate, sample_multivariate_normal


def inverse_2d_rotation_cert(sigma, norm_clean, norm_delta, inner, cross, n_samples_clean,
                             n_samples_pert, alpha=0.001) -> float:
    """Calculates smallest prediction probability s.t. robustness can be certified.

    This method uses the inverse Monte Carlo certification procedure
    specified in Appendix I, i.e. adjust threshold s.t. perturbed prediction probability is 0.5.
    and then bound corresponding clean prediction probability.
    Holm correction is used to ensure that these two bounds, together
    with the bound for p_lower hold with significance alpha.

    Args:
        sigma: Smoothing standard deviation.
        norm_clean: Norm of unperturbed point cloud.
        norm_delta: Norm of perturbation.
        inner: Frobenius inner product between perturbed and unperturbed pointcloud
            (eps_1 from paper).
        cross: Frobenius inner product  between perturbed and unperturbed pointcloud
            after rotation by 90 degrees (eps_2 from paper)
        n_samples_clean: Number of samples used to bound required clean prediction probability.
        n_samples_pert: Number of samples used to bound classification threshold.
        p_lower: Clean prediction probability.
        alpha: Significance between 0 and 1.

    Returns:
        A float between 0 and 1 corresponding to the smallest prediction probability.
    """

    ratios_clean = sample_2d_likelihood_ratio_logs(True, sigma,
                                                   norm_clean, norm_delta,
                                                   inner, cross, n_samples_clean)

    ratios_pert = sample_2d_likelihood_ratio_logs(False, sigma,
                                                  norm_clean, norm_delta,
                                                  inner, cross, n_samples_pert)

    cdf_pert_lower = proportion_confint(np.arange(1, n_samples_pert + 1),
                                        n_samples_pert, alpha=2 * alpha / 3, method='beta')[0]

    # smallest threshold that ensures that cert > 0.5
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


def forward_2d_rotation_cert(sigma: float,
                             norm_clean: float, norm_delta: float, inner: float, cross: float,
                             n_samples_clean: int, n_samples_pert: int,
                             p_lower: float, alpha: float = 0.001,
                             alpha_clean: Optional[float] = None,
                             alpha_pert: Optional[float] = None) -> tuple[float, float, float]:
    """Calculates tight lower bound on prediction probability under attack.

    This method uses the Monte Carlo certification procedure specified in Appendix F.5,
    i.e. using samples to bound classification threshold and then uses second round of samples
    to bound the perturbed prediction probability given this threshold.
    By default, Holm correction is used to ensure that these two bounds, together
    with the bound for p_lower hold with significance alpha.

    Args:
        sigma: Smoothing standard deviation.
        norm_clean: Norm of unperturbed point cloud.
        norm_delta: Norm of perturbation.
        inner: Frobenius inner product between perturbed and unperturbed pointcloud
            (eps_1 from paper).
        cross: Frobenius inner product  between perturbed and unperturbed pointcloud
            after rotation by 90 degrees (eps_2 from paper)
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

    if norm_delta == 0:
        return p_lower

    if alpha_clean is None:
        alpha_clean = alpha / 2
    if alpha_pert is None:
        alpha_pert = alpha / 3

    ratios_clean = sample_2d_likelihood_ratio_logs(True, sigma,
                                                   norm_clean, norm_delta,
                                                   inner, cross, n_samples_clean)

    ratios_pert = sample_2d_likelihood_ratio_logs(False, sigma,
                                                  norm_clean, norm_delta,
                                                  inner, cross, n_samples_pert)

    p_cert_estimates = [
        forward_mc_estimate(ratios_clean, ratios_pert, p_lower, estimate_type,
                            alpha_clean, alpha_pert)
        for estimate_type in ['lower', 'mean', 'upper']
    ]

    return tuple(p_cert_estimates)


def sample_2d_likelihood_ratio_logs(
        clean: bool, sigma: float,
        norm_clean: float, norm_delta: float, inner: float, cross: float,
        n_samples: int) -> np.ndarray:
    """Samples logarithm of likelihood ratios under clean or perturbed smoothing distribution.

    The log-likelihood distributions are those specified in Theorem 6 of Appendix F.4.2.

    Args:
        clean: If True, sample from clean smoothing distribution.
            Otherwise from perturbed  distribution.
        norm_clean: Norm of unperturbed point cloud.
        norm_delta: Norm of perturbation.
        inner: Frobenius inner product between perturbed and unperturbed pointcloud
            (eps_1 from paper).
        cross: Frobenius inner product  between perturbed and unperturbed pointcloud
            after rotation by 90 degrees (eps_2 from paper)
        n_samples: Number of likelihood ratios to sample.

    Returns:
        Array of length n_samples containing logarithms of sampled likelihood ratios.
    """
    a = gmpy2.mpfr(float(norm_clean)) ** 2
    b = gmpy2.mpfr(float(norm_delta)) ** 2
    c = gmpy2.mpfr(float(inner))
    d = gmpy2.mpfr(float(cross))

    # Parameters Theorem 6, without the factor 1 / (std^2) to avoid numerical issues.
    if clean:
        mean = np.array([a+c, -d, a, 0])
        cov = np.array([
            [a+b+2*c, 0, a+c, d],
            [0, a+b+2*c, -d, a+c],
            [a+c, -d, a, 0],
            [d, a+c, 0, a]
        ])

    else:
        mean = np.array([a+b+2*c, 0, a+c, d])
        cov = np.array([
            [a+b+2*c, 0, a+c, d],
            [0, a+b+2*c, -d, a+c],
            [a+c, -d, a, 0],
            [d, a+c, 0, a]
        ])

    # Samples from N(mean, cov * sigma^2)
    K = sample_multivariate_normal(mean, cov, sigma, n_samples)

    # These norms correspond to 1 / (sigma^2) * N(mean, cov*sigma^2) = N(mean/std^2, cov/std^2).
    q_1 = np.linalg.norm(K[:, :2], axis=1) / (sigma ** 2)
    q_2 = np.linalg.norm(K[:, 2:], axis=1) / (sigma ** 2)

    # Use exp modified Bessel functions to avoid numerical issues for large arguments
    ratio_log = (np.log(exp_modified_bessel(0, q_1)) + q_1
                 - np.log(exp_modified_bessel(0, q_2)) - q_2)

    assert not np.any(np.isnan(ratio_log))

    return ratio_log
