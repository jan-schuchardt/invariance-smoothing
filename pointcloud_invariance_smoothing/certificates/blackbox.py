"""This module implements the certificate and inverse certificate from Cohen et al. 2019."""

from scipy.stats import norm


def inverse_cert_cohen(sigma: float, norm_delta: float) -> float:
    """Calculates smallest prediction probability s.t. robustness can be certified.

    Args:
        sigma: Smoothing standard deviation.
        norm_delta: Perturbation norm.

    Returns:
        A float between 0 and 1 corresponding to the smallest prediction probability.
    """
    return norm.cdf(norm_delta / sigma)


def forward_cert_cohen(sigma: float, norm_delta: float, p_lower: float) -> float:
    """Calculates tight lower bound on prediction probability under attack.

    Args:.
        sigma: Smoothing standard deviation
        norm_delta: Perturbation norm.
        p_lower: Clean prediction probability.

    Returns:
        A float between 0 and 1 corresponding to the tight lower bound.
    """
    return norm.cdf(norm.ppf(p_lower) - norm_delta / sigma)
