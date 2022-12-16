from multiprocessing.sharedctypes import Value

import gmpy2
import numpy as np
from statsmodels.stats.proportion import proportion_confint


def sample_multivariate_normal(
        mean: np.ndarray, cov_wo_sigma: np.ndarray, sigma: float, n_samples: int) -> np.ndarray:
    """Sampling from multivariate normal distribution with scaled covariance matrix.

    Yields samples from N(mean, cov_wo_sigma * sigma^2).
    Avoids some numerical issues with very small standard deviations.

    Args:
        mean: Mean of the normal distribution of shape D
        cov_wo_sigma: Covariance matrix of shape D x D
        sigma: Scaling factor for covariance matrix
        n_samples: Number of samples

    Returns:
        Array of samples of shape n_samples x D
    """
    gmpy2.get_context().precision = 100

    cov = cov_wo_sigma

    # Avoid numerical issues by scaling mean and covariance, then multiplying sample
    if np.any(cov != 0):
        cov_max = gmpy2.mpfr(1.0) * np.max(np.abs(cov[cov != 0]))
        cov = cov / cov_max
        mean = mean / gmpy2.sqrt(cov_max)
        mean /= sigma

    mean = mean.astype('float')
    cov = cov.astype('float')

    # This now has distribution sigma * sqrt(cov_max) * N(mean/sigma/sqrt(cov_max), cov/cov_max)
    # = sigma * N(mean/sigma, cov) = N(mean, cov * sigma^2)
    return (sigma
            * float(gmpy2.sqrt(cov_max))
            * np.random.multivariate_normal(mean, cov, size=n_samples))


def forward_mc_estimate(
        ratios_clean: np.ndarray, ratios_pert: np.ndarray,
        p_lower: float, mode: str,
        alpha_clean: float, alpha_pert: float) -> float:
    """Calculates certified lower bounds on prediction probability using sampled likelihood ratios.

    Implements the Monte Carlo certification algorithm from Appendix F.5.

    Args:
        ratios_clean: Array of length N_2, containing ratios sampled from clean distribution.
        ratios_pert: Array of length N_3, containing ratios sampled from perturbed distribution.
        p_lower: Clean prediction probability.
        mode: Whether to use pessimistic (lower) bounds, optimistic (upper) bounds
            or just use the monte carlo samples directly (mean).I
            Importantly, all of the certificates yield lower bounds
            on the perturbed prediction probability.
            These parameters just control how we perform Monte Carlo estimation
            of the certification threshold and the resulting  perturbed prediction probability.
        alpha_clean: Significance level to use for boudning classification threshold.
        alpha_pert: Significance level to use for bounding perturbed prediction probability.

    Returns:
        Lower bound on preturbed prediction probability.
    """

    if mode not in ['lower', 'upper', 'mean']:
        raise ValueError("Supported modes are 'lower', 'upper' and 'mean'")
    n_samples_clean = len(ratios_clean)
    n_samples_pert = len(ratios_pert)

    if mode == 'lower':
        # Overestimate cdf to make threshold necessary for reaching p_lower smaller
        cdf_clean = proportion_confint(np.arange(1, n_samples_clean + 1),
                                       n_samples_clean, alpha=2 * alpha_clean, method='beta')[1]

    elif mode == 'upper':
        # Underestimate cdf to make threshold necessary for reaching p_lower bigger
        cdf_clean = proportion_confint(np.arange(1, n_samples_clean + 1),
                                       n_samples_clean, alpha=2 * alpha_clean, method='beta')[0]

        # We know that the threshold infty always contains 100% prob mass
        cdf_clean = np.append(cdf_clean, 1.0)
        ratios_clean = np.append(ratios_clean, np.inf)

    elif mode == 'mean':
        cdf_clean = np.arange(1, n_samples_clean + 1) / n_samples_clean

    else:
        assert False

    # largest threshold that covers at most p_lower% of samples w.h.p
    threshold_idx = max((cdf_clean >= p_lower).argmax() - 1, 0)
    threshold = np.sort(ratios_clean)[threshold_idx]

    if mode == 'lower':
        return proportion_confint((ratios_pert <= threshold).sum(),
                                  n_samples_pert, alpha=2 * alpha_pert, method='beta')[0]

    elif mode == 'upper':
        return proportion_confint((ratios_pert <= threshold).sum(),
                                  n_samples_pert, alpha=2 * alpha_pert, method='beta')[1]

    elif mode == 'mean':
        return (ratios_pert <= threshold).sum() / n_samples_pert

    else:
        assert False
