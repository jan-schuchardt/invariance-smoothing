"""This module implements the certificate and inverse certificate for 3D rotation invariance

forward_3d_rotation_cert: Lower bound on perturbed prediction probability.

sample_3d_likelihood_ratio_logs: Sample likelihood ratios (for finding worst-case classifier).

sample_3d_weights: Sample the 16 (or 18) parameters controlling cert (q from Appendix F.4.3.)

calc_trig: Evaluate cos and sin of multiple lists of angles.
"""

from functools import partial
from typing import Optional

import gmpy2
import numpy as np
import quadpy as qp
from scipy.linalg import block_diag
from scipy.special import ive as exp_modified_bessel
from statsmodels.stats.proportion import proportion_confint

from .utils import forward_mc_estimate, sample_multivariate_normal


def forward_3d_rotation_cert(sigma, X_clean, X_pert,
                             n_samples_clean, n_samples_pert,
                             p_lower,
                             quadrature_method: str = 'clenshaw_curtis',
                             quadrature_degree: int = 50,
                             use_mpf: bool = False, mpf_precision: int = 100,
                             alpha: float = 0.001, alpha_clean: Optional[float] = None,
                             alpha_pert: Optional[float] = None) -> tuple[float, float, float]:
    """Calculates tight lower bound on prediction probability under attack.

    This method uses the Monte Carlo certification procedure specified in Appendix F.5,
    i.e. using samples to bound classification threshold and then uses second round of samples
    to bound the perturbed prediction probability given this threshold.
    By default, Holm correction is used to ensure that these two bounds, together
    with the bound for p_lower hold with significance alpha.
    For the 3d certificate, we have to use numerical integration to evaluate the likelihood ratios.

    Args:
        sigma: Smoothing standard deviation.
        X_clean: The unperturbed point cloud
        X_pert: The perturbed point cloud.
        n_samples_clean: Number of samples used to bound classification threshold.
        n_samples_pert: Number of samples used to obtain lower bound on prediciton probability.
        p_lower: Clean prediction probability.
        quadratue_method: The quadpy quadrature method to use.
        quadrature_degree: The degree of the quadrature method.
        use_mpf: Whether to use mixed-precision floating point numbers during integration.
        mpf_precision: Precision when using mixed-precision floating point numbers.
        alpha: Significance between 0 and 1.
        alpha_clean: If not None overrides the significance for the threshold bound.
        alpha_pert: If not None overrides the significance for the perturbed probability bound.

    Returns:
        Three floats, which return the tight lower bound using:
        1.) the most pessimistic threshold and prediction probability at the significance level,
        2.) the raw Monte Carlo estimates for the threshold and the prediction probability,
        3.) the most optimistic threshold and prediction probability.
    """

    if np.all(np.isclose(X_clean, X_pert)):
        return p_lower

    if alpha_clean is None:
        alpha_clean = alpha / 2
    if alpha_pert is None:
        alpha_pert = alpha / 3

    ratios_clean = sample_3d_likelihood_ratio_logs(
                                    True, sigma, X_clean, X_pert, n_samples_clean,
                                    quadrature_method, quadrature_degree,
                                    use_mpf, mpf_precision)

    ratios_pert = sample_3d_likelihood_ratio_logs(
                                    False, sigma, X_clean, X_pert, n_samples_clean,
                                    quadrature_method, quadrature_degree,
                                    use_mpf, mpf_precision)

    p_cert_estimates = [
        forward_mc_estimate(ratios_clean, ratios_pert, p_lower, estimate_type,
                            alpha_clean, alpha_pert)
        for estimate_type in ['lower', 'mean', 'upper']
    ]

    return tuple(p_cert_estimates)


def sample_3d_likelihood_ratio_logs(clean: bool, sigma: float,
                                    X_clean: np.ndarray, X_pert: np.ndarray,
                                    n_samples: int,
                                    quadrature_method: str = 'clenshaw_curtis',
                                    quadrature_degree: int = 50,
                                    use_mpf: bool = False, mpf_precision: int = 100) -> np.ndarray:
    """Samples logarithm of likelihood ratios under clean or perturbed smoothing distribution.

    The log-likelihood distributions are those specified in Appendix F.4.3.
    For the 3d certificate, we have to use numerical integration to evaluate the likelihood ratios.

    Args:
        clean: If True, sample from clean smoothing distribution.
            Otherwise from perturbed  distribution.
        X_clean: The unperturbed point cloud of shape N x D.
        X_pert: The perturbed point cloud of shape N x D.
        n_samples: Number of likelihood ratios to sample.
        quadratue_method: The quadpy quadrature method to use.
        quadrature_degree: The degree of the quadrature method.
        use_mpf: Whether to use mixed-precision floating point numbers during integration.
        mpf_precision: Precision when using mixed-precision floating point numbers.

    Returns:
        Array of length n_samples containing logarithms of sampled likelihood ratios.
    """

    method_dict = {
        'chebyshev_gauss_1': qp.c1.chebyshev_gauss_1,
        'chebyshev_gauss_2': qp.c1.chebyshev_gauss_2,
        'clenshaw_curtis': qp.c1.clenshaw_curtis,
        'fejer_1': qp.c1.fejer_1,
        'fejer_2': qp.c1.fejer_2,
        'gauss_kronrod': qp.c1.gauss_kronrod,
        'gauss_legendre': qp.c1.gauss_legendre,
        'gauss_lobatto': qp.c1.gauss_lobatto,
        'gauss_radau': qp.c1.gauss_radau,
    }

    # Sample q-vectors, reshaped as 3x3 arrays,
    # from distribution specified at end of Appendix F.4.3.
    # (To simplify code, we also sample X_{:2} even though it is not used in certificate)
    A_pert, A_clean = sample_3d_weights(clean, sigma, X_clean, X_pert, n_samples)

    scheme = qp.c2.product(method_dict[quadrature_method](quadrature_degree))

    # For the taken samples, calculate their likelihood under the clean and perturbed distribution 
    integrand_pert = Integrand(A_pert, use_mpf, mpf_precision)

    likelihood_pert = scheme.integrate(
        integrand_pert.eval,
        qp.c2.rectangle_points([- np.pi / 2, np.pi / 2], [0.0, 2 * np.pi]),
    )

    integrand_clean = Integrand(A_clean, use_mpf, mpf_precision)

    likelihood_clean = scheme.integrate(
        integrand_clean.eval,
        qp.c2.rectangle_points([- np.pi / 2, np.pi / 2], [0.0, 2 * np.pi]),
    )

    # Calculate log-likelihood ratios from the pairs of likelihoods
    if use_mpf:
        mpf_log = np.frompyfunc(gmpy2.log, 1, 1)

        ratio_log_unnormalized = (
                mpf_log(likelihood_pert) - mpf_log(likelihood_clean)).astype('float')

    else:
        ratio_log_unnormalized = np.log(likelihood_pert) - np.log(likelihood_clean)

    assert not np.any(np.isnan(ratio_log_unnormalized))

    # Use sum-log-exp trick "normalizers" to avoid numerical issues
    return (ratio_log_unnormalized
            + integrand_pert.cached_normalizers - integrand_clean.cached_normalizers)


def sample_3d_weights(clean: bool, sigma: float,
                      X_clean: np.ndarray, X_pert: np.ndarray,
                      n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample from the low-dimensional normal distribution that worst-case classifier depends on.

    These parameters from the normal distribution (vector "q")
    are specified at the end of Appendix F.4.3.
    Instead of sampling from a 16-dimensional normal distribution,
    we sample from an 18-dimensional normal distribution which contains two values that are
    not actually used by the worst-case classifier.
    This is just to simplify vectorization and does not have any deeper motiviation.

    Args:
        clean: If True, sample from clean smoothing distribution.
            Otherwise from perturbed  distribution.
        X_clean: The unperturbed point cloud of shape N x D.
        X_pert: The perturbed point cloud of shape N x D.
        n_samples: Number of likelihood ratios to sample.
        quadratue_method: The quadpy quadrature method to use.
        quadrature_degree: The degree of the quadrature method.
        use_mpf: Whether to use mixed-precision floating point numbers during integration.
        mpf_precision: Precision when using mixed-precision floating point numbers.

    Returns:
        Two arrays of shape n_samples x 3 x 3.

        The first array are the q-parameters that the numerator
        of the worst-case classifier depends on.
        The second array are the q-parameters that the denominator
        of the worst-case classifier depends on. 
    """
    gmpy2.get_context().precision = 100

    # This is the block matrix from the end of Appendix F.4.3, but with additional columns
    # [X'_{:,2} 0 0]^T and [X_{:,2} 0 0]^T
    W = np.vstack(
                [block_diag(
                    *np.repeat(col[np.newaxis, :], 3, axis=0))
                 for col in np.hstack((X_pert, X_clean)).T])

    assert np.all(W.shape == np.array([18, X_clean.shape[0] * 3]))

    if clean:
        orig_mean = X_clean.flatten('f')  # Columns stacked above each other
    else:
        orig_mean = X_pert.flatten('f')  # Columns stacked above each other

    mean = W @ orig_mean
    cov = W @ W.T

    # Sample from N(mean, cov * sigma^2)
    A = sample_multivariate_normal(mean, cov, sigma, n_samples)

    A_pert = A[:, :9].reshape((-1, 3, 3))
    A_clean = A[:, 9:].reshape((-1, 3, 3))

    return A_pert, A_clean


class Integrand():
    """The integrand beta from Appendix F.4.3., for use with quadpy integration.

    Attributes:
        A: Array of shape n_samples x 3 x 3, corresponding to vector q.
        use_mpf: If true, use mixed-precision floating points.
        mpf_precision: Precision of mixed-precision floating points.
        cached_normalizers: Stores largest exponent to use sum-log-exp-trick.
    """
    def __init__(self, A, use_mpf=False, mpf_precision=100):
        self.A = A
        self.use_mpf = use_mpf
        self.mpf_precision = mpf_precision

        self.cached_normalizers = None

    def eval(self, x):
        """Evaluates integrand for different angles omega_2 and omega_3 from Appendix F.4.3."""
        gmpy2.get_context().precision = self.mpf_precision

        # A shape: N x 3 x 3
        A = self.A

        omega_2 = x[0, :]
        omega_3 = x[1, :]

        # These are shorthands for cos(omega_2), cos(omega_3), sin(omega_2), sin(omega_3)
        c_2, c_3, s_2, s_3 = calc_trig(omega_2, omega_3)

        chi_1 = np.einsum(
                    'ij,ik->kj',
                    np.array([c_2, s_2 * s_3, c_3 * s_2, c_3, -s_3]),  # 5 x S,
                    np.array([A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 1, 1], A[:, 1, 2]])  # 5 x N
                )  # N x S

        chi_2 = np.einsum(
                    'ij,ik->kj',
                    np.array([-c_3, s_3, c_2, s_2 * s_3, c_3 * s_2]),  # 5 x S,
                    np.array([A[:, 0, 1], A[:, 0, 2], A[:, 1, 0], A[:, 1, 1], A[:, 1, 2]])  # 5 x N,
                )  # N x S

        bessel_arg = np.sqrt((chi_1 ** 2) + (chi_2 ** 2))

        exponent = np.einsum(
            'ij,ik->kj',
            np.array([-s_2, c_2 * s_3, c_2 * c_3]),  # 3 x S
            np.array([A[:, 2, 0], A[:, 2, 1], A[:, 2, 2]])  # 3 x N
        )  # N x S

        exponent += np.log(exp_modified_bessel(0, bessel_arg)) + bessel_arg

        self.cached_normalizers = np.max(exponent, axis=1)
        exponent -= self.cached_normalizers[:, np.newaxis]

        if self.use_mpf:
            gmpy2.get_context().precision = self.mpf_precision
            ret = 2 * np.pi * c_2 * np.frompyfunc(gmpy2.exp, 1, 1)(exponent)
            return ret
        else:
            return 2 * np.pi * c_2 * np.exp(exponent)


def calc_trig(*angles) -> np.ndarray:  # arbitrary number of arrays of same size
    """Simultaneously calculate cosine and sine of N arrays of length L.

    This is needed as a helper function for numerical integration over Euler angles.

    Args:
        angles: Numpy arrays of identical length L.

    Returns:
        An array of shape (2 * N) x L, where first N rows are cosine and last N rows
        are sin.
    """
    cosines = np.cos(np.array(list(angles)))
    sines = np.sin(np.array(list(angles)))

    x = np.concatenate((cosines, sines), axis=0)

    return x
