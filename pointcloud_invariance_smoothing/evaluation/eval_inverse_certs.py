"""Functions for finding the smallest prediction probability s.t. robustness can be certified.

eval_multiple_norms_multiple_inner_cross: Cartesian product of all 4 2D-rotation cert params.

eval_multiple_norms_single_inner_cross: Cartesian product of all clean norm and perturbation norm.

eval_multiple_inner_cross: Cartesian product of eps_1 and eps_2 for fixed norms.

eval_rotation_only: Evaluation on values of eps_1 and eps_2 corresponding to adversarial rotation.

compute_rotation_inner_cross: Finding values of eps_1 and eps_2 corresponding to adv. rotation.
"""

import itertools

import numpy as np
from pointcloud_invariance_smoothing.certificates.blackbox import \
    inverse_cert_cohen
from pointcloud_invariance_smoothing.certificates.rotation_2d import \
    inverse_2d_rotation_cert
from tqdm import tqdm


def eval_multiple_norms_multiple_inner_cross(
        sigma: float, norms_clean: list[float], norms_delta: list[float],
        n_points_inner: int, n_points_cross: int,
        n_samples_clean: int, n_samples_pert: int, alpha: float) -> dict:
    """Evaluates inverse certificate for 2D rotation invariance on grid of all 4 parameters

    The inverse certificate is the smallest clean prediction probability such that
    robustness can be certified.
    It is computed using Algorithm 3 from Appendix I.
    It is evaluated on the Cartesian product of the specified clean data norms,
    perturbations norms and a linspace of certificate parameters eps_1 (inner product)
    and eps_2 (inner product after rotation by 90 degrees).
    The parameters eps_1 and eps_2 are evaluated on a linspace from
    [-norm_clean * norm_delta, norm_clean * norm_delta].

    Args:
        sigma: Smoothing standard deviaton
        norms_clean: List of unperturbed data norms ||X|| to evaluate on
        norms_delta: List of perturbation norms ||Delta|| to evaluate on
        n_points_inner: Number of points for linspace in feasible range of eps_1
        n_points_cross: Number of points for linspace in feasible range of eps_2
        n_samples_clean: Number of MC samples to use for finding minimum prediction probability.
        n_samples_pert: Number of MC samples to use for finding classification threshold.
        alpha: Significance level for confidence bounds.

    Returns:
        A dictionary whose keys are tuples of clean data norms and perturbations norms.
        Each entry is another dictionary with five entries:
            1.) 'inverse_cert_cohen': Float value of the inverse black-box certificate,
                which only depends on perturbation norm
            2.) 'inverse_ours': Float array of shape n_points_inner x n_points_cross,
                with values from inverse 2D rotation cert
            3.) 'valid_params_mask': Boolean array of shape n_points_inner x n_points_cross
                indicating which entries correspond to
                values of eps_1 and eps_2 that are feasible (see Appendix J.1)
            4.) 'inner_vals': The linspace for eps_1
            5.) 'cross_vals': The linspace for eps_2
            6.) 'inverse_rotation_only': The inverse certificate evaluated on the two combinations
                of eps_1 and eps_2 corresponding to adversarial rotations (see Appendix J.2),
                as well as the value of the inverse certificate evaluated on these two points.
    """

    result_dict = {'sigma': sigma}

    for norm_clean, norm_delta in tqdm(itertools.product(norms_clean, norms_delta)):

        norm_result_dict = {
            'inverse_cohen': inverse_cert_cohen(sigma, norm_delta)
        }

        inner_vals = np.linspace(-1 * norm_clean * norm_delta,
                                 norm_clean * norm_delta, n_points_inner)

        cross_vals = np.linspace(-1 * norm_clean * norm_delta,
                                 norm_clean * norm_delta, n_points_cross)

        inverse_ours, valid_params_mask = eval_multiple_inner_cross(
                                                 sigma, norm_clean, norm_delta,
                                                 inner_vals, cross_vals,
                                                 n_samples_clean, n_samples_pert, alpha)

        norm_result_dict['inverse_ours'] = inverse_ours
        norm_result_dict['valid_params_mask'] = valid_params_mask
        norm_result_dict['inner_vals'] = inner_vals
        norm_result_dict['cross_vals'] = cross_vals

        if norm_delta <= 2*norm_clean:

            inverse_rotation_only = eval_rotation_only(sigma, norm_clean, norm_delta,
                                                       n_samples_clean, n_samples_pert, alpha)

            norm_result_dict['inverse_rotation_only'] = inverse_rotation_only

        result_dict[(norm_clean, norm_delta)] = norm_result_dict

    return result_dict


def eval_multiple_norms_single_inner_cross(
                        sigma: float, norms_clean: list[float], norms_delta: list[float],
                        inner: float, cross: float,
                        relative_inner_cross: bool,
                        n_samples_clean: int, n_samples_pert: int, alpha: float) -> dict:
    """Evaluates inverse cert for 2D rotation invariance for different data and perturbation norms

    This function only takes a single value for inner (eps_1) and cross (eps_2).
    Other than that, it is virtually identical to eval_multiple_norms_multiple_inner_cross,
    but allows us to also specify inner (eps_1) and cross (eps_2) relative to
    norms_clean * norms_delta, which we use for the experiments on adversarial scaling
    (see Fig. 2).

    Args:
        sigma: Smoothing standard deviaton
        norms_clean: List of unperturbed data norms ||X|| to evaluate on
        norms_delta: List of perturbation norms ||Delta|| to evaluate on
        inner: Inner product between clean point cloud and perturbation vector (eps_1)
        cross: Inner product between clean point cloud and perturbation vector after rotation
            by 90 degrees (eps_2)
        relative_inner_cross: If True, inner and cross are interpreted as fractions of
            norms_clean * norms_delta, i.e. tilde(eps)_1 and tilde(eps)_2 from paper.
        n_samples_clean: Number of MC samples to use for finding minimum prediction probability.
        n_samples_pert: Number of MC samples to use for finding classification threshold.
        alpha: Significance level for confidence bounds.

    Returns:
        A dictionary whose keys are tuples of clean data norms and perturbations norms.
        Each entry is another dictionary with five entries:
            1.) 'inverse_cert_cohen': Float value of the inverse black-box certificate,
                which only depends on perturbation norm
            2.) 'inverse_ours': Float array of shape 1 x 1,
                with values from inverse 2D rotation cert
            3.) 'valid_params_mask': Boolean array of shape 1x1 indicating whether the combination
                of eps_1 and eps_2 is feasible (see Appendix J.1)
            4.) 'inner_vals': The linspace for eps_1
            5.) 'cross_vals': The linspace for eps_2
            6.) 'inverse_rotation_only': A dictionary containing the two combinations
                of eps_1 and eps_2 corresponding to adversarial rotations (see Appendix J.2),
                as well as the value of the inverse certificate evaluated on these two points.
    """

    result_dict = {'sigma': sigma}

    for norm_clean, norm_delta in tqdm(itertools.product(norms_clean, norms_delta)):

        norm_result_dict = {
            'inverse_cohen': inverse_cert_cohen(sigma, norm_delta)
        }

        if relative_inner_cross:
            if not (-1 <= inner <= 1):
                raise ValueError('Inner must be in [0,1] for relative mode')
            if not (-1 <= cross <= 1):
                raise ValueError('Inner must be in [0,1] for relative mode')

            inner_vals = np.array([inner * norm_clean * norm_delta])
            cross_vals = np.array([cross * norm_clean * norm_delta])

        else:
            inner_vals = np.array([inner])
            cross_vals = np.array([cross])

        inverse_ours, valid_params_mask = eval_multiple_inner_cross(
                                                 sigma, norm_clean, norm_delta,
                                                 inner_vals, cross_vals,
                                                 n_samples_clean, n_samples_pert, alpha)

        norm_result_dict['inverse_ours'] = inverse_ours
        norm_result_dict['valid_params_mask'] = valid_params_mask
        norm_result_dict['inner_vals'] = inner_vals
        norm_result_dict['cross_vals'] = cross_vals

        if norm_delta <= 2*norm_clean:

            inverse_rotation_only = eval_rotation_only(sigma, norm_clean, norm_delta,
                                                       n_samples_clean, n_samples_pert, alpha)

            norm_result_dict['inverse_rotation_only'] = inverse_rotation_only

        result_dict[(norm_clean, norm_delta)] = norm_result_dict

    return result_dict


def eval_multiple_inner_cross(sigma: float, norm_clean: float, norm_delta: float,
                              inner_vals: list[float], cross_vals: list[float],
                              n_samples_clean: int, n_samples_pert: int, alpha: float
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates inverse cert for 2D rotation invariance for fixed data and perturbation norm

    Here, the certificate is evaluated for the Cartesian product of the specified
    inner_vals (eps_1) and cross_vals (eps_2). 

    Args:
        sigma: Smoothing standard deviaton
        norms_clean: List of unperturbed data norms ||X|| to evaluate on
        norms_delta: List of perturbation norms ||Delta|| to evaluate on
        inner: Inner product between clean point cloud and perturbation vector (eps_1)
        cross: Inner product between clean point cloud and perturbation vector after rotation
            by 90 degrees (eps_2)
        relative_inner_cross: If True, inner and cross are interpreted as fractions of
            norms_clean * norms_delta, i.e. tilde(eps)_1 and tilde(eps)_2 from paper.
        n_samples_clean: Number of MC samples to use for finding minimum prediction probability.
        n_samples_pert: Number of MC samples to use for finding classification threshold.
        alpha: Significance level for confidence bounds.

    Returns:
        Returns two arrays of shape len(inner_vals) x len(cross_vals):
            1.) Values of the 2D rotation cert, with entry i, j corresponding to
                inner_vals[i] and cross_vals[j]
            2.) Boolean array, with entry i, j indicating whether inner_vals[i] and cross_vals[j]
                are within the feasible range of eps_1 and eps_2 (see Appendix J.1)
    """
    inverse_certs = np.zeros((len(inner_vals), len(cross_vals)))
    valid_params_mask = np.ones((len(inner_vals), len(cross_vals)), dtype='bool')

    for i, inner in enumerate(inner_vals):
        for j, cross in enumerate(cross_vals):

            if np.sqrt(inner ** 2 + cross ** 2) <= norm_clean * norm_delta:

                inverse_certs[i, j] = inverse_2d_rotation_cert(
                        sigma, norm_clean, norm_delta,
                        inner, cross, n_samples_clean, n_samples_pert, alpha)

            else:
                valid_params_mask[i, j] = False

    return inverse_certs, valid_params_mask


def eval_rotation_only(sigma: float, norm_clean: float, norm_delta: float,
                       n_samples_clean: int, n_samples_pert: int, alpha: float) -> dict:
    """Evaluates inverse cert for 2D rotation invariance on  points corresponding to adv rotations

    The formula for the two points can be found in Appendix J.2.

    Args:
        sigma: Smoothing standard deviaton
        norm_clean: Unperturbed data norm ||X|| to evaluate on
        norm_delta: Perturbation norm ||Delta|| to evaluate on
        n_samples_clean: Number of MC samples to use for finding minimum prediction probability.
        n_samples_pert: Number of MC samples to use for finding classification threshold.
        alpha: Significance level for confidence bounds.

    Returns:
        Dictionary with five entries:
            1.) 'inner': The value of eps_1 corresopnding to adversarial rotation
            2.) 'cross_1': The first value of eps_1 corresponding to adversarial rotation
            3.) 'cross_2': The second value of eps_2 corresponding to adversarial rotation
            4.) 'cert_1': The inverse certificate evaluated at (inner, cross_1)
            5.) 'cert_2': The inverse certificate evaluated at (inner, cross_2)
    """
    inner, cross_1, cross_2 = compute_rotation_inner_cross(norm_clean, norm_delta)

    cert_1 = inverse_2d_rotation_cert(sigma,
                                      norm_clean, norm_delta,
                                      inner, cross_1,
                                      n_samples_clean, n_samples_pert, alpha)

    cert_2 = inverse_2d_rotation_cert(sigma,
                                      norm_clean, norm_delta,
                                      inner, cross_2,
                                      n_samples_clean, n_samples_pert, alpha)

    return {
        'inner': inner,
        'cross_1': cross_1,
        'cross_2': cross_2,
        'cert_1': cert_1,
        'cert_2': cert_2,
    }


def compute_rotation_inner_cross(
        norm_clean: float, norm_delta: float) -> tuple[float, float, float]:
    """Finds two points corresponding to adversarial rotations, see Appendix J.2

    Args:
        norm_clean: Unperturbed data norm ||X|| to evaluate on
        norm_delta: Perturbation norm ||Delta|| to evaluate on

    Returns:
        Three floats:
            1.) The value of eps_1 corresopnding to adversarial rotation
            2.) The first value of eps_1 corresponding to adversarial rotation
            3.) The second value of eps_2 corresponding to adversarial rotation
    """
    a = norm_clean ** 2
    b = norm_delta ** 2

    angle_1 = np.arccos(1 - b / (2*a))
    angle_2 = - angle_1

    inner = a * (np.cos(angle_1) - 1)
    assert np.isclose(inner, -b / 2)

    cross_1 = -a * np.sin(angle_1)
    cross_2 = -a * np.sin(angle_2)

    assert np.isclose(cross_1, -1 / 2 * np.sqrt(b * (4 * a - b)))
    assert np.isclose(cross_2, 1 / 2 * np.sqrt(b * (4 * a - b)))

    return inner, cross_1, cross_2
