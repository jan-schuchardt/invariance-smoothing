"""This module implements functionality for evaluating the different certificates.

Here, we evaluate "forward" certificates, i.e. computing the worst-case perturbed
prediction probability, given an attack (budget) and a clean prediction probability.

eval_dataset_threatmodel: Evaluate on entire point cloud classification dataset

eval_datapoint_threatmodel: Evaluate on single point cloud with specific threat model

calc_p_cert_tight: Evaluate tight certificate (calls invariance-specific certs)

calc_p_cert_tight_2D: Lower bound on perturbed prediction probability for 2D rotation

calc_p_cert_tight_3D: Lower bound on perturbed prediction probability for 3D rotation

calc_p_cert_tight_elemental: Lower bound for invariance to rotation around coordinate axes

calc_p_cert_tight_elemental_single_axis: Lower bound for invariance to rotation around specific axis

calc_p_cert_baseline: Lower bound from black-box randomized smoothing

calc_p_cert_preprocessing: Lower bound from orbit-based certificate

sample_perturbed_input: Calls other perturbed functions based on parameter dictionary.

generate_parallel_perturbation: Performs adversarial scaling of the input point cloud.

generate_random_perturbation: Adds random perturbation and then randomly rotates input.

generate_random_rotation_matrix: Random rotation matrix for coordinate axis or arbitrary rotation

get_norms_delta: Parses parameter dictionary to determine perturbation norms for evaluation.

align_perturbed_pointcloud: Rotate perturbed point cloud to minimize dist. to clean one (for orbit).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm

from pointcloud_invariance_smoothing.certificates.blackbox import \
    forward_cert_cohen
from pointcloud_invariance_smoothing.certificates.rotation_2d import \
    forward_2d_rotation_cert
from pointcloud_invariance_smoothing.certificates.rotation_3d import \
    forward_3d_rotation_cert
from pointcloud_invariance_smoothing.certificates.rotation_elemental import \
    forward_elemental_rotation_cert


def eval_dataset_threatmodel(
         std: float,
         dataset: torch.utils.data.Dataset,
         pred_sample_results: dict,
         threat_model_params: dict, certificate_params: dict,
         n_datapoints: dict, n_samples_per_datapoint: dict) -> dict:
    """Evaluates specified certificates for given threat model on dataset.

    Args:
        std: Smoothing standard deviation
        dataset: The dataset (must match dataset used for generating pred_sample_results)
        threat_model_params: Parameters of the threat model, see seml/scripts/eval_forward.py
        certificate_params: Parameters for the evaluation of the certs, see eval_forward.py
        n_datapoints: How many datapoints to evaluate the certificate on
        n_samples_per_datapoint: How many randomly perturbed inputs to generate per datapoint

    Returns:
        A nested dictionary. With perturbation budget at first level,
        datapoint index at second level
        and different types of certificate results at third level
        (see eval_datapoint_threatmodel below).
    """

    results_dict = {}

    # Determine list of perturbation norms that certificates should be evaluated on
    # Either a linspace or a list of user-specified values.
    norms_delta = get_norms_delta(threat_model_params)

    for norm_delta in norms_delta:

        results_dict_delta = {}

        for i, idx in tqdm(enumerate(pred_sample_results['datapoint_idx'][:n_datapoints])):
            x = dataset[idx][0]
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            assert x.ndim == 2 and (x.shape[-1] == 2 or x.shape[-1] == 3)

            target = dataset[idx][1]
            assert target == pred_sample_results['targets'][i]

            cert_dict = eval_datapoint_threatmodel(
                         std, x, norm_delta,
                         target,
                         pred_sample_results['votes_pred'][i], pred_sample_results['votes_cert'][i],
                         threat_model_params['distribution_params'],
                         certificate_params,
                         n_samples_per_datapoint)

            results_dict_delta[idx] = cert_dict

        results_dict[norm_delta] = results_dict_delta

    return results_dict


def eval_datapoint_threatmodel(
        std: float,
        x: np.ndarray, norm_delta: float,
        target: int, votes_pred: np.ndarray, votes_cert: np.ndarray,
        distribution_params: dict, certificate_params: dict,
        n_samples_per_datapoint: int) -> dict:
    """Evaluates specified certificates for given datapoint.

    Args:
        std: Smoothing standard deviation
        x: The input datapoint
        norm_delta: Norm of the adversarial perturbation
        target: The ground truth label
        votes_pred: Array containing number of predictions per class (for prediction)
        votes_cert: Array containing number of predictions per class (for certification)
        distribution_params: Parameters specifying random input perturbations, see eval_forward.py
        certificate_params: Parameters for the evaluation of the certs, see eval_forward.py
        n_samples_per_datapoint: How many randomly perturbed inputs to generate per datapoint

    Returns:
        A dictionary with 5 keys:
        1.) correct: Whether the prediction is correct
        2.) abstain: Whether we have to abstain from making a prediction due to low consistency
        3.) p_certs_baseline: Perturbed probability lower bound from black-box cert
            (one per perturbed input)
        4.) p_certs_preprocessing: Perturbed probability lower bound from orbit-based cert
            (three per perturbed input, corresponding to using pessimistic confidence bound,
            directly using Monte Carlo estimate and using optimistic confidence bound
            for our Monte Carlo evaluation procedure from Algorithm 1).
            See also .py files in certificates module).
        5.) p_certs_tight: Perturbed probability lower bound from our tight invariance-aware cert
            (three per perturbed input, same as p_certs_preprocessing).
    """

    pred = votes_pred.argmax()
    correct_pred = (pred == target)

    n_consistent_pred = votes_cert[pred]
    n_samples_cert = votes_cert.sum()
    # Confidence lower bound on clean prediction probability, like in normal randomized smoothing
    p_clean_lower = proportion_confint(n_consistent_pred, n_samples_cert,
                                       certificate_params['alpha'] * 2, method='beta')[0]

    abstain = (p_clean_lower <= 0.5)

    p_certs_baseline = []
    p_certs_preprocessing = []
    p_certs_tight = []

    # Create n_samples_per_datapoint different perturbed inputs and evaluate certificates on them.
    for _ in range(n_samples_per_datapoint):
        x_pert = sample_perturbed_input(x, norm_delta, distribution_params)

        assert np.all(x_pert.shape == x.shape)

        assert isinstance(certificate_params['baseline'], bool)
        assert isinstance(certificate_params['preprocessing'], bool)
        assert isinstance(certificate_params['tight'], bool)
        if certificate_params['baseline']:

            p_certs_baseline.append(
                calc_p_cert_baseline(std, x, x_pert, p_clean_lower, certificate_params))

        if certificate_params['preprocessing']:

            p_certs_preprocessing.append(
                calc_p_cert_preprocessing(std, x, x_pert, p_clean_lower, certificate_params))

        if certificate_params['tight']:

            p_certs_tight.append(
                calc_p_cert_tight(std, x, x_pert, p_clean_lower, certificate_params))

    cert_dict = {
        'correct': correct_pred,
        'abstain': abstain,
        'p_certs_baseline': np.array(p_certs_baseline),
        'p_certs_preprocessing': np.array(p_certs_preprocessing),
        'p_certs_tight': np.array(p_certs_tight)
    }

    return cert_dict


def calc_p_cert_tight(std: float, x: np.ndarray, x_pert: np.ndarray,
                      p_clean: float, certificate_params: dict) -> tuple[float, float, float]:
    """Evaluates tight certificate for rotation-invariance for pair of clean and perturbed input.

    Depending on the data dimensionality, either evaluates the certificate for rotation
    invariance in 2D or 3D.
    In 3D, either evaluates the certificate for invariance to arbitrary rotations or
    rotation around a single axis

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'elemental' is True and D == 3, evaluate certificate for rotation invariance
            around single axis. Otherwise, evaluate certificate for invariance to arbitrary
            rotations.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.

    Returns:
        A tuple of three floats, corresponding to lower bound on perturbed prediction probability
        when using 1.) pessimistic confidence bounds, 2.) raw Monte Carlo estimate
        3.) optimistic confidence bounds in our Monte Carlo evaluation procedure from Algorithm 1.
    """
    D = x.shape[-1]
    assert D == 2 or D == 3

    if D == 2:
        return calc_p_cert_tight_2d(std, x, x_pert, p_clean, certificate_params)
    if (D == 3) and certificate_params['elemental']:
        return calc_p_cert_tight_elemental(std, x, x_pert, p_clean, certificate_params)
    if (D == 3) and not certificate_params['elemental']:
        return calc_p_cert_tight_3d(std, x, x_pert, p_clean, certificate_params)


def calc_p_cert_tight_2d(std: float, x: np.ndarray, x_pert: np.ndarray,
                         p_clean: float, certificate_params: dict) -> tuple[float, float, float]:
    """Evaluates tight certificate for 2D rotation-invariance for pair of clean and perturbed input.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.
            'n_samples_clean': Number of MC samples for finding classification threshold.
            'n_samples_pert: Number of MC samples for estimating perturbed prediction probability.
            'alpha': Significance for used confidence intervals.

    Returns:
        A tuple of three floats, corresponding to lower bound on perturbed prediction probability
        when using 1.) pessimistic confidence bounds, 2.) raw Monte Carlo estimate
        3.) optimistic confidence bounds in our Monte Carlo evaluation procedure from Algorithm 1.
    """
    if certificate_params['preprocess_translation']:
        x = x.copy()
        x_pert = x_pert.copy()
        x = x - x.mean(axis=0)
        x_pert = x_pert - x_pert.mean(axis=0)

    delta = x_pert - x
    norm_clean = np.linalg.norm(x)
    norm_delta = np.linalg.norm(x_pert - x)
    inner = np.sum(x * delta)  # eps_1 from paper
    cross = np.sum(x[:, 1] * delta[:, 0]) - np.sum(x[:, 0] * delta[:, 1])  # eps_2 from paper

    return forward_2d_rotation_cert(
        std,
        norm_clean, norm_delta, inner, cross,
        certificate_params['n_samples_clean'], certificate_params['n_samples_pert'],
        p_clean,
        alpha=certificate_params['alpha']
    )


def calc_p_cert_tight_3d(std: float, x: np.ndarray, x_pert: np.ndarray,
                         p_clean: float, certificate_params: dict) -> tuple[float, float, float]:
    """Evaluates tight certificate for 3D rotation-invariance for pair of clean and perturbed input.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.
            'n_samples_clean': Number of MC samples for finding classification threshold.
            'n_samples_pert: Number of MC samples for estimating perturbed prediction probability.
            'alpha': Significance for used confidence intervals.
            'quadrature_method': quadpy method to use for numerical integration of
                worst-case classifier.
            'quadrature_degree': Degree of the quadpy quadrature method.


    Returns:
        A tuple of three floats, corresponding to lower bound on perturbed prediction probability
        when using 1.) pessimistic confidence bounds, 2.) raw Monte Carlo estimate
        3.) optimistic confidence bounds in our Monte Carlo evaluation procedure from Algorithm 1.
    """
    if certificate_params['preprocess_translation']:
        x = x.copy()
        x_pert = x_pert.copy()
        x = x - x.mean(axis=0)
        x_pert = x_pert - x_pert.mean(axis=0)

    # 1. Worst-case classifier is rotation-invariant w.r.t. to its input
    # 2. Expected prediction of rotation-invariant classifier is invariant w.r.t.
    #    rotation of distribution mean
    # Thus, we can first rotate the perturbed data without changing the result,
    # which allows us to avoid numerical issues with very large / small likelihood ratios.

    x_pert = x_pert.copy()
    x_pert = align_perturbed_pointcloud(x, x_pert)

    return forward_3d_rotation_cert(
        std, x, x_pert,
        certificate_params['n_samples_clean'], certificate_params['n_samples_pert'],
        p_clean,
        quadrature_method=certificate_params['quadrature_method'],
        quadrature_degree=certificate_params['quadrature_degree'],
        alpha=certificate_params['alpha']
    )


def calc_p_cert_tight_elemental(std: float, x: np.ndarray, x_pert: np.ndarray,
                                p_clean: float,
                                certificate_params: dict) -> tuple[float, float, float]:
    """Evaluates tight certificate for invariance to rotation around a single, arbitrary axis.

    This method evaluates the certificate for invariance to rotation around each of the
    three axes separately and then picks the best one.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.
            'n_samples_clean': Number of MC samples for finding classification threshold.
            'n_samples_pert: Number of MC samples for estimating perturbed prediction probability.
            'alpha': Significance for used confidence intervals.

    Returns:
        A tuple of three floats, corresponding to lower bound on perturbed prediction probability
        when using 1.) pessimistic confidence bounds, 2.) raw Monte Carlo estimate
        3.) optimistic confidence bounds in our Monte Carlo evaluation procedure from Algorithm 1.
    """
    if certificate_params['preprocess_translation']:
        x = x.copy()
        x_pert = x_pert.copy()
        x = x - x.mean(axis=0)
        x_pert = x_pert - x_pert.mean(axis=0)

    p_certs = [calc_p_cert_tight_elemental_single_axis(
                    std, x, x_pert, p_clean, certificate_params, axis)
               for axis in [0, 1, 2]]

    return np.max(p_certs, axis=0)


def calc_p_cert_tight_elemental_single_axis(
        std: float, x: np.ndarray, x_pert: np.ndarray,
        p_clean: float, certificate_params: dict, axis: int) -> tuple[float, float, float]:
    """Evaluates tight certificate for invariance to rotation around a single, specific axis.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.
            'n_samples_clean': Number of MC samples for finding classification threshold.
            'n_samples_pert: Number of MC samples for estimating perturbed prediction probability.
            'alpha': Significance for used confidence intervals.
        axis: Number between 0 and 2, indicating the rotation axis.

    Returns:
        A tuple of three floats, corresponding to lower bound on perturbed prediction probability
        when using 1.) pessimistic confidence bounds, 2.) raw Monte Carlo estimate
        3.) optimistic confidence bounds in our Monte Carlo evaluation procedure from Algorithm 1.
    """

    axis_mask = np.zeros(3, dtype='bool')
    axis_mask[axis] = True

    x_rot_axis = x[:, axis_mask]
    x_pert_rot_axis = x_pert[:, axis_mask]
    # norm along rotation axis
    norm_delta_rot_axis = np.linalg.norm(x_pert_rot_axis - x_rot_axis)

    x_others = x[:, ~axis_mask]
    x_pert_others = x_pert[:, ~axis_mask]

    # Parameters for 2D rotation certificate, evaluated on two remaining axes
    delta_others = x_pert_others - x_others
    norm_clean_others = np.linalg.norm(x_others)
    norm_delta_others = np.linalg.norm(x_pert_others - x_others)
    inner_others = np.sum(x_others * delta_others)  # eps_1 from paper
    cross_others = (np.sum(x_others[:, 1] * delta_others[:, 0])
                    - np.sum(x_others[:, 0] * delta_others[:, 1]))  # eps_2 from paper

    return forward_elemental_rotation_cert(
        std,
        norm_delta_rot_axis,
        norm_clean_others, norm_delta_others, inner_others, cross_others,
        certificate_params['n_samples_clean'], certificate_params['n_samples_pert'],
        p_clean,
        alpha=certificate_params['alpha']
    )


def calc_p_cert_baseline(std: float, x: np.ndarray, x_pert: np.ndarray,
                         p_clean: float, certificate_params: dict) -> float:
    """Evaluates black-box randomized smothing for pair of clean and perturbed input.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.

    Returns:
        The lower bound on the perturbed prediction probability obtained via black-box randomized
        smoothing.
    """
    if certificate_params['preprocess_translation']:
        x = x.copy()
        x_pert = x_pert.copy()
        x = x - x.mean(axis=0)
        x_pert = x_pert - x_pert.mean(axis=0)

    norm_delta = np.linalg.norm(x_pert - x)

    return forward_cert_cohen(std, norm_delta, p_clean)


def calc_p_cert_preprocessing(
        std: float, x: np.ndarray, x_pert: np.ndarray,
        p_clean: float, certificate_params: dict) -> float:
    """Evaluates orbit-based certificate for rotation invariance, given clean and perturbed input.

    Args:
        std: Smoothing standard deviation
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D
        p_clean: Clean prediction probability
        certificate_params: Dictionary with certificate parameters.
            if 'preprocess_translation' is True, first subtract column-wise average
            from x and x_pert.

    Returns:
        The lower bound on the perturbed prediction probability obtained via black-box randomized
        smoothing.
    """
    if certificate_params['preprocess_translation']:
        x = x.copy()
        x_pert = x_pert.copy()
        x = x - x.mean(axis=0)
        x_pert = x_pert - x_pert.mean(axis=0)

    x_pert = x_pert.copy()
    x_pert = align_perturbed_pointcloud(x, x_pert)

    norm_delta = np.linalg.norm(x_pert - x)

    return forward_cert_cohen(std, norm_delta, p_clean)


def sample_perturbed_input(
        x: np.ndarray, norm_delta: float, distribution_params: dict) -> np.ndarray:
    """Generates randomly perturbed or adversarially scaled input with given perturbation norm.

    Either performs adversarial scaling or generates a perturbed input by
    1.) adding Gaussian noise rescaled to specified norm
    2.) randomly rotating by specified angle, either around a coordinate axis
    or a randomly sampled axis.

    Args:
        x: Clean input of shape N x D
        norm_delta: Perturbation norm
        distribution_params: Dictionary specifying type of perturbation.
            'parallel': If True, perform adversarial scaling of the input, i.e. no rotation.
                Otherwise, perform random perturbation
            'angle': Angle in degrees for random rotation
            'axis_parallel': If true and D=3, randomly pick a single coordinate axis
            to rotate around. Otherwise sample random rotation axis.

    Returns:
        Perturbed input of shape N x D
    """
    assert x.ndim == 2 and (x.shape[-1] == 2 or x.shape[-1] == 3)

    if distribution_params['parallel']:
        if distribution_params['angle'] != 0:
            raise ValueError('Cant rotate and perform parallel permutation at same time')

        x_pert = generate_parallel_perturbation(x, norm_delta)

    else:
        x_pert = generate_random_perturbation(
                x, norm_delta, distribution_params['angle'], distribution_params['axis_parallel'])

    return x_pert


def generate_parallel_perturbation(x: np.ndarray, norm_delta: float) -> np.ndarray:
    """Generates (random) perturbed input with given data norm by scaling the input

    Args:
        x: Clean input of shape N x D
        norm_delta: Perturbation norm

    Returns:
        Adversarially scaled input of shape N x D
    """
    assert x.shape[-1] == 2

    assert np.linalg.norm(x) > 0

    delta = (x / np.linalg.norm(x)) * norm_delta
    x_pert = x + delta

    return x_pert


def generate_random_perturbation(
        x: np.ndarray, norm_delta: float, angle: float, axis_parallel: bool) -> np.ndarray:
    """Generates randomly perturbed or adversarially scaled input with given perturbation norm.

    Either generates a perturbed input by adding Gaussian noise rescaled to specified norm
    and either rotating by specified angle, either around a coordinate axis
    or a randomly sampled axis.

    Args:
        x: Clean input of shape N x D
        norm_delta: Perturbation norm
        distribution_params: Dictionary specifying type of perturbation.
            'angle': Angle in degrees for random rotation
            'axis_parallel': If true and D=3, randomly pick a single coordinate axis
            to rotate around. Otherwise sample random rotation axis.

    Returns:
        Perturbed input of shape N x D
    """
    delta = np.random.normal(0, 1, x.shape)
    assert np.linalg.norm(delta) > 0
    delta = norm_delta * (delta / np.linalg.norm(delta))  # rescale noise magnitude to norm_delta

    x_pert = x + delta

    if angle > 0:
        R = generate_random_rotation_matrix(x, angle, axis_parallel)
        x_pert = x_pert @ R.T

    return x_pert


def generate_random_rotation_matrix(x: np.ndarray, angle: float, axis_parallel: bool) -> np.ndarray:
    """Generates random rotation matrix with dimensions matching that of given input x.

    In 3D, either randomly samples rotation axis uniformly at random or randomly chooses
    one of the three coordinate axes.

    Args:
        x: Clean input of shape N x D with D=2 or D=3
        angle: Angle in degrees to rotate by
        'axis_parallel': If true and D=3, randomly pick a single coordinate axis
            to rotate around. Otherwise sample random rotation axis.

    Returns:
        Rotation matrix of shape DxD.
    """
    D = x.shape[-1]

    if angle == 0:
        return np.eye(D)

    sign = 2 * np.random.randint(0, 2) - 1

    # Convert from degrees to radians
    angle = sign * angle
    angle = angle / 180 * np.pi

    if D == 2:
        # In 2D, there is only one possible rotation matrix for given angle
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    if D == 3:
        if axis_parallel:
            # Pick one of three rotation axes
            rotation_axis = np.random.randint(0, 3)
            rotation_vector = np.zeros(3)
            rotation_vector[rotation_axis] = angle
        else:
            rotation_vector = np.random.normal(0, 1, 3)
            assert np.linalg.norm(rotation_vector) > 0
            rotation_vector = angle * (rotation_vector / np.linalg.norm(rotation_vector))

        R = Rotation.from_rotvec(rotation_vector).as_matrix()

    return R


def get_norms_delta(threat_model_params: dict) -> np.ndarray:
    """Process threat model parameters from config file to obtain perturbation norms"""

    norms_delta_params = threat_model_params['norms_delta_params']

    if norms_delta_params['linspace']:
        norms_delta = np.linspace(norms_delta_params['min'],
                                  norms_delta_params['max'],
                                  norms_delta_params['steps'])
    else:
        norms_delta = norms_delta_params['values']

    return norms_delta


def align_perturbed_pointcloud(x: np.ndarray, x_pert: np.ndarray) -> np.ndarray:
    """Aligns perturbed point cloud to clean ponit cloud using solution to orthogonal Procrustes

    Args:
        x: Clean input of shape N x D
        x_pert: Perturbed input of shape N x D

    Returns:
        Perturbed input rotated s.t. Frobenius distance to clean input is minimized
    """
    U, _, Vt = np.linalg.svd(x_pert.T @ x)
    V = Vt.T

    s_modified = np.eye(x.shape[-1])
    s_modified[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ s_modified @ U.T

    return x_pert @ R.T
