"""This module contains a few helper functions for use by the scripts in seml/scripts"""

import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dict_to_dot(x: dict):
    """Converts nested dict to list of tuples using dot notation.

    The second entry of each tuple is a value from the lowest level of a nested dictionary.
    The first entry of each tuple are the keys leading to this value, in dot notation."""

    ret = []

    for key, value in x.items():
        assert isinstance(key, str)
        if isinstance(value, dict):
            child_ret = dict_to_dot(value)
            ret.extend([
                (f'{key}.{child_dotstring}', child_value)
                for child_dotstring, child_value in child_ret
            ])
        else:
            ret.append((key, value))

    return ret
