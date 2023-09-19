import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        raise Exception("ERROR: Noise type {} not supported".format(type))
    return noise
