import numpy as np


def all_zeros(weight_dimensions):
    return np.zeros(weight_dimensions)


def uniform(weight_dimensions, low=-1, high=1):
    return np.random.uniform(low=low, high=high, size=weight_dimensions)


def gaussian(weight_dimensions, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=weight_dimensions)


def xavier(weight_dimensions, low=-1, high=1):
    return np.random.uniform(low=low, high=high, size=weight_dimensions) * np.sqrt(1/weight_dimensions[0])


def kaiming(weight_dimensions):
    return np.random.normal(loc=0.0, scale=np.sqrt(2/weight_dimensions[0]), size=weight_dimensions)

