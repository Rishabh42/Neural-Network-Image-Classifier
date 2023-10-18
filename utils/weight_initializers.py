import numpy as np


def all_zeros(weights_dim):
    return np.zeros(weights_dim)


def uniform(weights_dim, low=-1, high=1):
    return np.random.uniform(low=low, high=high, size=weights_dim)


def gaussian(weights_dim, mean=0, std=1):
    return np.random.normal(loc=mean, scale=std, size=weights_dim)


def xavier_uniform(weights_dim):
    # ref: https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
    # weight dims should be a tuple of two values (num_inputs, num_outputs)
    limit = np.sqrt(6 / (weights_dim[0] + weights_dim[1]))
    return np.random.uniform(low=-limit, high=limit, size=weights_dim)


def xavier_normal(weights_dim):
    std = np.sqrt(2 / (weights_dim[0] + weights_dim[1]))
    return np.random.normal(loc=0.0, scale=std, size=weights_dim)


def kaiming(weights_dim, mode='fan_in', nonlinearity='relu'):
    # ref: https://medium.com/@shauryagoel/kaiming-he-initialization-a8d9ed0b5899
    fan = np.prod(weights_dim[1:]) if mode == 'fan_in' else np.prod(weights_dim)
    std = np.sqrt(2.0 / fan) if nonlinearity == 'relu' else np.sqrt(1.0 / fan)
    return np.random.normal(loc=0.0, scale=std, size=weights_dim)

