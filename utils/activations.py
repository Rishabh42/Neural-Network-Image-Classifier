from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):

    @abstractmethod
    def backward(self, Z: np.ndarray) -> np.ndarray:
        """Computes the value of the activation function at point :Z:"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, Z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the activation at point :Z:"""
        raise NotImplementedError


class Logistic(Activation):

    def __init__(self):
        pass

    def forward(self, Z):

        return 1. / (1 + np.exp(-Z))

    def backward(self, Z):
        l = self.forward(Z)
        return l * (1 - l)


class Tanh(Activation):

    def __init__(self):
        self.logistic = lambda Z: 1. / (1 + np.exp(-Z))
        pass

    def forward(self, Z):
        return 2 * self.logistic(Z) - 1

    def backward(self, Z):
        return 1 - self.logistic(Z)**2


class ReLU(Activation):

    def __init__(self):
        pass

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z):
        A = np.copy(Z)
        A[Z <= 0] = 0
        A[Z > 0] = 1
        return A


class Softmax(Activation):
    def __init__(self):
        pass

    def forward(self, z):
        """Implement softmax to avoid overflow"""
        eps = 1e-8
        return np.exp(z - np.max(z, axis=1, keepdims=True)) / (
                np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1,
                       keepdims=True) + eps)

    def backward(self, z):
        return self.forward(z) * self.forward(1 - z)
