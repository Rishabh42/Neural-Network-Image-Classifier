import numpy as np

def logistic_function(Z: np.ndarray) -> np.ndarray:
    """Computes the logistic of Z, handling large inputs by avoiding numerical overflow."""
    return 1. / (1 + np.exp(-np.clip(Z, -500, 500)))

class Activation:
    """Base class for activation functions."""

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """Computes the value of the activation function at point :Z:"""
        raise NotImplementedError

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the activation at point :Z:"""
        raise NotImplementedError


class Logistic(Activation):

    def forward(self, Z):

        return 1. / (1 + np.exp(-Z))

    def backward(self, Z):
        l = self.forward(Z)
        return l * (1 - l)


class Tanh(Activation):

    def forward(self, Z):
        return 2 * self.logistic(Z) - 1

    def backward(self, Z):
        return 1 - self.logistic(Z)**2


class ReLU(Activation):

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, Z):
        A = np.copy(Z)
        A[Z <= 0] = 0
        A[Z > 0] = 1
        return A


class Softmax(Activation):

    def forward(self, z):
        """Implement softmax to avoid overflow"""
        eps = 1e-8
        return np.exp(z - np.max(z, axis=1, keepdims=True)) / (
                np.sum(np.exp(z - np.max(z, axis=1, keepdims=True)), axis=1,
                       keepdims=True) + eps)

    def backward(self, z):
        return self.forward(z) * self.forward(1 - z)
