import numpy as np
from abc import ABC, abstractmethod

def logistic_function(Z: np.ndarray) -> np.ndarray:
    """Computes the logistic of Z, handling large inputs by avoiding numerical overflow."""
    return 1. / (1 + np.exp(-np.clip(Z, -500, 500)))

class ActivationFunction(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Computes the value of the activation function at point :Z:"""
        raise NotImplementedError

    @abstractmethod
    def gradient(self, Z: np.ndarray) -> np.ndarray:
        """Computes the gradient of the activation at point :Z:"""
        raise NotImplementedError


class Logistic(ActivationFunction):

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return logistic_function(Z)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        l = self.forward(Z)
        return l * (1 - l)


class Tanh(ActivationFunction):

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return 2 * logistic_function(2 * Z) - 1

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return 1 - np.square(self.forward(Z))


class ReLU(ActivationFunction):

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def gradient(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(int)


class Softmax(ActivationFunction):

    def forward(self, Z: np.ndarray) -> np.ndarray:
        Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return Z_exp / np.sum(Z_exp, axis=1, keepdims=True)

    def gradient(self, Z):
        #return self.forward(z) * self.forward(1 - z)
        # usually you compute the gradient of the loss function with respect to the inputs of the Softmax function during backprop
        raise NotImplementedError
