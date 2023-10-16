from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true, y_pred):
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred):
        pass


class CrossEntropyLoss(LossFunction):

    def loss(self, y_true, z, eps=1e-8):
        """Compute the negative log likelihood for multi-class classification, given by the equation in slide 37 of:
        https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/4-logisticregression.pdf"""
        zbar = np.max(z, axis=1, keepdims=True)

        loss = - np.trace(np.dot(y_true, z.transpose())) + np.sum(
            zbar + np.log(np.sum(np.exp(z - zbar), axis=1, keepdims=True)))
        return loss

    def gradient(self, y_true, y_pred):
        return y_pred - y_true