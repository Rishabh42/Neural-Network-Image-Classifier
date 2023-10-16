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

    def __init__(self, eps=1e-8):
        self.eps = eps

    def loss(self, y_true, y_pred):
        """Compute the negative log likelihood for multi-class classification, given by the equation in slide 37 of:
        https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/4-logisticregression.pdf"""
        
        zbar = np.max(y_pred, axis=1, keepdims=True)
        loss = -np.trace(np.dot(y_true, y_pred.transpose())) + np.sum(
            zbar + np.log(np.sum(np.exp(y_pred - zbar), axis=1, keepdims=True)) + self.eps)
        return loss

    def gradient(self, y_true, y_pred):
        return y_pred - y_true