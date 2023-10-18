import numpy as np

def evaluate_acc(y_true, y_pred):
    """Compute the accuracy. :y_true: is one-hot encoded and :y_pred: may be a probabilistic prediction or one-hot encoded.
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))