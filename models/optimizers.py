import numpy as np


def evaluate_acc(y_true, y_preds):
    """Computes accuracy of true an predicted labels. Assumes :y_true: and :y_pred: are one-hot encoded vectors."""
    y_true_classes = np.argmax(y_true, axis=1, keepdims=True)
    y_pred_classes = np.argmax(y_preds, axis=1, keepdims=True)

    accuracy_score = np.sum(y_true_classes == y_pred_classes) / y_true_classes.shape[0]
    return accuracy_score


def one_hot(y):
    """one-hot encodes probabilistic predictions"""
    pred_labels = y.argmax(axis=1)
    n_classes = pred_labels.max() + 1
    y_oh = np.eye(n_classes)[pred_labels].reshape(pred_labels.shape[0], -1)

    return y_oh


def softmax(Z):
    """Implement softmax to avoid overflow"""
    eps = 1e-8
    return np.exp(Z - np.max(Z, axis=1, keepdims=True)) / (np.sum(np.exp(Z-np.max(Z, axis=1, keepdims=True)), axis=1,
                                                                  keepdims=True) + eps)


class StochasticGradientDescent:
    """
    Stochastic gradient descent with momentum. Reduces to standard gradient descent when :batch_size: is equal to the
    number of samples in the training set. The parameter :beta: controls the momentum. When :beta: is equal to zero,
    the optimizer runs normal stochastic gradient descent. The StochasticGradientDescent class was modified from the 
    GradientDescent class defined in the gradient descent tutorial:
    https://github.com/rabbanyk/comp551-notebooks/blob/master/GradientDescent.ipynb
    """

    def __init__(self, learning_rate=0.1, n_epochs=100, epsilon=1e-8, batch_size=1, regularization=None, lambd=None, record_loss=False, verbose=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.verbose = verbose
        self.batch_size = batch_size
        self.record_loss = record_loss
        self.loss_history = None
        self.regularization = regularization

        if self.regularization is not None:
            assert lambd is not None, 'Must include regularization parameter.'
            self.lambd = lambd

    def regularization_term(self, param):
        """Regularizes according to regularization parameter"""

        if self.regularization == 'l1':
            return self.lambd * np.sign(param)
        elif self.regularization == 'l2':
            return self.lambd * param

        return 0    # if no regualrization

    def run(self, X, y, w, b, forward_fn, backward_fn, loss_fn):

        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        ix_list = [i for i in range(X.shape[0])]    # possible indices for each mini batch
        n_layers = len(list(w.keys())) + 1

        steps_per_epoch = X.shape[0] // self.batch_size

        if self.record_loss:
            self.loss_history = np.empty(self.n_epochs * steps_per_epoch)

        for epoch in range(int(self.n_epochs)):

            for t in range(steps_per_epoch):
                batch = np.random.choice(ix_list, size=self.batch_size, replace=False)

                z, a = forward_fn(X[batch])
                dz = backward_fn(z, a, y[batch], loss_fn)

                if self.record_loss:
                    self.loss_history[(epoch * steps_per_epoch) + t] = loss_fn.loss(y[batch], a[n_layers - 1])

                # update params
                for i in range(1, n_layers):

                    dw = (np.dot(a[i - 1].T, dz[i]) + self.regularization_term(w[i])) / self.batch_size
                    db = (np.mean(dz[i], axis=0).reshape(-1, 1) + self.regularization_term(b[i])) / self.batch_size

                    w[i] = w[i] - self.learning_rate * dw
                    b[i] = b[i] - self.learning_rate * db

                if self.verbose and (t == 0) and (epoch % 10 == 0):
                    acc = evaluate_acc(y[batch], softmax(a[n_layers-1]))
                    print(f'Epoch {epoch} loss: {loss_fn.loss(y[batch], a[n_layers - 1])}, accuracy: {acc}')

        return w
    
    
class Adam:
    """
    Implements the Adam gradient descent method according to the weight update in slide 35 of:
    https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf
    """

    def __init__(self, learning_rate=0.1, n_epochs=1e5, epsilon=1e-8, batch_size=1, regularization=None, lambd=None, record_loss=False, verbose=True,
                 beta_1=0.9, beta_2=0.9):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.verbose = verbose
        self.batch_size = batch_size
        self.beta_1 = beta_1    # used for moving average of the first moment
        self.beta_2 = beta_2    # used for moving average of the second moment

        self.record_loss = record_loss
        self.loss_history = None

        self.regularization = regularization
        if self.regularization is not None:
            assert lambd is not None, 'Must supply regularization term'
            self.lambd = lambd


    def regularization_term(self, param):
        """Regularizes according to regularization parameter"""

        if self.regularization == 'l1':
            return self.lambd * np.sign(param)
        elif self.regularization == 'l2':
            return self.lambd * param

        return 0    # if no regualrization

    def run(self, X, y, w, b, forward_fn, backward_fn, loss_fn):

        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        ix_list = [i for i in range(X.shape[0])]    # possible indices for each mini batch

        n_layers = len(list(w.keys())) + 1

        steps_per_epoch = X.shape[0] // self.batch_size

        if self.record_loss:
            self.loss_history = np.empty(self.n_epochs * steps_per_epoch)

        prev_M = {}
        prev_S = {}
        prev_M_b = {}
        prev_S_b = {}
        M = {}
        S = {}
        M_b = {}
        S_b = {}

        for i in range(1, n_layers):
            prev_M[i] = np.zeros(w[i].shape)
            prev_S[i] = np.zeros(w[i].shape)
            prev_M_b[i] = np.zeros(b[i].shape)
            prev_S_b[i] = np.zeros(b[i].shape)
            M[i] = np.zeros(w[i].shape)
            S[i] = np.zeros(w[i].shape)
            M_b[i] = np.zeros(b[i].shape)
            S_b[i] = np.zeros(b[i].shape)

        beta_1_t = 1
        beta_2_t = 1
        for epoch in range(int(self.n_epochs)):

            for t in range(steps_per_epoch):
                batch = np.random.choice(ix_list, size=self.batch_size, replace=False)
                # update weights according to the equation in slide 30 of:
                # https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf

                z, a = forward_fn(X[batch])
                dz = backward_fn(z, a, y[batch], loss_fn)

                if self.record_loss:
                    self.loss_history[(epoch * steps_per_epoch) + t] = loss_fn.loss(y[batch], a[n_layers - 1])

                beta_1_t *= self.beta_1
                beta_2_t *= self.beta_2

                # Update weights in each layer
                for i in range(1, n_layers):

                    dw = (np.dot(a[i - 1].T, dz[i]) + self.regularization_term(w[i])) / self.batch_size
                    db = (np.mean(dz[i], axis=0).reshape(-1, 1) + self.regularization_term(b[i])) / self.batch_size

                    # Compute weighted moving average of first and second moment of the cost gradient
                    M[i] = self.beta_1 * prev_M[i] + (1-self.beta_1) * dw
                    S[i] = self.beta_2 * prev_S[i] + (1-self.beta_2) * dw**2
                    M_b[i] = self.beta_1 * prev_M_b[i] + (1-self.beta_1) * db
                    S_b[i] = self.beta_2 * prev_S_b[i] + (1-self.beta_2) * db**2

                    prev_M[i] = M[i]
                    prev_S[i] = S[i]
                    prev_M_b[i] = M_b[i]
                    prev_S_b[i] = S_b[i]

                    M_hat = M[i] / (1 - beta_1_t)
                    S_hat = S[i] / (1 - beta_2_t)
                    M_hat_b = M_b[i] / (1 - beta_1_t)
                    S_hat_b = S_b[i] / (1 - beta_2_t)

                    w[i] = w[i] - (self.learning_rate * M_hat / np.sqrt(S_hat + self.epsilon))
                    b[i] = b[i] - (self.learning_rate * M_hat_b / np.sqrt(S_hat_b + self.epsilon))

                    #if (self.verbose) and (_ == 0) and (epoch % 10 == 0):
                    #    print(f'gradient norm at epoch {epoch}: ', np.linalg.norm(dz[i]))

                if self.verbose and (t == 0) and (epoch % 10 == 0):
                    acc = evaluate_acc(y[batch], softmax(a[n_layers-1]))
                    print(f'Epoch {epoch} loss: {loss_fn.loss(y[batch], a[n_layers - 1])}, accuracy: {acc}')

        if self.record_loss:
            pass

        return w
