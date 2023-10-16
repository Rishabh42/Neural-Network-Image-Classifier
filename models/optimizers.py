import numpy as np


def evaluate_acc(y_true, y_preds_probs):
    """Computes accuracy of true an predicted labels. Assumes :y_true: and :y_pred: are one-hot encoded vectors."""
    y_pred_labels = np.argmax(y_preds_probs, axis=1)

    # check if y_true is one-hot encoded and in cas convert to label encoding
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    accuracy = np.mean(y_pred_labels == y_true)
    return accuracy


class StochasticGradientDescent:
    """
    Stochastic gradient descent with momentum. Reduces to standard gradient descent when :batch_size: is equal to the
    number of samples in the training set. The parameter :beta: controls the momentum. When :beta: is equal to zero,
    the optimizer runs normal stochastic gradient descent. The StochasticGradientDescent class was modified from the 
    GradientDescent class defined in the gradient descent tutorial:
    https://github.com/rabbanyk/comp551-notebooks/blob/master/GradientDescent.ipynb
    """

    def __init__(self, learning_rate=0.1, n_epochs=100, epsilon=1e-8, batch_size=1, momentum=0.0, 
                 regularization=None, lambd=None, record_loss=False, verbose=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.verbose = verbose
        self.batch_size = batch_size
        self.record_loss = record_loss
        self.loss_history = [] if record_loss else None
        self.regularization = regularization
        self.lambd = lambd if regularization else 0

        self.momentum = momentum # reduces to standard gradient descent when momentum = 0
        self.velocity_w = None
        self.velocity_b = None

    def regularization_term(self, param):
        """Regularizes according to regularization parameter"""

        if self.regularization == 'l1':
            return self.lambd * np.sign(param)
        elif self.regularization == 'l2':
            return self.lambd * param

        return 0    # if no regualrization
    
    def update_params(self, w, b, dw, db, layer):
        """updates params with momentum"""
        if self.velocity_w is None or self.velocity_b is None:
            self.velocity_w, self.velocity_b = {}, {}
            for l in w.keys():
                self.velocity_w[l] = np.zeros_like(w[l])
                self.velocity_b[l] = np.zeros_like(b[l])

        # update velocity   
        self.velocity_w[layer] = self.momentum * self.velocity_w[layer] - self.learning_rate * dw
        self.velocity_b[layer] = self.momentum * self.velocity_b[layer] - self.learning_rate * db

        # update params
        w[layer] += self.velocity_w[layer]
        b[layer] += self.velocity_b[layer]
        
        return w, b

    def run(self, X, y, w, b, forward_fn, backward_fn, loss_fn):

        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        n_layers = len(w) + 1  # assuming w is a dictionary of weights per layer
        steps_per_epoch = X.shape[0] // self.batch_size

        for epoch in range(self.n_epochs):
            for t in range(steps_per_epoch):
                batch_idc = np.random.choice(X.shape[0], self.batch_size, replace=False)
                X_batch, y_batch = X[batch_idc], y[batch_idc]

                # fwd and bwd pass
                z, a = forward_fn(X_batch)
                dz = backward_fn(z, a, y_batch, loss_fn)

                if self.record_loss:
                    current_loss = loss_fn.loss(y_batch, a[n_layers - 1])
                    self.loss_history.append(current_loss)

                # update params
                for i in range(1, n_layers):

                    dw = (np.dot(a[i - 1].T, dz[i]) + self.regularization_term(w[i])) / self.batch_size
                    db = np.mean(dz[i], axis=0, keepdims=True) + self.regularization_term(b[i])

                    self.update_params(w, b, dw, db, i)

                if self.verbose and (t == 0) and (epoch % 10 == 0):
                    acc = evaluate_acc(y_batch, a[n_layers-1])
                    print(f'Epoch {epoch} loss: {current_loss}, accuracy: {acc}')

        return w, self.loss_history if self.record_loss else None
    
    
class Adam:
    """
    Implements the Adam gradient descent method according to the weight update in slide 35 of:
    https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf
    """
        
    def __init__(self, learning_rate=0.1, n_epochs=1000, epsilon=1e-8, batch_size=1, 
                 beta_1=0.9, beta_2=0.999, regularization=None, lambd=0.01, record_loss=False, verbose=False):
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.beta_1 = beta_1    # used for moving average of the first moment
        self.beta_2 = beta_2    # used for moving average of the second moment

        self.verbose = verbose
        self.record_loss = record_loss
        self.loss_history = [] if record_loss else None

        self.regularization = regularization
        self.lambd = lambd if regularization else 0


    def regularization_term(self, param):
        """Regularizes according to regularization parameter"""

        if self.regularization == 'l1':
            return self.lambd * np.sign(param)
        elif self.regularization == 'l2':
            return self.lambd * param

        return 0    # if no regualrization
    
    def initialize_moments(self, w, b):
        """init the first and second moment for w and b"""
        moments = {}
        for i in range(1, len(w) + 1):
            moments[i] = {'M': np.zeros_like(w[i]), 'S': np.zeros_like(w[i]),
                          'M_b': np.zeros_like(b[i]), 'S_b': np.zeros_like(b[i])}
        return moments
    
    def update_moments(self, moments, dw, db, layer_index):
        """Updates the moment estimates with new gradients"""
        moments[layer_index]['M'] = (self.beta_1 * moments[layer_index]['M']) + (1 - self.beta_1) * dw
        moments[layer_index]['S'] = (self.beta_2 * moments[layer_index]['S']) + (1 - self.beta_2) * (dw ** 2)
        moments[layer_index]['M_b'] = (self.beta_1 * moments[layer_index]['M_b']) + (1 - self.beta_1) * db
        moments[layer_index]['S_b'] = (self.beta_2 * moments[layer_index]['S_b']) + (1 - self.beta_2) * (db ** 2)

    def update_params(self, w, b, moments, t):
        """Updates weights and biases using the moment estimates."""
        # compute time step dependent learning rate
        lr_t = self.learning_rate * (np.sqrt(1 - self.beta_2 ** t) / (1 - self.beta_1 ** t))
        for i in range(1, len(w) + 1):
            w[i] -= lr_t * moments[i]['M'] / (np.sqrt(moments[i]['S']) + self.epsilon) + (self.learning_rate * self.regularization_term(w[i]))
            b[i] -= lr_t * moments[i]['M_b'] / (np.sqrt(moments[i]['S_b']) + self.epsilon)


    def run(self, X, y, w, b, forward_fn, backward_fn, loss_fn):

        assert self.batch_size <= X.shape[0], f'Error, batch size must be smaller than {X.shape[0]}'
        n_layers = len(w) + 1  # assuming w is a dictionary of weights per layer
        steps_per_epoch = X.shape[0] // self.batch_size

        moments = self.initialize_moments(w, b)

        for epoch in range(self.n_epochs):
            for t in range(steps_per_epoch):
                # note: random sampling without replacement does not make sure that all samples are used
                batch_idc = np.random.choice(X.shape[0], self.batch_size, replace=False) 
                X_batch, y_batch = X[batch_idc], y[batch_idc]
                # update weights according to the equation in slide 30 of:
                # https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/5-gradientdescent.pdf

                z, a = forward_fn(X_batch, w, b)
                dz = backward_fn(a, z, w, X_batch, y_batch, loss_fn)

                for layer_idx, (dw, db) in dz.items():
                    self.update_moments(moments, dw, db, layer_idx)

                self.update_params(w, b, moments, epoch*steps_per_epoch + t + 1)

                if self.record_loss:
                    current_loss = loss_fn.loss(y_batch, a[-1])
                    self.loss_history.append(current_loss)

                if self.verbose and (t == 0) and (epoch % 10 == 0):
                    acc = evaluate_acc(y_batch, a[-1])
                    print(f'Epoch {epoch} loss: {current_loss}, accuracy: {acc}')

        return w, self.loss_history if self.record_loss else None
