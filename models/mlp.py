import numpy as np

def softmax(X):
    """Implement softmax to avoid overflow"""
    eps = 1e-8
    return np.exp(X - np.max(X, axis=1, keepdims=True)) / (np.sum(np.exp(X-np.max(X, axis=1, keepdims=True)), axis=1,
                                                                  keepdims=True) + eps)

def one_hot(y):
    """one-hot encodes probabilistic predictions"""
    y_one_hot = np.zeros(y.shape)
    pred_labels = y.argmax(axis=1)
    for i in range(y.shape[0]):
        y_one_hot[i, pred_labels[i]] = 1
    return y_one_hot


class MLP:
    def __init__(self, layer_sizes, activations, weight_initializer):
        self.n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes

        # Initialize weights, biases and activations
        self.w = {}
        self.b = {}
        self.activations = {}

        for i in range(1, self.n_layers):
            self.w[i] = weight_initializer((layer_sizes[i - 1], layer_sizes[i]))
            self.b[i] = weight_initializer((layer_sizes[i], 1))

            self.activations[i] = activations[i-1]

    def forward(self, X, w=None, b=None):
        """Compute the activation and hidden layer values"""
        # z_i = (w_i . a_{i-1}) + b_i
        z = {}

        # activation: f_i(z)
        a = {0: X}

        for i in range(1, self.n_layers):
            if w is not None:
                assert b is not None, 'Must supply both weights and biases.'
                z[i] = np.dot(a[i-1], w[i]) + b[i].squeeze()
            else:
                z[i] = np.dot(a[i - 1], self.w[i]) + self.b[i].squeeze()

            a[i] = self.activations[i].forward(z[i])

        return z, a

    def backward(self, z, a, y, loss_fn):
        """Computes propagated error for each parameter"""
        error = loss_fn.gradient(y, a[self.n_layers - 1])

        # dJ / dz
        dz = {}

        for i in reversed(range(1, self.n_layers)):
            # Compute error wrt. dZ_{i}
            dz[i] = error * self.activations[i].backward(z[i])
            error = np.dot(dz[i], self.w[i].T)

        return dz

    def fit(self, X, y, optimizer, loss_fn):

        optimizer.run(X, y, self.w, self.b, self.forward, self.backward, loss_fn)

        return optimizer.w_history

    def predict(self, X, w=None, b=None):

        if w is not None:
            assert b is not None, 'Must supply w and b'
            _, a = self.forward(X, w=w, b=b)
        else:
            _, a = self.forward(X)

        y_preds = one_hot(a[self.n_layers - 1])
        return y_preds

