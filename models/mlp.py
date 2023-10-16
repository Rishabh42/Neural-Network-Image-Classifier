import numpy as np

class MLP:
    def __init__(self, layer_sizes, act_fn, loss_fn, softmax_fn, weight_initializer):
        self.layer_sizes = layer_sizes
        self.act_fn = act_fn
        self.loss_fn = loss_fn
        self.weight_initializer = weight_initializer
        self.softmax = softmax_fn

        # init weights and biases
        self.w = [self.weight_initializer((layer_sizes[i], layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)]

    def forward(self, X):
        a = [X] # input layer
        
        for i in range(len(self.w)-1): # hidden layers
            z = a[-1] @ self.w[i] + self.b[i]
            a.append(self.act_fn.forward(z))
        
        z = a[-1] @ self.w[-1] + self.b[-1] # output layer
        a.append(self.softmax.forward(z))
        return a

    def backward(self, X, y, a):    
        m = X.shape[0] # for averaging
        grads = {}
        output_error = self.loss_fn.gradient(y, a[-1])
        da = output_error

        for i in reversed(range(len(self.w))):
            dz = da * self.act_fn.gradient(a[i+1]) if i < len(self.w)-1 else da # da for output layer
            grads[f'W{i+1}'] = a[i].T @ dz / m
            grads[f'b{i+1}'] = np.sum(dz, axis=0, keepdims=True) / m
            da = dz @ self.w[i].T # derivative of loss wrt a[i]

        return grads

    def fit(self, X, y, optimizer, epochs=1000, batch_size=32, print_every=10):
        num_samples = X.shape[0]
        accuracy = []

        for epoch in range(epochs):
            permuted_indices = np.random.permutation(num_samples)
            X_shuffled = X[permuted_indices]
            y_shuffled = y[permuted_indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                a = self.forward(X_batch)
                loss = self.loss_fn.loss(y_batch, a[-1])
                grads = self.backward(X_batch, y_batch, a)
                optimizer.update(self.w, self.b, grads)

            if epoch % print_every == 0:
                acc = self.evaluate_acc(y, self.predict(X))
                accuracy.append(acc)
                print(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        return accuracy

    def predict(self, X):
        a = self.forward(X)
        y_preds = np.argmax(a[-1], axis=1)
        return y_preds

    def evaluate_acc(self, y_true, y_pred):
        # Convert one-hot encoded y_true to class labels
        y_true_labels = np.argmax(y_true, axis=1)
        
        correct_preds = np.sum(y_true_labels == y_pred)
        return correct_preds / y_true.shape[0]